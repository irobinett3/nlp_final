#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# training_pipeline/pipeline.py

"""
Main orchestration for risk detection training pipeline - OPTIMIZED VERSION

Coordinates:
1. Multi-source labeling (FinBERT + Phrasebank + LLM)
2. BIO tagging for span detection
3. Historical ground truth integration
4. LLM baseline evaluation

Each run saves to a unique timestamped directory.
"""

import sys
import os
import sqlite3
import time
import multiprocessing
import json
import warnings
from datetime import datetime
from typing import Optional, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .config import DB_PATH, HIGH_QUALITY_CONFIDENCE, MEDIUM_QUALITY_CONFIDENCE, get_run_directory
from .labelers import MultiSourceLabeler
from .bio_tagger import BIOTagger
from .ground_truth_loader import load_ground_truth_documents
from .llm_evaluator import LLMEvaluator
from .data_saver import save_all_datasets
from .report_generator import generate_report

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Global model cache for workers
_LABELER_CACHE = None
_BIO_TAGGER_CACHE = None

def _get_labeler():
    """Get or create cached labeler instance"""
    global _LABELER_CACHE
    if _LABELER_CACHE is None:
        from training_pipeline.labelers import MultiSourceLabeler
        _LABELER_CACHE = MultiSourceLabeler()
    return _LABELER_CACHE

def _get_bio_tagger():
    """Get or create cached BIO tagger instance"""
    global _BIO_TAGGER_CACHE
    if _BIO_TAGGER_CACHE is None:
        from training_pipeline.bio_tagger import BIOTagger
        _BIO_TAGGER_CACHE = BIOTagger()
    return _BIO_TAGGER_CACHE

# Module-level functions for multiprocessing (must be picklable)
def _label_single_doc_worker(doc):
    """Worker function for parallel document labeling"""
    try:
        labeler = _get_labeler()
        
        labels = labeler.label_document(
            doc['raw_text'],
            metadata={
                'source': doc.get('source'),
                'company': doc.get('company'),
                'published_at': doc.get('published_at')
            }
        )
        return {**doc, 'labels': labels}
    except Exception as e:
        # Silent failure - just return None
        return None

def _tag_single_doc_worker(doc):
    """Worker function for parallel BIO tagging"""
    try:
        bio_tagger = _get_bio_tagger()
        labeler = _get_labeler()
        
        bio_result = bio_tagger.generate_bio_tags(
            doc['raw_text'],
            finbert_labeler=labeler.finbert
        )
        return {**doc, 'bio_tags': bio_result}
    except Exception as e:
        # Silent failure - just return None
        return None

class RiskTrainingPipeline:
    """Main pipeline orchestrator - OPTIMIZED"""
    
    def __init__(self, company: Optional[str] = None, ticker: Optional[str] = None):
        print("="*80)
        print("RISK DETECTION TRAINING DATA GENERATION PIPELINE (OPTIMIZED)")
        print("="*80)
        
        # Generate unique output directory for this run
        self.output_dir = get_run_directory(company, ticker)
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"   All datasets will be saved here")
        
        # Initialize components (only once in main process)
        print(f"\n🔧 Initializing components...")
        self.labeler = MultiSourceLabeler()
        self.bio_tagger = BIOTagger()
        self.llm_evaluator = LLMEvaluator()
        print(f"   ✓ Components ready")
    
    def run(
        self,
        company: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 500,
        skip_llm_eval: bool = True  # Changed default to True for speed
    ) -> Dict:
        """
        Execute full pipeline
        
        Args:
            company: Filter by company name
            ticker: Filter by ticker symbol
            limit: Max documents to process
            skip_llm_eval: Skip LLM baseline evaluation (default: True for speed)
            
        Returns:
            Dictionary with all datasets and metrics
        """
        
        pipeline_start = time.time()
        
        # Step 1: Load documents
        step_start = time.time()
        documents = self._load_documents(company, ticker, limit)
        print(f"  ⏱️  Step 1 completed in {time.time() - step_start:.1f}s")
        
        # Step 2: Multi-source labeling (PARALLEL)
        step_start = time.time()
        labeled_data = self._label_documents(documents)
        print(f"  ⏱️  Step 2 completed in {time.time() - step_start:.1f}s")
        
        # Step 3: BIO tagging (PARALLEL)
        step_start = time.time()
        bio_tagged_data = self._generate_bio_tags(documents)
        print(f"  ⏱️  Step 3 completed in {time.time() - step_start:.1f}s")
        
        # Step 4: Load ground truth
        step_start = time.time()
        ground_truth_data = self._load_ground_truth()
        print(f"  ⏱️  Step 4 completed in {time.time() - step_start:.1f}s")
        
        # Step 5: Create train/val/test splits
        step_start = time.time()
        datasets = self._create_splits(labeled_data, bio_tagged_data, ground_truth_data)
        print(f"  ⏱️  Step 5 completed in {time.time() - step_start:.1f}s")
        
        # Step 6: LLM baseline evaluation (OPTIONAL - expensive)
        if not skip_llm_eval:
            step_start = time.time()
            datasets['llm_baseline'] = self._evaluate_llm_baseline(datasets)
            print(f"  ⏱️  Step 6 completed in {time.time() - step_start:.1f}s")
        else:
            datasets['llm_baseline'] = {}
            print(f"  ⏭️  Step 6 skipped (use --no-skip-llm to enable)")
        
        # Step 7: Save datasets
        step_start = time.time()
        save_all_datasets(datasets, output_dir=self.output_dir)
        print(f"  ⏱️  Step 7 completed in {time.time() - step_start:.1f}s")
       
        # Step 8: Generate report
        step_start = time.time()
        report = generate_report(datasets, output_dir=self.output_dir)
        print(f"  ⏱️  Step 8 completed in {time.time() - step_start:.1f}s")

        # Step 9: Save run metadata
        step_start = time.time()
        self._save_run_metadata(company, ticker, limit, datasets, pipeline_start)
        print(f"  ⏱️  Step 9 completed in {time.time() - step_start:.1f}s")
        
        # Summary
        self._print_summary(datasets, pipeline_start)
        
        return datasets
    
    def _load_documents(
        self,
        company: Optional[str],
        ticker: Optional[str],
        limit: int
    ) -> List[Dict]:
        """Load documents from database"""
        
        print(f"\n[Step 1/9] Loading documents from database...")
        
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cursor = con.cursor()
        
        # Only fetch needed columns and filter short texts
        query = """
            SELECT id, raw_text, source, company, ticker, published_at 
            FROM documents 
            WHERE raw_text IS NOT NULL 
            AND length(raw_text) > 100
        """
        params = []
        
        if company:
            query += " AND company LIKE ?"
            params.append(f"%{company}%")
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        documents = [dict(row) for row in cursor.fetchall()]
        con.close()
        
        print(f"  ✓ Loaded {len(documents)} documents")
        
        return documents
    
    def _label_documents(self, documents: List[Dict]) -> List[Dict]:
        """Apply multi-source labeling with parallel processing"""
        
        print(f"\n[Step 2/9] Multi-source labeling (parallel)...")
        
        # Determine optimal worker count
        max_workers = min(multiprocessing.cpu_count() - 1, len(documents), 8)
        max_workers = max(1, max_workers)
        
        print(f"  Using {max_workers} parallel workers...")
        print(f"  Processing {len(documents)} documents...")
        
        labeled_data = []
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(_label_single_doc_worker, doc): doc 
                          for doc in documents}
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(documents), desc="Labeling"):
                    result = future.result()
                    if result is not None:
                        labeled_data.append(result)
        
        except Exception as e:
            print(f"  ⚠️ Parallel processing error: {e}")
            print(f"  Falling back to sequential processing...")
            
            # Fallback to sequential
            for doc in tqdm(documents, desc="Labeling (sequential)"):
                result = _label_single_doc_worker(doc)
                if result:
                    labeled_data.append(result)
        
        # Quality filtering
        high_quality = [
            d for d in labeled_data 
            if d.get('labels', {}).get('confidence', 0) >= HIGH_QUALITY_CONFIDENCE
        ]
        medium_quality = [
            d for d in labeled_data 
            if MEDIUM_QUALITY_CONFIDENCE <= d.get('labels', {}).get('confidence', 0) < HIGH_QUALITY_CONFIDENCE
        ]
        
        print(f"  ✓ Successfully labeled: {len(labeled_data)}/{len(documents)} documents")
        print(f"  ✓ High quality (high agreement): {len(high_quality)}")
        print(f"  ✓ Medium quality: {len(medium_quality)}")
        
        return labeled_data
    
    def _generate_bio_tags(self, documents: List[Dict]) -> List[Dict]:
        """Generate BIO tags for span detection with parallel processing"""
        
        print(f"\n[Step 3/9] BIO tagging for span detection (parallel)...")
        
        # Limit for efficiency (you can increase this if needed)
        sample_size = min(200, len(documents))
        sample_docs = documents[:sample_size]
        
        print(f"  Processing {sample_size} documents for BIO tagging...")
        
        # Determine optimal worker count
        max_workers = min(multiprocessing.cpu_count() - 1, sample_size, 8)
        max_workers = max(1, max_workers)
        
        print(f"  Using {max_workers} parallel workers...")
        
        bio_tagged_data = []
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(_tag_single_doc_worker, doc): doc 
                          for doc in sample_docs}
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(sample_docs), desc="BIO tagging"):
                    result = future.result()
                    if result is not None:
                        bio_tagged_data.append(result)
        
        except Exception as e:
            print(f"  ⚠️ Parallel processing error: {e}")
            print(f"  Falling back to sequential processing...")
            
            for doc in tqdm(sample_docs, desc="BIO tagging (sequential)"):
                result = _tag_single_doc_worker(doc)
                if result:
                    bio_tagged_data.append(result)
        
        total_sentences = sum(
            d.get('bio_tags', {}).get('total_sentences', 0) 
            for d in bio_tagged_data
        )
        print(f"  ✓ Successfully tagged: {len(bio_tagged_data)}/{len(sample_docs)} documents")
        print(f"  ✓ Generated BIO tags for {total_sentences} sentences")
        
        return bio_tagged_data
    
    def _load_ground_truth(self) -> List[Dict]:
        """Load historical ground truth"""
        
        print(f"\n[Step 4/9] Loading historical ground truth...")
        
        try:
            ground_truth = load_ground_truth_documents()
            print(f"  ✓ Loaded {len(ground_truth)} ground truth examples")
            return ground_truth
        except Exception as e:
            print(f"  ⚠️ No ground truth data available: {e}")
            return []
    
    def _create_splits(
        self,
        labeled_data: List[Dict],
        bio_tagged_data: List[Dict],
        ground_truth_data: List[Dict]
    ) -> Dict:
        """Create train/val/test splits"""
        
        print(f"\n[Step 5/9] Creating train/val/test splits...")
        
        # Classification dataset - combine ground truth with high quality labels
        high_quality = [
            d for d in labeled_data 
            if d.get('labels', {}).get('confidence', 0) >= HIGH_QUALITY_CONFIDENCE
        ]
        
        if ground_truth_data:
            all_classification = ground_truth_data + high_quality
            print(f"  ✓ Using {len(ground_truth_data)} ground truth + {len(high_quality)} labeled examples")
        else:
            all_classification = high_quality
            print(f"  ✓ Using {len(high_quality)} labeled examples")
        
        if not all_classification:
            print(f"  ⚠️ No classification data available!")
            return {
                'classification': {'train': [], 'val': [], 'test': []},
                'span_detection': {'train': [], 'val': [], 'test': []},
                'attributions': []
            }
        
        # Check class distribution
        from collections import Counter
        class_counts = Counter(d['labels']['risk_category'] for d in all_classification)
        print(f"  Class distribution: {dict(class_counts)}")
        
        # Check if we can stratify
        can_stratify = len(all_classification) >= 3 and all(count >= 2 for count in class_counts.values())
        
        # Split classification data
        if len(all_classification) < 3:
            print(f"  ⚠️ Too few examples for splits, using all for training")
            train_clf, val_clf, test_clf = all_classification, [], []
        else:
            try:
                stratify_param = [d['labels']['risk_category'] for d in all_classification] if can_stratify else None
                
                train_clf, temp = train_test_split(
                    all_classification,
                    test_size=0.3,
                    random_state=42,
                    stratify=stratify_param
                )
                
                if len(temp) >= 2:
                    temp_stratify = [d['labels']['risk_category'] for d in temp] if can_stratify else None
                    val_clf, test_clf = train_test_split(
                        temp,
                        test_size=0.5,
                        random_state=42,
                        stratify=temp_stratify
                    )
                else:
                    val_clf, test_clf = temp, []
                    
            except Exception as e:
                print(f"  ⚠️ Split error, using simple split: {e}")
                train_clf, temp = train_test_split(all_classification, test_size=0.3, random_state=42)
                val_clf, test_clf = train_test_split(temp, test_size=0.5, random_state=42) if len(temp) >= 2 else (temp, [])
        
        # Split BIO tagged data
        if len(bio_tagged_data) >= 3:
            train_bio, temp_bio = train_test_split(bio_tagged_data, test_size=0.3, random_state=42)
            val_bio, test_bio = train_test_split(temp_bio, test_size=0.5, random_state=42) if len(temp_bio) >= 2 else (temp_bio, [])
        else:
            train_bio, val_bio, test_bio = bio_tagged_data, [], []
        
        print(f"  ✓ Classification: {len(train_clf)} train, {len(val_clf)} val, {len(test_clf)} test")
        print(f"  ✓ BIO tagging: {len(train_bio)} train, {len(val_bio)} val, {len(test_bio)} test")
        
        return {
            'classification': {'train': train_clf, 'val': val_clf, 'test': test_clf},
            'span_detection': {'train': train_bio, 'val': val_bio, 'test': test_bio},
            'attributions': []
        }
    
    def _evaluate_llm_baseline(self, datasets: Dict) -> Dict:
        """Evaluate LLM baseline"""
        
        print(f"\n[Step 6/9] LLM baseline evaluation...")
        
        test_set = (
            datasets['classification']['test'][-30:] if datasets['classification']['test'] 
            else datasets['classification']['train'][-30:]
        )
        
        if not test_set:
            print("  ⚠️ No test data available for LLM evaluation")
            return {}
        
        print(f"  Evaluating on {len(test_set)} examples...")
        return self.llm_evaluator.evaluate_baseline(test_set)
    
    def _save_run_metadata(self, company: Optional[str], ticker: Optional[str], 
                           limit: int, datasets: Dict, pipeline_start: float):
        """Save metadata about this run"""
        
        total_time = time.time() - pipeline_start
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'company': company,
            'ticker': ticker,
            'limit': limit,
            'database': DB_PATH,
            'output_dir': self.output_dir,
            'execution_time_seconds': round(total_time, 2),
            'execution_time_minutes': round(total_time / 60, 2),
            'statistics': {
                'total_documents': len(datasets['classification']['train']) + 
                                 len(datasets['classification']['val']) + 
                                 len(datasets['classification']['test']),
                'train_examples': len(datasets['classification']['train']),
                'val_examples': len(datasets['classification']['val']),
                'test_examples': len(datasets['classification']['test']),
                'bio_sentences': sum(
                    d.get('bio_tags', {}).get('total_sentences', 0) 
                    for d in datasets['span_detection']['train']
                )
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'run_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[Step 9/9] Saved run metadata to {metadata_path}")
    
    def _print_summary(self, datasets: Dict, pipeline_start: float):
        """Print summary statistics"""
        
        total_time = time.time() - pipeline_start
        
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETE!")
        print("="*80)
        
        clf = datasets['classification']
        bio = datasets['span_detection']
        llm = datasets.get('llm_baseline', {})
        attr = datasets.get('attributions', [])
        
        print(f"\n⏱️  Total execution time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        
        print(f"\n📁 All outputs saved to: {self.output_dir}")
        print(f"   - training_report.md")
        print(f"   - run_metadata.json")
        print(f"   - classification/*.jsonl")
        print(f"   - span_detection/*.conll")
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Classification train: {len(clf['train'])} examples")
        print(f"   Classification val: {len(clf['val'])} examples")
        print(f"   Classification test: {len(clf['test'])} examples")
        print(f"   Span detection train: {len(bio['train'])} documents")
        print(f"   Attribution examples: {len(attr)}")
        
        if llm and llm.get('best_configuration'):
            best = llm['best_configuration']
            print(f"   LLM baseline: F1={best.get('f1', 0):.3f}, cost=${best.get('cost_per_doc', 0):.4f}/doc")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Review {self.output_dir}/training_report.md")
        print(f"   2. Train models using the generated datasets")
        print(f"   3. Compare fine-tuned model vs LLM baseline")


def main():
    """CLI entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive training data for risk detection (OPTIMIZED)"
    )
    parser.add_argument(
        '--company',
        help='Filter by company name',
        default=None
    )
    parser.add_argument(
        '--ticker',
        help='Filter by ticker symbol',
        default=None
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=500,
        help='Max documents to process (default: 500)'
    )
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        default=True,
        help='Skip LLM baseline evaluation (default: True for speed)'
    )
    parser.add_argument(
        '--no-skip-llm',
        action='store_true',
        help='Enable LLM baseline evaluation (slow and expensive)'
    )
    
    args = parser.parse_args()
    
    # Handle the no-skip-llm flag
    skip_llm = args.skip_llm and not args.no_skip_llm
    
    # Check API keys
    if not skip_llm:
        from .config import OPENAI_API_KEY
        if not OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY not set. LLM features will be limited.")
            print("   Set environment variable or use --skip-llm flag")
            skip_llm = True
    
    # Run pipeline
    try:
        print(f"\n🚀 Starting optimized pipeline...")
        print(f"   Limit: {args.limit} documents")
        print(f"   LLM evaluation: {'ENABLED' if not skip_llm else 'DISABLED (use --no-skip-llm to enable)'}")
        
        pipeline = RiskTrainingPipeline(company=args.company, ticker=args.ticker)
        datasets = pipeline.run(
            company=args.company,
            ticker=args.ticker,
            limit=args.limit,
            skip_llm_eval=skip_llm
        )
        
        print("\n✓ Success! Training data generated.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()