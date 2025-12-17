#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# training_pipeline/fast_pipeline.py

"""
ULTRA-FAST training data generation pipeline
Maximizes throughput for large-scale data ingestion
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
from .data_saver import save_all_datasets

# Suppress all warnings for speed
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Global model cache
_LABELER_CACHE = None

def _get_labeler():
    """Get or create cached labeler instance"""
    global _LABELER_CACHE
    if _LABELER_CACHE is None:
        from training_pipeline.labelers import MultiSourceLabeler
        _LABELER_CACHE = MultiSourceLabeler()
    return _LABELER_CACHE

def _label_single_doc_worker(doc):
    """Fast worker function - labels only, no BIO tagging"""
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
    except Exception:
        return None

class FastTrainingPipeline:
    """Ultra-fast pipeline - labels only, no BIO tagging or LLM eval"""
    
    def __init__(self):
        print("="*80)
        print("FAST RISK DETECTION TRAINING PIPELINE")
        print("="*80)
        
        self.output_dir = get_run_directory(None, None)
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n📁 Output: {self.output_dir}")
        print(f"🚀 Speed mode: No BIO tagging, no LLM eval")
    
    def run(self, limit: int = 10000) -> Dict:
        """Run ultra-fast labeling pipeline"""
        
        pipeline_start = time.time()
        
        # Step 1: Load ALL documents
        print(f"\n[1/4] Loading up to {limit} documents...")
        step_start = time.time()
        documents = self._load_documents(limit)
        print(f"  ✓ Loaded {len(documents)} documents ({time.time() - step_start:.1f}s)")
        
        # Step 2: Label in parallel (FAST)
        print(f"\n[2/4] Labeling with maximum parallelism...")
        step_start = time.time()
        labeled_data = self._label_documents_fast(documents)
        print(f"  ✓ Labeled {len(labeled_data)} docs ({time.time() - step_start:.1f}s)")
        
        # Step 3: Create splits
        print(f"\n[3/4] Creating splits...")
        step_start = time.time()
        datasets = self._create_splits(labeled_data)
        print(f"  ✓ Splits created ({time.time() - step_start:.1f}s)")
        
        # Step 4: Save
        print(f"\n[4/4] Saving datasets...")
        step_start = time.time()
        save_all_datasets(datasets, output_dir=self.output_dir)
        self._save_metadata(limit, datasets, pipeline_start)
        print(f"  ✓ Saved ({time.time() - step_start:.1f}s)")
        
        total_time = time.time() - pipeline_start
        print(f"\n✓ COMPLETE! {total_time/60:.1f} min ({len(labeled_data)} docs)")
        print(f"📁 Output: {self.output_dir}")
        
        return datasets
    
    def _load_documents(self, limit: int) -> List[Dict]:
        """Load documents - fetch ALL available up to limit"""
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cursor = con.cursor()
        
        query = """
            SELECT id, raw_text, source, company, ticker, published_at 
            FROM documents 
            WHERE raw_text IS NOT NULL 
            AND length(raw_text) > 100
            LIMIT ?
        """
        
        cursor.execute(query, (limit,))
        documents = [dict(row) for row in cursor.fetchall()]
        con.close()
        
        return documents
    
    def _label_documents_fast(self, documents: List[Dict]) -> List[Dict]:
        """Ultra-fast parallel labeling"""
        
        # Max out workers for speed
        max_workers = multiprocessing.cpu_count()
        
        print(f"  Using {max_workers} workers...")
        
        labeled_data = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_label_single_doc_worker, doc): doc 
                      for doc in documents}
            
            for future in tqdm(as_completed(futures), total=len(documents), desc="Labeling"):
                result = future.result()
                if result is not None:
                    labeled_data.append(result)
        
        # Quality stats
        high_quality = [d for d in labeled_data 
                       if d.get('labels', {}).get('confidence', 0) >= HIGH_QUALITY_CONFIDENCE]
        medium_quality = [d for d in labeled_data 
                         if MEDIUM_QUALITY_CONFIDENCE <= d.get('labels', {}).get('confidence', 0) < HIGH_QUALITY_CONFIDENCE]
        
        print(f"  High quality: {len(high_quality)}")
        print(f"  Medium quality: {len(medium_quality)}")
        
        return labeled_data
    
    def _create_splits(self, labeled_data: List[Dict]) -> Dict:
        """Create train/val/test splits"""
        
        # Use high + medium quality
        all_data = [d for d in labeled_data 
                   if d.get('labels', {}).get('confidence', 0) >= MEDIUM_QUALITY_CONFIDENCE]
        
        if len(all_data) < 10:
            return {
                'classification': {'train': all_data, 'val': [], 'test': []},
                'span_detection': {'train': [], 'val': [], 'test': []},
                'attributions': []
            }
        
        # 70/15/15 split
        train, temp = train_test_split(all_data, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        from collections import Counter
        train_dist = Counter(d['labels']['risk_category'] for d in train)
        
        print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        print(f"  Class distribution: {dict(train_dist)}")
        
        return {
            'classification': {'train': train, 'val': val, 'test': test},
            'span_detection': {'train': [], 'val': [], 'test': []},
            'attributions': []
        }
    
    def _save_metadata(self, limit: int, datasets: Dict, start_time: float):
        """Save run metadata"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'fast_ingestion',
            'limit': limit,
            'execution_time_minutes': round((time.time() - start_time) / 60, 2),
            'statistics': {
                'total_labeled': len(datasets['classification']['train']) + 
                               len(datasets['classification']['val']) + 
                               len(datasets['classification']['test']),
                'train': len(datasets['classification']['train']),
                'val': len(datasets['classification']['val']),
                'test': len(datasets['classification']['test'])
            }
        }
        
        with open(os.path.join(self.output_dir, 'run_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-fast training data generation")
    parser.add_argument('--limit', type=int, default=10000, 
                       help='Max documents to process (default: 10000)')
    
    args = parser.parse_args()
    
    try:
        print(f"\n🚀 FAST MODE: Processing up to {args.limit} documents...")
        
        pipeline = FastTrainingPipeline()
        datasets = pipeline.run(limit=args.limit)
        
        print("\n✓ Ready for model training!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()