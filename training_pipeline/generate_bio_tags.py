#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# training_pipeline/generate_bio_tags.py

"""
Standalone script to generate BIO tags for span detection
Loads documents from database and creates BIO-tagged CoNLL files
"""

import sys
import os
import sqlite3
import time
import multiprocessing
import warnings
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from .config import DB_PATH, get_run_directory
from .bio_tagger import BIOTagger
from .labelers import FinBERTLabeler

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Global cache
_BIO_TAGGER_CACHE = None
_FINBERT_CACHE = None

def _get_bio_tagger():
    """Get or create cached BIO tagger"""
    global _BIO_TAGGER_CACHE
    if _BIO_TAGGER_CACHE is None:
        from training_pipeline.bio_tagger import BIOTagger
        _BIO_TAGGER_CACHE = BIOTagger()
    return _BIO_TAGGER_CACHE

def _get_finbert():
    """Get or create cached FinBERT"""
    global _FINBERT_CACHE
    if _FINBERT_CACHE is None:
        from training_pipeline.labelers import FinBERTLabeler
        _FINBERT_CACHE = FinBERTLabeler()
    return _FINBERT_CACHE

def _tag_single_doc_worker(doc):
    """Worker function for parallel BIO tagging"""
    try:
        bio_tagger = _get_bio_tagger()
        finbert = _get_finbert()
        
        bio_result = bio_tagger.generate_bio_tags(
            doc['raw_text'],
            finbert_labeler=finbert
        )
        return {**doc, 'bio_tags': bio_result}
    except Exception as e:
        print(f"[ERROR] Failed to tag doc {doc.get('id')}: {e}")
        return None


class BIOTagGenerator:
    """Generate BIO tags for span detection"""
    
    def __init__(self, output_dir: str = None):
        print("="*80)
        print("BIO TAG GENERATION FOR SPAN DETECTION")
        print("="*80)
        
        # Create output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("training_data") / f"bio_tags_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Output directory: {self.output_dir}")
        
        # Initialize components
        print(f"\n🔧 Initializing BIO tagger...")
        self.bio_tagger = BIOTagger()
        self.finbert = FinBERTLabeler()
        print(f"   ✓ Components ready")
    
    def load_documents(self, limit: int = 5000) -> List[Dict]:
        """Load documents from database"""
        print(f"\n📂 Loading up to {limit} documents from database...")
        
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
        
        print(f"  ✓ Loaded {len(documents)} documents")
        
        return documents
    
    def generate_bio_tags(self, documents: List[Dict]) -> List[Dict]:
        """Generate BIO tags for all documents in parallel"""
        print(f"\n🏷️  Generating BIO tags (parallel)...")
        
        # Max out workers
        max_workers = min(multiprocessing.cpu_count() - 1, len(documents), 8)
        max_workers = max(1, max_workers)
        
        print(f"  Using {max_workers} parallel workers...")
        
        bio_tagged_data = []
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_tag_single_doc_worker, doc): doc 
                          for doc in documents}
                
                for future in tqdm(as_completed(futures), total=len(documents), desc="BIO tagging"):
                    result = future.result()
                    if result is not None:
                        bio_tagged_data.append(result)
        
        except Exception as e:
            print(f"  ⚠️ Parallel processing error: {e}")
            print(f"  Falling back to sequential...")
            
            for doc in tqdm(documents, desc="BIO tagging (sequential)"):
                result = _tag_single_doc_worker(doc)
                if result:
                    bio_tagged_data.append(result)
        
        total_sentences = sum(
            d.get('bio_tags', {}).get('total_sentences', 0) 
            for d in bio_tagged_data
        )
        
        print(f"  ✓ Successfully tagged: {len(bio_tagged_data)}/{len(documents)} documents")
        print(f"  ✓ Generated BIO tags for {total_sentences} sentences")
        
        return bio_tagged_data
    
    def save_conll_files(self, bio_tagged_data: List[Dict]):
        """Save BIO-tagged data to CoNLL format files"""
        print(f"\n💾 Saving CoNLL files...")
        
        from sklearn.model_selection import train_test_split
        
        # Split data: 70/15/15
        if len(bio_tagged_data) >= 3:
            train, temp = train_test_split(bio_tagged_data, test_size=0.3, random_state=42)
            val, test = train_test_split(temp, test_size=0.5, random_state=42)
        else:
            train, val, test = bio_tagged_data, [], []
        
        print(f"  Train: {len(train)} documents")
        print(f"  Val: {len(val)} documents")
        print(f"  Test: {len(test)} documents")
        
        # Save each split
        splits = {'train': train, 'val': val, 'test': test}
        
        for split_name, split_data in splits.items():
            filepath = self.output_dir / f'{split_name}.conll'
            
            with open(filepath, 'w') as f:
                for doc in split_data:
                    bio_tags = doc.get('bio_tags', {})
                    
                    # Get BIO tagged sentences
                    bio_sentences = bio_tags.get('bio_tagged_sentences', [])
                    
                    for sent in bio_sentences:
                        words = sent.get('words', [])
                        tags = sent.get('tags', [])
                        
                        # Write word-tag pairs
                        for word, tag in zip(words, tags):
                            f.write(f"{word}\t{tag}\n")
                        
                        # Blank line between sentences
                        f.write("\n")
            
            # Count sentences
            sentence_count = sum(
                len(d.get('bio_tags', {}).get('bio_tagged_sentences', [])) 
                for d in split_data
            )
            
            print(f"  ✓ Saved {filepath} ({sentence_count} sentences)")
        
        print(f"\n✅ All CoNLL files saved to: {self.output_dir}")
    
    def run(self, limit: int = 5000):
        """Run complete BIO tag generation pipeline"""
        start_time = time.time()
        
        # Load documents
        documents = self.load_documents(limit)
        
        # Generate BIO tags
        bio_tagged_data = self.generate_bio_tags(documents)
        
        # Save to CoNLL files
        self.save_conll_files(bio_tagged_data)
        
        elapsed = time.time() - start_time
        
        print(f"\n" + "="*80)
        print(f"✓ COMPLETE!")
        print(f"="*80)
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
        print(f"📁 Output: {self.output_dir}")
        print(f"\n🚀 Next step: Train span detection model")
        print(f"   python -m training_pipeline.train_span_detector \\")
        print(f"       --data-dir {self.output_dir} \\")
        print(f"       --output-dir models/span_detector_v1")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate BIO tags for span detection"
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=5000,
        help='Max documents to process (default: 5000)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for CoNLL files'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"\n🚀 Starting BIO tag generation...")
        print(f"   Limit: {args.limit} documents")
        
        generator = BIOTagGenerator(output_dir=args.output_dir)
        generator.run(limit=args.limit)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()