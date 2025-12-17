#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data_saver.py

"""Save training datasets to disk"""

import os
import json
from typing import Dict, List, Optional


def save_all_datasets(datasets: Dict, output_dir: Optional[str] = None):
    """Save all datasets to disk in appropriate formats
    
    Args:
        datasets: Dictionary containing all dataset splits
        output_dir: Optional output directory. If None, uses current directory
    """
    
    # Use provided output_dir or current directory
    if output_dir is None:
        from .config import OUTPUT_DIR
        output_dir = OUTPUT_DIR
    
    print(f"\n[DataSaver] Writing datasets to {output_dir}...")
    
    # Create subdirectories
    os.makedirs(os.path.join(output_dir, 'classification'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'span_detection'), exist_ok=True)
    
    # Classification data (JSONL)
    _save_classification_data(datasets['classification'], output_dir)
    
    # BIO tagged data (CoNLL format)
    _save_bio_data(datasets['span_detection'], output_dir)
    
    # LLM results (JSON)
    _save_llm_results(datasets.get('llm_baseline', {}), output_dir)
    
    # Attribution data (JSON)
    _save_attribution_data(datasets.get('attributions', []), output_dir)
    
    print("[DataSaver] All datasets saved")


def _save_classification_data(classification_splits: Dict, output_dir: str):
    """Save classification data as JSONL"""
    
    for split in ['train', 'val', 'test']:
        filepath = os.path.join(output_dir, 'classification', f'{split}.jsonl')
        
        with open(filepath, 'w') as f:
            for item in classification_splits[split]:
                # Remove raw_text to save space, keep only essentials
                output_item = {
                    'id': item.get('id'),
                    'company': item.get('company'),
                    'ticker': item.get('ticker'),
                    'source': item.get('source'),
                    'title': item.get('title'),
                    'url': item.get('url'),
                    'published_at': item.get('published_at'),
                    'labels': item['labels'],
                    'text_preview': item.get('raw_text', '')[:500]
                }
                f.write(json.dumps(output_item) + '\n')
        
        print(f"  ✓ Saved {filepath}")


def _save_bio_data(bio_splits: Dict, output_dir: str):
    """Save BIO tagged data in CoNLL format"""
    
    for split in ['train', 'val', 'test']:
        filepath = os.path.join(output_dir, 'span_detection', f'{split}.conll')
        
        with open(filepath, 'w') as f:
            for doc in bio_splits[split]:
                bio_tags = doc.get('bio_tags', {})
                for sent in bio_tags.get('bio_tagged_sentences', []):
                    for word, tag in zip(sent['words'], sent['tags']):
                        f.write(f"{word}\t{tag}\n")
                    f.write("\n")  # Blank line between sentences
        
        print(f"  ✓ Saved {filepath}")


def _save_llm_results(llm_results: Dict, output_dir: str):
    """Save LLM baseline results as JSON"""
    
    filepath = os.path.join(output_dir, 'llm_baseline_results.json')
    
    with open(filepath, 'w') as f:
        json.dump(llm_results, f, indent=2)
    
    print(f"  ✓ Saved {filepath}")


def _save_attribution_data(attributions: List[Dict], output_dir: str):
    """Save attribution data as JSON"""
    
    filepath = os.path.join(output_dir, 'attributions.json')
    
    with open(filepath, 'w') as f:
        json.dump(attributions, f, indent=2)
    
    print(f"  ✓ Saved {filepath}")