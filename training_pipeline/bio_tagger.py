#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# bio_tagger.py

"""BIO tagging for risk span detection"""

import re
from typing import Dict, List, Optional

from .config import COMPILED_PATTERNS, MAX_SENTENCE_LIMIT
from .utils import sentence_tokenize, merge_overlapping_spans, calculate_word_positions


class BIOTagger:
    """Generate BIO tags for risk spans using weak supervision"""
    
    def __init__(self):
        print("[BIOTagger] Initializing...")
        self.patterns = COMPILED_PATTERNS
        print("[BIOTagger] Ready")
    
    def generate_bio_tags(
        self,
        text: str,
        finbert_labeler: Optional[object] = None
    ) -> Dict:
        """
        Generate BIO tags using:
        1. Regex patterns (weak supervision)
        2. FinBERT attribution spans (if available)
        """
        
        # Sentence tokenization
        sentences = sentence_tokenize(text)
        
        bio_tagged = []
        for sent in sentences[:MAX_SENTENCE_LIMIT]:
            words = sent.split()
            if not words:
                continue
            
            # Initialize tags as Outside
            tags = ['O'] * len(words)
            
            # Method 1: Pattern matching
            pattern_spans = self._find_pattern_spans(sent)
            
            # Method 2: FinBERT attribution (if available)
            finbert_spans = []
            if finbert_labeler:
                try:
                    result = finbert_labeler.label_with_attribution(sent)
                    finbert_spans = result.get('risk_spans', [])
                except Exception as e:
                    print(f"[BIOTagger] FinBERT error: {e}")
            
            # Merge spans
            all_spans = self._merge_spans_for_bio(pattern_spans, finbert_spans, sent)
            
            # Apply BIO tags
            tags = self._apply_bio_tags(words, all_spans, sent)
            
            bio_tagged.append({
                'sentence': sent,
                'words': words,
                'tags': tags,
                'span_count': len([t for t in tags if t != 'O'])
            })
        
        return {
            'text': text,
            'bio_tagged_sentences': bio_tagged,
            'total_sentences': len(bio_tagged),
            'total_spans': sum(s['span_count'] for s in bio_tagged)
        }
    
    def _find_pattern_spans(self, text: str) -> List[Dict]:
        """Find risk spans using regex patterns"""
        
        spans = []
        for risk_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    spans.append({
                        'start': match.start(),
                        'end': match.end(),
                        'text': match.group(),
                        'risk_type': risk_type,
                        'source': 'pattern',
                        'confidence': 0.7
                    })
        return spans
    
    def _merge_spans_for_bio(
        self,
        pattern_spans: List[Dict],
        finbert_spans: List[Dict],
        text: str
    ) -> List[Dict]:
        """Merge spans from different sources"""
        
        # Convert FinBERT spans to same format
        finbert_formatted = []
        for span in finbert_spans:
            span_text = span['text']
            start = text.lower().find(span_text.lower())
            if start != -1:
                finbert_formatted.append({
                    'start': start,
                    'end': start + len(span_text),
                    'text': span_text,
                    'risk_type': 'general',
                    'source': 'finbert',
                    'confidence': span['avg_attribution']
                })
        
        all_spans = pattern_spans + finbert_formatted
        
        # Remove duplicates and overlaps
        merged = merge_overlapping_spans(all_spans)
        
        return merged
    
    def _apply_bio_tags(
        self,
        words: List[str],
        spans: List[Dict],
        text: str
    ) -> List[str]:
        """Apply BIO tags to words based on spans"""
        
        tags = ['O'] * len(words)
        
        # Calculate word positions
        word_positions = calculate_word_positions(words, text)
        
        # Apply spans to words
        for span in spans:
            span_start = span['start']
            span_end = span['end']
            risk_type = span['risk_type'].upper()
            
            first_token = True
            for i, (w_start, w_end) in enumerate(word_positions):
                # Check if word overlaps with span
                if w_start < span_end and w_end > span_start:
                    if first_token:
                        tags[i] = f'B-{risk_type}'
                        first_token = False
                    else:
                        tags[i] = f'I-{risk_type}'
        
        return tags