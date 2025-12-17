#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py

"""Utility functions for risk training pipeline"""

import re
from typing import List, Dict, Tuple


def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    """Split text into chunks at sentence boundaries"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in sentences:
        sent_len = len(sent.split())
        if current_length + sent_len > chunk_size and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sent]
            current_length = sent_len
        else:
            current_chunk.append(sent)
            current_length += sent_len
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks if chunks else [text]


def sentence_tokenize(text: str) -> List[str]:
    """Simple sentence tokenization"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def sentiment_to_risk_score(sentiment: str, confidence: float) -> float:
    """Convert sentiment to risk score [0, 1]"""
    if sentiment == "negative":
        return confidence
    elif sentiment == "neutral":
        return 0.3
    else:  # positive
        return 0.0


def score_to_category(score: float) -> str:
    """Convert risk score to category"""
    if score < 0.2:
        return 'none'
    elif score < 0.4:
        return 'low'
    elif score < 0.6:
        return 'medium'
    elif score < 0.8:
        return 'high'
    else:
        return 'catastrophic'


def merge_overlapping_spans(spans: List[Dict]) -> List[Dict]:
    """Remove duplicate and overlapping spans"""
    if not spans:
        return []
    
    merged = []
    sorted_spans = sorted(spans, key=lambda x: x['start'])
    
    for span in sorted_spans:
        overlaps = False
        for existing in merged:
            if (span['start'] <= existing['end'] and 
                span['end'] >= existing['start']):
                overlaps = True
                # Boost confidence of existing span
                existing['confidence'] = min(existing['confidence'] * 1.2, 1.0)
                break
        
        if not overlaps and span['confidence'] > 0.5:
            merged.append(span)
    
    return merged


def calculate_word_positions(words: List[str], text: str) -> List[Tuple[int, int]]:
    """Calculate character positions for each word in text"""
    positions = []
    current_pos = 0
    
    for word in words:
        start = text.find(word, current_pos)
        if start == -1:
            start = current_pos
        end = start + len(word)
        positions.append((start, end))
        current_pos = end
    
    return positions