#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""
Risk Detection Training Pipeline

A modular pipeline for generating high-quality training data for financial
risk detection models.

Components:
- config: Central configuration
- utils: Shared utility functions
- labelers: Multi-source labeling (FinBERT, Phrasebank, LLM)
- bio_tagger: BIO tag generation for span detection
- ground_truth_loader: Historical ground truth integration
- llm_evaluator: LLM baseline benchmarking
- data_saver: Dataset persistence
- report_generator: Comprehensive reporting
- pipeline: Main orchestration

Usage:
    from training_pipeline import RiskTrainingPipeline
    
    pipeline = RiskTrainingPipeline()
    datasets = pipeline.run(company="Apple Inc.", ticker="AAPL")
"""

from .pipeline import RiskTrainingPipeline, main
from .labelers import MultiSourceLabeler, FinBERTLabeler, PhrasebankLabeler, LLMLabeler
from .bio_tagger import BIOTagger
from .ground_truth_loader import load_ground_truth_documents
from .llm_evaluator import LLMEvaluator

__version__ = "1.0.0"
__all__ = [
    'RiskTrainingPipeline',
    'MultiSourceLabeler',
    'FinBERTLabeler',
    'PhrasebankLabeler',
    'LLMLabeler',
    'BIOTagger',
    'load_ground_truth_documents',
    'LLMEvaluator',
    'main'
]