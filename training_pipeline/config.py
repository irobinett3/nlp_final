#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# training_pipeline/config.py

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =========================
# Database Configuration
# =========================

DB_PATH = os.environ.get("RISKRADAR_DB", "riskradar_ingest.db")

# =========================
# API Keys
# =========================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# =========================
# Output Configuration
# =========================


# Add to training_pipeline/config.py

DEFAULT_CHUNK_SIZE = 512  # Tokens per chunk (for FinBERT which has 512 max)
MAX_CHUNKS_PER_DOC = 10 
# Attribution threshold for FinBERT attention scores
ATTRIBUTION_THRESHOLD = 0.1  # Minimum attention score to consider token relevant

import re

# Risk pattern matching (for BIO tagger)
RISK_PATTERNS = {
    'financial': [
        r'\b(bankruptcy|insolvent|debt\s+default|financial\s+distress)\b',
        r'\b(revenue\s+decline|profit\s+loss|cash\s+flow\s+negative)\b',
    ],
    'legal': [
        r'\b(lawsuit|litigation|legal\s+action|settlement)\b',
        r'\b(fraud|investigation|regulatory\s+violation)\b',
    ],
    'operational': [
        r'\b(recall|safety\s+issue|supply\s+chain\s+disruption)\b',
        r'\b(data\s+breach|cybersecurity|ransomware)\b',
    ],
    'reputational': [
        r'\b(scandal|controversy|public\s+backlash)\b',
        r'\b(boycott|negative\s+press|reputation\s+damage)\b',
    ],
}

# Compile patterns for efficiency
COMPILED_PATTERNS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in RISK_PATTERNS.items()
}

# BIO tagger limits
MAX_SENTENCE_LIMIT = 100  # Max sentences to tag per document

def get_run_directory(company: str = None, ticker: str = None) -> str:
    """
    Generate a unique output directory for this training run.
    
    Format: training_data/run_YYYYMMDD_HHMMSS_{company}_{ticker}
    Example: training_data/run_20251214_130855_microsoft_msft
    
    Args:
        company: Company name (optional, for descriptive folder name)
        ticker: Ticker symbol (optional, for descriptive folder name)
        
    Returns:
        Path to unique run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build descriptive suffix
    parts = ["run", timestamp]
    
    if company:
        # Clean company name for filesystem
        clean_company = company.lower().replace(" ", "_").replace(".", "")
        clean_company = "".join(c for c in clean_company if c.isalnum() or c == "_")
        parts.append(clean_company[:20])  # Limit length
    
    if ticker:
        parts.append(ticker.lower())
    
    run_id = "_".join(parts)
    return os.path.join("training_data", run_id)


# Base output directory (will be overridden per run)
OUTPUT_DIR = "training_data"

# =========================
# Labeling Configuration
# =========================

# Confidence thresholds
HIGH_QUALITY_CONFIDENCE = 0.67  # 2+ sources agree
MEDIUM_QUALITY_CONFIDENCE = 0.34  # 1+ sources agree

# Risk categories
RISK_CATEGORIES = ["high_risk", "medium_risk", "low_risk", "no_risk"]

# =========================
# Model Configuration
# =========================

PHRASEBANK_MODEL = "ProsusAI/finbert"  # Or whatever model you want to use

# FinBERT model for sentiment/risk analysis
FINBERT_MODEL = "yiyanghkust/finbert-tone"

# LLM configuration
LLM_MODEL = "gpt-4o-mini"  # Cost-effective for baseline
LLM_TEMPERATURE = 0.1  # Low temperature for consistent labels
LLM_MAX_TOKENS = 500