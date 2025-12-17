#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# Database Configuration
# =========================

DB_PATH = os.environ.get("RISKRADAR_DB", "riskradar_ingest.db")

# =========================
# API Configuration
# =========================

SEC_UA = os.environ.get(
    "SEC_USER_AGENT",
    "riskradar-mvp/0.1 (example@example.com)"  # replace with your contact email/domain
)

MARKETAUX_TOKEN = os.environ.get("MARKETAUX_API_TOKEN", None)

# =========================
# Request Configuration
# =========================

REQUEST_TIMEOUT = 20

# =========================
# SEC Endpoints
# =========================

SEC_BASE = "https://data.sec.gov"
SEC_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

HEADERS_DEFAULT = {
    "User-Agent": SEC_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# =========================
# Retry Configuration
# =========================

RETRY_STATUSES = {429, 502, 503, 504}