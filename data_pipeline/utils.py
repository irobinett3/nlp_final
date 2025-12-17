#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py

import re
import time
import random
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup

try:
    import dateparser
except ImportError:
    dateparser = None

from .config import HEADERS_DEFAULT, REQUEST_TIMEOUT

# =========================
# Text Processing
# =========================

def is_relevant(text: str, company: str, ticker: str | None) -> bool:
    """
    Score text relevance to a company based on mentions and keywords.
    Returns True if score >= 4.
    """
    if not text:
        return False

    text_l = text.lower()
    company_l = company.lower()

    score = 0

    # Strong signal: company name
    if company_l in text_l:
        score += 3
        score += min(5, text_l.count(company_l))  # frequency bonus

    # Medium signal: ticker
    if ticker:
        ticker_l = ticker.lower()
        if ticker_l in text_l:
            score += 2
            score += min(3, text_l.count(ticker_l))

    # Weak but common business signals
    keywords = [
        "earnings", "revenue", "guidance", "quarter",
        "sec", "regulatory", "antitrust",
        "lawsuit", "acquisition", "merger",
        "ceo", "cfo", "board", "investor",
        "press release"
    ]
    score += sum(1 for k in keywords if k in text_l)

    return score >= 4


def clean_text(html_or_text: str) -> str:
    """
    Clean HTML or text content, removing scripts/styles and normalizing whitespace.
    """
    if not html_or_text:
        return ""
    s = html_or_text.strip()

    # Detect XML (doctype or root tag)
    if s.startswith("<?xml") or s.lower().startswith("<rss") or s.lower().startswith("<feed"):
        soup = BeautifulSoup(s, "xml")  # use XML parser
    else:
        soup = BeautifulSoup(s, "lxml")  # HTML parser

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


# =========================
# Date/Time Utilities
# =========================

def now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def parse_date(s: Optional[str]) -> Optional[str]:
    """
    Parse various date formats into ISO format.
    Returns ISO string or None if parsing fails.
    """
    if not s:
        return None
    # Try ISO first
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    # Try dateparser if available
    if dateparser:
        dt = dateparser.parse(s)
        if dt:
            if not dt.tzinfo:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
    # Fallback: None
    return None


# =========================
# Hashing
# =========================

def sha1(text: str) -> str:
    """Return SHA1 hash of text."""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


# =========================
# HTTP Utilities
# =========================

def http_get(url: str, headers: Dict[str, str] = None, params: Dict[str, Any] = None) -> requests.Response:
    """
    Make HTTP GET request with default headers and timeout.
    Raises HTTPError on non-2xx status.
    """
    h = dict(HEADERS_DEFAULT)
    if headers:
        h.update(headers)
    r = requests.get(url, headers=h, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r


def backoff_sleep(attempt: int):
    """
    Jittered exponential backoff sleep.
    ~0.3s, 0.6s, 1.2s, 2.4s...
    """
    time.sleep(0.3 * (2 ** attempt) + random.random() * 0.2)