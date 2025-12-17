#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# orchestrator.py

from typing import Optional, Dict

from .database import init_db
from .sec_ingestion import resolve_company, ingest_sec
from .press_release_ingestion import ingest_globenewswire
from .news_ingestion import ingest_marketaux, ingest_gnews, ingest_yahoo_finance
from .transcript_ingestion import ingest_yahoo_earnings_transcripts, ingest_seekingalpha_rss_transcripts

# =========================
# Main Orchestration
# =========================

def ingest_company(company: str, ticker: Optional[str] = None,
                   take: Dict[str, int] = None):
    """
    Ingest all sources for a given company.
    
    Args:
        company: Company name (string)
        ticker: Optional ticker (string)
        take: Dict to control max items per source, e.g.
              {"sec": 25, "press": 50, "news": 50, "transcripts": 10}
    """
    init_db()
    take = take or {"sec": 25, "press": 50, "news": 50, "transcripts": 10}

    norm_company, norm_ticker, cik = resolve_company(company, ticker)
    print(f"Resolved -> company: {norm_company} | ticker: {norm_ticker} | cik: {cik}")

    # 1) SEC filings (US only)
    try:
        ingest_sec(norm_company or company, norm_ticker, cik, max_filings=take.get("sec", 25))
    except Exception as e:
        print(f"[SEC] Top-level error: {e}")

    # 2) Press releases (GlobeNewswire RSS)
    try:
        ingest_globenewswire(norm_company or company, norm_ticker, max_items=take.get("press", 50))
    except Exception as e:
        print(f"[PR] Top-level error: {e}")

    # 3) News (MarketAux + GNews)
    try:
        ingest_marketaux(norm_company or company, norm_ticker, max_items=take.get("news", 50))
    except Exception as e:
        print(f"[News] MarketAux top-level error: {e}")
    
    try:
        ingest_gnews(norm_company or company, norm_ticker, max_items=take.get("news", 50))
    except Exception as e:
        print(f"[News] GNews top-level error: {e}")

    # Optional: Yahoo Finance news (commented out in original)
    # try:
    #     ingest_yahoo_finance(norm_company or company, norm_ticker, max_items=take.get("news", 50))
    # except Exception as e:
    #     print(f"[News] Yahoo Finance top-level error: {e}")

    # 4) Earnings call transcripts
    try:
        ingest_yahoo_earnings_transcripts(
            norm_company or company, 
            norm_ticker, 
            max_items=take.get("transcripts", 10)
        )
    except Exception as e:
        print(f"[Transcript] Yahoo top-level error: {e}")
    
    try:
        ingest_seekingalpha_rss_transcripts(
            norm_company or company,
            norm_ticker,
            max_items=take.get("transcripts", 10)
        )
    except Exception as e:
        print(f"[Transcript] SeekingAlpha top-level error: {e}")

    print("\n=== Ingestion complete ===")