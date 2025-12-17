#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ingest_data.py

"""
RiskRadar Data Ingestion CLI

This script orchestrates the ingestion of financial data from multiple sources:
- SEC filings (8-K, 10-K, 10-Q, 6-K)
- Press releases (GlobeNewswire)
- News articles (MarketAux, Google News, Yahoo Finance)
- Earnings call transcripts (Yahoo Finance, SeekingAlpha)

Usage:
    python ingest_data.py --company "Apple Inc." --ticker AAPL
    python ingest_data.py --company "Tesla Inc." --ticker TSLA --sec 50 --news 100
"""

import argparse
from .orchestrator import ingest_company

def main():
    parser = argparse.ArgumentParser(
        description="Ingest SEC, PR, News, and Transcripts for a company.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_data.py --company "Apple Inc." --ticker AAPL
  python ingest_data.py --company "Tesla Inc." --ticker TSLA --sec 50 --news 100
  python ingest_data.py --company "Microsoft Corporation" --ticker MSFT --transcripts 20
        """
    )
    
    parser.add_argument(
        "--company", 
        required=True, 
        help="Company name, e.g., 'Apple Inc.'"
    )
    parser.add_argument(
        "--ticker", 
        default=None, 
        help="Ticker, e.g., 'AAPL' (optional but recommended)"
    )
    parser.add_argument(
        "--sec", 
        type=int, 
        default=25, 
        help="Max SEC filings to fetch (default: 25)"
    )
    parser.add_argument(
        "--press", 
        type=int, 
        default=50, 
        help="Max press releases to fetch (default: 50)"
    )
    parser.add_argument(
        "--news", 
        type=int, 
        default=50, 
        help="Max news articles to fetch (default: 50)"
    )
    parser.add_argument(
        "--transcripts", 
        type=int, 
        default=10, 
        help="Max transcripts to fetch (default: 10)"
    )
    
    args = parser.parse_args()

    take = {
        "sec": args.sec, 
        "press": args.press, 
        "news": args.news, 
        "transcripts": args.transcripts
    }
    
    print(f"\n{'='*60}")
    print(f"RiskRadar Data Ingestion")
    print(f"{'='*60}")
    print(f"Company: {args.company}")
    print(f"Ticker: {args.ticker or 'N/A'}")
    print(f"Limits: SEC={args.sec}, Press={args.press}, News={args.news}, Transcripts={args.transcripts}")
    print(f"{'='*60}\n")
    
    ingest_company(args.company, args.ticker, take)

if __name__ == "__main__":
    main()