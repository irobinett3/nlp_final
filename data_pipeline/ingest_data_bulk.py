#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ingest_data_bulk.py

"""
RiskRadar BULK Data Ingestion - Optimized for Maximum Throughput

Ingests data for multiple companies in parallel to maximize database population.
Prioritizes speed and volume over granular control.

Usage:
    # Single company - max data
    python ingest_data_bulk.py --company "Apple Inc." --ticker AAPL --max
    
    # Multiple companies from file
    python ingest_data_bulk.py --companies-file sp500.txt --max
    
    # Top N S&P 500 companies
    python ingest_data_bulk.py --top-sp500 50 --max
"""

import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from pathlib import Path

from .orchestrator import ingest_company


# Pre-defined lists for bulk ingestion
SP500_TOP_100 = [
    ("Apple Inc.", "AAPL"),
    ("Microsoft Corporation", "MSFT"),
    ("Amazon.com Inc.", "AMZN"),
    ("NVIDIA Corporation", "NVDA"),
    ("Alphabet Inc.", "GOOGL"),
    ("Meta Platforms Inc.", "META"),
    ("Tesla Inc.", "TSLA"),
    ("Berkshire Hathaway Inc.", "BRK-B"),
    ("Visa Inc.", "V"),
    ("JPMorgan Chase & Co.", "JPM"),
    ("Walmart Inc.", "WMT"),
    ("Eli Lilly and Company", "LLY"),
    ("UnitedHealth Group Incorporated", "UNH"),
    ("Exxon Mobil Corporation", "XOM"),
    ("Johnson & Johnson", "JNJ"),
    ("Mastercard Incorporated", "MA"),
    ("Procter & Gamble Company", "PG"),
    ("Broadcom Inc.", "AVGO"),
    ("Home Depot Inc.", "HD"),
    ("Chevron Corporation", "CVX"),
    ("AbbVie Inc.", "ABBV"),
    ("Merck & Co. Inc.", "MRK"),
    ("Costco Wholesale Corporation", "COST"),
    ("Bank of America Corporation", "BAC"),
    ("Coca-Cola Company", "KO"),
    ("Pepsico Inc.", "PEP"),
    ("Adobe Inc.", "ADBE"),
    ("Netflix Inc.", "NFLX"),
    ("Salesforce Inc.", "CRM"),
    ("Thermo Fisher Scientific Inc.", "TMO"),
    ("McDonald's Corporation", "MCD"),
    ("Cisco Systems Inc.", "CSCO"),
    ("Oracle Corporation", "ORCL"),
    ("Accenture plc", "ACN"),
    ("Pfizer Inc.", "PFE"),
    ("Intel Corporation", "INTC"),
    ("Nike Inc.", "NKE"),
    ("Abbott Laboratories", "ABT"),
    ("Comcast Corporation", "CMCSA"),
    ("Walt Disney Company", "DIS"),
    ("Verizon Communications Inc.", "VZ"),
    ("Danaher Corporation", "DHR"),
    ("Qualcomm Incorporated", "QCOM"),
    ("United Parcel Service Inc.", "UPS"),
    ("Philip Morris International Inc.", "PM"),
    ("Amgen Inc.", "AMGN"),
    ("Boeing Company", "BA"),
    ("T-Mobile US Inc.", "TMUS"),
    ("Advanced Micro Devices Inc.", "AMD"),
    ("Honeywell International Inc.", "HON"),
    ("IBM", "IBM"),
    ("Caterpillar Inc.", "CAT"),
    ("General Electric Company", "GE"),
    ("American Express Company", "AXP"),
    ("Goldman Sachs Group Inc.", "GS"),
    ("Morgan Stanley", "MS"),
    ("Lowe's Companies Inc.", "LOW"),
    ("CVS Health Corporation", "CVS"),
    ("Starbucks Corporation", "SBUX"),
    ("Lockheed Martin Corporation", "LMT"),
    ("Raytheon Technologies Corporation", "RTX"),
    ("3M Company", "MMM"),
    ("Deere & Company", "DE"),
    ("BlackRock Inc.", "BLK"),
    ("Target Corporation", "TGT"),
    ("Wells Fargo & Company", "WFC"),
    ("Citigroup Inc.", "C"),
    ("American Tower Corporation", "AMT"),
    ("NextEra Energy Inc.", "NEE"),
    ("Duke Energy Corporation", "DUK"),
    ("Southern Company", "SO"),
    ("Moderna Inc.", "MRNA"),
    ("Vertex Pharmaceuticals Incorporated", "VRTX"),
    ("Regeneron Pharmaceuticals Inc.", "REGN"),
    ("Gilead Sciences Inc.", "GILD"),
    ("Schlumberger Limited", "SLB"),
    ("ConocoPhillips", "COP"),
    ("Marathon Petroleum Corporation", "MPC"),
    ("Valero Energy Corporation", "VLO"),
    ("Ford Motor Company", "F"),
    ("General Motors Company", "GM"),
    ("Delta Air Lines Inc.", "DAL"),
    ("American Airlines Group Inc.", "AAL"),
    ("Marriott International Inc.", "MAR"),
    ("Booking Holdings Inc.", "BKNG"),
    ("PayPal Holdings Inc.", "PYPL"),
    ("Block Inc.", "SQ"),
    ("Snowflake Inc.", "SNOW"),
    ("Airbnb Inc.", "ABNB"),
    ("Uber Technologies Inc.", "UBER"),
    ("Lyft Inc.", "LYFT"),
    ("DoorDash Inc.", "DASH"),
    ("Shopify Inc.", "SHOP"),
    ("Zoom Video Communications Inc.", "ZM"),
    ("DocuSign Inc.", "DOCU"),
    ("Palantir Technologies Inc.", "PLTR"),
    ("Coinbase Global Inc.", "COIN"),
]


class BulkIngestionManager:
    """Manages parallel ingestion across multiple companies"""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.results = []
        
    def ingest_single_company(
        self, 
        company: str, 
        ticker: str, 
        limits: Dict[str, int]
    ) -> Dict:
        """Ingest data for a single company"""
        
        start_time = time.time()
        
        try:
            print(f"\n🔄 Starting: {company} ({ticker})")
            
            ingest_company(company, ticker, limits)
            
            elapsed = time.time() - start_time
            
            result = {
                'company': company,
                'ticker': ticker,
                'status': 'success',
                'time': elapsed
            }
            
            print(f"✅ Completed: {company} ({ticker}) in {elapsed:.1f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            
            result = {
                'company': company,
                'ticker': ticker,
                'status': 'failed',
                'error': str(e),
                'time': elapsed
            }
            
            print(f"❌ Failed: {company} ({ticker}) - {e}")
        
        return result
    
    def ingest_bulk(
        self, 
        companies: List[tuple], 
        limits: Dict[str, int]
    ):
        """Ingest data for multiple companies in parallel"""
        
        print("="*80)
        print("BULK DATA INGESTION - PARALLEL MODE")
        print("="*80)
        print(f"\nCompanies to process: {len(companies)}")
        print(f"Parallel workers: {self.max_workers}")
        print(f"Limits per company: {limits}")
        print("="*80)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self.ingest_single_company, 
                    company, 
                    ticker, 
                    limits
                ): (company, ticker)
                for company, ticker in companies
            }
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
        
        total_time = time.time() - start_time
        
        # Summary
        self._print_summary(total_time)
    
    def _print_summary(self, total_time: float):
        """Print ingestion summary"""
        
        successes = [r for r in self.results if r['status'] == 'success']
        failures = [r for r in self.results if r['status'] == 'failed']
        
        print("\n" + "="*80)
        print("INGESTION SUMMARY")
        print("="*80)
        print(f"\nTotal companies processed: {len(self.results)}")
        print(f"✅ Successful: {len(successes)}")
        print(f"❌ Failed: {len(failures)}")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"⏱️  Avg time per company: {total_time/len(self.results):.1f}s")
        
        if failures:
            print("\n❌ Failed companies:")
            for f in failures:
                print(f"  - {f['company']} ({f['ticker']}): {f['error']}")
        
        print("\n✅ Ready for training pipeline!")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk data ingestion for RiskRadar - Optimized for maximum throughput",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single company - maximum data
  python ingest_data_bulk.py --company "Apple Inc." --ticker AAPL --max

  # Top 50 S&P 500 companies
  python ingest_data_bulk.py --top-sp500 50 --max --workers 10

  # All pre-loaded S&P 100
  python ingest_data_bulk.py --all-sp100 --max --workers 8

  # From custom file (one company per line: "Company Name,TICKER")
  python ingest_data_bulk.py --companies-file companies.txt --max
        """
    )
    
    # Single company mode
    parser.add_argument("--company", help="Single company name")
    parser.add_argument("--ticker", help="Single company ticker")
    
    # Bulk modes
    parser.add_argument("--top-sp500", type=int, help="Process top N S&P 500 companies")
    parser.add_argument("--all-sp100", action="store_true", help="Process all pre-loaded S&P 100 companies")
    parser.add_argument("--companies-file", help="Path to file with companies (CSV: Company Name,TICKER)")
    
    # Limits
    parser.add_argument("--max", action="store_true", help="Maximum data per company (recommended)")
    parser.add_argument("--sec", type=int, default=100, help="Max SEC filings (default: 100)")
    parser.add_argument("--press", type=int, default=200, help="Max press releases (default: 200)")
    parser.add_argument("--news", type=int, default=200, help="Max news articles (default: 200)")
    parser.add_argument("--transcripts", type=int, default=50, help="Max transcripts (default: 50)")
    
    # Performance
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers (default: 5)")
    
    args = parser.parse_args()
    
    # Determine limits
    if args.max:
        limits = {
            "sec": 500,
            "press": 500,
            "news": 500,
            "transcripts": 100
        }
    else:
        limits = {
            "sec": args.sec,
            "press": args.press,
            "news": args.news,
            "transcripts": args.transcripts
        }
    
    # Determine companies to process
    companies = []
    
    if args.company and args.ticker:
        # Single company mode
        companies = [(args.company, args.ticker)]
        
    elif args.top_sp500:
        # Top N S&P 500
        companies = SP500_TOP_100[:args.top_sp500]
        
    elif args.all_sp100:
        # All pre-loaded S&P 100
        companies = SP500_TOP_100
        
    elif args.companies_file:
        # Load from file
        file_path = Path(args.companies_file)
        if not file_path.exists():
            print(f"❌ File not found: {args.companies_file}")
            return
        
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    companies.append((parts[0].strip(), parts[1].strip()))
    
    else:
        print("❌ Please specify companies to process:")
        print("   --company + --ticker (single)")
        print("   --top-sp500 N (top N companies)")
        print("   --all-sp100 (all S&P 100)")
        print("   --companies-file FILE (custom list)")
        return
    
    # Run ingestion
    manager = BulkIngestionManager(max_workers=args.workers)
    manager.ingest_bulk(companies, limits)


if __name__ == "__main__":
    main()