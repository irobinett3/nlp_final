#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# press_release_ingestion.py

from typing import Optional, List
import feedparser

from .utils import http_get, clean_text, parse_date, is_relevant
from .database import upsert_document

# =========================
# GlobeNewswire RSS
# =========================

def find_globenewswire_feeds(company: str) -> List[str]:
    """
    Return list of GlobeNewswire RSS feed URLs to check.
    Simple heuristic: GlobeNewswire has company/category feeds but not always predictable.
    For MVP, search a site-wide feed and filter by company name, plus a few common endpoints.
    """
    return [
        "https://www.globenewswire.com/RssFeed/industry/Technology.xml",
        "https://www.globenewswire.com/RssFeed/industry/Financial%20Services.xml",
        "https://www.globenewswire.com/RssFeed/industry/Healthcare.xml",
        "https://www.globenewswire.com/RssFeed/industry/Consumer%20Goods.xml",
        # fallback mega feed:
        "https://www.globenewswire.com/RssFeed/Top%20News%20Releases.xml",
    ]


def ingest_globenewswire(company: str, ticker: Optional[str], max_items: int = 50):
    """
    Ingest press releases from GlobeNewswire RSS feeds.
    
    Args:
        company: Company name
        ticker: Stock ticker (optional)
        max_items: Maximum number of items to fetch
    """
    feeds = find_globenewswire_feeds(company)
    company_lc = company.lower()
    seen = 0
    
    for feed_url in feeds:
        try:
            fp = feedparser.parse(feed_url)
            for entry in fp.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                link = entry.get("link", "")
                published = entry.get("published") or entry.get("updated")
                
                if company_lc in (title + " " + summary).lower():
                    # fetch full page for body text
                    text = ""
                    try:
                        r = http_get(link)
                        text = clean_text(r.text)
                        if not is_relevant(text, company, ticker):
                            continue
                    except Exception:
                        # fall back to summary if fetch fails
                        text = clean_text(summary)
                    
                    doc = {
                        "company": company,
                        "ticker": ticker,
                        "source": "PressRelease",
                        "title": title,
                        "url": link,
                        "published_at": parse_date(published) or published,
                        "raw_text": text,
                        "metadata": {"feed": feed_url}
                    }
                    upsert_document(doc)
                    seen += 1
                    print(f"[PR] Saved: {title}")
                    if seen >= max_items:
                        return
        except Exception as e:
            print(f"[PR] Feed error {feed_url}: {e}")