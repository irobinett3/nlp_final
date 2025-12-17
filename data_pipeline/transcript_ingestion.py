#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# transcript_ingestion.py

import time
from typing import Optional
from urllib.parse import urljoin
import feedparser
from bs4 import BeautifulSoup

from .utils import http_get, clean_text, parse_date, is_relevant
from .database import upsert_document

# =========================
# Yahoo Finance Earnings Transcripts
# =========================

def ingest_yahoo_earnings_transcripts(company: str, ticker: str, max_items=5):
    """
    Ingest earnings call transcripts from Yahoo Finance.
    
    Args:
        company: Company name
        ticker: Stock ticker (required for Yahoo)
        max_items: Maximum number of transcripts to fetch
    """
    if not ticker:
        return

    base = f"https://finance.yahoo.com/quote/{ticker}/earnings"
    try:
        r = http_get(base)
    except Exception as e:
        print(f"[Transcript] Yahoo earnings page error: {e}")
        return

    soup = BeautifulSoup(r.text, "lxml")

    links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if "earnings-call-transcript" in href:
            if href.startswith("/"):
                href = urljoin("https://finance.yahoo.com", href)
            links.append(href)

    links = list(dict.fromkeys(links))[:max_items]

    for link in links:
        try:
            rr = http_get(link)
            ss = BeautifulSoup(rr.text, "lxml")

            title = ss.find("h1")
            title = title.get_text(strip=True) if title else "Earnings Call Transcript"

            body = ""
            article = ss.find("article")
            if article:
                body = article.get_text(" ", strip=True)
            else:
                body = clean_text(rr.text)

            if not is_relevant(body, company, ticker):
                continue

            tnode = ss.find("time")
            published = tnode.get("datetime") if tnode else None

            upsert_document({
                "company": company,
                "ticker": ticker,
                "source": "Transcript",
                "title": title,
                "url": link,
                "published_at": parse_date(published),
                "raw_text": body,
                "metadata": {"source": "yahoo_earnings"}
            })

            print(f"[Transcript] Yahoo: {title}")

        except Exception as e:
            print(f"[Transcript] Yahoo transcript error: {e}")


# =========================
# SeekingAlpha RSS Transcripts
# =========================

def ingest_seekingalpha_rss_transcripts(company: str, ticker: Optional[str], max_items=10):
    """
    Ingest earnings call transcripts from SeekingAlpha RSS feed.
    
    Args:
        company: Company name
        ticker: Stock ticker (required for SeekingAlpha)
        max_items: Maximum number of transcripts to fetch
    """
    if not ticker:
        return

    feed_url = f"https://seekingalpha.com/api/sa/combined/{ticker.upper()}.xml"

    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        print(f"[Transcript] SeekingAlpha RSS error: {e}")
        return

    count = 0
    for entry in feed.entries:
        if count >= max_items:
            break

        title = entry.get("title", "")
        link = entry.get("link", "")
        published = entry.get("published")

        # HARD FILTER: transcripts only
        if "earnings call transcript" not in title.lower():
            continue

        try:
            rr = http_get(link, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(rr.text, "lxml")

            article = soup.find("article")
            body = article.get_text(" ", strip=True) if article else clean_text(rr.text)

            if not is_relevant(body, company, ticker):
                continue

            upsert_document({
                "company": company,
                "ticker": ticker,
                "source": "Transcript",
                "title": title,
                "url": link,
                "published_at": parse_date(published),
                "raw_text": body,
                "metadata": {"source": "seekingalpha_rss"}
            })

            print(f"[Transcript] SeekingAlpha RSS: {title}")
            count += 1
            time.sleep(1)

        except Exception as e:
            print(f"[Transcript] SeekingAlpha article error: {e}")