#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# news_ingestion.py

from typing import Optional
from bs4 import BeautifulSoup
from gnews import GNews

from .config import MARKETAUX_TOKEN
from .utils import http_get, clean_text, parse_date, is_relevant
from .database import upsert_document

# =========================
# Google News (GNews)
# =========================

def ingest_gnews(company: str, ticker: str | None, max_items=50):
    """
    Ingest news articles from Google News using the GNews library.
    
    Args:
        company: Company name
        ticker: Stock ticker (optional)
        max_items: Maximum number of items to fetch
    """
    google_news = GNews(
        language='en',
        country='US',
        max_results=max_items,
        period='7d'
    )
    queries = [
        company,
        f"{company} earnings",
        f"{company} acquisition",
        f"{company} lawsuit",
    ]

    results = []
    for q in queries:
        results.extend(google_news.get_news(q))
    
    for a in results:
        url = a.get("url")
        title = a.get("title")
        published = a.get("published date")
        try:
            r = http_get(url)
            text = clean_text(r.text)
        except Exception:
            text = clean_text(a.get("description", ""))

        if not is_relevant(text, company, ticker):
            continue

        upsert_document({
            "company": company,
            "ticker": ticker,
            "source": "News",
            "title": title,
            "url": url,
            "published_at": parse_date(published),
            "raw_text": text,
            "metadata": {"source": "gnews"}
        })
        print(f"[News] GNews: {title}")


# =========================
# MarketAux API
# =========================

def ingest_marketaux(company: str, ticker: Optional[str], max_items: int = 50):
    """
    Ingest news articles from MarketAux API.
    
    Args:
        company: Company name
        ticker: Stock ticker (optional)
        max_items: Maximum number of items to fetch
    """
    if not MARKETAUX_TOKEN:
        print("[News] MARKETAUX_API_TOKEN not set; skipping MarketAux.")
        return
    
    base = "https://api.marketaux.com/v1/news/all"
    queries = [
        company,
        f"{company} earnings",
        f"{company} lawsuit",
        f"{company} acquisition",
        f"{company} guidance",
        f"{company} CEO",
    ]

    for q in queries:
        params = {
            "api_token": MARKETAUX_TOKEN,
            "language": "en",
            "limit": min(max_items, 50),
        }

        if ticker:
            params["symbols"] = ticker.upper()
            params["search"] = q
            params["filter_entities"] = "true"
        else:
            params["search"] = q

        try:
            r = http_get(base, params=params)
            data = r.json()
            articles = data.get("data", [])
            for a in articles[:max_items]:
                title = a.get("title", "")
                url = a.get("url", "")
                published_at = a.get("published_at") or a.get("published")
                # fetch full article html for robust text (API summaries are often short)
                text = ""
                try:
                    rr = http_get(url)
                    text = clean_text(rr.text)
                    if not is_relevant(text, company, ticker):
                        continue
                except Exception:
                    # fallback to description from API
                    text = clean_text(a.get("description") or "")
                
                doc = {
                    "company": company,
                    "ticker": ticker,
                    "source": "News",
                    "title": title,
                    "url": url,
                    "published_at": parse_date(published_at) or published_at,
                    "raw_text": text,
                    "metadata": {
                        "marketaux": {k: a.get(k) for k in ("source", "entities", "snippet", "similar", "sentiment")}
                    }
                }
                upsert_document(doc)
                print(f"[News] MarketAux: {title}")
        except Exception as e:
            print(f"[News] MarketAux error: {e}")


# =========================
# Yahoo Finance News
# =========================

def ingest_yahoo_finance(company: str, ticker: str, max_items=30):
    """
    Ingest news articles from Yahoo Finance.
    
    Args:
        company: Company name
        ticker: Stock ticker (required for Yahoo)
        max_items: Maximum number of items to fetch
    """
    if not ticker:
        return

    url = f"https://finance.yahoo.com/quote/{ticker}/news"

    try:
        r = http_get(url)
    except Exception as e:
        print(f"[News] Yahoo Finance unavailable ({ticker}): {e}")
        return

    soup = BeautifulSoup(r.text, "lxml")

    links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if "/news/" in href:
            if href.startswith("/"):
                from urllib.parse import urljoin
                href = urljoin("https://finance.yahoo.com", href)
            links.append(href)

    links = list(dict.fromkeys(links))[:max_items]

    for link in links:
        try:
            rr = http_get(link)
            text = clean_text(rr.text)

            if not is_relevant(text, company, ticker):
                continue

            title = BeautifulSoup(rr.text, "lxml").title
            title = title.get_text(strip=True) if title else "Yahoo Finance News"

            upsert_document({
                "company": company,
                "ticker": ticker,
                "source": "News",
                "title": title,
                "url": link,
                "raw_text": text,
                "metadata": {"source": "yahoo_finance"}
            })
            print(f"[News] Yahoo: {title}")

        except Exception:
            continue