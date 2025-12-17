#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# sec_ingestion.py

import time
import random
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

from .config import SEC_BASE, SEC_COMPANY_TICKERS, SEC_UA, RETRY_STATUSES
from .utils import http_get, clean_text, parse_date, backoff_sleep
from .database import upsert_document

# =========================
# SEC: CIK/Ticker Resolution
# =========================

def load_sec_company_index() -> List[Dict[str, Any]]:
    """
    Returns list of dicts: {cik_str, ticker, title}
    """
    resp = http_get(SEC_COMPANY_TICKERS)
    # The JSON is indexed by number strings: {"0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}, ...}
    data = resp.json()
    rows = []
    for _, v in data.items():
        rows.append({
            "cik_str": str(v.get("cik_str")).zfill(10),
            "ticker": v.get("ticker"),
            "title": v.get("title"),
        })
    return rows


def resolve_company(company: Optional[str], ticker: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (normalized_company_title, ticker, cik)
    """
    index = load_sec_company_index()
    by_ticker = {(row["ticker"] or "").upper(): row for row in index if row.get("ticker")}
    if ticker:
        row = by_ticker.get(ticker.upper())
        if row:
            return row["title"], row["ticker"], row["cik_str"]

    # name match (simple heuristic)
    norm = (company or "").lower()
    best = None
    for row in index:
        title = (row["title"] or "").lower()
        if norm and norm in title:
            best = row
            break

    if best:
        return best["title"], best["ticker"], best["cik_str"]
    return company, ticker, None


# =========================
# SEC: Submissions and Filings
# =========================

def fetch_sec_submissions(cik: str) -> Dict[str, Any]:
    """Fetch SEC submissions JSON for a given CIK."""
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    time.sleep(0.2)  # be polite
    r = http_get(url, headers={"User-Agent": SEC_UA})
    return r.json()


def list_filing_docs(cik: str, accession_no: str) -> List[str]:
    """
    Return absolute URLs for all files in a filing directory, using JSON first,
    then falling back to parsing the HTML directory listing.
    """
    acc_no_nodashes = accession_no.replace("-", "")
    base_dir = f"{SEC_BASE}/Archives/edgar/data/{int(cik)}/{acc_no_nodashes}/"
    index_json = base_dir + "index.json"

    # 1) Try JSON index with small retries for transient issues
    for attempt in range(3):
        try:
            r = http_get(index_json, headers={"User-Agent": SEC_UA})
            items = r.json().get("directory", {}).get("item", [])
            if items:
                return [base_dir + it["name"] for it in items if it.get("name")]
            break
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in RETRY_STATUSES:
                backoff_sleep(attempt)
                continue
            if status == 404:
                # fall through to HTML listing
                break
            # Other hard errors: give up JSON, try HTML
            break
        except Exception:
            break

    # 2) Fallback: fetch the HTML directory listing and parse anchors
    dir_html = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_nodashes}/"
    for attempt in range(3):
        rr = http_get(dir_html, headers={"User-Agent": SEC_UA})
        soup = BeautifulSoup(rr.text, "lxml")
        urls = []
        for a in soup.select("a[href]"):
            href = a["href"].strip()
            if not href or href.startswith("?"):
                continue
            if not href.lower().endswith((".htm", ".html", ".txt", ".pdf", ".xml")):
                continue
            # Normalize to absolute
            if href.startswith("/"):
                full = urljoin("https://www.sec.gov", href)
            else:
                full = urljoin(dir_html, href)
            urls.append(full)
        # dedupe
        seen = set()
        out = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out
    return []


def pick_best_filing_url_from_list(candidates: List[str], primary_doc: str | None, ticker: str | None = None) -> str | None:
    """
    Choose the most reliable document to parse from a list of filing URLs.
    Preference order:
      1) <ticker>-<yyyymmdd>.htm style main file
      2) filing-detail.html
      3) complete submission .txt
      4) primaryDocument (when present)
      5) any .htm/.html
      6) any .txt
      7) any .pdf
    """
    lower = [u.lower() for u in candidates]

    # 0) a <ticker>-<yyyymmdd>.htm style main file
    if ticker:
        for i, u in enumerate(lower):
            name = u.rsplit("/", 1)[-1]
            if name.startswith(ticker.lower()) and name.endswith((".htm", ".html")):
                return candidates[i]

    # 1) filing-detail.html
    for i, u in enumerate(lower):
        if "filing-detail" in u and u.endswith(".html"):
            return candidates[i]

    # 2) complete submission .txt
    for i, u in enumerate(lower):
        if u.endswith(".txt"):
            return candidates[i]

    # 3) primary doc (if present)
    if primary_doc:
        for i, u in enumerate(lower):
            if u.rsplit("/", 1)[-1] == primary_doc.lower():
                return candidates[i]

    # 4) any .htm/.html
    for i, u in enumerate(lower):
        if u.endswith((".htm", ".html")):
            return candidates[i]

    # 5) any .txt
    for i, u in enumerate(lower):
        if u.endswith(".txt"):
            return candidates[i]

    # (optionally consider PDFs last)
    for i, u in enumerate(lower):
        if u.endswith(".pdf"):
            return candidates[i]

    return None


def fetch_and_parse_sec_doc(cik: str, accession_no: str, primary_doc: str | None, ticker: str | None = None) -> Tuple[str | None, str | None]:
    """
    Get a robust list of files, pick the best one, fetch it, and return (url, cleaned_text).
    """
    candidates = list_filing_docs(cik, accession_no)
    if not candidates:
        return None, None
    tried = set()
    for _ in range(3):
        final_url = pick_best_filing_url_from_list(candidates, primary_doc, ticker)
        if not final_url or final_url in tried:
            break
        tried.add(final_url)
        r = http_get(final_url, headers={"User-Agent": SEC_UA})
        text = clean_text(r.text)
        if text and len(text) > 200:   # bump threshold a bit
            return final_url, text
        # remove this url and try next best
        candidates = [u for u in candidates if u != final_url]
    return None, None


# =========================
# SEC: Main Ingestion
# =========================

def ingest_sec(company: str, ticker: str | None, cik: str | None,
               forms: Tuple[str, ...] = ("8-K", "10-K", "10-Q", "6-K"),
               max_filings: int = 30):
    """
    Ingest SEC filings for a company.
    
    Args:
        company: Company name
        ticker: Stock ticker (optional)
        cik: CIK number (required)
        forms: Tuple of form types to fetch
        max_filings: Maximum number of filings to fetch
    """
    if not cik:
        print("[SEC] No CIK resolved; skipping SEC for this company.")
        return

    subs = fetch_sec_submissions(cik)
    recent = subs.get("filings", {}).get("recent", {})
    form_list = recent.get("form", [])
    acc_list = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    pulled = 0
    for i, form in enumerate(form_list):
        if form not in forms:
            continue
        if pulled >= max_filings:
            break

        acc = acc_list[i]
        fdate = filing_dates[i]
        primary_doc = primary_docs[i] if i < len(primary_docs) else None

        final_url, text = fetch_and_parse_sec_doc(cik, acc, primary_doc, ticker)
        if not final_url or not text:
            print(f"[SEC] Skipping {form} {acc} — no fetchable body (JSON+HTML fallback failed).")
            continue

        upsert_document({
            "company": company,
            "ticker": ticker,
            "source": "SEC",
            "title": f"{company} - {form}",
            "url": final_url,
            "published_at": parse_date(fdate) or fdate,
            "raw_text": text,
            "metadata": {"form": form, "cik": cik, "accession": acc, "chosen_url": final_url}
        })
        pulled += 1
        print(f"[SEC] Saved {form} {acc} -> {final_url}")
        time.sleep(0.25)  # be polite