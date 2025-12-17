#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# database.py

import json
import sqlite3
from typing import Dict, Any

from .config import DB_PATH
from .utils import now_utc, sha1

# =========================
# Database Setup
# =========================

def init_db(db_path: str = DB_PATH):
    """Initialize database schema."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT,
        ticker TEXT,
        source TEXT,               -- 'SEC', 'PressRelease', 'News', 'Transcript'
        title TEXT,
        url TEXT UNIQUE,
        published_at TEXT,
        fetched_at TEXT,
        raw_text TEXT,
        text_hash TEXT,
        metadata_json TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_company ON documents(company)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_published ON documents(published_at)")
    con.commit()
    con.close()


# =========================
# Document Operations
# =========================

def upsert_document(doc: Dict[str, Any]):
    """
    Insert document if URL not present; otherwise skip.
    
    Args:
        doc: Dictionary containing document fields:
            - company: str
            - ticker: str | None
            - source: str
            - title: str
            - url: str
            - published_at: str | None
            - raw_text: str
            - metadata: dict
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    fields = ("company", "ticker", "source", "title", "url", "published_at", 
              "fetched_at", "raw_text", "text_hash", "metadata_json")
    placeholders = ",".join(["?"] * len(fields))
    values = [
        doc.get("company"),
        doc.get("ticker"),
        doc.get("source"),
        doc.get("title"),
        doc.get("url"),
        doc.get("published_at"),
        doc.get("fetched_at", now_utc().isoformat()),
        doc.get("raw_text"),
        doc.get("text_hash") or sha1((doc.get("raw_text") or "")[:500000]),
        json.dumps(doc.get("metadata", {}), ensure_ascii=False)
    ]
    try:
        cur.execute(f"INSERT INTO documents ({','.join(fields)}) VALUES ({placeholders})", values)
        con.commit()
    except sqlite3.IntegrityError:
        # URL already present
        pass
    finally:
        con.close()