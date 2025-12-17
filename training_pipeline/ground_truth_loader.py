#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ground_truth_loader.py

"""Load historical ground truth events for training"""

import sqlite3
from typing import List, Dict

from .config import DB_PATH
from ground_truth_events import GROUND_TRUTH_EVENTS


def load_ground_truth_documents(db_path: str = DB_PATH) -> List[Dict]:
    """
    Load documents from historical events with known outcomes
    
    Returns:
        List of documents with perfect labels from ground truth events
    """
    print("[GroundTruth] Loading historical events...")
    
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cursor = con.cursor()
    
    ground_truth_docs = []
    
    for event in GROUND_TRUTH_EVENTS:
        # Fetch documents from the time period
        cursor.execute("""
            SELECT * FROM documents
            WHERE company = ?
            AND published_at BETWEEN ? AND ?
            ORDER BY published_at
        """, (
            event['company'],
            event['date_range'][0],
            event['date_range'][1]
        ))
        
        rows = cursor.fetchall()
        
        for row in rows:
            doc = dict(row)
            doc['ground_truth_event'] = event
            doc['labels'] = {
                'risk_category': event['severity'],
                'risk_types': event['risk_type'],
                'confidence': 1.0,  # Perfect confidence - we know the outcome
                'ground_truth': True
            }
            doc['quality_tier'] = 0  # Highest quality
            doc['weight'] = 1.5  # Extra weight in training
            
            ground_truth_docs.append(doc)
    
    con.close()
    
    print(f"[GroundTruth] Loaded {len(ground_truth_docs)} documents from {len(GROUND_TRUTH_EVENTS)} events")
    
    return ground_truth_docs