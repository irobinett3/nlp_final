#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# labelers.py

"""Labeling modules for risk detection"""

import json
import torch
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from captum.attr import LayerIntegratedGradients
import torch.nn.functional as F

from .config import *
from .utils import *

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class FinBERTLabeler:
    """FinBERT sentiment analysis with token attribution"""
    
    def __init__(self):
        print("[FinBERT] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        self.model.eval()
        
        # Attribution
        self.attribution = LayerIntegratedGradients(
            self.model,
            self.model.bert.embeddings
        )
        print("[FinBERT] Ready")
    
    def label_with_attribution(self, text: str) -> Dict:
        """Get sentiment + token attributions"""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            sentiment_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][sentiment_idx].item()
        
        labels = ["positive", "negative", "neutral"]
        sentiment = labels[sentiment_idx]
        
        # Compute attributions for negative sentiment
        attributions = None
        risk_spans = []
        
        if sentiment == "negative":
            # Try to compute attributions, but don't fail if it doesn't work
            # Attributions are nice-to-have for explainability, not critical
            try:
                # Simple approach: just use the attention weights as a proxy
                # This is much more stable than integrated gradients
                with torch.no_grad():
                    # Get model outputs with attention
                    outputs_with_attn = self.model(**inputs, output_attentions=True)
                    
                    # Average attention across all heads and layers
                    # attentions is a tuple of (num_layers,) each (batch, num_heads, seq_len, seq_len)
                    if hasattr(outputs_with_attn, 'attentions') and outputs_with_attn.attentions:
                        # Take last layer, average across heads
                        last_layer_attn = outputs_with_attn.attentions[-1]  # (batch, heads, seq, seq)
                        avg_attn = last_layer_attn.mean(dim=1)  # (batch, seq, seq)
                        
                        # Get attention to [CLS] token (first token)
                        cls_attn = avg_attn[0, 0, :]  # (seq,)
                        
                        # Extract tokens
                        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                        
                        # Extract high-attention spans
                        risk_spans = self._extract_risk_spans(tokens, cls_attn, ATTRIBUTION_THRESHOLD)
                        attributions = cls_attn.cpu().numpy()

                
            except Exception as e:
                print(f"[FinBERT] Attribution failed: {e}")
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'risk_score': sentiment_to_risk_score(sentiment, confidence),
            'attributions': attributions,
            'risk_spans': risk_spans
        }
    
    def _extract_risk_spans(
        self,
        tokens: List[str],
        attributions: torch.Tensor,
        threshold: float
    ) -> List[Dict]:
        """Extract continuous spans of high-attribution tokens"""
        
        spans = []
        current_span = []
        current_scores = []
        
        # Ensure attributions is 1D
        if attributions.dim() > 1:
            attributions = attributions.squeeze()
        
        # Ensure we have the right length
        min_len = min(len(tokens), len(attributions))
        
        for i in range(min_len):
            token = tokens[i]
            attr = attributions[i]
            
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            if attr > threshold:
                current_span.append(token)
                current_scores.append(attr.item() if isinstance(attr, torch.Tensor) else float(attr))
            else:
                if current_span:
                    text = self.tokenizer.convert_tokens_to_string(current_span)
                    spans.append({
                        'text': text,
                        'avg_attribution': float(np.mean(current_scores)),
                        'token_count': len(current_span)
                    })
                    current_span = []
                    current_scores = []
        
        # Last span
        if current_span:
            text = self.tokenizer.convert_tokens_to_string(current_span)
            spans.append({
                'text': text,
                'avg_attribution': float(np.mean(current_scores)),
                'token_count': len(current_span)
            })
        
        return sorted(spans, key=lambda x: x['avg_attribution'], reverse=True)


class PhrasebankLabeler:
    """Financial Phrasebank sentiment analyzer"""
    
    def __init__(self):
        print("[Phrasebank] Loading model...")
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=PHRASEBANK_MODEL
            )
            self.available = True
            print("[Phrasebank] Ready")
        except Exception as e:
            print(f"[Phrasebank] Failed to load: {e}")
            self.pipeline = None
            self.available = False
    
    def label(self, text: str) -> Dict:
        """Get sentiment from Financial Phrasebank"""
        
        if not self.available:
            return {
                'risk_score': 0.5,
                'sentiment': 'neutral',
                'confidence': 0.5
            }
        
        try:
            result = self.pipeline(text[:512], truncation=True)[0]
            sentiment = result['label'].lower()
            confidence = result['score']
            risk_score = sentiment_to_risk_score(sentiment, confidence)
            
            return {
                'risk_score': risk_score,
                'sentiment': sentiment,
                'confidence': confidence
            }
        except Exception as e:
            print(f"[Phrasebank] Error: {e}")
            return {
                'risk_score': 0.5,
                'sentiment': 'neutral',
                'confidence': 0.5
            }


class LLMLabeler:
    """LLM-based risk assessment (GPT-4)"""
    
    def __init__(self):
        print("[LLM] Initializing...")
        self.available = HAS_OPENAI and OPENAI_API_KEY
        if self.available:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            print("[LLM] Ready")  # Only here!
        else:
            self.client = None
            print("[LLM] Not available (missing API key or library)")
    
    def label(self, text: str, metadata: Dict = None) -> Dict:
        """Get risk assessment from LLM"""
        
        if not self.available:
            return {
                'risk_category': 'medium',
                'risk_score': 0.5,
                'confidence': 0.5,
                'evidence': [],
                'reasoning': 'LLM unavailable'
            }
        
        metadata = metadata or {}
        
        prompt = f"""You are a financial risk analyst. Analyze this document excerpt for potential risks.

Document Type: {metadata.get('source', 'Unknown')}
Company: {metadata.get('company', 'Unknown')}
Date: {metadata.get('published_at', 'Unknown')}

Excerpt (max 2000 chars):
{text[:2000]}

Analyze for these risk categories:
1. Financial Risk (liquidity, debt, revenue decline)
2. Operational Risk (supply chain, management changes, cybersecurity)
3. Legal/Regulatory Risk (lawsuits, investigations, compliance)
4. Reputational Risk (scandals, negative publicity)
5. Market Risk (competition, market conditions)

Return ONLY valid JSON (no markdown, no explanation):
{{
    "risk_category": "none|low|medium|high|catastrophic",
    "risk_types": ["financial", "operational", etc.],
    "risk_score": 0.0-1.0,
    "evidence": ["quote 1", "quote 2"],
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return {
                'risk_category': 'medium',
                'risk_score': 0.5,
                'confidence': 0.5,
                'evidence': [],
                'reasoning': f'Error: {str(e)}'
            }


class MultiSourceLabeler:
    """Aggregate labels from FinBERT, Phrasebank, and LLM"""
    
    def __init__(self):
        print("[MultiSource] Initializing all labelers...")
        self.finbert = FinBERTLabeler()
        self.phrasebank = PhrasebankLabeler()
        self.llm = None  # Disabled for speed
        print("[MultiSource] Ready (2-source labeling)")
    
    def label_document(self, text: str, metadata: Dict = None) -> Dict:
        """Generate comprehensive labels with attribution"""
        
        # Chunk long documents
        chunks = chunk_text(text, DEFAULT_CHUNK_SIZE)
        
        chunk_results = []
        for chunk in chunks[:MAX_CHUNKS_PER_DOC]:
            # Get labels from each source (FinBERT + Phrasebank only)
            finbert_result = self.finbert.label_with_attribution(chunk)
            phrasebank_result = self.phrasebank.label(chunk)
            llm_result = {'risk_category': 'N/A', 'risk_score': 0.5}  # Dummy result
            
            # Aggregate
            aggregated = self._aggregate_labels(
                finbert_result,
                phrasebank_result,
                llm_result
            )
            
            chunk_results.append(aggregated)
        
        # Combine chunks
        final_result = self._combine_chunk_results(chunk_results, text)
        
        return final_result
    
    def _aggregate_labels(
        self,
        finbert_result: Dict,
        phrasebank_result: Dict,
        llm_result: Dict
    ) -> Dict:
        """Aggregate labels from all sources with voting"""
        
        # Convert to risk categories
        fb_risk = score_to_category(finbert_result['risk_score'])
        pb_risk = score_to_category(phrasebank_result['risk_score'])
        
        # Only use 2 sources (no LLM)
        votes = [fb_risk, pb_risk]
        total_sources = 2
        
        vote_counts = Counter(votes)
        majority = vote_counts.most_common(1)[0]
        agreement_level = majority[1]
        
        # Confidence based on agreement
        confidence = agreement_level / total_sources  # Use actual source count!
        if agreement_level >= 2:
            confidence = min(confidence * 1.2, 1.0)
        
        return {
            'risk_category': majority[0],
            'confidence': confidence,
            'agreement_level': agreement_level,
            'sources': {
                'finbert': fb_risk,
                'phrasebank': pb_risk,
                'llm': 'N/A'  # Not used
            },
            'finbert_score': finbert_result['risk_score'],
            'phrasebank_score': phrasebank_result['risk_score'],
            'llm_score': 0.5,  # Not used
            'token_attributions': finbert_result.get('attributions'),
            'risk_spans': finbert_result.get('risk_spans', []),
            'llm_evidence': [],
            'llm_reasoning': 'LLM disabled for speed'
        }
    
    def _combine_chunk_results(self, chunk_results: List[Dict], full_text: str) -> Dict:
        """Combine results from multiple chunks"""
        
        if not chunk_results:
            return {
                'risk_category': 'none',
                'confidence': 0.0,
                'agreement_level': 0
            }
        
        # Average scores
        avg_finbert = np.mean([r['finbert_score'] for r in chunk_results])
        avg_phrasebank = np.mean([r['phrasebank_score'] for r in chunk_results])
        avg_llm = np.mean([r['llm_score'] for r in chunk_results])
        
        # Most common category
        categories = [r['risk_category'] for r in chunk_results]
        final_category = Counter(categories).most_common(1)[0][0]
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in chunk_results])
        
        # Combine risk spans
        all_spans = []
        for r in chunk_results:
            all_spans.extend(r.get('risk_spans', []))
        
        # Sort by attribution and keep top 10
        all_spans = sorted(all_spans, key=lambda x: x['avg_attribution'], reverse=True)[:10]
        
        return {
            'risk_category': final_category,
            'confidence': float(avg_confidence),
            'agreement_level': int(np.mean([r['agreement_level'] for r in chunk_results])),
            'finbert_score': float(avg_finbert),
            'phrasebank_score': float(avg_phrasebank),
            'llm_score': float(avg_llm),
            'risk_spans': all_spans,
            'chunk_count': len(chunk_results)
        }