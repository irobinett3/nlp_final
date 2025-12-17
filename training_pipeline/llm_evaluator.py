#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# llm_evaluator.py

"""LLM baseline evaluation for risk detection"""

import os
import json
import time
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from .config import OPENAI_API_KEY, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Import evaluation metrics if available
try:
    from .evaluation_metrics import ClassificationMetrics, LLMBaselineMetrics
except ImportError:
    # Fallback if evaluation_metrics doesn't exist
    class ClassificationMetrics:
        def compute_all_metrics(self, y_true, y_pred):
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_weighted': precision,
                'recall_weighted': recall,
                'f1_weighted': f1,
            }
    
    class LLMBaselineMetrics:
        def compute_cost_metrics(self, predictions, model, input_tokens, output_tokens):
            # gpt-4o-mini pricing
            input_cost = sum(input_tokens) / 1_000_000 * 0.15
            output_cost = sum(output_tokens) / 1_000_000 * 0.60
            total_cost = input_cost + output_cost
            return {
                'total_cost': total_cost,
                'cost_per_document': total_cost / len(predictions) if predictions else 0,
                'total_input_tokens': sum(input_tokens),
                'total_output_tokens': sum(output_tokens),
            }
        
        def compute_latency_metrics(self, latencies):
            return {
                'avg_latency': np.mean(latencies) if latencies else 0,
                'median_latency': np.median(latencies) if latencies else 0,
                'p95_latency': np.percentile(latencies, 95) if latencies else 0,
            }

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not installed. LLM evaluation will be limited.")

# LLM strategies
LLM_STRATEGIES = ['zero_shot', 'few_shot', 'chain_of_thought', 'structured_output']
LLM_MAX_TEST_SAMPLES = 30


class LLMEvaluator:
    """Comprehensive LLM baseline evaluation (OpenAI only)"""
    
    def __init__(self):
        print("[LLMEval] Initializing...")
        
        self.models = []
        self.client = None
        
        if HAS_OPENAI and OPENAI_API_KEY:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.models.append('gpt-4o-mini')
            print(f"[LLMEval] ✅ OpenAI client initialized")
        else:
            print(f"[LLMEval] ⚠️  No OpenAI API key found")
        
        self.strategies = LLM_STRATEGIES
        
        print(f"[LLMEval] Will evaluate {len(self.models)} models × {len(self.strategies)} strategies")
    
    def evaluate_baseline(
        self,
        test_dataset: List[Dict],
        max_samples: int = None
    ) -> Dict:
        """Evaluate LLM performance on test set"""
        
        if not self.models:
            print("[LLMEval] ⚠️  No models available - skipping LLM evaluation")
            return {'error': 'No LLM models available (missing OpenAI API key)'}
        
        # Sample for cost control
        max_samples = max_samples or LLM_MAX_TEST_SAMPLES
        test_sample = test_dataset[:max_samples]
        
        print(f"[LLMEval] Testing on {len(test_sample)} examples")
        
        results = {}
        
        for model in self.models:
            for strategy in self.strategies:
                key = f"{model}_{strategy}"
                print(f"\n[LLMEval] Evaluating {key}...")
                
                try:
                    metrics = self._evaluate_model_strategy(
                        model,
                        strategy,
                        test_sample
                    )
                    results[key] = metrics
                    
                    print(f"  ✅ F1: {metrics.get('f1_weighted', 0):.3f}, Cost: ${metrics.get('total_cost', 0):.3f}")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    results[key] = {'error': str(e)}
        
        # Find best configuration
        best = self._find_best_config(results)
        
        return {
            'individual_results': results,
            'best_configuration': best,
            'summary': self._create_summary(results)
        }
    
    def _evaluate_model_strategy(
        self,
        model: str,
        strategy: str,
        test_data: List[Dict]
    ) -> Dict:
        """Evaluate specific model + strategy"""
        
        predictions = []
        ground_truth = []
        latencies = []
        input_tokens = []
        output_tokens = []
        
        for doc in tqdm(test_data, desc=f"{model} {strategy}"):
            start = time.time()
            
            # Generate prompt
            prompt = self._generate_prompt(strategy, doc)
            
            # Call LLM
            try:
                result = self._call_llm(model, prompt, strategy)
                predictions.append(result)
                ground_truth.append(doc['labels'])
                
                # Track tokens (estimate)
                input_tokens.append(len(prompt.split()) * 1.3)
                output_tokens.append(len(str(result).split()) * 1.3)
                
            except Exception as e:
                print(f"    ⚠️  Error on doc {doc.get('id', '?')}: {e}")
                predictions.append({'risk_level': 'medium_risk', 'confidence': 0.5})
                ground_truth.append(doc['labels'])
                input_tokens.append(1000)
                output_tokens.append(100)
            
            latencies.append(time.time() - start)
            
            # Rate limiting (be nice to API)
            time.sleep(0.1)
        
        # Compute metrics
        y_true = [gt['risk_category'] for gt in ground_truth]
        y_pred = [p.get('risk_level', 'medium_risk') for p in predictions]
        
        clf_metrics = ClassificationMetrics().compute_all_metrics(y_true, y_pred)
        
        # Cost metrics
        llm_metrics = LLMBaselineMetrics()
        cost_metrics = llm_metrics.compute_cost_metrics(
            predictions, model, input_tokens, output_tokens
        )
        latency_metrics = llm_metrics.compute_latency_metrics(latencies)
        
        return {
            **clf_metrics,
            **cost_metrics,
            **latency_metrics,
            'model': model,
            'strategy': strategy
        }
    
    def _generate_prompt(self, strategy: str, doc: Dict) -> str:
        """Generate prompt based on strategy"""
        
        text = doc.get('raw_text', '')[:3000]  # Limit to 3000 chars
        
        if strategy == 'zero_shot':
            return f"""Analyze this financial document for risk. Rate as: no_risk, low_risk, medium_risk, high_risk, or catastrophic_risk.

Document: {text}

Risk Level:"""
        
        elif strategy == 'few_shot':
            examples = """Example 1: "Company reported 50% revenue decline" → Risk: high_risk
Example 2: "Strong quarterly results exceeded expectations" → Risk: no_risk
Example 3: "SEC investigation announced" → Risk: catastrophic_risk

"""
            return f"""{examples}Now analyze:

Document: {text}

Risk Level:"""
        
        elif strategy == 'chain_of_thought':
            return f"""Analyze this financial document step-by-step:

Document: {text}

Think step by step:
1. What type of document is this?
2. Key financial indicators?
3. Any concerning trends?
4. Overall risk level (no_risk, low_risk, medium_risk, high_risk, catastrophic_risk)?

Answer:"""
        
        elif strategy == 'structured_output':
            return f"""Analyze this document. Return JSON only:

Document: {text}

{{
  "risk_level": "no_risk|low_risk|medium_risk|high_risk|catastrophic_risk",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
        
        return text
    
    def _call_llm(self, model: str, prompt: str, strategy: str) -> Dict:
        """Call LLM API (OpenAI only)"""
        
        if not self.client:
            raise ValueError("No OpenAI client available")
        
        # Prepare request
        messages = [{"role": "user", "content": prompt}]
        
        # Add response format for structured output
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        }
        
        if strategy == 'structured_output':
            kwargs["response_format"] = {"type": "json_object"}
        
        # Make API call
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        
        # Parse response
        if strategy == 'structured_output':
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return self._parse_text_response(content)
        else:
            return self._parse_text_response(content)
    
    def _parse_text_response(self, text: str) -> Dict:
        """Parse free-form text response"""
        text_lower = text.lower()
        
        # Map to our risk categories
        if 'catastrophic' in text_lower:
            level = 'catastrophic_risk'
        elif 'high' in text_lower:
            level = 'high_risk'
        elif 'medium' in text_lower:
            level = 'medium_risk'
        elif 'low' in text_lower:
            level = 'low_risk'
        elif 'no' in text_lower or 'none' in text_lower:
            level = 'no_risk'
        else:
            level = 'medium_risk'  # Default
        
        return {
            'risk_level': level,
            'confidence': 0.7,
            'reasoning': text[:200]
        }
    
    def _find_best_config(self, results: Dict) -> Dict:
        """Find best model + strategy combination"""
        
        best_f1 = 0
        best_config = None
        
        for key, metrics in results.items():
            if 'error' in metrics:
                continue
            
            f1 = metrics.get('f1_weighted', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_config = {
                    'config': key,
                    'model': metrics['model'],
                    'strategy': metrics['strategy'],
                    'f1': f1,
                    'precision': metrics.get('precision_weighted', 0),
                    'recall': metrics.get('recall_weighted', 0),
                    'cost_per_doc': metrics.get('cost_per_document', 0),
                    'total_cost': metrics.get('total_cost', 0),
                }
        
        return best_config or {}
    
    def _create_summary(self, results: Dict) -> Dict:
        """Create summary statistics"""
        
        valid_results = [r for r in results.values() if 'error' not in r]
        
        if not valid_results:
            return {}
        
        return {
            'num_configs_tested': len(valid_results),
            'avg_f1': np.mean([r.get('f1_weighted', 0) for r in valid_results]),
            'best_f1': max(r.get('f1_weighted', 0) for r in valid_results),
            'avg_cost': np.mean([r.get('cost_per_document', 0) for r in valid_results]),
            'min_cost': min(r.get('cost_per_document', float('inf')) for r in valid_results),
            'total_cost_all_configs': sum(r.get('total_cost', 0) for r in valid_results),
        }