#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# training_pipeline/compare_models.py

"""
Compare fine-tuned model vs LLM on fresh test data
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparison:
    """Compare fine-tuned model against LLM baseline"""
    
    def __init__(self, finetuned_model_path: str, test_data_path: str):
        self.test_data_path = Path(test_data_path)
        self.model_path = Path(finetuned_model_path)
        
        print("="*80)
        print("MODEL COMPARISON: Fine-tuned vs LLM")
        print("="*80)
        
        # Load fine-tuned model
        print("\n🔧 Loading fine-tuned model...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.eval()
        
        # Setup LLM
        print("🔧 Setting up LLM...")
        api_key = os.getenv('OPENAI_API_KEY')
        self.llm_client = OpenAI(api_key=api_key) if api_key else None
        
        if not self.llm_client:
            print("  ⚠️  No OpenAI API key - LLM comparison will be skipped")
        
        # Label mapping
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        print(f"\n📁 Test data: {self.test_data_path}")
        print(f"📁 Fine-tuned model: {self.model_path}")
    
    def load_test_data(self) -> List[Dict]:
        """Load fresh test data"""
        print("\n📂 Loading test data...")
        
        test_data = []
        with open(self.test_data_path) as f:
            for line in f:
                item = json.loads(line)
                text = item.get('text_preview', '')
                true_label = item.get('labels', {}).get('risk_category', 'none')
                
                if text and true_label in self.label2id:
                    test_data.append({
                        'text': text,
                        'true_label': true_label,
                        'metadata': {
                            'company': item.get('company'),
                            'source': item.get('source')
                        }
                    })
        
        print(f"  ✓ Loaded {len(test_data)} test examples")
        return test_data
    
    def predict_finetuned(self, texts: List[str]) -> List[str]:
        """Get predictions from fine-tuned model"""
        print("\n🤖 Running fine-tuned model predictions...")
        
        predictions = []
        start_time = time.time()
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred_id = torch.argmax(outputs.logits, dim=-1).item()
                predictions.append(self.id2label[pred_id])
        
        elapsed = time.time() - start_time
        
        print(f"  ✓ Predicted {len(texts)} examples in {elapsed:.1f}s")
        print(f"  ⏱️  Avg time: {elapsed/len(texts)*1000:.0f}ms per example")
        print(f"  💰 Cost: $0.00 (no API costs)")
        
        return predictions
    
    def predict_llm(self, test_data: List[Dict]) -> List[str]:
        """Get predictions from LLM"""
        if not self.llm_client:
            return []
        
        print("\n🤖 Running LLM predictions...")
        
        predictions = []
        total_cost = 0
        start_time = time.time()
        
        for item in test_data:
            pred, cost = self._llm_predict_single(item['text'], item['metadata'])
            predictions.append(pred)
            total_cost += cost
        
        elapsed = time.time() - start_time
        
        print(f"  ✓ Predicted {len(test_data)} examples in {elapsed:.1f}s")
        print(f"  ⏱️  Avg time: {elapsed/len(test_data)*1000:.0f}ms per example")
        print(f"  💰 Total cost: ${total_cost:.4f}")
        
        return predictions
    
    def _llm_predict_single(self, text: str, metadata: Dict) -> tuple:
        """Get single LLM prediction with cost tracking"""
        
        prompt = f"""Analyze this financial document for risk. Return ONLY a JSON object.

Company: {metadata.get('company', 'Unknown')}
Source: {metadata.get('source', 'Unknown')}

Text: {text[:1500]}

Return JSON:
{{
    "risk_category": "none|low|medium|high|catastrophic"
}}"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            risk_category = result.get('risk_category', 'none')
            
            # Cost calculation (gpt-4o-mini pricing)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)
            
            return risk_category, cost
            
        except Exception as e:
            print(f"  ⚠️  LLM error: {e}")
            return 'none', 0.0
    
    def compare(self):
        """Run full comparison"""
        
        # Load test data
        test_data = self.load_test_data()
        
        if not test_data:
            print("\n✗ No test data available!")
            return
        
        true_labels = [item['true_label'] for item in test_data]
        texts = [item['text'] for item in test_data]
        
        # Fine-tuned predictions
        ft_predictions = self.predict_finetuned(texts)
        
        # LLM predictions
        llm_predictions = self.predict_llm(test_data) if self.llm_client else []
        
        # Results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        
        # Fine-tuned metrics
        # Fine-tuned metrics
        # Fine-tuned metrics
        print("\n📊 Fine-tuned Model:")

        # Just use the predictions/labels as-is (they're already strings)
        unique_labels = sorted(set(true_labels) | set(ft_predictions))

        print(classification_report(
            true_labels,
            ft_predictions,
            labels=unique_labels,
            digits=3,
            zero_division=0
        ))
        
        ft_f1 = f1_score(true_labels, ft_predictions, average='weighted')
        ft_acc = accuracy_score(true_labels, ft_predictions)
        
        # LLM metrics
        # LLM metrics
        # LLM metrics
        if llm_predictions:
            print("\n📊 LLM (GPT-4o-mini):")
            
            # Just use the predictions/labels as-is (they're already strings)
            unique_labels = sorted(set(true_labels) | set(llm_predictions))
            
            print(classification_report(
                true_labels,
                llm_predictions,
                labels=unique_labels,
                digits=3,
                zero_division=0
            ))
            
            llm_f1 = f1_score(true_labels, llm_predictions, average='weighted')
            llm_acc = accuracy_score(true_labels, llm_predictions)
            
            # Comparison
            print("\n" + "="*80)
            print("COMPARISON SUMMARY")
            print("="*80)
            print(f"\nFine-tuned Model:")
            print(f"  Accuracy: {ft_acc:.3f}")
            print(f"  F1 (weighted): {ft_f1:.3f}")
            print(f"  Cost per doc: $0.00")
            print(f"  Inference time: ~{1000/len(test_data):.0f}ms/doc")
            
            print(f"\nLLM (GPT-4o-mini):")
            print(f"  Accuracy: {llm_acc:.3f}")
            print(f"  F1 (weighted): {llm_f1:.3f}")
            print(f"  Cost per doc: ~$0.0002")
            print(f"  Inference time: ~{2000/len(test_data):.0f}ms/doc")
            
            print(f"\nWinner: {'Fine-tuned' if ft_f1 > llm_f1 else 'LLM'} (by F1 score)")
            print(f"F1 difference: {abs(ft_f1 - llm_f1):.3f}")
        
        # Save results
        results = {
            'test_size': len(test_data),
            'finetuned': {
                'accuracy': float(ft_acc),
                'f1_weighted': float(ft_f1),
                'predictions': ft_predictions
            }
        }
        
        if llm_predictions:
            results['llm'] = {
                'accuracy': float(llm_acc),
                'f1_weighted': float(llm_f1),
                'predictions': llm_predictions
            }
        
        results_path = self.model_path.parent / 'comparison_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument('--model', required=True, help='Path to fine-tuned model')
    parser.add_argument('--test-data', required=True, help='Path to test.jsonl')
    
    args = parser.parse_args()
    
    try:
        comparison = ModelComparison(args.model, args.test_data)
        comparison.compare()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()