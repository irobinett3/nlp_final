#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# training_pipeline/train_span_detector.py

"""
Train a BIO span detection model for highlighting risk phrases in documents
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

# Config
MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 512
BATCH_SIZE = 4  # Small for M1 Pro
LEARNING_RATE = 2e-5
EPOCHS = 5
WARMUP_STEPS = 100


class SpanDetectionTrainer:
    """Train BIO span detection model"""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("models") / f"span_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Data: {self.data_dir}")
        print(f"📁 Output: {self.output_dir}")
        
        # BIO labels
        self.label_list = ["O", "B-RISK", "I-RISK"]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        
        # Load tokenizer
        print("\n🔧 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
    def load_data(self):
        """Load and prepare BIO tagged datasets from CoNLL format"""
        print("\n📂 Loading BIO datasets...")
        
        def read_conll(filepath):
            """Read CoNLL format file"""
            sentences = []
            labels = []
            
            current_tokens = []
            current_labels = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    if not line:  # Empty line = end of sentence
                        if current_tokens:
                            sentences.append(current_tokens)
                            labels.append(current_labels)
                            current_tokens = []
                            current_labels = []
                    else:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            token, label = parts
                            current_tokens.append(token)
                            current_labels.append(label)
                
                # Don't forget last sentence
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
            
            return sentences, labels
        
        # Load train/val/test
        train_sents, train_labels = read_conll(self.data_dir / 'span_detection' / 'train.conll')
        val_sents, val_labels = read_conll(self.data_dir / 'span_detection' / 'val.conll')
        test_sents, test_labels = read_conll(self.data_dir / 'span_detection' / 'test.conll')
        
        print(f"  Train: {len(train_sents)} sentences")
        print(f"  Val: {len(val_sents)} sentences")
        print(f"  Test: {len(test_sents)} sentences")
        
        # Create datasets
        self.train_dataset = Dataset.from_dict({
            'tokens': train_sents,
            'ner_tags': [[self.label2id.get(l, 0) for l in labels] for labels in train_labels]
        })
        
        self.val_dataset = Dataset.from_dict({
            'tokens': val_sents,
            'ner_tags': [[self.label2id.get(l, 0) for l in labels] for labels in val_labels]
        })
        
        self.test_dataset = Dataset.from_dict({
            'tokens': test_sents,
            'ner_tags': [[self.label2id.get(l, 0) for l in labels] for labels in test_labels]
        })
        
        # Tokenize
        print("\n🔤 Tokenizing and aligning labels...")
        self.train_dataset = self.train_dataset.map(
            self._tokenize_and_align_labels,
            batched=True
        )
        self.val_dataset = self.val_dataset.map(
            self._tokenize_and_align_labels,
            batched=True
        )
        self.test_dataset = self.test_dataset.map(
            self._tokenize_and_align_labels,
            batched=True
        )
    
    def _tokenize_and_align_labels(self, examples):
        """Tokenize and align BIO labels with subword tokens"""
        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            truncation=True,
            is_split_into_words=True,
            max_length=MAX_LENGTH,
            padding=False
        )
        
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                # Special tokens have a word id that is None
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For other tokens in a word, use -100 (ignored in loss)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def train(self):
        """Train the span detection model"""
        print("\n🚀 Training span detection model...")
        
        # Load model
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Training args
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=WARMUP_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            save_total_limit=2,
            fp16=False,  # Disable for M1 Pro stability
            report_to="none"
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer
        )
        
        # Train
        trainer.train()
        
        # Save
        print(f"\n💾 Saving model to {self.output_dir}")
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        return trainer
    
    def _compute_metrics(self, eval_pred):
        """Compute seqeval metrics for BIO tagging"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            'precision': precision_score(true_labels, true_predictions, scheme=IOB2),
            'recall': recall_score(true_labels, true_predictions, scheme=IOB2),
            'f1': f1_score(true_labels, true_predictions, scheme=IOB2),
        }
    
    def evaluate(self, trainer):
        """Evaluate on test set"""
        print("\n📊 Evaluating on test set...")
        
        predictions, labels, _ = trainer.predict(self.test_dataset)
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Classification report
        report = classification_report(
            true_labels,
            true_predictions,
            scheme=IOB2,
            digits=3
        )
        
        print("\nSpan Detection Report:")
        print(report)
        
        # Save report
        with open(self.output_dir / 'span_evaluation_report.txt', 'w') as f:
            f.write(report)
        
        # Extract some example predictions
        print("\n📋 Example predictions:")
        for i in range(min(3, len(true_predictions))):
            print(f"\nExample {i+1}:")
            tokens = self.test_dataset[i]['tokens']
            pred_tags = true_predictions[i]
            true_tags = true_labels[i]
            
            for token, pred, true in zip(tokens[:20], pred_tags[:20], true_tags[:20]):
                match = "✓" if pred == true else "✗"
                print(f"  {match} {token:15s} | Pred: {pred:10s} | True: {true:10s}")
        
        return {
            'predictions': true_predictions,
            'labels': true_labels,
            'report': report
        }
    
    def demo_prediction(self, text: str):
        """Demo: Highlight risk spans in new text"""
        print(f"\n🔍 Demo prediction on: '{text[:100]}...'")
        
        # Load model
        model = AutoModelForTokenClassification.from_pretrained(str(self.output_dir))
        model.eval()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].numpy()
        
        # Get tokens and predictions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Extract risk spans
        risk_spans = []
        current_span = []
        
        for token, pred_id in zip(tokens, predictions):
            pred_label = self.id2label[pred_id]
            
            if pred_label == "B-RISK":
                if current_span:
                    risk_spans.append(self.tokenizer.convert_tokens_to_string(current_span))
                current_span = [token]
            elif pred_label == "I-RISK" and current_span:
                current_span.append(token)
            else:
                if current_span:
                    risk_spans.append(self.tokenizer.convert_tokens_to_string(current_span))
                    current_span = []
        
        if current_span:
            risk_spans.append(self.tokenizer.convert_tokens_to_string(current_span))
        
        print("\n🎯 Detected risk spans:")
        for span in risk_spans:
            print(f"  - {span}")
        
        return risk_spans


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BIO span detection model")
    parser.add_argument('--data-dir', required=True, help='Path to training data directory')
    parser.add_argument('--output-dir', help='Where to save the model')
    parser.add_argument('--demo-text', help='Optional: Test prediction on this text')
    
    args = parser.parse_args()
    
    try:
        # Train
        trainer_obj = SpanDetectionTrainer(args.data_dir, args.output_dir)
        trainer_obj.load_data()
        trainer = trainer_obj.train()
        trainer_obj.evaluate(trainer)
        
        print(f"\n✓ Training complete! Model saved to {trainer_obj.output_dir}")
        
        # Demo
        if args.demo_text:
            trainer_obj.demo_prediction(args.demo_text)
        else:
            # Default demo
            demo_text = "The company faces significant liquidity concerns and potential bankruptcy proceedings due to mounting debt obligations."
            trainer_obj.demo_prediction(demo_text)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()