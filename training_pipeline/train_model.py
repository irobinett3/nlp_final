#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# training_pipeline/train_model.py

"""
Train a fine-tuned FinBERT model for risk classification
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Config
# Config
MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 512
BATCH_SIZE = 4  # ← REDUCED for M1 Pro memory
LEARNING_RATE = 2e-5
EPOCHS = 5
WARMUP_STEPS = 100  # ← Also reduced since we have fewer batches


class RiskModelTrainer:
    """Train fine-tuned risk classification model"""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("models") / f"risk_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Data: {self.data_dir}")
        print(f"📁 Output: {self.output_dir}")
        
        # Load tokenizer
        print("\n🔧 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Label mapping
        self.label2id = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'catastrophic': 4}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def load_data(self):
        """Load and prepare datasets"""
        print("\n📂 Loading datasets...")
        
        def load_jsonl(filepath):
            data = []
            with open(filepath) as f:
                for line in f:
                    item = json.loads(line)
                    # Extract text preview and label
                    text = item.get('text_preview', '')
                    label = item.get('labels', {}).get('risk_category', 'none')
                    if text and label in self.label2id:
                        data.append({'text': text, 'label': label})
            return data
        
        train_data = load_jsonl(self.data_dir / 'classification' / 'train.jsonl')
        val_data = load_jsonl(self.data_dir / 'classification' / 'val.jsonl')
        test_data = load_jsonl(self.data_dir / 'classification' / 'test.jsonl')
        
        print(f"  Train: {len(train_data)} examples")
        print(f"  Val: {len(val_data)} examples")
        print(f"  Test: {len(test_data)} examples")
        
        # Convert to HuggingFace datasets
        self.train_dataset = Dataset.from_dict({
            'text': [d['text'] for d in train_data],
            'label': [self.label2id[d['label']] for d in train_data]
        })
        
        self.val_dataset = Dataset.from_dict({
            'text': [d['text'] for d in val_data],
            'label': [self.label2id[d['label']] for d in val_data]
        })
        
        self.test_dataset = Dataset.from_dict({
            'text': [d['text'] for d in test_data],
            'label': [self.label2id[d['label']] for d in test_data]
        })
        
        # Tokenize
        print("\n🔤 Tokenizing...")
        self.train_dataset = self.train_dataset.map(self._tokenize, batched=True)
        self.val_dataset = self.val_dataset.map(self._tokenize, batched=True)
        self.test_dataset = self.test_dataset.map(self._tokenize, batched=True)
        
        # Set format
        self.train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        self.val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        self.test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    def _tokenize(self, examples):
        """Tokenize texts"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH
        )
    
    def train(self):
        """Train the model"""
        print("\n🚀 Training model...")
        
        # Load model - ignore pre-trained head (3 classes -> 5 classes)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True  # ← ADD THIS LINE
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
            logging_steps=100,
            eval_strategy="epoch",  # ← CHANGED from evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        trainer.train()
        
        # Save
        print(f"\n💾 Saving model to {self.output_dir}")
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        return trainer
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics during training"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import accuracy_score, f1_score
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1_macro': f1_score(labels, predictions, average='macro'),
            'f1_weighted': f1_score(labels, predictions, average='weighted')
        }
    
    def evaluate(self, trainer):
        """Evaluate on test set"""
        print("\n📊 Evaluating on test set...")
        
        predictions = trainer.predict(self.test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Classification report
        report = classification_report(
            true_labels,
            pred_labels,
            target_names=list(self.id2label.values()),
            digits=3
        )
        
        print("\nClassification Report:")
        print(report)
        
        # Save report
        with open(self.output_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        self._plot_confusion_matrix(cm)
        
        return {
            'predictions': pred_labels.tolist(),
            'true_labels': true_labels.tolist(),
            'report': report
        }
    
    def _plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.id2label.values()),
            yticklabels=list(self.id2label.values())
        )
        plt.title('Confusion Matrix - Fine-tuned Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        print(f"  ✓ Saved confusion matrix to {self.output_dir / 'confusion_matrix.png'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train risk classification model")
    parser.add_argument('--data-dir', required=True, help='Path to training data directory')
    parser.add_argument('--output-dir', help='Where to save the model')
    
    args = parser.parse_args()
    
    try:
        trainer_obj = RiskModelTrainer(args.data_dir, args.output_dir)
        trainer_obj.load_data()
        trainer = trainer_obj.train()
        trainer_obj.evaluate(trainer)
        
        print(f"\n✓ Training complete! Model saved to {trainer_obj.output_dir}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()