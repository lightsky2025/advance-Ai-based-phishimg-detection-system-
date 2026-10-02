"""
train_bert.py
=============
Fine-tunes a BERT Transformer model for Phishing Email Detection.
Supports both fast-sample training for quick experimentation on CPU
and full-dataset training for production deployment.
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# Ensure UTF-8 console output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class PhishingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def load_data(sample_size=None):
    """Load cleaned email dataset."""
    data_path = 'datasets/email_clean.csv'
    if not os.path.exists(data_path):
        print(f"[Error] Cleaned dataset '{data_path}' not found! Run preprossing.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)

    if sample_size and 0 < sample_size < len(df):
        print(f"[Data] Sampling {sample_size} balanced examples for rapid CPU training...")
        safe_df = df[df['label'] == 0].sample(n=min(sample_size // 2, (df['label'] == 0).sum()), random_state=42)
        phish_df = df[df['label'] == 1].sample(n=min(sample_size // 2, (df['label'] == 1).sum()), random_state=42)
        df = pd.concat([safe_df, phish_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[Data] Total training dataset size: {len(df)} samples")
    print(f"       Legitimate: {(df['label'] == 0).sum()} | Phishing: {(df['label'] == 1).sum()}")
    return df


def train_bert(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==========================================")
    print(f"       BERT Fine-Tuning Pipeline          ")
    print(f"==========================================")
    print(f"Device       : {device}")
    print(f"Base Model   : {args.model_name}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Output Dir   : {args.output_dir}\n")

    # 1. Load data
    df = load_data(sample_size=args.sample_size)
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_labels = list(train_labels)
    val_labels = list(val_labels)

    # 2. Tokenize
    print(f"\n[Tokenizer] Loading tokenizer for '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("[Tokenizer] Tokenizing datasets...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")

    train_dataset = PhishingDataset(train_encodings, train_labels)
    val_dataset = PhishingDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model setup
    print(f"[Model] Initializing sequence classification model with 2 classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "LEGITIMATE", 1: "PHISHING"},
        label2id={"LEGITIMATE": 0, "PHISHING": 1}
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Training Loop
    print("\n--- Starting Training Loop ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_tensor = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_tensor
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if step % 20 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # 5. Validation Loop
        model.eval()
        val_preds = []
        val_targets = []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_tensor = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_tensor
                )
                total_val_loss += outputs.loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels_tensor.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        acc = accuracy_score(val_targets, val_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary', zero_division=0)

        epoch_duration = time.time() - epoch_start
        print(f"\n--- Epoch {epoch} Results ({epoch_duration:.1f}s) ---")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")

        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            print(f"  >>> Best F1 improved to {best_val_f1:.4f}! Saving checkpoint to '{args.output_dir}'...")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    total_duration = time.time() - start_time
    print(f"\n==========================================")
    print(f"Training Complete in {total_duration/60:.2f} minutes!")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Model saved at     : {args.output_dir}")
    print(f"==========================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Email Phishing Detection")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pretrained BERT model name or path (default: bert-base-uncased)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum token sequence length (default: 256)')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Sample size for training (default: 1000 for fast CPU demo; set to 0 or omit for full dataset)')
    parser.add_argument('--output_dir', type=str, default='models/bert_phishing',
                        help='Directory to save fine-tuned model (default: models/bert_phishing)')

    args = parser.parse_args()
    if args.sample_size == 0:
        args.sample_size = None

    train_bert(args)
