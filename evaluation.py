"""
evaluation.py
=============
Evaluates and benchmarks all models:
- Traditional ML: Naive Bayes, Logistic Regression, Random Forest
- Deep Learning: BERT Transformer Phishing Detector
Generates comparative metrics table, confusion matrices, and visualizations.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report, confusion_matrix)

from bert_classifier import get_bert_detector

print("Loading test data and baseline models...")
X_test_e, y_test_e, X_test_u, y_test_u = joblib.load('models/test_data.pkl')

nb_email = joblib.load('models/nb_email.pkl')
lr_email = joblib.load('models/lr_email.pkl')
rf_email = joblib.load('models/rf_email.pkl')
rf_url   = joblib.load('models/rf_url.pkl')

results = []

def evaluate_model(model, X_test, y_test, name, arch_type="Traditional ML"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*60}")
    print(f"  {name} ({arch_type})")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    results.append({
        'Model': name,
        'Architecture': arch_type,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc
    })

    return y_pred, y_prob


# 1. Evaluate Email Traditional Models
print("\n--- Evaluating Traditional Baseline Models on Email ---")
evaluate_model(nb_email, X_test_e, y_test_e, "Naive Bayes", "TF-IDF + Bayes")
evaluate_model(lr_email, X_test_e, y_test_e, "Logistic Regression", "TF-IDF + Linear")
evaluate_model(rf_email, X_test_e, y_test_e, "Random Forest", "TF-IDF + Ensembles")

# 2. Evaluate BERT Transformer Model on Email
print("\n--- Evaluating BERT Transformer on Email ---")
email_df = pd.read_csv('datasets/email_clean.csv')
email_df.dropna(subset=['text', 'label'], inplace=True)

# Generate matching test split
_, test_texts, _, test_labels = train_test_split(
    email_df['text'].tolist(), email_df['label'].tolist(),
    test_size=0.2, random_state=42, stratify=email_df['label'].tolist()
)

# For fast evaluation on CPU, evaluate on test samples
sample_eval_count = min(500, len(test_texts))
print(f"Running BERT inference on {sample_eval_count} test email samples...")
sample_texts = test_texts[:sample_eval_count]
sample_labels = np.array(test_labels[:sample_eval_count])

bert = get_bert_detector()
bert_preds, bert_probs = bert.predict_batch(sample_texts, batch_size=32, max_length=256)

bert_acc = accuracy_score(sample_labels, bert_preds)
bert_prec = precision_score(sample_labels, bert_preds, zero_division=0)
bert_rec = recall_score(sample_labels, bert_preds, zero_division=0)
bert_f1 = f1_score(sample_labels, bert_preds, zero_division=0)
bert_roc = roc_auc_score(sample_labels, bert_probs)

print(f"\n{'='*60}")
print(f"  BERT Transformer Phishing Detector (Deep Learning)")
print(f"{'='*60}")
print(f"  Accuracy : {bert_acc:.4f} ({bert_acc*100:.2f}%)")
print(f"  Precision: {bert_prec:.4f}")
print(f"  Recall   : {bert_rec:.4f}")
print(f"  F1 Score : {bert_f1:.4f}")
print(f"  ROC-AUC  : {bert_roc:.4f}")
print(classification_report(sample_labels, bert_preds, target_names=['Legitimate', 'Phishing']))

results.append({
    'Model': 'BERT Transformer',
    'Architecture': 'Deep Self-Attention (SOTA)',
    'Accuracy': bert_acc,
    'Precision': bert_prec,
    'Recall': bert_rec,
    'F1-Score': bert_f1,
    'ROC-AUC': bert_roc
})

# 3. Evaluate URL Model
print("\n--- Evaluating URL Random Forest Model ---")
evaluate_model(rf_url, X_test_u, y_test_u, "URL Random Forest", "Lexical + Ensembles")

# 4. Summary Comparison Table
comparison_df = pd.DataFrame(results)
print("\n" + "="*80)
print("             OVERALL MODEL PERFORMANCE COMPARISON TABLE")
print("="*80)
print(comparison_df.to_string(index=False))

# Save summary to CSV
comparison_df.to_csv('models/model_comparison_results.csv', index=False)
print("\nComparison results saved to 'models/model_comparison_results.csv'.")

# 5. Plot Comparison Chart
email_models_df = comparison_df[comparison_df['Model'] != 'URL Random Forest']

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(email_models_df))
width = 0.2

ax.bar(x - width*1.5, email_models_df['Accuracy'], width, label='Accuracy', color='#3498db')
ax.bar(x - width*0.5, email_models_df['Precision'], width, label='Precision', color='#2ecc71')
ax.bar(x + width*0.5, email_models_df['Recall'], width, label='Recall', color='#e67e22')
ax.bar(x + width*1.5, email_models_df['F1-Score'], width, label='F1-Score', color='#9b59b6')

ax.set_ylabel('Score (0.0 - 1.0)')
ax.set_title('Email Phishing Detection: BERT vs Traditional ML Baselines')
ax.set_xticks(x)
ax.set_xticklabels(email_models_df['Model'])
ax.set_ylim(0.70, 1.02)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('models/accuracy_comparison_chart.png', dpi=300)
print("Comparison chart saved as 'models/accuracy_comparison_chart.png'.")
plt.close()
