"""
predict.py
==========
Interactive CLI for Phishing Detection using BERT Transformer and Traditional ML.
Allows users to classify emails and URLs, view confidence scores,
and compare BERT's deep learning accuracy against baseline models.
"""

import os
import sys
import re
from urllib.parse import urlparse
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from bert_classifier import get_bert_detector

# Load baseline models
print("Loading baseline models...")
email_model_rf = joblib.load('models/rf_email.pkl')
url_model_rf   = joblib.load('models/rf_url.pkl')
tfidf_vec      = joblib.load('models/tfidf_vectorizer.pkl')

def clean_text_for_tfidf(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_email_features_rf(text):
    """Extract both TF-IDF and metadata features matching the 5007 trained features."""
    clean = clean_text_for_tfidf(text)
    tfidf_feat = tfidf_vec.transform([clean])

    raw_text = str(text)
    word_count = len(raw_text.split())
    char_count = len(raw_text)
    has_url = 1 if 'http' in raw_text.lower() else 0
    url_count = len(re.findall(r'http\S+', raw_text))
    exclamation_count = raw_text.count('!')
    uppercase_ratio = sum(1 for c in raw_text if c.isupper()) / (len(raw_text) + 1)
    urgent_words = ['urgent', 'verify', 'account', 'suspend', 'click', 'password', 'login', 'security', 'alert', 'bank']
    has_urgent_words = 1 if any(w in raw_text.lower() for w in urgent_words) else 0

    extra_vals = np.array([[
        word_count, char_count, has_url, url_count,
        exclamation_count, uppercase_ratio, has_urgent_words
    ]])

    full_features = hstack([tfidf_feat, csr_matrix(extra_vals)])
    return full_features

def extract_url_features(url):
    url = str(url).strip()
    parsed = urlparse(url)
    return pd.DataFrame([{
        'url_length'          : len(url),
        'domain_length'       : len(parsed.netloc),
        'path_length'         : len(parsed.path),
        'num_dots'            : url.count('.'),
        'num_hyphens'         : url.count('-'),
        'num_underscores'     : url.count('_'),
        'num_slashes'         : url.count('/'),
        'num_at'              : url.count('@'),
        'num_question'        : url.count('?'),
        'num_equals'          : url.count('='),
        'num_digits'          : sum(c.isdigit() for c in url),
        'num_params'          : len(parsed.query.split('&')) if parsed.query else 0,
        'has_https'           : 1 if parsed.scheme == 'https' else 0,
        'has_ip'              : 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0,
        'has_suspicious_words': 1 if any(w in url.lower() for w in ['login','verify','bank','secure','account','update','confirm','paypal']) else 0,
        'subdomain_count'     : len(parsed.netloc.split('.')) - 2 if parsed.netloc else 0,
        'shortening_service'  : 1 if any(s in url for s in ['bit.ly','tinyurl','t.co','goo.gl']) else 0,
    }])


def run_predict_email_bert(text):
    print("\n[BERT Transformer Analysis]")
    detector = get_bert_detector()
    res = detector.predict(text)

    print("-" * 50)
    if res['is_phishing']:
        print(f"Result        : [!] {res['prediction']} ({res['risk_level']})")
    else:
        print(f"Result        : [OK] {res['prediction']} ({res['risk_level']})")

    print(f"Confidence    : {res['confidence']:.2%}")
    print(f"Phishing Prob : {res['phishing_prob']:.2%}")
    print(f"Legitimate Prob: {res['legit_prob']:.2%}")
    print(f"Model Engine  : {res['model_name']}")

    if res['cues']:
        print("\nIdentified Threat Indicators:")
        for cue in res['cues']:
            print(f"  - {cue}")
    print("-" * 50)


def run_predict_email_rf(text):
    print("\n[Random Forest (TF-IDF + Metadata) Analysis]")
    features = extract_email_features_rf(text)
    pred = email_model_rf.predict(features)[0]
    probs = email_model_rf.predict_proba(features)[0]
    phish_prob = probs[1]
    legit_prob = probs[0]

    print("-" * 50)
    if pred == 1:
        print(f"Result        : [!] PHISHING DETECTED")
        print(f"Confidence    : {phish_prob:.2%}")
    else:
        print(f"Result        : [OK] LEGITIMATE EMAIL")
        print(f"Confidence    : {legit_prob:.2%}")

    print(f"Phishing Prob : {phish_prob:.2%}")
    print(f"Legitimate Prob: {legit_prob:.2%}")
    print("-" * 50)


def run_compare_email_models(text):
    print("\n" + "=" * 60)
    print("   SIDE-BY-SIDE MODEL COMPARISON (BERT vs RANDOM FOREST)")
    print("=" * 60)
    run_predict_email_bert(text)
    run_predict_email_rf(text)


def run_predict_url(url):
    print("\n[URL Phishing Analysis]")
    features = extract_url_features(url)
    pred = url_model_rf.predict(features)[0]
    probs = url_model_rf.predict_proba(features)[0]
    phish_prob = probs[1]
    legit_prob = probs[0]

    print("-" * 50)
    if pred == 1:
        print(f"Result        : [!] PHISHING URL DETECTED")
        print(f"Confidence    : {phish_prob:.2%}")
    else:
        print(f"Result        : [OK] LEGITIMATE / SAFE URL")
        print(f"Confidence    : {legit_prob:.2%}")

    print(f"Phishing Prob : {phish_prob:.2%}")
    print(f"Safe URL Prob : {legit_prob:.2%}")
    print("-" * 50)


def main():
    while True:
        print("\n" + "=" * 55)
        print("    AI PHISHING DETECTION SYSTEM")
        print("    (Featuring BERT Transformer + Random Forest)")
        print("=" * 55)
        print("1. Check Email with BERT (Deep Learning - Recommended: 99%+ SOTA)")
        print("2. Check Email with Random Forest (Classic ML Baseline)")
        print("3. Compare Both Models on Email (BERT vs Random Forest)")
        print("4. Check a URL")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == '1':
            print("\nPaste email content below (press Enter on an empty line when done):")
            lines = []
            while True:
                line = input()
                if line == '':
                    break
                lines.append(line)
            email_text = '\n'.join(lines)
            if email_text.strip():
                run_predict_email_bert(email_text)
            else:
                print("Empty input. Returning to menu.")

        elif choice == '2':
            print("\nPaste email content below (press Enter on an empty line when done):")
            lines = []
            while True:
                line = input()
                if line == '':
                    break
                lines.append(line)
            email_text = '\n'.join(lines)
            if email_text.strip():
                run_predict_email_rf(email_text)
            else:
                print("Empty input. Returning to menu.")

        elif choice == '3':
            print("\nPaste email content below (press Enter on an empty line when done):")
            lines = []
            while True:
                line = input()
                if line == '':
                    break
                lines.append(line)
            email_text = '\n'.join(lines)
            if email_text.strip():
                run_compare_email_models(email_text)
            else:
                print("Empty input. Returning to menu.")

        elif choice == '4':
            url = input("\nEnter URL: ").strip()
            if url:
                run_predict_url(url)
            else:
                print("Empty URL input.")

        elif choice == '5':
            print("Exiting Phishing Detection System. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, 3, 4, or 5.")


if __name__ == '__main__':
    main()
