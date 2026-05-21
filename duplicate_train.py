"""
Training Script for Phishing Detection System
Handles data loading, preprocessing, model training, and evaluation
Supports separate datasets for EMAILS and URLS
UPDATED: Added proper Kaggle dataset support with domain-aware splitting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import json
import os
import pickle
from urllib.parse import urlparse
import re
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, TRAINING_CONFIG
from feature_engineering import CombinedFeatureExtractor, URLFeatureExtractor, EmailFeatureExtractor
from models import EmailClassifier, BERTURLClassifier, EnsemblePhishingDetector

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ==================== DATA LEAKAGE DIAGNOSIS ====================

def diagnose_data_leakage(df, label_col='label'):
    """Identify which features are causing 100% accuracy"""
    print("\n" + "="*60)
    print("🔍 DATA LEAKAGE DIAGNOSIS")
    print("="*60)
    
    # Separate features and label
    feature_cols = [col for col in df.columns if col != label_col]
    
    # Check each numeric feature for perfect separation
    print("\n📊 Checking for features that perfectly separate classes:")
    print("-" * 50)
    
    leaking_features = []
    
    # Handle string labels if needed
    if df[label_col].dtype == 'object':
        unique_labels = df[label_col].unique()
        label_map = {label: i for i, label in enumerate(unique_labels)}
        temp_labels = df[label_col].map(label_map)
    else:
        temp_labels = df[label_col]
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            # Get unique values per class
            class_0_vals = set(df[temp_labels == 0][col].dropna().unique())
            class_1_vals = set(df[temp_labels == 1][col].dropna().unique())
            
            # Check if values don't overlap (perfect separation)
            if class_0_vals and class_1_vals and class_0_vals.isdisjoint(class_1_vals):
                leaking_features.append(col)
                print(f"  🚨 {col}: Class 0={class_0_vals}, Class 1={class_1_vals}")
                print(f"     → NO OVERLAP! This feature perfectly separates classes!")
            elif len(class_0_vals) == 1 and len(class_1_vals) == 1:
                print(f"  ⚠️ {col}: Class 0={class_0_vals}, Class 1={class_1_vals}")
    
    # Check for suspicious column names (likely leaks)
    print("\n📊 Checking for suspicious column names (potential leaks):")
    print("-" * 50)
    
    leak_keywords = ['score', 'result', 'detected', 'prediction', 'is_', 
                      'flag', 'verdict', 'classification', 'threat', 'risk']
    
    for col in feature_cols:
        col_lower = col.lower()
        for keyword in leak_keywords:
            if keyword in col_lower:
                print(f"  ⚠️ Potentially leaking column: {col}")
                if col not in leaking_features:
                    leaking_features.append(col)
                break
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS:")
    print("="*60)
    
    if leaking_features:
        print(f"\n❌ REMOVE these leaking features:\n   {', '.join(leaking_features)}")
    else:
        print("\n✅ No obvious leakage found.")
    
    print("\n⚠️ IMPORTANT: Even without feature leaks, you may have DOMAIN LEAKAGE!")
    print("   Same domains in train AND test sets cause overfitting.")
    print("   Use domain-aware splitting to fix this.")
    
    return leaking_features

# ==================== DOMAIN-AWARE SPLITTING ====================

def get_domain_from_url(url: str) -> str:
    """Extract registered domain from URL"""
    if not url or not isinstance(url, str):
        return None
    try:
        url = str(url)
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        if domain.startswith('www.'):
            domain = domain[4:]
        if ':' in domain:
            domain = domain.split(':')[0]
        return domain.lower() if domain else None
    except:
        return None

def domain_aware_split(df, url_col='url', test_size=0.2, random_state=42):
    """Split data by unique domains to prevent domain leakage"""
    print("\n" + "="*60)
    print("🔀 DOMAIN-AWARE DATA SPLIT")
    print("="*60)
    
    df = df.copy()
    df['_domain'] = df[url_col].apply(get_domain_from_url)
    
    original_len = len(df)
    df = df.dropna(subset=['_domain'])
    print(f"Removed {original_len - len(df)} rows with invalid/missing domains")
    
    unique_domains = df['_domain'].unique()
    print(f"Total unique domains: {len(unique_domains)}")
    
    train_domains, test_domains = train_test_split(
        unique_domains, test_size=test_size, random_state=random_state
    )
    
    train_df = df[df['_domain'].isin(train_domains)]
    test_df = df[df['_domain'].isin(test_domains)]
    
    train_df = train_df.drop(columns=['_domain'])
    test_df = test_df.drop(columns=['_domain'])
    
    print(f"\n📊 Split Results:")
    print(f"   Train: {len(train_df)} samples from {len(train_domains)} domains")
    print(f"   Test:  {len(test_df)} samples from {len(test_domains)} domains")
    
    overlap = set(train_domains) & set(test_domains)
    if overlap:
        print(f"   ⚠️ WARNING: {len(overlap)} domains appear in both sets!")
    else:
        print(f"   ✅ No domain overlap - good!")
    
    return train_df, test_df

# ==================== KAGGLE URL DATASET LOADER ====================

def load_kaggle_url_dataset(csv_path: str, binary_classification: bool = True):
    """Load the Kaggle Malicious URL Detection Dataset with pre-computed features"""
    csv_path = str(csv_path).replace('\\', '/')
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at {csv_path}")
        return None, None, None, None
    
    print(f"\n🔗 Loading Kaggle URL Dataset from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    print(f"\n📋 Available columns ({len(df.columns)} total):")
    print(f"   {list(df.columns)[:10]}... (showing first 10)")
    
    label_col = None
    for col in df.columns:
        if col.lower() in ['label', 'type', 'class', 'category']:
            label_col = col
            break
    
    if label_col is None:
        print(f"❌ Error: No label column found! Columns: {list(df.columns)}")
        return None, None, None, None
    
    print(f"\n✅ Found label column: '{label_col}'")
    print(f"   Unique labels: {df[label_col].unique()}")
    
    url_col = None
    for col in df.columns:
        if col.lower() in ['url', 'link', 'website', 'domain']:
            url_col = col
            break
    
    if binary_classification:
        benign_labels = ['benign', 'legitimate', 'safe', 'good', '0', 'normal']
        malicious_labels = ['phishing', 'malware', 'defacement', 'malicious', 'spam', 'fraud', '1']
        
        def convert_label(val):
            val_str = str(val).lower().strip()
            if val_str in benign_labels:
                return 0
            elif val_str in malicious_labels:
                return 1
            else:
                try:
                    return int(float(val_str))
                except:
                    print(f"⚠️ Unknown label: {val_str}, defaulting to 0")
                    return 0
        
        df['label_numeric'] = df[label_col].apply(convert_label)
        
        benign_count = sum(df['label_numeric'] == 0)
        malicious_count = sum(df['label_numeric'] == 1)
        print(f"\n📊 Class Distribution (Binary):")
        print(f"   Benign (0): {benign_count} ({benign_count/len(df)*100:.1f}%)")
        print(f"   Malicious (1): {malicious_count} ({malicious_count/len(df)*100:.1f}%)")
    else:
        unique_labels = df[label_col].unique()
        label_map = {label: i for i, label in enumerate(unique_labels)}
        df['label_numeric'] = df[label_col].map(label_map)
        print(f"\n📊 Class Distribution (Multi-class):")
        for label, idx in label_map.items():
            count = sum(df['label_numeric'] == idx)
            print(f"   {label} ({idx}): {count} ({count/len(df)*100:.1f}%)")
    
    exclude_patterns = ['label', 'url', 'link', 'domain', 'timestamp', 'date', 
                        'id', 'index', 'row_num', 'source', 'file']
    
    feature_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col == label_col or col == 'label_numeric':
            continue
        if url_col and col == url_col:
            continue
        skip = False
        for pattern in exclude_patterns:
            if pattern in col_lower:
                skip = True
                break
        if not skip:
            if df[col].dtype in ['int64', 'float64']:
                feature_cols.append(col)
    
    print(f"\n🔧 Using {len(feature_cols)} pre-computed features")
    print(f"   Features: {feature_cols[:10]}...")
    
    leaking_features = diagnose_data_leakage(df, label_col='label_numeric')
    
    if leaking_features:
        print(f"\n🗑️ Removing {len(leaking_features)} leaking features...")
        feature_cols = [f for f in feature_cols if f not in leaking_features]
        print(f"   Remaining features: {len(feature_cols)}")
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['label_numeric'].values
    
    urls = df[url_col].tolist() if url_col else [''] * len(df)
    
    print(f"\n✅ Final dataset:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Feature matrix shape: {X.shape}")
    
    return X, y, urls, feature_cols

# ==================== EMAIL DATASET LOADING ====================

def load_email_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load email phishing detection dataset from CSV file"""
    csv_path = str(csv_path).replace('\\', '/')
    
    if not os.path.exists(csv_path):
        print(f"Warning: Email dataset not found at {csv_path}")
        return None, None, None
    
    print(f"\n📧 Loading Email Dataset from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin1')
        except:
            df = pd.read_csv(csv_path, encoding='cp1252')
    
    print(f"Loaded {len(df)} email rows with columns: {list(df.columns)}")
    
    subject_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'subject' in col_lower:
            subject_col = col
            break
    
    body_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'body' in col_lower or 'content' in col_lower or 'message' in col_lower or col_lower == 'text':
            body_col = col
            break
    
    if subject_col is None and body_col is not None:
        print(f"  ℹ️ No subject column found. Using '{body_col}' for both subject and body.")
        subject_col = body_col
    
    label_col = None
    for col in df.columns:
        if col.lower() == 'label':
            label_col = col
            print(f"  ✅ Found exact label column: '{col}'")
            break
    
    if label_col is None:
        label_variants = ['is_phishing', 'phishing', 'malicious', 'spam', 'class', 'type']
        for variant in label_variants:
            for col in df.columns:
                if col.lower() == variant:
                    label_col = col
                    print(f"  ✅ Found label column: '{col}'")
                    break
            if label_col:
                break
    
    if subject_col is None:
        print(f"Error: No subject or body/text column found. Available columns: {list(df.columns)}")
        return None, None, None
    
    if label_col is None:
        print(f"Error: No label column found. Available columns: {list(df.columns)}")
        return None, None, None
    
    df[subject_col] = df[subject_col].fillna('').astype(str)
    if body_col:
        df[body_col] = df[body_col].fillna('').astype(str)
    
    print(f"\n✅ Using columns: Content='{subject_col}', Label='{label_col}'")
    
    data = []
    labels = []
    texts = []
    label_mapping = {}
    
    for idx, row in df.iterrows():
        if body_col and body_col != subject_col:
            content = f"{str(row[subject_col])} {str(row[body_col])}"
        else:
            content = str(row[subject_col])
        
        email_data = {
            'subject': str(row[subject_col]) if subject_col else content[:100],
            'body': content,
            'sender': 'unknown@example.com',
            'headers': {},
            'html_content': '',
            'num_attachments': 0
        }
        
        label_val = row[label_col]
        original_label = label_val
        
        if isinstance(label_val, str):
            label_val = label_val.lower().strip()
            if label_val in ['phishing', 'fraud', 'spam', '1', 'true', 'yes', 'malicious', 'phish', 'bad']:
                label = 1
            elif label_val in ['legitimate', 'safe', 'ham', '0', 'false', 'no', 'benign', 'good']:
                label = 0
            else:
                try:
                    label = int(float(label_val))
                except:
                    print(f"Warning: Unrecognized label value '{original_label}' at row {idx}, defaulting to 0")
                    label = 0
        else:
            try:
                label = int(float(label_val)) if pd.notna(label_val) else 0
            except:
                label = 0
        
        if original_label not in label_mapping:
            label_mapping[original_label] = label
        
        data.append(email_data)
        labels.append(label)
        texts.append(content)
    
    phishing_count = sum(labels)
    legitimate_count = len(labels) - phishing_count
    
    print(f"\n📊 Email Dataset Statistics:")
    print(f"  - Total emails: {len(data)}")
    print(f"  - Phishing (label=1): {phishing_count} ({phishing_count/len(data)*100:.2f}%)")
    print(f"  - Legitimate (label=0): {legitimate_count} ({legitimate_count/len(data)*100:.2f}%)")
    
    if phishing_count == 0 or legitimate_count == 0:
        print("\n❌ ERROR: Dataset missing one or both classes!")
        return None, None, None
    
    return pd.DataFrame(data), pd.Series(labels), texts

# ==================== SYNTHETIC DATA GENERATORS ====================

def generate_synthetic_email_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Generate synthetic email phishing dataset for demonstration"""
    np.random.seed(TRAINING_CONFIG["random_state"])
    
    phishing_subjects = [
        "URGENT: Verify your account immediately",
        "Your account has been suspended",
        "Action Required: Update your payment info",
        "Congratulations! You've won a prize",
        "Security Alert: Suspicious activity detected",
    ]
    
    legitimate_subjects = [
        "Weekly team meeting agenda",
        "Project update - Q4 goals",
        "Lunch tomorrow?",
        "Quarterly report attached",
        "Feedback on your presentation",
    ]
    
    phishing_bodies = [
        "Dear customer, your account has been compromised. Click here immediately to verify: http://suspicious-site.xyz/verify",
        "Your account will be closed. Verify now: http://fake-site.tk/verify",
        "Update payment info: http://billing-update.ml/payment",
    ]
    
    legitimate_bodies = [
        "Hi team, reminder about our weekly standup at 10 AM tomorrow.",
        "I've attached the quarterly report for your review.",
        "A few of us are going to lunch at noon. Want to join?",
    ]
    
    data = []
    labels = []
    texts = []
    
    for i in range(n_samples):
        is_phishing = np.random.random() < 0.5
        
        if is_phishing:
            subject = np.random.choice(phishing_subjects)
            body = np.random.choice(phishing_bodies)
            sender = f"security{np.random.randint(1,999)}@gmail.com"
            label = 1
        else:
            subject = np.random.choice(legitimate_subjects)
            body = np.random.choice(legitimate_bodies)
            sender = f"user{np.random.randint(1,999)}@company.com"
            label = 0
        
        email_data = {
            'subject': subject,
            'body': body,
            'sender': sender,
            'headers': {},
            'html_content': '',
            'num_attachments': 0
        }
        
        data.append(email_data)
        labels.append(label)
        texts.append(f"{subject} {body}")
    
    return pd.DataFrame(data), pd.Series(labels), texts

def generate_synthetic_url_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate synthetic URL phishing dataset for demonstration"""
    np.random.seed(TRAINING_CONFIG["random_state"])
    
    legitimate_urls = [
        "https://www.google.com",
        "https://github.com",
        "https://stackoverflow.com",
        "https://www.linkedin.com",
        "https://docs.python.org",
    ]
    
    phishing_urls = [
        "http://paypal-verify.xyz/login",
        "http://secure-bank-verify.tk/account",
        "http://amazon-prize-claim.ml/winner",
        "http://apple-id-locked.cf/unlock",
        "http://netflix-billing-update.xyz/payment",
    ]
    
    urls = []
    labels = []
    
    for i in range(n_samples):
        is_phishing = np.random.random() < 0.5
        if is_phishing:
            url = np.random.choice(phishing_urls)
            label = 1
        else:
            url = np.random.choice(legitimate_urls)
            label = 0
        urls.append(url)
        labels.append(label)
    
    url_extractor = URLFeatureExtractor()
    feature_names = url_extractor.get_feature_names()
    
    feature_list = []
    valid_urls = []
    valid_labels = []
    
    for url, label in zip(urls, labels):
        try:
            features = url_extractor.extract_features(url)
            feature_values = [features.get(name, 0.0) for name in feature_names]
            feature_list.append(feature_values)
            valid_urls.append(url)
            valid_labels.append(label)
        except:
            continue
    
    X = np.array(feature_list)
    y = np.array(valid_labels)
    
    return X, y, valid_urls

# ==================== MODEL TRAINING FUNCTIONS ====================

def train_email_models(email_data: pd.DataFrame, email_labels: pd.Series, email_texts: List[str]):
    """Train models specifically for email phishing detection"""
    print("\n" + "="*60)
    print("📧 TRAINING EMAIL PHISHING DETECTION MODELS")
    print("="*60)
    
    unique_labels = email_labels.unique()
    if len(unique_labels) < 2:
        print(f"\n❌ ERROR: Dataset only contains one class: {unique_labels}")
        return None, None
    
    extractor = CombinedFeatureExtractor()
    feature_dicts = []
    
    LEAKING_FEATURES_TO_REMOVE = [
        'x_spam_score', 'spf_result', 'dkim_result', 'dmarc_result',
        'spam_score', 'phishing_score', 'x_spam', 'is_spam', 'is_phishing',
    ]
    
    print("Extracting email features...")
    successful_extractions = 0
    
    for idx, row in email_data.iterrows():
        email_dict = row.to_dict()
        
        if 'subject' not in email_dict:
            email_dict['subject'] = ''
        if 'body' not in email_dict:
            email_dict['body'] = email_dict.get('body_plain', '')
        if 'sender' not in email_dict:
            email_dict['sender'] = email_dict.get('from_address', 'unknown@example.com')
        if 'headers' not in email_dict:
            email_dict['headers'] = {}
        if 'html_content' not in email_dict:
            email_dict['html_content'] = ''
        if 'num_attachments' not in email_dict:
            email_dict['num_attachments'] = email_dict.get('has_attachments', 0)
        
        try:
            features = extractor.extract_from_email_with_urls(email_dict)
            
            filtered_features = {}
            for key, value in features.items():
                is_leaking = False
                for leak_pattern in LEAKING_FEATURES_TO_REMOVE:
                    if leak_pattern in key.lower():
                        is_leaking = True
                        break
                if not is_leaking:
                    filtered_features[key] = value
            
            feature_dicts.append(filtered_features)
            successful_extractions += 1
            
            if successful_extractions % 1000 == 0:
                print(f"   Processed {successful_extractions}/{len(email_data)} emails...")
        except Exception as e:
            continue
    
    if not feature_dicts:
        print("❌ No features extracted successfully! Using fallback...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X = vectorizer.fit_transform(email_texts).toarray()
        y = email_labels.values
        feature_names = vectorizer.get_feature_names_out()
        filtered_texts = email_texts
    else:
        feature_names = list(feature_dicts[0].keys())
        X = np.array([[fd.get(name, 0.0) for name in feature_names] for fd in feature_dicts])
        y = email_labels[:len(X)].values
        filtered_texts = [email_texts[i] for i in range(len(feature_dicts))]
        
        print(f"✅ Extracted {len(feature_names)} features from {len(X)} emails")
    
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y, filtered_texts, test_size=0.2, random_state=42, stratify=y
    )
    
    results = {}
    
    print("\n📊 Training XGBoost Classifier...")
    try:
        xgb_model = EmailClassifier(use_xgboost=True)
        xgb_model.train(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        results['xgboost'] = {
            'model': xgb_model,
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred, zero_division=0),
            'recall': recall_score(y_test, xgb_pred, zero_division=0),
            'f1': f1_score(y_test, xgb_pred, zero_division=0)
        }
        print(f"  ✅ XGBoost - Accuracy: {results['xgboost']['accuracy']:.4f}, F1: {results['xgboost']['f1']:.4f}")
    except Exception as e:
        print(f"  ❌ XGBoost failed: {e}")
        results['xgboost'] = None
    
    print("\n📊 Training Random Forest Classifier...")
    try:
        rf_model = EmailClassifier(use_xgboost=False)
        rf_model.train(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        results['random_forest'] = {
            'model': rf_model,
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
            'f1': f1_score(y_test, rf_pred, zero_division=0)
        }
        print(f"  ✅ Random Forest - Accuracy: {results['random_forest']['accuracy']:.4f}, F1: {results['random_forest']['f1']:.4f}")
    except Exception as e:
        print(f"  ❌ Random Forest failed: {e}")
        results['random_forest'] = None
    
    results = {k: v for k, v in results.items() if v is not None}
    
    if results:
        os.makedirs("models", exist_ok=True)
        email_metrics = {}
        for model_name, model_info in results.items():
            email_metrics[f"{model_name}_email"] = {
                "accuracy": float(model_info['accuracy']),
                "precision": float(model_info['precision']),
                "recall": float(model_info['recall']),
                "f1": float(model_info['f1'])
            }
        with open("models/email_training_metrics.json", "w") as f:
            json.dump(email_metrics, f, indent=4)
    
    return results, feature_names

def train_url_models_from_features(X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
    """Train URL models using pre-computed features"""
    print("\n" + "="*60)
    print("🔗 TRAINING URL PHISHING DETECTION MODELS")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    results = {}
    
    print("\n📊 Training XGBoost URL Classifier...")
    try:
        xgb_model = EmailClassifier(use_xgboost=True)
        xgb_model.train(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        results['xgboost_url'] = {
            'model': xgb_model,
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred, zero_division=0),
            'recall': recall_score(y_test, xgb_pred, zero_division=0),
            'f1': f1_score(y_test, xgb_pred, zero_division=0)
        }
        print(f"  ✅ XGBoost URL - Accuracy: {results['xgboost_url']['accuracy']:.4f}, F1: {results['xgboost_url']['f1']:.4f}")
    except Exception as e:
        print(f"  ❌ XGBoost failed: {e}")
    
    print("\n📊 Training Random Forest URL Classifier...")
    try:
        rf_model = EmailClassifier(use_xgboost=False)
        rf_model.train(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        results['random_forest_url'] = {
            'model': rf_model,
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
            'f1': f1_score(y_test, rf_pred, zero_division=0)
        }
        print(f"  ✅ Random Forest URL - Accuracy: {results['random_forest_url']['accuracy']:.4f}, F1: {results['random_forest_url']['f1']:.4f}")
    except Exception as e:
        print(f"  ❌ Random Forest failed: {e}")
    
    return results

def save_url_model(model, path: str):
    """Save URL model to disk"""
    model_data = {
        'model': model.model if hasattr(model, 'model') else model,
        'scaler': model.scaler if hasattr(model, 'scaler') else None,
        'use_xgboost': model.use_xgboost if hasattr(model, 'use_xgboost') else True,
        'is_fitted': model.is_fitted if hasattr(model, 'is_fitted') else True
    }
    
    url_model_path = f"models/{path}_url_model.pkl"
    with open(url_model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  💾 URL model saved to {url_model_path}")

def create_sample_datasets():
    """Create sample dataset files for both emails and URLs"""
    email_data = {
        'subject': [
            'Weekly team meeting agenda',
            'URGENT: Your account has been suspended',
            'Project update - Q4 goals',
            'Verify your PayPal account now',
            'Lunch tomorrow?',
        ],
        'body': [
            'Hi team, reminder about our weekly standup at 10 AM tomorrow.',
            'Your account will be closed. Click here to verify: http://fake-site.xyz/verify',
            'I have attached the Q4 report for your review.',
            'We detected unusual activity. Confirm your identity: http://paypal-verify.tk/login',
            'A few of us are going to lunch at noon. Want to join?',
        ],
        'sender': [
            'john.smith@company.com',
            'security@gmail.com',
            'sarah@company.com',
            'security@paypa1-verify.com',
            'mike@company.com',
        ],
        'label': [0, 1, 0, 1, 0]
    }
    
    email_df = pd.DataFrame(email_data)
    email_df.to_csv('data/sample_email_dataset.csv', index=False)
    print(f"📧 Sample Email Dataset created at 'data/sample_email_dataset.csv'")
    
    url_data = {
        'url': [
            'https://www.google.com',
            'http://paypal-verify.xyz/login',
            'https://github.com',
            'http://secure-bank-verify.tk/account',
            'https://stackoverflow.com',
        ],
        'label': [0, 1, 0, 1, 0]
    }
    
    url_df = pd.DataFrame(url_data)
    url_df.to_csv('data/sample_url_dataset.csv', index=False)
    print(f"\n🔗 Sample URL Dataset created at 'data/sample_url_dataset.csv'")
    
    print("\n" + "="*50)
    print("Sample datasets created!")

def print_metrics_summary(results: Dict, model_type: str):
    """Print summary of model metrics"""
    print(f"\n📊 {model_type} MODEL PERFORMANCE SUMMARY:")
    print("-"*40)
    for model_name, metrics in results.items():
        print(f"\n  {model_name.upper()}:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1 Score:  {metrics['f1']:.4f}")

# ==================== MAIN TRAINING FUNCTION ====================

def main():
    """Main training pipeline - Supports separate email and URL datasets"""
    print("="*60)
    print("🛡️ PHISHING DETECTION SYSTEM - MODEL TRAINING")
    print("="*60)
    
    print("\n📁 Dataset Options:")
    print("  1. Use synthetic data (auto-generated for testing)")
    print("  2. Load EMAIL dataset from CSV")
    print("  3. Load KAGGLE URL dataset with pre-computed features (RECOMMENDED)")
    print("  4. Load raw URL dataset (legacy - extracts features)")
    print("  5. Create sample dataset files")
    
    choice = input("\n👉 Enter your choice (1-5): ").strip()
    
    if choice == '5':
        create_sample_datasets()
        return
    
    elif choice == '2':
        csv_path = input("Enter path to email CSV file: ").strip()
        csv_path = csv_path.strip('"').strip("'")
        
        email_data, email_labels, email_texts = load_email_dataset(csv_path)
        
        if email_data is None:
            print("\n⚠️ Email dataset not found. Using synthetic data...")
            email_data, email_labels, email_texts = generate_synthetic_email_data(1000)
        
        email_results, feature_names = train_email_models(email_data, email_labels, email_texts)
        
        if email_results:
            print("\n💾 Saving email models...")
            os.makedirs("models", exist_ok=True)
            for model_name, model_info in email_results.items():
                if model_name in ['xgboost', 'random_forest']:
                    model_info['model'].save(f"models/email_{model_name}.pkl")
            print("✅ Email models saved to models/email_*.pkl")
            print_metrics_summary(email_results, "EMAIL")

        if email_texts is not None and email_labels is not None:
            print("\n🤖 Starting BERT training on email data...")
            bert_model = train_bert_model(
            email_texts, 
            email_labels.tolist(),  # ← Convert Series to list
            save_path="models/bert_email_model"
        )
            if bert_model is not None:
                evaluate_bert_model(bert_model, email_texts, email_labels.tolist())
    
    elif choice == '3':
        csv_path = input("Enter path to Kaggle URL CSV file: ").strip()
        csv_path = csv_path.strip('"').strip("'")
        
        X, y, urls, feature_names = load_kaggle_url_dataset(csv_path, binary_classification=True)
        
        if X is None:
            print("\n❌ Failed to load dataset!")
            return
        
        # Apply domain-aware split if URL column exists
        if urls and urls[0]:
            try:
                df = pd.read_csv(csv_path)
                train_df, test_df = domain_aware_split(df, url_col='url', test_size=0.2)
                
                # Align features with split
                label_col = 'label_numeric' if 'label_numeric' in df.columns else 'label'
                if label_col not in train_df.columns:
                    # Binary conversion
                    benign_labels = ['benign', 'legitimate', 'safe', 'good', '0', 'normal']
                    def convert_label(val):
                        val_str = str(val).lower().strip()
                        return 0 if val_str in benign_labels else 1
                    train_df['label_temp'] = train_df['label'].apply(convert_label)
                    test_df['label_temp'] = test_df['label'].apply(convert_label)
                    label_col = 'label_temp'
                
                # Use only the feature columns we identified
                available_features = [f for f in feature_names if f in train_df.columns]
                X_train = train_df[available_features].values.astype(np.float32)
                y_train = train_df[label_col].values
                X_test = test_df[available_features].values.astype(np.float32)
                y_test = test_df[label_col].values
                
                print(f"\n✅ Domain-aware split applied!")
                print(f"   Train: {len(X_train)} samples")
                print(f"   Test: {len(X_test)} samples")
                
                # Train with the split data
                results = {}
                
                print("\n📊 Training XGBoost URL Classifier...")
                try:
                    xgb_model = EmailClassifier(use_xgboost=True)
                    xgb_model.train(X_train, y_train)
                    xgb_pred = xgb_model.predict(X_test)
                    results['xgboost_url'] = {
                        'model': xgb_model,
                        'accuracy': accuracy_score(y_test, xgb_pred),
                        'precision': precision_score(y_test, xgb_pred, zero_division=0),
                        'recall': recall_score(y_test, xgb_pred, zero_division=0),
                        'f1': f1_score(y_test, xgb_pred, zero_division=0)
                    }
                    print(f"  ✅ XGBoost - Accuracy: {results['xgboost_url']['accuracy']:.4f}, F1: {results['xgboost_url']['f1']:.4f}")
                except Exception as e:
                    print(f"  ❌ XGBoost failed: {e}")
                
                print("\n📊 Training Random Forest URL Classifier...")
                try:
                    rf_model = EmailClassifier(use_xgboost=False)
                    rf_model.train(X_train, y_train)
                    rf_pred = rf_model.predict(X_test)
                    results['random_forest_url'] = {
                        'model': rf_model,
                        'accuracy': accuracy_score(y_test, rf_pred),
                        'precision': precision_score(y_test, rf_pred, zero_division=0),
                        'recall': recall_score(y_test, rf_pred, zero_division=0),
                        'f1': f1_score(y_test, rf_pred, zero_division=0)
                    }
                    print(f"  ✅ Random Forest - Accuracy: {results['random_forest_url']['accuracy']:.4f}, F1: {results['random_forest_url']['f1']:.4f}")
                except Exception as e:
                    print(f"  ❌ Random Forest failed: {e}")
                
                if results:
                    print("\n💾 Saving URL models...")
                    for model_name, model_info in results.items():
                        save_url_model(model_info['model'], model_name.replace('_url', ''))
                    
                    metrics = {name: {k: v for k, v in info.items() if k != 'model'} 
                              for name, info in results.items()}
                    with open('models/url_training_metrics.json', 'w') as f:
                        json.dump(metrics, f, indent=2)
                    
                    print_metrics_summary(results, "KAGGLE URL")
            except Exception as e:
                print(f"⚠️ Domain split failed: {e}, using regular split...")
                url_results = train_url_models_from_features(X, y, feature_names)
                if url_results:
                    for model_name, model_info in url_results.items():
                        save_url_model(model_info['model'], model_name.replace('_url', ''))
                    print_metrics_summary(url_results, "KAGGLE URL")
        else:
            url_results = train_url_models_from_features(X, y, feature_names)
            if url_results:
                for model_name, model_info in url_results.items():
                    save_url_model(model_info['model'], model_name.replace('_url', ''))
                print_metrics_summary(url_results, "KAGGLE URL")
    
    elif choice == '4':
        csv_path = input("Enter path to raw URL CSV file: ").strip()
        csv_path = csv_path.strip('"').strip("'")
        
        X, y, urls = load_url_dataset_legacy(csv_path)
        
        if X is None:
            print("\n⚠️ URL dataset not found. Using synthetic data...")
            X, y, urls = generate_synthetic_url_data(1000)
        
        url_results = train_url_models_from_features(X, y)
        
        if url_results:
            print("\n💾 Saving URL models...")
            for model_name, model_info in url_results.items():
                save_url_model(model_info['model'], model_name.replace('_url', ''))
            
            metrics = {name: {k: v for k, v in info.items() if k != 'model'} 
                      for name, info in url_results.items()}
            with open('models/url_training_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print_metrics_summary(url_results, "URL")
    
    else:
        print("\n📧 Generating synthetic email data...")
        email_data, email_labels, email_texts = generate_synthetic_email_data(1000)
        email_results, _ = train_email_models(email_data, email_labels, email_texts)
        
        print("\n🔗 Generating synthetic URL data...")
        X_url, y_url, urls = generate_synthetic_url_data(1000)
        url_results = train_url_models_from_features(X_url, y_url)
        
        if email_results:
            for model_name, model_info in email_results.items():
                if model_name in ['xgboost', 'random_forest']:
                    model_info['model'].save(f"models/email_{model_name}.pkl")
        # --- ADD THIS BLOCK right after email_results are saved ---
        if email_results and email_texts is not None:
            print("\n🤖 Starting BERT training on email data...")
        # email_texts is the list of combined subject+body strings
        # email_labels is the corresponding list of 0/1 labels
            train_bert_model(email_texts, email_labels, save_path="models/bert_email_model")
        if url_results:
            for model_name, model_info in url_results.items():
                save_url_model(model_info['model'], model_name.replace('_url', ''))
        
        if email_results:
            print_metrics_summary(email_results, "EMAIL")
        if url_results:
            print_metrics_summary(url_results, "URL")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print("\n✅ Models saved to 'models/' folder")
    print("\n🚀 Next steps:")
    print("   1. Run 'streamlit run app.py' to start the web interface")
    print("   2. The system will automatically use your trained models")
def evaluate_bert_model(bert_model, texts, labels, test_size=0.2):
    """Evaluate BERT model and print accuracy metrics"""
    print("\n" + "="*60)
    print("📊 BERT MODEL EVALUATION")
    print("="*60)
    
    # Split same way as other models
    _, texts_test, _, labels_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    predictions = []
    correct = 0
    total = len(texts_test)
    
    print(f"Evaluating on {total} test samples...")
    
    for i, text in enumerate(texts_test):
        try:
            result = bert_model.predict(text)
            
            # Handle different return formats from BERTURLClassifier
            if isinstance(result, dict):
                pred_label = result.get('prediction', result.get('label', 0))
                if isinstance(pred_label, str):
                    pred_label = 1 if pred_label.lower() in ['phishing', 'malicious', '1'] else 0
            elif isinstance(result, (list, tuple)):
                pred_label = int(result[0])
            else:
                pred_label = int(result)
            
            predictions.append(pred_label)
            
            if pred_label == labels_test[i]:
                correct += 1
                
        except Exception as e:
            predictions.append(0)  # Default to legitimate on error
            if i == 0:
                print(f"  ⚠️ Prediction error: {e}")
    
    # Calculate metrics
    accuracy = accuracy_score(labels_test, predictions)
    precision = precision_score(labels_test, predictions, zero_division=0)
    recall = recall_score(labels_test, predictions, zero_division=0)
    f1 = f1_score(labels_test, predictions, zero_division=0)
    
    print(f"\n  BERT EMAIL MODEL RESULTS:")
    print(f"  {'='*40}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels_test, predictions)
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Legit  Phish")
    print(f"  Actual Legit  [{cm[0][0]:5d}  {cm[0][1]:5d}]")
    print(f"  Actual Phish  [{cm[1][0]:5d}  {cm[1][1]:5d}]")
    
    # Save BERT metrics
    bert_metrics = {
        "bert_email": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    }
    
    with open("models/bert_training_metrics.json", "w") as f:
        json.dump(bert_metrics, f, indent=4)
    print(f"\n  💾 BERT metrics saved to models/bert_training_metrics.json")
    
    return bert_metrics

# Legacy URL dataset loader (for backward compatibility)
def load_url_dataset_legacy(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Legacy function for loading raw URLs and extracting features"""
    csv_path = str(csv_path).replace('\\', '/')
    
    if not os.path.exists(csv_path):
        print(f"Warning: URL dataset not found at {csv_path}")
        return None, None, None
    
    print(f"\n🔗 Loading URL Dataset from {csv_path} (legacy mode)...")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin1')
        except:
            df = pd.read_csv(csv_path, encoding='cp1252')
    
    print(f"Loaded {len(df)} URL rows with columns: {list(df.columns)}")
    
    url_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'url' in col_lower or 'link' in col_lower or 'website' in col_lower:
            url_col = col
        if 'label' in col_lower or 'class' in col_lower or 'type' in col_lower:
            label_col = col
    
    if url_col is None:
        print(f"Error: No URL column found. Available columns: {list(df.columns)}")
        return None, None, None
    
    if label_col is None:
        print(f"Error: No label column found. Available columns: {list(df.columns)}")
        return None, None, None
    
    url_extractor = URLFeatureExtractor()
    feature_names = url_extractor.get_feature_names()
    
    feature_list = []
    urls = []
    labels = []
    
    for idx, row in df.iterrows():
        url = str(row[url_col]) if pd.notna(row[url_col]) else ""
        if not url or len(url) < 5:
            continue
        
        label_val = row[label_col]
        if isinstance(label_val, str):
            label_val = label_val.lower()
            label = 1 if label_val in ['phishing', 'fraud', 'malicious', '1', 'true', 'yes'] else 0
        else:
            label = int(label_val) if pd.notna(label_val) else 0
        
        try:
            features = url_extractor.extract_features(url)
            feature_values = [features.get(name, 0.0) for name in feature_names]
            feature_list.append(feature_values)
            urls.append(url)
            labels.append(label)
        except Exception as e:
            continue
    
    if len(feature_list) == 0:
        print("Error: No valid URLs found in dataset")
        return None, None, None
    
    X = np.array(feature_list)
    y = np.array(labels)
    
    print(f"📊 URL Dataset Statistics:")
    print(f"  - Total URLs: {len(X)}")
    print(f"  - Malicious (label=1): {sum(y)}")
    print(f"  - Benign (label=0): {len(y) - sum(y)}")
    print(f"  - Features extracted: {len(feature_names)}")
    
    return X, y, urls
def train_bert_model(texts, labels, save_path="models/bert_email_model"):
    """
    Train BERT on email texts and save to disk.
    Call this after training XGBoost/RandomForest on email data.
    
    Args:
        texts : list of strings — combined subject + body of each email
        labels: list of int    — 0 for legitimate, 1 for phishing
        save_path: directory where BERT model files will be saved
    """
    print("\n" + "="*60)
    print("🤖 TRAINING BERT EMAIL MODEL")
    print("="*60)

    bert = BERTURLClassifier(model_name="bert-base-uncased")
    bert.train(texts, list(labels))

    if not bert.transformers_available:
        print("❌ transformers library not found. Install with:")
        print("   pip install transformers torch")
        return None

    print(f"Training on {len(texts)} samples...")
    print("⚠️  This will take 10-30 minutes on CPU. Use GPU if available.")

    try:
        bert.train(texts, list(labels))
        bert.save(save_path)
        print(f"\n✅ BERT model saved to '{save_path}/'")
        return bert
    except Exception as e:
        print(f"❌ BERT training failed: {e}")
        return None

if __name__ == "__main__":
    main()