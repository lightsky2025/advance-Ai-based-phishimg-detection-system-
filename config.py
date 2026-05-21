"""
Configuration settings for the Phishing Detection System
"""

import os

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Model paths
MODEL_CONFIG = {
    "xgboost_path": "models/email_xgboost.pkl",
    "random_forest_path": "models/email_random_forest.pkl",
    "bert_model_name": "bert-base-uncased",
    "bert_url_model_path": "models/bert_url_model",
    "scaler_path": "models/scaler.pkl",
    "vectorizer_path": "models/tfidf_vectorizer.pkl"
}

# Feature configuration - ENHANCED VERSION
FEATURE_CONFIG = {
    "email_features": [
        "total_length",
        "num_caps",
        "num_digits",
        "num_special_chars",
        "num_urls",
        "num_emails",
        "num_ips",
        "num_links",
        "num_attachments",
        "suspicious_words_count",
        "urgency_score",
        "threat_score",
        "sender_reputation",
        "has_spf",
        "has_dkim",
        "has_dmarc",
        "spf_result_match",
        "html_ratio",
        "text_html_ratio",
        "avg_word_length",
        "unique_word_ratio"
    ],
    "url_features": [
        # ========== BASIC FEATURES (Keep these) ==========
        "url_length",
        "num_dots",
        "num_hyphens",
        "num_at_symbols",
        "num_question_marks",
        "num_percent",
        "num_equals",
        "num_slashes",
        "has_https",
        "has_suspicious_tld",
        "has_ip_address",
        "has_shortening_service",
        "num_subdomains",
        "url_entropy",
        "has_port",
        "has_authentication",
        "path_length",
        "query_length",
        "digit_url_ratio",
        "special_char_ratio",
        
        # ========== NEW URL FEATURES (Add these) ==========
        
        # Redirect and tracking features
        "num_redirects",           # Number of redirects in URL
        "has_tracking",            # Contains tracking parameters (utm_, fbclid, etc.)
        
        # Date and time features
        "has_current_year",        # Contains current year (2026)
        "has_any_year",            # Contains any year (2020-2026)
        
        # Path and structure features
        "path_depth",              # Depth of URL path (number of subdirectories)
        "digit_ratio_host",        # Ratio of digits in hostname
        "hyphens_in_host",         # Number of hyphens in hostname
        
        # Keyword detection
        "brand_keyword_count",     # Count of brand keywords (paypal, amazon, etc.)
        "suspicious_keyword_count", # Count of suspicious keywords (verify, login, etc.)
        
        # Character analysis
        "has_uppercase",           # Contains uppercase letters
        "num_query_params",        # Number of query parameters
        
        # TLD analysis
        "is_common_tld",           # Uses common TLD (.com, .org, .net, .edu, .gov)
        
        # Security indicators
        "has_at_symbol",           # Contains @ symbol (credential theft)
        "has_double_dot",          # Contains double dot (directory traversal)
        "has_encoding",            # Contains URL encoding (% signs)
        "encoding_count",          # Count of encoded characters
        
        # Advanced pattern detection
        "has_excessive_subdomains", # Has more than 2 subdomains
        "homograph_score",         # Score for homograph attacks (lookalike characters)
        "has_typosquatting",       # Contains typosquatting patterns
        "url_complexity"           # Combined entropy and length score
    ]
}

# Suspicious words for phishing detection
SUSPICIOUS_WORDS = [
    "urgent", "immediate", "action required", "verify", "suspended",
    "confirm", "update", "expired", "click here", "login", "password",
    "account", "bank", "security", "alert", "warning",
    "terminate", "limited time", "offer", "free", "winner", "prize",
    "congratulations", "inheritance", "million", "lottery", "wire",
    "transfer", "ssn", "social security", "credit card", "cvv"
]

# Threat keywords
THREAT_KEYWORDS = [
    "threat", "virus", "malware", "phishing", "scam", "fraud",
    "hack", "compromised", "breach", "infection", "trojan", "ransomware"
]

# Suspicious TLDs (expanded)
SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".club", ".work", ".date", ".bid", ".stream",
    ".download", ".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".cc",
    ".icu", ".click", ".link", ".website", ".site", ".online",
    ".space", ".tech", ".press", ".host", ".live", ".rocks", ".bar",
    ".cyou", ".men", ".win", ".bid", ".loan", ".review", ".trade"
]

# URL shortening services (expanded)
URL_SHORTENERS = [
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "short.link", "tiny.cc", "rb.gy",
    "cutt.ly", "shorturl.at", "rebrand.ly", "hyperurl.co", "shorte.st"
]

# URL entropy threshold for obfuscation detection
URL_ENTROPY_THRESHOLD = 4.5

# Model training parameters
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "xgboost_params": {
        "n_estimators": 200,  # Increased for better learning
        "max_depth": 8,       # Increased for complex patterns
        "learning_rate": 0.05, # Lower for better generalization
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "random_forest_params": {
        "n_estimators": 200,      # Increased
        "max_depth": 12,          # Increased
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "class_weight": "balanced"  # Handle class imbalance
    },
    "bert_config": {
        "max_length": 128,
        "batch_size": 32,
        "epochs": 3,
        "learning_rate": 2e-5
    }
}

# Streamlit UI configuration
UI_CONFIG = {
    "page_title": "AI Phishing Detection System",
    "page_icon": "🛡️",
    "layout": "wide",
    "primary_color": "#FF4B4B",
    "theme": {
        "primaryColor": "#FF4B4B",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730",
        "font": "sans serif"
    }
}

# Detection thresholds (lowered for better recall)
DETECTION_THRESHOLDS = {
    "high_risk": 0.70,    # Lowered from 0.80
    "medium_risk": 0.45,  # Lowered from 0.50
    "low_risk": 0.20      # Lowered from 0.30
}