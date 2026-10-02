import os
import sys
import re
from urllib.parse import urlparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import scipy.sparse
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

os.makedirs('models', exist_ok=True)

print("Loading cleaned datasets...")
email_df = pd.read_csv('datasets/email_clean.csv')
url_df   = pd.read_csv('datasets/url_clean.csv')

# ========== EMAIL FEATURES ==========
print("Extracting TF-IDF features for emails...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
email_clean_texts = email_df['clean_text'].fillna('')
X_email_tfidf = tfidf.fit_transform(email_clean_texts)

# Extra structural & lexical features for emails
print("Extracting metadata features for emails...")
def extract_email_features(df):
    features = pd.DataFrame()
    raw_texts = df['text'].fillna('')
    features['word_count']        = raw_texts.apply(lambda x: len(str(x).split()))
    features['char_count']        = raw_texts.apply(lambda x: len(str(x)))
    features['has_url']           = raw_texts.apply(lambda x: 1 if 'http' in str(x).lower() else 0)
    features['url_count']         = raw_texts.apply(lambda x: len(re.findall(r'http\S+', str(x))))
    features['exclamation_count'] = raw_texts.apply(lambda x: str(x).count('!'))
    features['uppercase_ratio']   = raw_texts.apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x))+1))
    urgent_words = ['urgent', 'verify', 'account', 'suspend', 'click', 'password', 'login', 'security', 'alert', 'bank']
    features['has_urgent_words']  = raw_texts.apply(lambda x: 1 if any(w in str(x).lower() for w in urgent_words) else 0)
    return features

email_extra   = extract_email_features(email_df)
X_email_final = hstack([X_email_tfidf, csr_matrix(email_extra.values)])
y_email       = email_df['label']

# ========== URL FEATURES ==========
print("Extracting lexical features for URLs...")
def extract_url_features(url):
    url    = str(url)
    parsed = urlparse(url)
    return {
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
    }

url_feature_list = [extract_url_features(u) for u in url_df['url']]
X_url = pd.DataFrame(url_feature_list)
y_url = url_df['label']

# Save all features
print("Saving engineered feature matrices to models/...")
scipy.sparse.save_npz('models/X_email_final.npz', X_email_final)
X_url.to_csv('models/X_url.csv', index=False)
y_email.to_csv('models/y_email.csv', index=False)
y_url.to_csv('models/y_url.csv', index=False)
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

print(f"✅ Features extracted successfully! Email shape: {X_email_final.shape}, URL shape: {X_url.shape}")
