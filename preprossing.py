import os
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

os.makedirs('datasets', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load Email Dataset
email_path = 'Phishing_Email.csv' if os.path.exists('Phishing_Email.csv') else 'datasets/email_dataset.csv'
print(f"Loading email dataset from: {email_path}")
email_df = pd.read_csv(email_path)

# Normalize column names
if 'Email Text' in email_df.columns:
    email_df = email_df.rename(columns={'Email Text': 'text', 'Email Type': 'label'})
elif 'text' not in email_df.columns:
    # Fallback to column indices if named differently
    email_df.columns = ['id', 'text', 'label'] if len(email_df.columns) == 3 else email_df.columns

# Handle missing values
email_df.dropna(subset=['text'], inplace=True)
email_df['text'] = email_df['text'].astype(str)

# Clean text for TF-IDF (keep original raw text for BERT!)
def clean_text_tfidf(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

email_df['clean_text'] = email_df['text'].apply(clean_text_tfidf)

# Standardize email labels: 1 = Phishing, 0 = Safe/Legitimate
def map_email_label(val):
    val_str = str(val).lower()
    if 'phish' in val_str or val_str in ['1', 'bad', 'phishing']:
        return 1
    return 0

email_df['label'] = email_df['label'].apply(map_email_label)

# 2. Load URL Dataset
url_path = 'phishing_site_urls.csv' if os.path.exists('phishing_site_urls.csv') else 'datasets/url_dataset.csv'
print(f"Loading URL dataset from: {url_path}")
url_df = pd.read_csv(url_path)

if 'URL' in url_df.columns:
    url_df = url_df.rename(columns={'URL': 'url', 'Label': 'label'})

url_df.dropna(subset=['url'], inplace=True)
url_df['url'] = url_df['url'].astype(str)

def map_url_label(val):
    val_str = str(val).lower()
    if 'bad' in val_str or 'phish' in val_str or val_str == '1':
        return 1
    return 0

url_df['label'] = url_df['label'].apply(map_url_label)

# Balance / sample URL dataset if very large to prevent memory overhead during feature extraction
if len(url_df) > 50000:
    print(f"Sampling URL dataset from {len(url_df)} to 50,000 balanced records for efficient feature extraction...")
    good_urls = url_df[url_df['label'] == 0].sample(n=25000, random_state=42)
    bad_urls = url_df[url_df['label'] == 1].sample(n=min(25000, (url_df['label'] == 1).sum()), random_state=42)
    url_df = pd.concat([good_urls, bad_urls]).sample(frac=1.0, random_state=42).reset_index(drop=True)

# 3. Save cleaned data
email_df.to_csv('datasets/email_clean.csv', index=False)
url_df.to_csv('datasets/url_clean.csv', index=False)

print("\n Preprocessing done! Cleaned files saved:")
print(f"- datasets/email_clean.csv: {email_df.shape} (Safe: {(email_df['label']==0).sum()}, Phishing: {(email_df['label']==1).sum()})")
print(f"- datasets/url_clean.csv: {url_df.shape} (Safe: {(url_df['label']==0).sum()}, Phishing: {(url_df['label']==1).sum()})")
print("\nEmail sample preview:")
print(email_df[['clean_text', 'label']].head())
