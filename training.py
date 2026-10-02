import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import scipy.sparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print("Loading feature matrices...")
X_email = scipy.sparse.load_npz('models/X_email_final.npz')
X_url   = pd.read_csv('models/X_url.csv')
y_email = pd.read_csv('models/y_email.csv').squeeze()
y_url   = pd.read_csv('models/y_url.csv').squeeze()

# Split datasets
print("Splitting train and test sets (80/20)...")
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_email, y_email, test_size=0.2, random_state=42, stratify=y_email
)
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X_url, y_url, test_size=0.2, random_state=42, stratify=y_url
)

# Train EMAIL models
print("\n--- Training Email Baseline Models ---")
print("1. Training Naive Bayes...")
nb_email = MultinomialNB()
nb_email.fit(X_train_e, y_train_e)

print("2. Training Logistic Regression...")
lr_email = LogisticRegression(max_iter=1000, random_state=42)
lr_email.fit(X_train_e, y_train_e)

print("3. Training Random Forest (n_estimators=100)...")
rf_email = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_email.fit(X_train_e, y_train_e)

# Train URL models
print("\n--- Training URL Baseline Models ---")
print("1. Training URL Random Forest...")
rf_url = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_url.fit(X_train_u, y_train_u)

print("2. Training URL Logistic Regression...")
lr_url = LogisticRegression(max_iter=1000, random_state=42)
lr_url.fit(X_train_u, y_train_u)

# Save all models + test data
print("\nSaving trained models to models/...")
joblib.dump(nb_email,  'models/nb_email.pkl')
joblib.dump(lr_email,  'models/lr_email.pkl')
joblib.dump(rf_email,  'models/rf_email.pkl')
joblib.dump(rf_url,    'models/rf_url.pkl')
joblib.dump(lr_url,    'models/lr_url.pkl')
joblib.dump((X_test_e, y_test_e, X_test_u, y_test_u), 'models/test_data.pkl')

print("[OK] All baseline models trained and saved!")
