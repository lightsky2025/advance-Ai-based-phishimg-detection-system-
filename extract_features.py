"""
Extract the correct 47 features from your trained model
"""
import pickle
import numpy as np
import pandas as pd

print("📂 Loading model and training data...")

# Load the model
with open('models/xgboost_url_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']

print(f"✅ Model loaded")

# Load a sample of your training data to get feature names
print("\n📂 Loading training data to get feature names...")

# Read just the first few rows of your CSV to see columns
df_sample = pd.read_csv('urlsTrain.csv', nrows=5)
print(f"Total columns in CSV: {len(df_sample.columns)}")

# These are the columns that should be excluded (non-features)
exclude_cols = ['url', 'type', 'label', 'domain', 'Date_inspection', 
                'Unnamed', 'index', 'id']

# Get the actual feature columns (these are your features)
feature_cols = []
for col in df_sample.columns:
    col_lower = col.lower()
    # Skip excluded columns
    skip = False
    for exclude in exclude_cols:
        if exclude.lower() in col_lower:
            skip = True
            break
    # Skip any column with 'label' or 'type' in name
    if 'label' in col_lower or 'type' in col_lower:
        skip = True
    if not skip:
        feature_cols.append(col)

print(f"\n✅ Found {len(feature_cols)} feature columns")
print(f"First 10 features: {feature_cols[:10]}")

# Save the correct feature list for future use
with open('models/feature_names.txt', 'w') as f:
    for col in feature_cols:
        f.write(col + '\n')

print(f"\n💾 Saved feature list to 'models/feature_names.txt'")

# Verify count
if len(feature_cols) == 47:
    print("✅ Perfect! 47 features found!")
else:
    print(f"⚠️ Got {len(feature_cols)} features (expected 47)")