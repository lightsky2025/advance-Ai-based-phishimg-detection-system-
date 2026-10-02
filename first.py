import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load raw datasets with fallback checks
email_file = 'Phishing_Email.csv' if os.path.exists('Phishing_Email.csv') else 'datasets/email_dataset.csv'
url_file = 'phishing_site_urls.csv' if os.path.exists('phishing_site_urls.csv') else 'datasets/url_dataset.csv'

email_df = pd.read_csv(email_file)
url_df = pd.read_csv(url_file)

print("=== EMAIL DATASET ===")
print("Shape:", email_df.shape)
print(email_df.head())
print("\nInfo:")
print(email_df.info())
print("\nMissing Values:")
print(email_df.isnull().sum())
email_label_col = 'Email Type' if 'Email Type' in email_df.columns else 'label'
print(email_df[email_label_col].value_counts())

print("\n=== URL DATASET ===")
print("Shape:", url_df.shape)
print(url_df.head())
url_label_col = 'Label' if 'Label' in url_df.columns else 'label'
print(url_df[url_label_col].value_counts())

# Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

email_df[email_label_col].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Email: Phishing vs Legitimate')
axes[0].set_xlabel('Label')

url_df[url_label_col].value_counts().plot(kind='bar', ax=axes[1], color=['green', 'red'])
axes[1].set_title('URL: Phishing vs Legitimate')
axes[1].set_xlabel('Label')

plt.tight_layout()
plt.savefig('datasets_distribution.png')
print("\nClass distribution plot saved as 'datasets_distribution.png'.")
plt.show()
