import pandas as pd
import numpy as np
import sys
import io

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Generate Synthetic Customer Metrics Data
np.random.seed(42)
n_samples = 1000

total_transactions = np.random.randint(1, 150, size=n_samples)
days_as_customer = np.random.randint(30, 1000, size=n_samples)
total_spent = total_transactions * np.random.uniform(5.0, 150.0, size=n_samples)
days_since_last_purchase = np.random.randint(1, 365, size=n_samples)
purchase_count = total_transactions

df = pd.DataFrame({
    'customer_id': [f"CUST_{i:04d}" for i in range(1, n_samples + 1)],
    'total_transactions': total_transactions,
    'days_as_customer': days_as_customer,
    'total_spent': total_spent,
    'days_since_last_purchase': days_since_last_purchase,
    'purchase_count': purchase_count
})

print("="*70)
print("ORIGINAL RAW CUSTOMER DATA")
print("="*70)
print(df.head())
print(df.dtypes)

# ----------------------------------------------------
# Task 1: Compute Ratio Features
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 1: COMPUTE RATIO FEATURES")
print("="*70)
df['transactions_per_month'] = df['total_transactions'] / (df['days_as_customer'] / 30)
df['avg_spend_per_transaction'] = df['total_spent'] / df['total_transactions']
df['lifetime_value_per_month'] = df['total_spent'] / (df['days_as_customer'] / 30)

print("Ratio Features Summary:")
print(df[['transactions_per_month', 'avg_spend_per_transaction', 'lifetime_value_per_month']].describe())

# ----------------------------------------------------
# Task 2: Binning with Equal-Width Bins
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 2: BINNING WITH CUSTOM INTERVALS (ENGAGEMENT TIER)")
print("="*70)
df['engagement_tier'] = pd.cut(
    df['transactions_per_month'],
    bins=[0, 2, 10, float('inf')],
    labels=['low', 'medium', 'high']
)
print("Engagement Tier Distribution:")
print(df['engagement_tier'].value_counts())

# ----------------------------------------------------
# Task 3: Binning with Quantiles
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 3: BINNING WITH QUANTILES (SPEND QUARTILE)")
print("="*70)
df['spend_quartile'] = pd.qcut(
    df['total_spent'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)
print("Spend Quartile Distribution:")
print(df['spend_quartile'].value_counts())

# ----------------------------------------------------
# Task 4: Composite Score (RFM)
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 4: COMPOSITE RFM SCORE")
print("="*70)
# Recency: shorter duration is better (score 5 is best/most recent)
df['recency_score'] = pd.qcut(df['days_since_last_purchase'], q=5, labels=[5, 4, 3, 2, 1])
# Frequency: higher count is better (score 5 is best)
df['frequency_score'] = pd.qcut(df['purchase_count'], q=5, labels=[1, 2, 3, 4, 5])
# Monetary: higher spent is better (score 5 is best)
df['monetary_score'] = pd.qcut(df['total_spent'], q=5, labels=[1, 2, 3, 4, 5])

df['rfm_score'] = (df['recency_score'].astype(int) + 
                   df['frequency_score'].astype(int) + 
                   df['monetary_score'].astype(int))

print(df[['customer_id', 'recency_score', 'frequency_score', 'monetary_score', 'rfm_score']].head())
print(f"\nRFM Score Distribution (Counts):")
print(df['rfm_score'].value_counts().sort_index())

# ----------------------------------------------------
# Task 5: Feature Validation
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 5: FEATURE VALIDATION")
print("="*70)
# Check ranges are sensible
print(f"Engagement tier distribution:\n{df['engagement_tier'].value_counts(dropna=False)}")
print(f"RFM score range: {df['rfm_score'].min()}-{df['rfm_score'].max()}")

# Ensure no NaNs introduced
nan_counts = df[['engagement_tier', 'spend_quartile', 'rfm_score']].isna().sum()
print(f"\nMissing values in engineered columns:\n{nan_counts}")

# Export the engineered feature set
df.to_csv('data/processed/customer_features.csv', index=False)
print("\n✓ Engineered features saved to data/processed/customer_features.csv")
