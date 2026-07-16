import pandas as pd
import numpy as np
import sys
import io
from scipy import stats

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Generate Synthetic Data with Outliers
np.random.seed(101)
n_records = 500

# Normal revenue centered around $500, normal age between 18 and 80
revenue = np.random.normal(500, 100, n_records - 3)  # Standard data
age = np.random.randint(18, 80, n_records - 3)

# Append extreme outliers
revenue_outliers = np.array([1000000.0, 1200000.0, 1500000.0]) # Extremely high revenue
age_outliers = np.array([150, 185, 210])                      # Impossible ages

all_revenue = np.concatenate([revenue, revenue_outliers])
all_age = np.concatenate([age, age_outliers])
customer_ids = [f"CUST_{i:03d}" for i in range(1, n_records + 1)]

df = pd.DataFrame({
    'customer_id': customer_ids,
    'revenue': all_revenue,
    'age': all_age
})

print("="*70)
print("ORIGINAL DATASET SUMMARY")
print("="*70)
print(df.describe())

# List to collect cleaning log events
cleaning_log_entries = []

# ----------------------------------------------------
# Task 1: Z-Score Outlier Detection (for Revenue)
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 1: Z-SCORE OUTLIER DETECTION")
print("="*70)
df['revenue_zscore'] = np.abs(stats.zscore(df['revenue']))
z_outliers = df[df['revenue_zscore'] > 3]
print(f"Z-score outliers (revenue > 3 SD): {len(z_outliers)}")
print(z_outliers[['customer_id', 'revenue', 'revenue_zscore']])

# ----------------------------------------------------
# Task 2: IQR Outlier Detection (for Revenue)
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 2: IQR OUTLIER DETECTION")
print("="*70)
Q1_rev = df['revenue'].quantile(0.25)
Q3_rev = df['revenue'].quantile(0.75)
IQR_rev = Q3_rev - Q1_rev

lower_rev = Q1_rev - 1.5 * IQR_rev
upper_rev = Q3_rev + 1.5 * IQR_rev

df['is_outlier_iqr'] = (df['revenue'] < lower_rev) | (df['revenue'] > upper_rev)
iqr_outliers_rev = df[df['is_outlier_iqr']]
print(f"IQR Outliers (revenue): {len(iqr_outliers_rev)}")
print(f"IQR Boundaries: Lower={lower_rev:.2f}, Upper={upper_rev:.2f}")

# ----------------------------------------------------
# Task 3: Cap Outliers at Boundaries (for Revenue)
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 3: CAP OUTLIERS AT BOUNDARIES")
print("="*70)
df['revenue_capped'] = df['revenue'].clip(lower=lower_rev, upper=upper_rev)

print(f"Before Capping: min={df['revenue'].min():.2f}, max={df['revenue'].max():.2f}")
print(f"After Capping:  min={df['revenue_capped'].min():.2f}, max={df['revenue_capped'].max():.2f}")

# ----------------------------------------------------
# Task 4: Flag Outliers with Binary Column (for Revenue)
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 4: FLAG OUTLIERS WITH BINARY COLUMN")
print("="*70)
df['is_outlier'] = (df['is_outlier_iqr']) | (df['revenue_zscore'] > 3)

normal = df[~df['is_outlier']]
anomalies = df[df['is_outlier']]

print(f"Normal records: {len(normal)}")
print(f"Anomalies: {len(anomalies)}")

# Log the revenue cleanup
cleaning_log_entries.append({
    'column': 'revenue',
    'method': 'IQR + Z-score',
    'action': 'cap',
    'threshold_lower': lower_rev,
    'threshold_upper': upper_rev,
    'affected_rows': int(df['is_outlier'].sum()),
    'date': pd.Timestamp.now()
})

# ----------------------------------------------------
# Outlier Detection & Capping for Age (Impossible values > 100 years)
# ----------------------------------------------------
print("\n" + "="*70)
print("ADDITIONAL OUTLIER HANDLING: AGE COLUMN")
print("="*70)
# Impossible age limit (e.g. standard maximum human age for customer databases is 100)
age_limit = 100
df['is_age_outlier'] = df['age'] > age_limit
print(f"Number of impossible age values (> {age_limit}): {df['is_age_outlier'].sum()}")
print(df[df['is_age_outlier']][['customer_id', 'age']])

# Strategy: Cap age at 80 years (the max logical standard customer age in our distribution) or remove.
# Let's cap impossible age values at 80.
age_cap_value = 80
df['age_cleaned'] = np.where(df['is_age_outlier'], age_cap_value, df['age'])

print(f"Before Age Cleanup: max={df['age'].max()}")
print(f"After Age Cleanup:  max={df['age_cleaned'].max()}")

# Log the age cleanup
cleaning_log_entries.append({
    'column': 'age',
    'method': 'Fixed Limit (>100)',
    'action': f'cap to {age_cap_value}',
    'threshold_lower': 0.0,
    'threshold_upper': float(age_limit),
    'affected_rows': int(df['is_age_outlier'].sum()),
    'date': pd.Timestamp.now()
})

# ----------------------------------------------------
# Task 5: Create Cleaning Log
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 5: GENERATE CLEANING LOG")
print("="*70)
log_df = pd.DataFrame(cleaning_log_entries)
log_df.to_csv('output/cleaning_log.csv', index=False)
print("Saved outlier cleaning log to output/cleaning_log.csv")
print("\nLog Contents:")
print(log_df)

# Save the final cleaned dataset
df.to_csv('data/processed/cleaned_outliers.csv', index=False)
print("\n✓ Cleaned and processed dataset saved to data/processed/cleaned_outliers.csv")
