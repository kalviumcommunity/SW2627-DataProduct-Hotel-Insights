import pandas as pd
import numpy as np
import json
import sys
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Generate Synthetic Customer Churn Dataset
np.random.seed(42)
n_samples = 1000

# Support tickets (e.g., customer complaints)
support_tickets = np.random.randint(0, 11, size=n_samples)

# Churn (highly correlated with support tickets)
churn_prob = support_tickets / 10.0
# Add a bit of noise to churn probability
churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05, size=n_samples), 0, 1)
churn = np.random.binomial(1, churn_prob)

# Transactions per month
transactions_per_month = np.random.randint(1, 50, size=n_samples)

# Engagement (collinear/redundant with transactions_per_month, target correlation ~ 0.92)
engagement = transactions_per_month * 2.0 + np.random.normal(0, 11, size=n_samples)

df = pd.DataFrame({
    'support_tickets': support_tickets,
    'transactions_per_month': transactions_per_month,
    'engagement': engagement,
    'churn': churn
})

# ----------------------------------------------------
# Task 1: Compute Pearson and Spearman Correlation
# ----------------------------------------------------
print("="*70)
print("TASK 1: COMPUTE PEARSON & SPEARMAN CORRELATION")
print("="*70)

# Pearson (linear relationships)
pearson_corr = df.corr(method='pearson')

# Spearman (monotonic, robust to outliers)
spearman_corr = df.corr(method='spearman')

# Compare which correlations differ
comparison = pd.DataFrame({
    'pearson': pearson_corr['churn'],
    'spearman': spearman_corr['churn']
})
print("Correlation comparison relative to Churn:")
print(comparison)

# ----------------------------------------------------
# Task 2: Visualize Correlation Heatmap
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 2: VISUALIZE CORRELATION HEATMAP")
print("="*70)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt=".2f")
ax.set_title('Feature Correlation Matrix (Pearson)')
plt.tight_layout()
plt.savefig('output/correlation_heatmap.png')
print("Saved correlation heatmap to output/correlation_heatmap.png")

# ----------------------------------------------------
# Task 3: Identify Strongly Correlated Pairs
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 3: IDENTIFY STRONGLY CORRELATED PAIRS")
print("="*70)
# Flatten and find strong correlations
corr_flat = pearson_corr.unstack()
strong = corr_flat[corr_flat.abs() > 0.7].sort_values(ascending=False)

# Exclude self-correlation (r=1.0)
strong_pairs = strong[strong < 0.999999]  # Exclude perfect self-joins
print("Top strongly correlated feature pairs (threshold absolute r > 0.7):")
print(strong_pairs)

# ----------------------------------------------------
# Task 4: Business Interpretation
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 4: BUSINESS CAUSATION ANALYSIS REPORT")
print("="*70)
# Reason about causation vs correlation
analysis = {
    'support_tickets <-> churn': {
        'correlation': round(pearson_corr.loc['support_tickets', 'churn'], 2),
        'possible_directions': [
            'support_tickets → churn (customer gives up after contacting support)',
            'churn → support_tickets (unhappy customers contact support before leaving)',
            'customer_pain → both (underlying system issues or service failures cause both)'
        ],
        'data_indicates': 'Likely customer_pain is the hidden confounder; tickets are a symptom, not the root cause.',
        'action': 'Focus on resolving service pain points and product bugs rather than attempting to reduce support ticket volumes.'
    }
}

print(json.dumps(analysis, indent=2))

# Save analysis report
with open('output/causation_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2)
print("\nSaved business analysis JSON report to output/causation_analysis.json")

# ----------------------------------------------------
# Task 5: Feature Selection Based on Correlation
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 5: FEATURE SELECTION (COLLINEARITY MITIGATION)")
print("="*70)
# High correlation between engagement and transactions_per_month means redundancy
# Keep the more direct, interpretable one: transactions_per_month
df_features = df[['engagement', 'transactions_per_month', 'support_tickets', 'churn']]

print("Correlations before dropping redundant collinear feature:")
print(df_features.corr())

# Drop redundant feature, keep the most interpretable one
df_features_selected = df_features.drop('engagement', axis=1)

print("\nCorrelations after dropping redundant feature 'engagement':")
print(df_features_selected.corr())

# Export the selected features dataset
df_features_selected.to_csv('data/processed/selected_churn_features.csv', index=False)
print("\n✓ Saved selected feature dataset to data/processed/selected_churn_features.csv")
