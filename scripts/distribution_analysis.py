import pandas as pd
import numpy as np
import sys
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Generate synthetic revenue data matching skewness description
# Target: mean ~ 500, median ~ 450
np.random.seed(42)
mu = 6.109
sigma = 0.458
n_customers = 2000
revenue = np.random.lognormal(mean=mu, sigma=sigma, size=n_customers)

# Add a few high-value enterprise outliers to increase skewness/kurtosis
outliers = np.array([5000.0, 7500.0, 10000.0, 15000.0])
revenue = np.concatenate([revenue, outliers])

df = pd.DataFrame({'revenue': revenue})

# ----------------------------------------------------
# Task 1: Distribution Plots
# ----------------------------------------------------
print("="*70)
print("TASK 1: DISTRIBUTION PLOTS")
print("="*70)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['revenue'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('Revenue Distribution (Histogram)')
axes[0].set_xlabel('Revenue ($)')
axes[0].set_ylabel('Frequency')

# KDE
df['revenue'].plot(kind='density', ax=axes[1], color='coral')
axes[1].set_title('Revenue Distribution (KDE)')
axes[1].set_xlabel('Revenue ($)')

plt.tight_layout()
plt.savefig('output/revenue_distribution.png')
print("Saved revenue distribution plot to output/revenue_distribution.png")

# ----------------------------------------------------
# Task 2: Compute Skewness and Kurtosis
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 2: COMPUTE SKEWNESS AND KURTOSIS")
print("="*70)
skewness = stats.skew(df['revenue'])
kurtosis = stats.kurtosis(df['revenue'])

print(f"Skewness: {skewness:.2f}")
print(f"Kurtosis: {kurtosis:.2f}")

if abs(skewness) > 1:
    print("Highly skewed - use median not mean for typical customer value")
if kurtosis > 3:
    print("Heavy tails (leptokurtic) - expect significant outliers")

# ----------------------------------------------------
# Task 3: Identify Abnormal Patterns
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 3: IDENTIFY ABNORMAL PATTERNS (PERCENTILES)")
print("="*70)
print("Summary Statistics:")
print(df['revenue'].describe())

percentiles = df['revenue'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print("\nPercentiles:")
print(percentiles)

# Check gap
gap_75_90 = percentiles[0.90] - percentiles[0.75]
print(f"\nGap between 75th and 90th percentile: ${gap_75_90:.2f}")

# ----------------------------------------------------
# Task 4: Compare Segment Distributions
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 4: COMPARE SEGMENT DISTRIBUTIONS")
print("="*70)
# Split by high-value vs low-value
q25 = df['revenue'].quantile(0.25)
q75 = df['revenue'].quantile(0.75)

high_value = df[df['revenue'] > q75]
low_value = df[df['revenue'] < q25]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograms
axes[0].hist(high_value['revenue'], bins=30, alpha=0.7, color='green', edgecolor='black', label='High-Value (>75th)')
axes[0].hist(low_value['revenue'], bins=30, alpha=0.7, color='red', edgecolor='black', label='Low-Value (<25th)')
axes[0].legend()
axes[0].set_title('Revenue: High vs Low Value Customers')
axes[0].set_xlabel('Revenue ($)')
axes[0].set_ylabel('Frequency')

# Boxplot comparison
axes[1].boxplot([low_value['revenue'], high_value['revenue']], tick_labels=['Low-Value', 'High-Value'])
axes[1].set_title('Revenue Spread Comparison')
axes[1].set_ylabel('Revenue ($)')

plt.tight_layout()
plt.savefig('output/revenue_segments.png')
print("Saved segment comparison plot to output/revenue_segments.png")

print(f"\nHigh-value segment (>75th percentile): mean=${high_value['revenue'].mean():.2f}, median=${high_value['revenue'].median():.2f}")
print(f"Low-value segment (<25th percentile):  mean=${low_value['revenue'].mean():.2f}, median=${low_value['revenue'].median():.2f}")

# ----------------------------------------------------
# Task 5: Business Interpretation
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 5: BUSINESS INTERPRETATION")
print("="*70)

interpretation = f"""
Revenue Distribution Analysis:

Skewness: {skewness:.2f} → {"Highly right-skewed" if skewness > 1 else "Moderate"}
Mean: ${df['revenue'].mean():.2f}
Median: ${df['revenue'].median():.2f}
Interpretation: {'Most customers are small; few are huge enterprise accounts (unbalanced spend pattern)' if skewness > 1 else 'Balanced customer spend pattern'}

Kurtosis: {kurtosis:.2f} → {"Fat tails (highly volatile outlier presence)" if kurtosis > 3 else "Normal tails"}
Max Customer Spend: ${df['revenue'].max():.2f}
Top 1% Cut-off: ${df['revenue'].quantile(0.99):.2f}

Business Action: {'Segment customers into Small/Medium (SMB) vs Enterprise categories to apply distinct pricing models and account management strategies.' if skewness > 1 else 'Apply uniform pricing and sales strategies across the customer base.'}
"""

print(interpretation)

# Save text report
with open('output/revenue_distribution_interpretation.txt', 'w', encoding='utf-8') as f:
    f.write(interpretation)
print("Saved interpretation report to output/revenue_distribution_interpretation.txt")
