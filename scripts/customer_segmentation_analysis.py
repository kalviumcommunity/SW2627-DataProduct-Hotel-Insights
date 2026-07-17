import pandas as pd
import numpy as np
import sys
import io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure output directory exists
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Generate Synthetic Customer Data
# Base sizes:
# Enterprise: 5% (500)
# SMB: 40% (4,000)
# Startup: 55% (5,500)
np.random.seed(42)
n_records = 10000

customer_types = (
    ['Enterprise'] * 500 +
    ['SMB'] * 4000 +
    ['Startup'] * 5500
)

# Churn rates:
# Enterprise: 1% churn (5 churned)
# SMB: 12% churn (480 churned)
# Startup: 8% churn (440 churned)
churn_flags = (
    [1] * 5 + [0] * 495 +
    [1] * 480 + [0] * 3520 +
    [1] * 440 + [0] * 5060
)

# Lifetime Value (LTV):
# Enterprise: ~$150,000
# SMB: ~$8,000
# Startup: ~$2,000
ltv = []
for c_type in customer_types:
    if c_type == 'Enterprise':
        ltv.append(np.random.normal(loc=150000.0, scale=10000.0))
    elif c_type == 'SMB':
        ltv.append(np.random.normal(loc=8000.0, scale=800.0))
    else:
        ltv.append(np.random.normal(loc=2000.0, scale=200.0))

# Support tickets:
# Enterprise: ~1.5, SMB: ~3.5, Startup: ~2.5
support_tickets = []
for c_type in customer_types:
    if c_type == 'Enterprise':
        support_tickets.append(np.random.poisson(lam=1.5))
    elif c_type == 'SMB':
        support_tickets.append(np.random.poisson(lam=3.5))
    else:
        support_tickets.append(np.random.poisson(lam=2.5))

# Retention days:
# Enterprise: ~730 days, SMB: ~365 days, Startup: ~180 days
retention_days = []
for c_type in customer_types:
    if c_type == 'Enterprise':
        retention_days.append(np.random.normal(loc=730.0, scale=50.0))
    elif c_type == 'SMB':
        retention_days.append(np.random.normal(loc=365.0, scale=30.0))
    else:
        retention_days.append(np.random.normal(loc=180.0, scale=20.0))

df = pd.DataFrame({
    'customer_id': [f"CUST_{i:05d}" for i in range(1, n_records + 1)],
    'customer_type': customer_types,
    'lifetime_value': ltv,
    'churn': churn_flags,
    'support_tickets': support_tickets,
    'retention_days': retention_days
})

# ======================================================================
# Task 1: Define Segments and Compute Metrics
# ======================================================================
print("="*70)
print("TASK 1: DEFINE SEGMENTS AND COMPUTE METRICS")
print("="*70)

segment_metrics = df.groupby('customer_type').agg({
    'lifetime_value': 'mean',
    'churn': 'mean',
    'support_tickets': 'mean',
    'retention_days': 'mean',
    'customer_id': 'count'
})

segment_metrics.columns = ['avg_ltv', 'churn_rate', 'avg_tickets', 'avg_retention', 'count']
print(segment_metrics)

# ======================================================================
# Task 2: Summary Statistics Table
# ======================================================================
print("\n" + "="*70)
print("TASK 2: SUMMARY STATISTICS TABLE")
print("="*70)

segment_summary = segment_metrics.copy()
segment_summary['ltv_rank'] = segment_summary['avg_ltv'].rank(ascending=False)
segment_summary['churn_rank'] = segment_summary['churn_rate'].rank(ascending=True)

# Format for readability
formatted_summary = segment_summary.copy()
formatted_summary['avg_ltv'] = formatted_summary['avg_ltv'].apply(lambda x: f"${x:,.2f}")
formatted_summary['churn_rate'] = formatted_summary['churn_rate'].apply(lambda x: f"{x:.1%}")
formatted_summary['avg_retention'] = formatted_summary['avg_retention'].apply(lambda x: f"{x:.1f} days")
formatted_summary['avg_tickets'] = formatted_summary['avg_tickets'].apply(lambda x: f"{x:.2f}")

print(formatted_summary[['avg_ltv', 'ltv_rank', 'churn_rate', 'churn_rank', 'avg_retention', 'count']])

# ======================================================================
# Task 3: Visual Comparison
# ======================================================================
print("\n" + "="*70)
print("TASK 3: VISUAL COMPARISON")
print("="*70)

plt.figure(figsize=(8, 6))
# Normalize/scale columns for heatmap visual balance or plot raw values as they are
sns.heatmap(segment_metrics[['avg_ltv', 'churn_rate', 'avg_tickets']], 
            annot=True, cmap='RdYlGn_r', cbar_kws={'label': 'Value'}, fmt=".2f")
plt.title('Segment Comparison Heatmap')
plt.tight_layout()
plt.savefig('output/segment_heatmap.png')
plt.savefig('segment_heatmap.png')  # Save to root as requested
plt.close()

print("Segment comparison heatmap created and saved to 'output/segment_heatmap.png' and 'segment_heatmap.png'.")

# ======================================================================
# Task 4: Top and Bottom Performer Analysis
# ======================================================================
print("\n" + "="*70)
print("TASK 4: TOP AND BOTTOM PERFORMER ANALYSIS")
print("="*70)

# Highest value segment
top_segment = segment_metrics['avg_ltv'].idxmax()
top_value = segment_metrics.loc[top_segment, 'avg_ltv']

# Highest churn segment
high_churn = segment_metrics['churn_rate'].idxmax()

insights = f"""
HIGHEST VALUE: {top_segment} = ${top_value:,.0f}
HIGHEST CHURN: {high_churn} = {segment_metrics.loc[high_churn, 'churn_rate']:.1%}
BEST RETENTION: {segment_metrics['avg_retention'].idxmax()}
"""

print(insights)

# ======================================================================
# Task 5: Business-Facing Insights
# ======================================================================
print("="*70)
print("TASK 5: BUSINESS-FACING INSIGHTS")
print("="*70)

business_summary = """
SEGMENT STRATEGY SUMMARY:

Enterprise (5% of base, $150k LTV, 1% churn):
- Highest value, lowest churn. Shows excellent retention (~730 days) and lowest support tickets per customer.
- Action: Maintain premium support tier, assign dedicated account managers, and focus on expanding contract value through upsell.

SMB (40% of base, $8k LTV, 12% churn):
- Middle value, high churn risk. Customers have the highest average support ticket volume (~3.5 tickets) and churn early.
- Action: Improve product onboarding tutorials and offer a lower-cost, self-service support tier to offset support costs while investigating pain points.

Startup (55% of base, $2k LTV, 8% churn):
- Lowest value, moderate churn. Largest base by customer count, but contributes low individual lifetime value (~$2k) and moderate support overhead.
- Action: Transition startups to automated self-service paths, leverage community education forums, and offer automated product guides to reduce ticket volume.
"""

print(business_summary)
