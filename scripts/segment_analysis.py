import pandas as pd
import numpy as np
import sys
import io
from pathlib import Path

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset of 10,000 customers
# Base sizes:
# Enterprise: 5% (500)
# SMB: 40% (4,000)
# Startup: 55% (5,500)
n_records = 10000

customer_types = (
    ['Enterprise'] * 500 +
    ['SMB'] * 4000 +
    ['Startup'] * 5500
)

# Churn flags (0 or 1)
# Enterprise: exactly 1% churn (5 churned)
# SMB: exactly 12% churn (480 churned)
# Startup: exactly 8% churn (440 churned)
churn_flags = (
    [1] * 5 + [0] * 495 +
    [1] * 480 + [0] * 3520 +
    [1] * 440 + [0] * 5060
)

# Revenues: Enterprise has 70% of total revenue. SMB & Startup have 15% each.
# Total revenue target: $1,000,000
# Enterprise: $700,000 total / 500 = $1400 each
# SMB: $150,000 total / 4000 = $37.5 each
# Startup: $150,000 total / 5500 = $27.272727 each
revenues = (
    [1400.0] * 500 +
    [37.5] * 4000 +
    [27.272727] * 5500
)

# Support tickets average: Enterprise: 1.5, SMB: 3.5, Startup: 2.5
support_tickets = (
    [1.5] * 500 +
    [3.5] * 4000 +
    [2.5] * 5500
)

# Products:
# Enterprise: Enterprise product
# SMB: Pro and Basic
# Startup: Pro and Basic
products = (
    ['Enterprise'] * 500 +
    ['Pro'] * 2000 + ['Basic'] * 2000 +
    ['Pro'] * 2200 + ['Basic'] * 3300
)

# Create DataFrame
df = pd.DataFrame({
    'customer_id': [f"CUST_{i:05d}" for i in range(1, n_records + 1)],
    'customer_type': customer_types,
    'churn': churn_flags,
    'revenue': revenues,
    'support_tickets': support_tickets,
    'product': products
})

print("="*70)
print("Task 1: Single-Level GroupBy with Multiple Aggregations")
print("="*70)
segment_metrics = df.groupby('customer_type').agg({
    'churn': 'mean',
    'revenue': 'sum',
    'customer_id': 'count',
    'support_tickets': 'mean'
})

segment_metrics.columns = ['churn_rate', 'total_revenue', 'customer_count', 'avg_support_tickets']
print(segment_metrics)

print("\n" + "="*70)
print("Task 2: Multi-Level GroupBy")
print("="*70)
# Two dimensions simultaneously
product_segment = df.groupby(['customer_type', 'product']).agg({
    'revenue': 'sum',
    'customer_id': 'count'
})

product_segment.columns = ['total_revenue', 'customer_count']

# Unstack for cleaner view
product_segment_pivot = product_segment.unstack()
print(product_segment_pivot)

print("\n" + "="*70)
print("Task 3: Pivot Table")
print("="*70)
# Two-dimensional view: customer_type rows, product columns
pivot = pd.pivot_table(
    df,
    values='revenue',
    index='customer_type',
    columns='product',
    aggfunc='sum'
)
print(pivot)

print("\n" + "="*70)
print("Task 4: Rank and Identify Top/Bottom Performers")
print("="*70)
# Rank segments by churn
segment_metrics['churn_rank'] = segment_metrics['churn_rate'].rank()

# Sort to see worst first
worst_first = segment_metrics.sort_values('churn_rate', ascending=False)
print(worst_first)

# Profit/revenue ranking
segment_metrics['revenue_contribution'] = (segment_metrics['total_revenue'] / segment_metrics['total_revenue'].sum() * 100)
print(segment_metrics[['revenue_contribution', 'churn_rate']])

print("\n" + "="*70)
print("Task 5: Surface Actionable Segment Insights")
print("="*70)
# Create insight summary
insights = []

for segment in segment_metrics.index:
    row = segment_metrics.loc[segment]
    
    insight = {
        'segment': segment,
        'customer_count': int(row['customer_count']),
        'churn_rate': f"{row['churn_rate']:.1%}",
        'total_revenue': f"${row['total_revenue']:.0f}",
        'revenue_contribution': f"{row['revenue_contribution']:.1f}%",
        'action': ''
    }
    
    # Action based on metrics
    if row['churn_rate'] > 0.10:
        insight['action'] = 'HIGH PRIORITY: Churn above 10%. Investigate pain points.'
    elif row['churn_rate'] < 0.02:
        insight['action'] = 'Healthy. Maintain current service level.'
    else:
        insight['action'] = 'Monitor. No immediate action needed.'
    
    insights.append(insight)

insights_df = pd.DataFrame(insights)
print(insights_df.to_string(index=False))

# Ensure output directory exists
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
insights_df.to_csv('output/segment_insights.csv', index=False)
