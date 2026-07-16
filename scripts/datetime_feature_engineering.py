import pandas as pd
import numpy as np
import sys
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Generate Synthetic Data spanning from Jan 2023 to June 2025 to allow seasonal decomposition (52 weeks period)
np.random.seed(42)
date_range = pd.date_range(start='2023-01-01 00:00:00', end='2025-06-01 23:59:59', freq='h')
# Pick 5000 random datetimes to ensure good distribution density
random_dates = np.random.choice(date_range, size=5000)

# Apply seasonal logic to make analysis insightful:
# 1. Busiest hour is mid-day (12:00 - 15:00)
# 2. Busiest day is Saturday
# 3. December has a significant sales spike
adjusted_dates = []
amounts = []
for d in random_dates:
    ts = pd.Timestamp(d)
    amt = np.random.uniform(10.0, 500.0)
    if ts.month == 12:
        amt *= 1.8 # December sales spike
    if ts.weekday() == 5: # Saturday
        amt *= 1.5 # Saturdays are busiest
    elif ts.weekday() == 6: # Sunday
        amt *= 0.5 # Sundays are slowest
    
    # Adjust hour probability (more purchases during lunch hour)
    if 12 <= ts.hour <= 14:
        amt *= 1.3
        
    adjusted_dates.append(ts)
    amounts.append(amt)

# Convert dates to string formatted raw timestamps
datetime_strings = [d.strftime('%Y-%m-%d %H:%M:%S') for d in adjusted_dates]
customer_ids = [f"CUST_{np.random.randint(1, 51):03d}" for _ in range(5000)]

df = pd.DataFrame({
    'customer_id': customer_ids,
    'transaction_date_raw': datetime_strings,
    'amount': amounts
})

print("="*70)
print("ORIGINAL DATASET (FIRST 5 ROWS)")
print("="*70)
print(df.head())
print(df.dtypes)

# ----------------------------------------------------
# Task 1: Parse Timestamp Strings with Explicit Format
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 1: PARSE TIMESTAMP STRINGS")
print("="*70)
df['transaction_date'] = pd.to_datetime(
    df['transaction_date_raw'],
    format='%Y-%m-%d %H:%M:%S'
)
print(f"Format specification: '%Y-%m-%d %H:%M:%S'")
print(f"Parsed Series Dtype: {df['transaction_date'].dtype}")

# ----------------------------------------------------
# Task 2: Extract Day-of-Week and Hour-of-Day
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 2: EXTRACT DAY-OF-WEEK AND HOUR-OF-DAY")
print("="*70)
df['day_of_week'] = df['transaction_date'].dt.day_name()
df['hour'] = df['transaction_date'].dt.hour

hourly_volume = df.groupby('hour').size()
print("Hourly Volume Distribution:")
print(hourly_volume)

# Plot hour distribution
plt.figure(figsize=(10, 6))
plt.hist(df['hour'], bins=24, range=(0, 24), color='skyblue', edgecolor='black', rwidth=0.8)
plt.title('Transaction Volume by Hour of Day')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Transaction Count')
plt.grid(axis='y', alpha=0.75)
plt.savefig('output/hourly_distribution.png')
print("\nSaved hourly volume histogram to output/hourly_distribution.png")

# ----------------------------------------------------
# Task 3: Compute Week Number and Resample Data
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 3: WEEK NUMBER AND RESAMPLING")
print("="*70)
df['week_num'] = df['transaction_date'].dt.isocalendar().week

# Set index for resampling
df_ts = df.set_index('transaction_date')
weekly_revenue = df_ts['amount'].resample('W').sum()

print("Weekly Revenue Trend (First 10 weeks):")
print(weekly_revenue.head(10))

# ----------------------------------------------------
# Task 4: Compute Days-Since-Event Metric
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 4: DAYS-SINCE-EVENT RECENCY METRIC")
print("="*70)
today = pd.Timestamp('2026-07-16 09:00:00') # Setting current reference timestamp

# Find last purchase date for each customer
customer_last_purchase = df.groupby('customer_id')['transaction_date'].max()

# Map last purchase date back to main dataframe to compute per-transaction recency relative to the customer's last purchase
# or compute current recency for each customer
df['days_since_last_purchase'] = df['customer_id'].map(customer_last_purchase)
df['days_since_last_purchase'] = (today - df['days_since_last_purchase']).dt.days

print("Recency Distribution (days since last purchase per customer):")
print(df['days_since_last_purchase'].describe())

# Identify inactive/churned customers (recency > 450 days in this static dataset)
churned_threshold = 450
churned_customers = df[df['days_since_last_purchase'] > churned_threshold]['customer_id'].unique()
print(f"\nNumber of customers with no recent activity (>{churned_threshold} days): {len(churned_customers)}")
print(f"Sample inactive customers: {churned_customers[:5]}")

# ----------------------------------------------------
# Task 5: Build Time-Indexed Aggregation
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 5: TIME-INDEXED AGGREGATION & HEATMAP")
print("="*70)

# Multi-level groupby
hourly_daily = df.groupby(['day_of_week', 'hour']).agg({
    'amount': ['sum', 'count', 'mean']
})
print("Sample multi-level groupby (Day of Week × Hour):")
print(hourly_daily.head(10))

# Pivot table showing hour × day-of-week heatmap
pivot_table = pd.pivot_table(
    df,
    values='amount',
    index='hour',
    columns='day_of_week',
    aggfunc='sum'
)

# Reorder columns to standard chronological order
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_table = pivot_table.reindex(columns=days_order)

print("\nHour × Day-of-Week Pivot Table (Total Sales Amount):")
print(pivot_table.round(2))

# Peak activity analysis
peak_cell = pivot_table.stack().idxmax()
print(f"\nPeak activity window: {peak_cell[1]} at {peak_cell[0]}:00 with total volume {pivot_table.loc[peak_cell]:.2f}")

# Save pivot to csv
pivot_table.to_csv('output/hourly_weekly_sales_pivot.csv')
print("Saved pivot table to output/hourly_weekly_sales_pivot.csv")

# ----------------------------------------------------
# Advanced Temporal Analysis: Seasonal Decomposition
# ----------------------------------------------------
print("\n" + "="*70)
print("ADVANCED TEMPORAL ANALYSIS: SEASONAL DECOMPOSITION")
print("="*70)

# Resample weekly total sales amount
ts_weekly = df_ts['amount'].resample('W').sum().fillna(0)

# Execute seasonal decomposition
# Since there are ~126 weeks (more than 2 full periods of 52 weeks), this is perfect
decomposition = seasonal_decompose(ts_weekly, model='additive', period=52)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Observed Weekly Sales')
decomposition.trend.plot(ax=axes[1], title='Trend Component')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component (52-week period)')
decomposition.resid.plot(ax=axes[3], title='Residual/Noise Component')
plt.tight_layout()
plt.savefig('output/seasonal_decomposition.png')
print("Saved seasonal decomposition plots to output/seasonal_decomposition.png")

# ----------------------------------------------------
# Testing Instructions & Verification
# ----------------------------------------------------
print("\n" + "="*70)
print("VERIFICATION AND TESTING")
print("="*70)
print(f"Min date in dataset: {df['transaction_date'].min()}")
print(f"Max date in dataset: {df['transaction_date'].max()}")
print(f"Days spanned in dataset: {(df['transaction_date'].max() - df['transaction_date'].min()).days}")
print(f"Hours with data: {sorted(df['hour'].unique())}")
print(f"Weeks in dataset: {df['week_num'].nunique()}")
print(f"Min days since last purchase: {df['days_since_last_purchase'].min()}")
print(f"Max days since last purchase: {df['days_since_last_purchase'].max()}")

# ----------------------------------------------------
# Edge Cases and Timezone Handling
# ----------------------------------------------------
print("\n" + "="*70)
print("TIMEZONE HANDLING AND FORMAT TESTING")
print("="*70)

test_dates = [
    '2025-01-15 14:30:45',        # Standard
    '2025-1-15 14:30:45',         # Single-digit month
    '15/01/2025 14:30:45',        # European format
    '2025-01-15T14:30:45Z',       # ISO format with Z
]

# We try to parse using multiple format matching or pd.to_datetime with format list / general parsing
for date_str in test_dates:
    try:
        # Standardize matching formats
        if 'Z' in date_str:
            parsed = pd.to_datetime(date_str, format='%Y-%m-%dT%H:%M:%SZ')
        elif '/' in date_str:
            parsed = pd.to_datetime(date_str, format='%d/%m/%Y %H:%M:%S')
        else:
            # Let pandas infer format or match standard format
            parsed = pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
        print(f"✓ {date_str} -> Parsed successfully as: {parsed}")
    except Exception as e:
        print(f"✗ {date_str} - format mismatch: {e}")
