import pandas as pd
import numpy as np
import sys
import io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure output directories exist
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Generate Synthetic Time Series Data for 1 year (365 days)
np.random.seed(42)
date_range = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
n_days = len(date_range)

# Create daily fluctuations around a base of $10,000 with a slight upward trend
base_revenue = 10000.0
trend = np.linspace(0, 5000, n_days)  # Revenue grows by up to $5000 by the end of the year
noise = np.random.normal(0, 2000, n_days)  # Daily noise with standard deviation of $2000
revenue = base_revenue + trend + noise
revenue = np.clip(revenue, 1000, None)  # Ensure no negative revenue

# Daily orders (correlated with revenue)
orders = (revenue / 100).astype(int) + np.random.randint(-5, 5, n_days)
orders = np.clip(orders, 10, None)

df = pd.DataFrame({
    'date': date_range,
    'revenue': revenue,
    'orders': orders
})

# ======================================================================
# Task 1: Resample Data by Time Period
# ======================================================================
print("="*70)
print("TASK 1: RESAMPLE DATA BY TIME PERIOD")
print("="*70)

df_ts = df.set_index('date')

# Weekly aggregation
weekly_revenue = df_ts['revenue'].resample('W').sum()
weekly_count = df_ts['orders'].resample('W').count()
weekly_avg = df_ts['revenue'].resample('W').mean()

print("Weekly Revenue Summary (First 5 weeks):")
print(weekly_revenue.head())
print("\nWeekly Order Count (First 5 weeks):")
print(weekly_count.head())
print("\nWeekly Average Daily Revenue (First 5 weeks):")
print(weekly_avg.head())

# Monthly aggregation
monthly_revenue = df_ts['revenue'].resample('ME').sum()
monthly_count = df_ts['orders'].resample('ME').count()
monthly_avg = df_ts['revenue'].resample('ME').mean()

print("\nMonthly Revenue Summary:")
print(monthly_revenue)

# Compare results: show which period has highest revenue
highest_weekly_period = weekly_revenue.idxmax().strftime('%Y-%m-%d')
highest_monthly_period = monthly_revenue.idxmax().strftime('%B %Y')

print(f"\nHighest Weekly Revenue: Week ending {highest_weekly_period} (${weekly_revenue.max():,.2f})")
print(f"Highest Monthly Revenue: {highest_monthly_period} (${monthly_revenue.max():,.2f})")

# ======================================================================
# Task 2: Compute Rolling Window Average
# ======================================================================
print("\n" + "="*70)
print("TASK 2: COMPUTE ROLLING WINDOW AVERAGE")
print("="*70)

df['revenue_ma7'] = df['revenue'].rolling(window=7).mean()
df['revenue_ma30'] = df['revenue'].rolling(window=30).mean()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['revenue'], label='Raw Revenue', color='lightgrey', alpha=0.6)
plt.plot(df['date'], df['revenue_ma7'], label='7-day Moving Average', color='orange', linewidth=1.5)
plt.plot(df['date'], df['revenue_ma30'], label='30-day Moving Average', color='teal', linewidth=2.0)
plt.title('Daily Revenue with 7-Day & 30-Day Moving Averages')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.tight_layout()
plt.savefig('output/rolling_avg.png')
plt.savefig('rolling_avg.png')  # Save to root as requested by task instruction
plt.close()

print("Rolling averages computed successfully. Plot saved to 'output/rolling_avg.png' and 'rolling_avg.png'.")
print("Trend Observation: The 30-day moving average smooths out the daily noise of $2k fluctuations and clearly reveals a steady upward baseline growth trend starting around $10k in Jan and ending near $15k in Dec.")

# ======================================================================
# Task 3: Calculate Month-over-Month Percentage Change
# ======================================================================
print("\n" + "="*70)
print("TASK 3: MONTH-OVER-MONTH PERCENTAGE CHANGE")
print("="*70)

mom_change = monthly_revenue.pct_change() * 100
mom_change_df = pd.DataFrame({
    'Revenue': monthly_revenue,
    'MoM_Change_%': mom_change
})
print(mom_change_df)

# Show which months had growth vs decline
growth_months = mom_change[mom_change > 0]
decline_months = mom_change[mom_change < 0]

print("\nMonths with Positive Growth:")
for date, val in growth_months.items():
    print(f"- {date.strftime('%B %Y')}: +{val:.2f}%")

print("\nMonths with Decline:")
for date, val in decline_months.items():
    print(f"- {date.strftime('%B %Y')}: {val:.2f}%")

print("\nPattern Explanation:")
print("The month-over-month growth shows a pattern of overall accelerating/sustainable expansion with minor fluctuations due to monthly length variations and noise. The majority of the months show positive growth, demonstrating stable upward trajectory.")

# ======================================================================
# Task 4: Compute Cumulative Sum
# ======================================================================
print("\n" + "="*70)
print("TASK 4: COMPUTE CUMULATIVE SUM")
print("="*70)

df['cumulative_revenue'] = df['revenue'].cumsum()

plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['cumulative_revenue'], color='indigo', linewidth=2.5)
plt.title('Cumulative Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Revenue ($)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('output/cumulative.png')
plt.savefig('cumulative.png')  # Save to root as requested by task instruction
plt.close()

print(f"Total revenue accumulated by end of period: ${df['cumulative_revenue'].iloc[-1]:,.2f}")
print("Cumulative plot saved to 'output/cumulative.png' and 'cumulative.png'.")

# ======================================================================
# Task 5: Identify Trend Pattern and Business Implications
# ======================================================================
print("\n" + "="*70)
print("TASK 5: IDENTIFY TREND PATTERN AND BUSINESS IMPLICATIONS")
print("="*70)

# Analyze rolling average trend
recent_ma30 = df['revenue_ma30'].iloc[-30:]
trend_direction = 'up' if recent_ma30.iloc[-1] > recent_ma30.iloc[0] else 'down'
trend_magnitude = ((recent_ma30.iloc[-1] - recent_ma30.iloc[0]) / recent_ma30.iloc[0]) * 100

analysis = f"""
TREND ANALYSIS:

Rolling Average Trend: {trend_direction.upper()}
Change over last 30 days: {trend_magnitude:.1f}%

Month-over-month growth: {mom_change.iloc[-1]:.1f}%

Business Implications:
- {['Accelerating growth - maintain current strategy', 'Declining momentum - investigate causes'][0 if trend_direction == 'up' else 1]}
- Revenue volatility: ${df['revenue'].std():.0f} (measure of noise)

Business Implication Details:
- The business is experiencing a healthy, accelerating growth trend when looking at the 30-day moving average.
- The high standard deviation (${df['revenue'].std():.0f}) indicates that looking only at raw daily fluctuations would suggest false crises or spikes, whereas the trend is positive.

Suggested Action:
- Scale marketing and operations progressively to match the accelerating demand trajectory.
- Use 30-day moving averages for forecasting rather than daily/weekly raw numbers to avoid overreacting to daily volatility.
"""

print(analysis)
