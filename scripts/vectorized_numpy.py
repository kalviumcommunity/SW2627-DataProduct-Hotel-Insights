import pandas as pd
import numpy as np
import time
import sys
import io

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Generate synthetic dataset of 100,000 customers
print("Generating 100k customer records...")
np.random.seed(42)
n_records = 100000
df = pd.DataFrame({
    'customer_id': [f"CUST_{i:06d}" for i in range(1, n_records + 1)],
    'revenue': np.random.exponential(scale=500.0, size=n_records) + 50.0
})

print("="*70)
print("TASK 1: MIN-MAX NORMALIZATION (LOOP VS NUMPY)")
print("="*70)

# SLOW: Loop Min-Max
start_loop_minmax = time.perf_counter()
normalized_loop = []
min_val = df['revenue'].min()
max_val = df['revenue'].max()
range_val = max_val - min_val
for val in df['revenue']:
    normalized_loop.append((val - min_val) / range_val)
loop_minmax_time = time.perf_counter() - start_loop_minmax

# FAST: NumPy Min-Max
start_np_minmax = time.perf_counter()
revenue_array = df['revenue'].values
normalized_np = (revenue_array - revenue_array.min()) / (revenue_array.max() - revenue_array.min())
df['revenue_normalized'] = normalized_np
np_minmax_time = time.perf_counter() - start_np_minmax

print(f"Loop Min-Max Time:  {loop_minmax_time:.6f}s")
print(f"NumPy Min-Max Time: {np_minmax_time:.6f}s")
minmax_speedup = loop_minmax_time / np_minmax_time if np_minmax_time > 0 else float('inf')
print(f"Min-Max Speedup:    {minmax_speedup:.1f}x")

# ----------------------------------------------------
# Task 2: Z-Score Normalization
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 2: Z-SCORE NORMALIZATION")
print("="*70)
revenue_array = df['revenue'].values
z_scores = (revenue_array - revenue_array.mean()) / revenue_array.std()
df['revenue_zscore'] = z_scores
print("Calculated Z-Scores successfully.")
print(df[['revenue', 'revenue_zscore']].head())

# ----------------------------------------------------
# Task 3: Bulk Ranking/Scoring
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 3: BULK RANKING")
print("="*70)
# Rank all customers by revenue (1 = highest revenue)
revenue_array = df['revenue'].values
rankings = np.argsort(-revenue_array)  # Negative for descending
ranks = np.empty_like(rankings)
ranks[rankings] = np.arange(1, len(rankings) + 1)
df['revenue_rank'] = ranks

print("Computed revenue ranks successfully.")
print(df[['customer_id', 'revenue', 'revenue_rank']].sort_values('revenue_rank').head(10))

# ----------------------------------------------------
# Task 4: Time Performance Comparison
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 4: GENERAL MULTIPLICATION PERFORMANCE TIMER")
print("="*70)

# Time loop version
start = time.perf_counter()
result_loop = []
for val in df['revenue']:
    result_loop.append(val * 1.1)
loop_time = time.perf_counter() - start

# Time NumPy version
start = time.perf_counter()
result_np = df['revenue'].values * 1.1
np_time = time.perf_counter() - start

print(f"Loop Multiplication:  {loop_time:.6f}s")
print(f"NumPy Multiplication: {np_time:.6f}s")
mult_speedup = loop_time / np_time if np_time > 0 else float('inf')
print(f"Multiplication Speedup: {mult_speedup:.1f}x")

# ----------------------------------------------------
# Task 5: Integrate Back to DataFrame & Verify
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 5: INTEGRATION AND VERIFICATION")
print("="*70)

df['revenue_normalized'] = normalized_np
df['revenue_zscore'] = z_scores
df['revenue_rank'] = ranks

# Verify types and shapes
print(f"Final DataFrame Shape: {df.shape}")
print(f"\nFinal DataFrame Dtypes:\n{df.dtypes}")

# Save output
df.to_csv('data/processed/vectorized_revenue.csv', index=False)
print("\n✓ Processed and verified dataset saved to data/processed/vectorized_revenue.csv")
