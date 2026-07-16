import pandas as pd
import numpy as np
import json
import sys
import io

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Generate Synthetic Data
np.random.seed(42)

# Generate 1000 customers
cust_ids = [f"CUST_{i:04d}" for i in range(1, 1001)]
df_customers = pd.DataFrame({
    'customer_id': cust_ids,
    'customer_name': [f"Customer {i}" for i in range(1, 1001)],
    'segment': np.random.choice(['B2B', 'SMB', 'Enterprise'], size=1000)
})

# Generate 5000 orders
# Let 950 customers be active. 50 customers will have no orders.
# Introduce 50 orphaned customer IDs for orders to simulate keys missing from customers table.
active_custs = cust_ids[:950]
orphaned_custs = [f"CUST_{i:04d}" for i in range(1001, 1051)] # Orphaned customer IDs (not in customers)

all_order_cust_ids = np.random.choice(active_custs + orphaned_custs, size=5000)

df_orders = pd.DataFrame({
    'order_id': [f"ORD_{i:05d}" for i in range(1, 5001)],
    'customer_id': all_order_cust_ids,
    'amount': np.random.uniform(5.0, 1000.0, size=5000),
    'order_date': pd.date_range(start='2025-01-01', periods=5000, freq='min').strftime('%Y-%m-%d %H:%M:%S')
})

# ----------------------------------------------------
# Task 1: Explicit Join with Row Count Validation
# ----------------------------------------------------
print("="*70)
print("TASK 1: EXPLICIT JOIN & ROW COUNT VALIDATION")
print("="*70)
print(f"Left (Customers): {len(df_customers)}")
print(f"Right (Orders): {len(df_orders)}")

df_merged = pd.merge(df_customers, df_orders, on='customer_id', how='left')

print(f"Merged (Left Join): {len(df_merged)}")
print(f"Change (Merged - Left): {len(df_merged) - len(df_customers)}")

# ----------------------------------------------------
# Task 2: Detect Unmatched Keys
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 2: DETECT UNMATCHED KEYS")
print("="*70)
unmatched_customers = df_customers[~df_customers['customer_id'].isin(df_orders['customer_id'])]
unmatched_orders = df_orders[~df_orders['customer_id'].isin(df_customers['customer_id'])]

print(f"Customers without orders (unmatched left): {len(unmatched_customers)}")
print(f"Orphaned orders (unmatched right): {len(unmatched_orders)}")

# Export unmatched keys to output
unmatched_customers.to_csv('output/unmatched_customers.csv', index=False)
unmatched_orders.to_csv('output/unmatched_orders.csv', index=False)
print("Saved unmatched records to output/unmatched_customers.csv and output/unmatched_orders.csv")

# ----------------------------------------------------
# Task 3: Compare Join Types
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 3: COMPARE JOIN TYPES")
print("="*70)
inner = pd.merge(df_customers, df_orders, on='customer_id', how='inner')
left = pd.merge(df_customers, df_orders, on='customer_id', how='left')
outer = pd.merge(df_customers, df_orders, on='customer_id', how='outer')

print(f"Inner Join Rows: {len(inner)}")
print(f"Left Join Rows:  {len(left)}")
print(f"Outer Join Rows: {len(outer)}")

# ----------------------------------------------------
# Task 4: Validate No Unexpected Duplication
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 4: VALIDATE NO UNEXPECTED DUPLICATION")
print("="*70)
print("Merged Columns:")
print(list(df_merged.columns))

# Verify if there are overlapping columns with suffixes (e.g., _x, _y)
overlapping_cols = [col for col in df_merged.columns if col.endswith('_x') or col.endswith('_y')]
if overlapping_cols:
    print(f"WARNING: Found overlapping columns with suffixes: {overlapping_cols}")
else:
    print("✓ No unexpected column conflicts or suffixes detected.")

key_counts = df_merged['customer_id'].value_counts()
print(f"Max orders per customer: {key_counts.max()}")

# ----------------------------------------------------
# Task 5: Document Join Decision
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 5: DOCUMENT JOIN DECISION REPORT")
print("="*70)
join_report = {
    'join_type': 'left',
    'left_table': 'customers',
    'right_table': 'orders',
    'join_key': 'customer_id',
    'left_rows': len(df_customers),
    'right_rows': len(df_orders),
    'result_rows': len(df_merged),
    'unmatched_left': len(unmatched_customers),
    'unmatched_right': len(unmatched_orders),
    'reasoning': 'Left join preserves all customer records regardless of order history, which prevents losing non-purchasing customers in downstream marketing segments, while flagging orphaned orders for data quality reviews.'
}

# Print report
print(json.dumps(join_report, indent=2))

# Save report
with open('output/join_decision_report.json', 'w') as f:
    json.dump(join_report, f, indent=2)
print("\n✓ Saved join report to output/join_decision_report.json")
