import pandas as pd
import numpy as np
import sys
import io

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Generate Synthetic Raw Data with typical errors
# e.g., age out of range, negative price, birth_date in future, missing customer_id, invalid email/phone, end_date before start_date
raw_data = {
    'customer_id': ['CUST01', 'CUST02', None, 'CUST04', 'CUST05', 'CUST06', 'CUST07'],
    'age': [25, 180, 34, 45, -5, 12, 60],
    'price': [150.0, 200.0, 99.9, -10.0, 50.0, 0.0, 1200.0],
    'birth_date': ['1999-05-15', '2050-01-01', '1989-11-20', '1975-03-30', '1910-06-12', '2012-07-08', '1965-12-01'],
    'email': ['alice@example.com', 'bob_example.com', 'charlie@gmail.com', None, 'eve@yahoo.com', 'frank@mail.com', 'grace@domain.com'],
    'phone': ['1234567890', '987654321', '5551234567', '4445556660', '222333', 'invalidphone', '9998887776'],
    'start_date': ['2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01', '2025-06-15', '2025-07-01'],
    'end_date': ['2025-01-10', '2025-01-15', '2025-03-10', '2025-04-15', '2025-04-30', '2025-06-20', '2025-07-15']
}

df = pd.DataFrame(raw_data)

# Convert date fields to datetime for proper validation comparisons
df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')

print("="*70)
print("ORIGINAL RAW DATASET")
print("="*70)
print(df)

# ----------------------------------------------------
# Task 1: Range Checks
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 1: RANGE CHECKS")
print("="*70)
df['valid_age'] = (df['age'] >= 0) & (df['age'] <= 150)
df['valid_price'] = df['price'] >= 0
# Convert string boundary to timestamp for robust comparison
birth_date_lower_boundary = pd.Timestamp('1920-01-01')
current_timestamp_boundary = pd.Timestamp.now()
df['valid_date'] = (df['birth_date'] >= birth_date_lower_boundary) & (df['birth_date'] <= current_timestamp_boundary)

print(f"Invalid ages (must be 0-150): {(~df['valid_age']).sum()}")
print(f"Invalid prices (must be >= 0): {(~df['valid_price']).sum()}")
print(f"Invalid birth dates (must be between 1920 and now): {(~df['valid_date']).sum()}")

# ----------------------------------------------------
# Task 2: Null Constraints
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 2: NULL CONSTRAINTS")
print("="*70)
df['valid_customer_id'] = df['customer_id'].notna()
df['valid_email'] = df['email'].notna()

print(f"Missing customer IDs: {(~df['valid_customer_id']).sum()}")
print(f"Missing emails: {(~df['valid_email']).sum()}")

# ----------------------------------------------------
# Task 3: Format Pattern Validation
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 3: FORMAT PATTERN VALIDATION")
print("="*70)
# Match standard email pattern containing '@'
df['valid_email_format'] = df['email'].str.contains('@', na=False)
# Match standard 10 digit phone numbers
df['valid_phone'] = df['phone'].str.match(r'^\d{10}$', na=False)

print(f"Invalid emails (must contain '@'): {(~df['valid_email_format']).sum()}")
print(f"Invalid phone numbers (must be exactly 10 digits): {(~df['valid_phone']).sum()}")

# ----------------------------------------------------
# Task 4: Business Rule Validation
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 4: BUSINESS RULE VALIDATION")
print("="*70)
df['valid_date_order'] = df['end_date'] >= df['start_date']
print(f"Invalid date ranges (end_date must be >= start_date): {(~df['valid_date_order']).sum()}")

# ----------------------------------------------------
# Task 5: Validation Report & Failure Isolation
# ----------------------------------------------------
print("\n" + "="*70)
print("TASK 5: VALIDATION REPORT")
print("="*70)

validation_cols = ['valid_age', 'valid_price', 'valid_date', 'valid_customer_id', 
                  'valid_email_format', 'valid_phone', 'valid_date_order']

df['passes_all_checks'] = df[validation_cols].all(axis=1)

# Isolate failures
failures = df[~df['passes_all_checks']]
failures.to_csv('output/validation_failures.csv', index=False)
print("Saved failure records to output/validation_failures.csv")

# Print Summary Report
print(f"Total Records Analyzed: {len(df)}")
print(f"Passed Clean Records:   {df['passes_all_checks'].sum()}")
print(f"Failed Records:         {(~df['passes_all_checks']).sum()}")

# Print detailed failed rows
if len(failures) > 0:
    print("\nFailed Records Details:")
    print(failures[['customer_id', 'age', 'price', 'email', 'phone', 'passes_all_checks']])

df_clean = df[df['passes_all_checks']]
df_clean.to_csv('data/processed/clean_validated_data.csv', index=False)
print("\n✓ Cleaned and validated dataset saved to data/processed/clean_validated_data.csv")
