import pandas as pd
import numpy as np
import sys
import io

# Ensure console supports UTF-8 characters on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Task 5: Reusable String Cleaning Function
def clean_text_column(series, lowercase=True, strip=True, 
                     remove_special=False, mapping=None):
    """Reusable text cleaning function for any string column."""
    result = series.copy()
    
    if result.isna().any():
        print(f"Warning: {result.isna().sum()} null values in column")
    
    if strip:
        # Fill NA temporarily with empty string or handle series.str functions which retain NaN
        result = result.str.strip()
    
    if lowercase:
        result = result.str.lower()
    
    if remove_special:
        result = result.str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    
    if mapping:
        result = result.map(mapping)
    
    return result

# Task 1: Strip Whitespace Function
def strip_all_strings(df):
    """Strip whitespace from all string columns."""
    df_cleaned = df.copy()
    string_cols = df_cleaned.select_dtypes(include=['object']).columns
    
    print("\n=== Task 1: Whitespace Removal ===")
    for col in string_cols:
        # Count before
        before = df_cleaned[col].nunique()
        
        # Apply strip
        df_cleaned[col] = df_cleaned[col].str.strip()
        
        # Count after
        after = df_cleaned[col].nunique()
        
        print(f"{col}: {before} → {after} unique values")
    
    return df_cleaned

# Task 2: Normalize Casing Function
def normalize_casing(df, columns_to_lower):
    """Normalize casing for specified columns."""
    df_cleaned = df.copy()
    print("\n=== Task 2: Casing Normalization ===")
    for col in columns_to_lower:
        df_cleaned[col] = df_cleaned[col].str.lower()
        print(f"Normalized {col} to lowercase")
    
    return df_cleaned

# Task 3: Remove Special Characters Function
def remove_special_characters(df, columns):
    """Remove special characters from specified columns."""
    df_cleaned = df.copy()
    print("\n=== Task 3: Special Character Removal ===")
    for col in columns:
        df_cleaned[col] = df_cleaned[col].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
        print(f"Removed special characters from {col}")
    
    return df_cleaned

if __name__ == "__main__":
    # Create sample dataset with messy text columns
    # Inconsistent whitespaces, casing, special chars, and variations
    data = {
        'product_name': [' Electronics ', 'electronics', 'ELECTRONICS', ' Furniture ', 'furniture', '  Software  ', None],
        'customer_name': ['JOHN', 'john', 'John', 'ALICE', 'alice', 'Alice', 'BOB'],
        'segment': ['b2b', 'b 2 b', 'b2 b', 'sme', 'small medium enterprise', 'small-medium enterprise', 'enterprise'],
        'location': ['São Paulo', 'Montréal', 'New York', 'São Paulo', 'Montréal', 'Paris', None]
    }
    
    df = pd.DataFrame(data)
    
    print("="*70)
    print("ORIGINAL MESSY DATASET")
    print("="*70)
    print(df)
    
    # ----------------------------------------------------
    # Task 1: Strip Whitespace Consistently
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("TASK 1: STRIP WHITESPACE")
    print("="*70)
    
    # Print value counts before for 2 columns to show consolidation later
    print("\nBefore strip - 'product_name' value counts:")
    print(df['product_name'].value_counts(dropna=False))
    print("\nBefore strip - 'location' value counts:")
    print(df['location'].value_counts(dropna=False))
    
    df_stripped = strip_all_strings(df)
    
    print("\nAfter strip - 'product_name' value counts:")
    print(df_stripped['product_name'].value_counts(dropna=False))
    print("\nAfter strip - 'location' value counts:")
    print(df_stripped['location'].value_counts(dropna=False))
    
    # Calculate fixed issues (difference in total characters or whitespace count)
    total_spaces_removed = 0
    for col in df.select_dtypes(include=['object']).columns:
        original_lengths = df[col].fillna('').astype(str).str.len().sum()
        stripped_lengths = df_stripped[col].fillna('').astype(str).str.len().sum()
        diff = original_lengths - stripped_lengths
        total_spaces_removed += diff
    print(f"\nSummary: Total whitespace issues fixed (spaces removed): {total_spaces_removed}")
    
    # ----------------------------------------------------
    # Task 2: Normalize Casing to Consistent Standard
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("TASK 2: NORMALIZE CASING")
    print("="*70)
    print("Business Decision: We use lowercase for categoricals ('product_name', 'customer_name', 'segment') to standardise matching, indexing, and prevent case-sensitive duplicates.")
    
    print("\nSample rows before casing normalization:")
    print(df_stripped[['product_name', 'customer_name', 'segment']].head(3))
    
    df_cased = normalize_casing(df_stripped, ['product_name', 'customer_name', 'segment'])
    
    print("\nSample rows after casing normalization:")
    print(df_cased[['product_name', 'customer_name', 'segment']].head(3))
    
    # ----------------------------------------------------
    # Task 3: Remove Special Characters Using Regex
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("TASK 3: REMOVE SPECIAL CHARACTERS")
    print("="*70)
    print("Regex Pattern Used: [^a-zA-Z0-9 ]")
    print("Explanation: Negated set matches any character NOT a letter (a-z, A-Z), digit (0-9), or space.")
    
    print("\nBefore special character removal (focusing on international locations):")
    print(df_cased['location'].value_counts(dropna=False))
    
    df_no_special = remove_special_characters(df_cased, ['location'])
    
    print("\nAfter special character removal (international characters stripped/modified):")
    print(df_no_special['location'].value_counts(dropna=False))
    
    # ----------------------------------------------------
    # Task 4: Standardize Categorical Labels Using Mapping Dictionary
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("TASK 4: STANDARDIZE CATEGORICAL LABELS")
    print("="*70)
    
    # Segment mapping dictionary
    segment_map = {
        'b2b': 'B2B',
        'b 2 b': 'B2B',
        'b2 b': 'B2B',
        'sme': 'SMB',
        'small medium enterprise': 'SMB',
        'small-medium enterprise': 'SMB',
        'enterprise': 'Enterprise'
    }
    
    print("Mapping Dictionary Definition:")
    for k, v in segment_map.items():
        print(f"  '{k}' -> '{v}'")
        
    print("\nBusiness Justifications:")
    print("  - B2B: Capitalized canonical form mapped from all lowercase/spaced formats for CRM compatibility.")
    print("  - SMB: Mapped from SME, small medium enterprise, and small-medium enterprise to consolidate standard industry acronyms.")
    print("  - Enterprise: Capitalized title-case standard.")
    
    print("\nBefore mapping - 'segment' value counts:")
    print(df_no_special['segment'].value_counts(dropna=False))
    
    df_mapped = df_no_special.copy()
    df_mapped['segment'] = df_mapped['segment'].map(segment_map)
    
    print("\nAfter mapping - 'segment' value counts:")
    print(df_mapped['segment'].value_counts(dropna=False))
    
    # ----------------------------------------------------
    # Task 5: Build Reusable String Cleaning Function
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("TASK 5: REUSABLE CLEANING PIPELINE APPLICATION")
    print("="*70)
    
    # Resetting test dataset
    raw_df = pd.DataFrame(data)
    cleaned_df = raw_df.copy()
    
    # Clean product_name: Strip, lowercase, remove special characters
    print("\nCleaning 'product_name': strip=True, lowercase=True, remove_special=True")
    cleaned_df['product_name'] = clean_text_column(
        raw_df['product_name'], strip=True, lowercase=True, remove_special=True
    )
    
    # Clean segment: Strip, lowercase, map using segment_map
    print("\nCleaning 'segment': strip=True, lowercase=True, mapping=segment_map")
    cleaned_df['segment'] = clean_text_column(
        raw_df['segment'], strip=True, lowercase=True, mapping=segment_map
    )
    
    # Clean location: Strip, lowercase=False, remove_special=True (removes diacritics)
    print("\nCleaning 'location': strip=True, lowercase=False, remove_special=True")
    cleaned_df['location'] = clean_text_column(
        raw_df['location'], strip=True, lowercase=False, remove_special=True
    )
    
    print("\nFinal Cleaned DataFrame:")
    print(cleaned_df)
    
    # Save cleaned data to processed folder
    cleaned_df.to_csv('data/processed/cleaned_strings.csv', index=False)
    print("\nCleaned data saved to data/processed/cleaned_strings.csv")
    
    # ----------------------------------------------------
    # Testing Instructions / Edge Case Test
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("PIPELINE TEST WITH EDGE CASES")
    print("="*70)
    test_cases = [
        '  Product A  ',      # Leading/trailing spaces
        'PRODUCT B',         # All caps
        'Product_C',         # Special char
        None,                # Null value
        ''                   # Empty string
    ]

    test_series = pd.Series(test_cases)
    result = clean_text_column(test_series, lowercase=True, strip=True, remove_special=True)
    print("Input test cases:")
    print(test_cases)
    print("\nCleaned output:")
    print(result)
