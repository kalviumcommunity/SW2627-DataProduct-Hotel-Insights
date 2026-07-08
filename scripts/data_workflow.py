"""Command-line data workflow for ingestion, processing, and output.

This script reads a sample CSV file, applies a small set of reusable
transformations, and writes an analysis-ready output file.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Use UTF-8 output so the success markers render correctly in Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ingest_data(filepath: Path):
    """Load tabular data from a CSV file and return a Pandas DataFrame.

    Input: A file path pointing to a CSV dataset with a header row.
    Output: A Pandas DataFrame containing the raw records.
    Assumptions: The file exists and is encoded as UTF-8 compatible text.
    """
    import pandas as pd

    # Read the raw CSV into a DataFrame so the rest of the workflow can reuse it.
    df = pd.read_csv(filepath)
    return df


def process_data(df):
    """Transform raw records into analysis-ready output.

    Input: A Pandas DataFrame containing the ingested raw dataset.
    Output: A cleaned Pandas DataFrame with duplicates removed and basic
    derived fields added when the source columns are available.
    Assumptions: Column names are stable enough for lightweight rule-based cleanup.
    """
    import pandas as pd

    processed = df.copy()

    # Remove exact duplicate rows so downstream counts are not inflated.
    processed = processed.drop_duplicates()

    # Trim column names to avoid subtle mismatches caused by leading or trailing spaces.
    processed.columns = [column.strip() for column in processed.columns]

    # Fill missing text values with a readable placeholder for easier inspection.
    text_columns = processed.select_dtypes(include=["object"]).columns
    for column in text_columns:
        processed[column] = processed[column].fillna("Unknown")

    # Fill missing numeric values with the median so the dataset stays usable.
    numeric_columns = processed.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        if processed[column].notna().any():
            processed[column] = processed[column].fillna(processed[column].median())

    # Add a small summary column when the source schema supports it.
    if {"total_spend", "stay_nights"}.issubset(processed.columns):
        processed["spend_per_night"] = processed["total_spend"] / processed["stay_nights"].replace(0, pd.NA)

    # Standardize any date-like booking column when present for consistent reporting.
    if "booking_date" in processed.columns:
        processed["booking_date"] = pd.to_datetime(processed["booking_date"], errors="coerce").dt.date

    return processed


def output_results(df, output_path: Path):
    """Save the processed dataset and print execution confirmation.

    Input: A Pandas DataFrame with processed records and a destination file path.
    Output: Writes a CSV file and prints status messages to standard output.
    Assumptions: The destination directory already exists or can be created.
    """
    # Ensure the output directory exists before writing the CSV file.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist the final dataset so the workflow can be consumed by other tools.
    df.to_csv(output_path, index=False)

    print("✓ Data successfully processed")
    print(f"✓ Rows processed: {len(df)}")
    print(f"✓ Output saved to {output_path}")


def main():
    """Run the end-to-end data workflow from the command line.

    Input: None directly. Uses the sample dataset bundled with the repository.
    Output: A processed CSV file written to the output directory.
    Assumptions: The repository layout matches the expected scripts/data/output paths.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_path = project_root / "data" / "raw" / "sample.csv"
    output_path = project_root / "output" / "processed.csv"

    # Read the sample data, transform it, and write the finished dataset.
    data = ingest_data(input_path)
    processed = process_data(data)
    output_results(processed, output_path)


if __name__ == "__main__":
    main()