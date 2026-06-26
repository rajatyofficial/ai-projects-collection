"""
Data Loading & Initial Exploration Module
==========================================
Handles loading the Airbnb dataset and provides functions for
initial data exploration (shape, info, head, tail, describe, etc.)
"""

import pandas as pd


def load_data(filepath: str = '../data/airnb.csv') -> pd.DataFrame:
    """
    Load the Airbnb CSV dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file. Defaults to '../data/airnb.csv'
        (relative to the Notebooks directory).
    
    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"[OK] Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Print a comprehensive initial exploration summary of the DataFrame.
    Includes: shape, info, head, tail, describe, duplicates, and unique counts.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to explore.
    """
    print("=" * 60)
    print("INITIAL DATA EXPLORATION")
    print("=" * 60)

    # Shape
    print(f"\n>> Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Info
    print("\n>> Data Types & Non-Null Counts:")
    print("-" * 40)
    df.info()

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print("\n>> Missing Values:")
    print("-" * 40)
    print(missing_df[missing_df['Missing'] > 0].to_string() if missing.any() else "  No missing values!")

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n>> Duplicate Rows: {dup_count} ({(dup_count / len(df) * 100):.2f}%)")

    # Unique value counts
    print("\n>> Unique Values Per Column:")
    print("-" * 40)
    print(df.nunique().to_string())

    print("\n" + "=" * 60)


def show_sample(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Display a quick sample of the DataFrame (head).
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to sample.
    n : int
        Number of rows to show.
    
    Returns
    -------
    pd.DataFrame
        The first n rows.
    """
    print(f"\n>> First {n} rows:")
    return df.head(n)


def show_column_summary(df: pd.DataFrame, column: str) -> None:
    """
    Show a detailed summary for a specific column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column.
    column : str
        The column name to summarize.
    """
    if column not in df.columns:
        print(f"[ERROR] Column '{column}' not found in DataFrame!")
        return

    col = df[column]
    print(f"\n>> Column Summary: '{column}'")
    print("-" * 40)
    print(f"  Type:       {col.dtype}")
    print(f"  Non-Null:   {col.notna().sum()} / {len(col)}")
    print(f"  Null:       {col.isna().sum()}")
    print(f"  Unique:     {col.nunique()}")

    if col.dtype in ['object', 'str', 'string']:
        print(f"\n  Top 10 Values:")
        print(col.value_counts().head(10).to_string())
    else:
        print(f"\n  Statistics:")
        print(col.describe().to_string())
