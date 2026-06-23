"""
Analysis & Visualization Module for Airbnb Listings Deep Dive
==============================================================
Provides:
  1. prepare_analysis_df() — drops redundant raw columns, reorders the
     remaining columns into logical groups, and returns a clean DataFrame
     ready for graphing and statistical analysis.

  2. save_analysis_df() / load_analysis_df() — persist the cleaned DataFrame
     to a .parquet file so the full pipeline doesn't need to re-run every time.

  3. (Future) Univariate, bivariate, and multivariate analysis functions
     that operate on the cleaned DataFrame returned by prepare_analysis_df().

Usage
-----
    from analysis import load_analysis_df

    # First time — run the full pipeline and save:
    from data_loader import load_data
    from feature_extraction import extract_all_features
    from data_cleaning import clean_all_columns
    from analysis import prepare_analysis_df, save_analysis_df

    df = load_data()
    df = extract_all_features(df)
    df = clean_all_columns(df)
    df = prepare_analysis_df(df)
    save_analysis_df(df)

    # Subsequent times — just load directly:
    df = load_analysis_df()   # reads from ../data/cleaned_and_processed_data.parquet
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# COLUMNS THAT BECOME REDUNDANT AFTER FEATURE EXTRACTION & CLEANING
# =============================================================================

# These are the raw string columns whose information has been fully extracted
# into typed, analysis-ready columns by feature_extraction.py and data_cleaning.py.
_RAW_COLUMNS_TO_DROP = [
    "Title",                   # → Property_Type, City, State, Country
    "Detail",                  # → amenity/location flags, Theme, Property_Subtype, text metrics
    "Date",                    # → Month, Start_Day, End_Day, Duration_Nights
    "Price(in dollar)",        # → Price (float)
    "Offer price(in dollar)",  # → Offer_Price (float)
    "Review and rating",       # → Rating, Review_Count, Is_New_Listing
    "Number of bed",           # → Bed_Count, Bed_Type
]

# Columns with extremely low fill-rates or limited analytical value
_LOW_VALUE_COLUMNS_TO_DROP = [
    "Bedroom_Count",  # Only 43/953 non-null (4.5%) — too sparse to be useful
    "Start_Day",      # Rarely needed on its own; Duration_Nights captures the useful info
    "End_Day",        # Same as above
]

# Logical grouping and preferred column order for the analysis DataFrame
_ANALYSIS_COLUMN_ORDER = [
    # --- Identity / Location ---
    "Property_Type",
    "Property_Subtype",
    "City",
    "State",
    "Country",

    # --- Pricing ---
    "Price",
    "Offer_Price",
    "Discount_Pct",
    "Price_Per_Night",

    # --- Reviews ---
    "Rating",
    "Review_Count",
    "Is_New_Listing",

    # --- Beds ---
    "Bed_Count",
    "Bed_Type",

    # --- Dates ---
    "Month",
    "Duration_Nights",

    # --- Amenity Flags ---
    "Has_Hot_Tub",
    "Has_Sauna",
    "Has_Pool",
    "Has_BBQ",
    "Has_Fireplace",
    "Has_Parking",
    "Has_Spa",

    # --- Location Flags ---
    "Has_Waterfront",
    "Is_Near_Beach",
    "Is_Mountain",
    "Is_Rural",
    "Is_Urban",

    # --- Classification ---
    "Theme",

    # --- Text Metrics ---
    "Detail_Word_Count",
    "Detail_Char_Count",
    "Has_Promo",
]


# =============================================================================
# CORE FUNCTION: Prepare a clean, analysis-ready DataFrame
# =============================================================================

def prepare_analysis_df(df: pd.DataFrame, drop_low_value: bool = True) -> pd.DataFrame:
    """
    Remove redundant / low-value columns and reorder the remaining columns
    into logical groups for analysis and visualization.

    Parameters
    ----------
    df : pd.DataFrame
        The 42-column DataFrame produced by the full pipeline
        (load_data → extract_all_features → clean_all_columns).
    drop_low_value : bool, default True
        If True, also drops columns with very low fill-rates
        (Bedroom_Count, Start_Day, End_Day).  Set to False if you
        specifically need those columns for a targeted analysis.

    Returns
    -------
    pd.DataFrame
        A clean DataFrame with columns logically ordered and redundant
        raw columns removed.  Typically 32 columns when drop_low_value=True.
    """
    df = df.copy()

    # --- Drop redundant raw columns ---
    cols_to_drop = [c for c in _RAW_COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # --- Optionally drop low-value columns ---
    if drop_low_value:
        low_value = [c for c in _LOW_VALUE_COLUMNS_TO_DROP if c in df.columns]
        df = df.drop(columns=low_value)

    # --- Reorder columns (keep any unexpected extras at the end) ---
    ordered = [c for c in _ANALYSIS_COLUMN_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    print(f"[OK] Analysis DataFrame ready: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# =============================================================================
# HELPER: Quick summary of the analysis DataFrame
# =============================================================================

def analysis_overview(df: pd.DataFrame) -> None:
    """Print a compact overview of the analysis-ready DataFrame."""
    print("=" * 60)
    print("ANALYSIS DATAFRAME OVERVIEW")
    print("=" * 60)

    # Column groups
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in bool_cols + num_cols]

    print(f"\n  Total columns : {df.shape[1]}")
    print(f"  Numeric       : {len(num_cols)}  → {', '.join(num_cols)}")
    print(f"  Boolean flags : {len(bool_cols)}  → {', '.join(bool_cols)}")
    print(f"  Categorical   : {len(cat_cols)}  → {', '.join(cat_cols)}")

    # Missing-value snapshot (only columns with missing data)
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(f"\n  Columns with missing values ({len(missing)}):")
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"    {col:<20s} {count:>4d} missing  ({pct:5.1f}%)")
    else:
        print("\n  No missing values!")

    print("\n" + "=" * 60)


# =============================================================================
# PARQUET PERSISTENCE  (save / load the analysis-ready DataFrame)
# =============================================================================

# Path to the parquet file, relative to this script's location (Notebooks/)
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
_PARQUET_FILENAME = 'cleaned_and_processed_data.parquet'
_PARQUET_PATH = os.path.join(_DATA_DIR, _PARQUET_FILENAME)


def save_analysis_df(df: pd.DataFrame, path: str = _PARQUET_PATH) -> str:
    """
    Save the analysis-ready DataFrame to a .parquet file using fastparquet.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame (output of prepare_analysis_df).
    path : str, optional
        Destination file path.  Defaults to
        ``../data/cleaned_and_processed_data.parquet``.

    Returns
    -------
    str
        The absolute path the file was written to.
    """
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine='fastparquet')
    abs_path = os.path.abspath(path)
    print(f"[OK] Saved analysis DataFrame → {abs_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return abs_path


def load_analysis_df(path: str = _PARQUET_PATH) -> pd.DataFrame:
    """
    Load the analysis-ready DataFrame from the .parquet file using fastparquet.

    If the parquet file does not exist, raises FileNotFoundError with a
    helpful message telling the user to run the full pipeline first.

    Parameters
    ----------
    path : str, optional
        Path to the parquet file.  Defaults to
        ``../data/cleaned_and_processed_data.parquet``.

    Returns
    -------
    pd.DataFrame
        The previously saved analysis DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Parquet file not found at: {os.path.abspath(path)}\n"
            f"Run the full pipeline first:\n"
            f"  df = load_data()\n"
            f"  df = extract_all_features(df)\n"
            f"  df = clean_all_columns(df)\n"
            f"  df = prepare_analysis_df(df)\n"
            f"  save_analysis_df(df)"
        )
    df = pd.read_parquet(path, engine='fastparquet')
    print(f"[OK] Loaded analysis DataFrame ← {os.path.abspath(path)}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return df


def parquet_exists(path: str = _PARQUET_PATH) -> bool:
    """Check whether the processed parquet file already exists."""
    return os.path.exists(path)


# =============================================================================
# UNIVARIATE ANALYSIS
# =============================================================================
# TODO: Add functions for:
#   - Distribution plots (histograms, KDE) for numeric columns
#   - Count plots / bar charts for categorical columns
#   - Boolean flag frequency summary

def univariate_bar_plot(df: pd.DataFrame, column: str, top_n: int = 10,
                        palette: str = "viridis", figsize: tuple = (12, 8)) -> None:
    """
    Plot the value counts of any categorical column as a horizontal bar chart.

    Shows the top N most frequent values with counts annotated on each bar.
    Works with any categorical (string/object) column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    column : str
        Name of the categorical column to plot.
    top_n : int, optional
        Number of top categories to display. Defaults to 10.
    palette : str, optional
        Seaborn color palette name. Defaults to "viridis".
    figsize : tuple, optional
        Figure size as (width, height). Defaults to (12, 8).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available: {list(df.columns)}")

    plt.figure(figsize=figsize)

    # Get the top N value counts
    top_values = df[column].value_counts().nlargest(top_n)

    # Create a horizontal bar plot
    sns.barplot(
        y=top_values.index,
        x=top_values.values,
        palette=palette,
        legend=False
    )

    # Format the column name for display (e.g. Property_Type → Property Type)
    display_name = column.replace("_", " ")

    # Add labels and title
    plt.xlabel("Count", fontsize=12)
    plt.ylabel(display_name, fontsize=12)
    plt.title(f"Top {top_n} — {display_name}", fontsize=14, fontweight="bold")

    # Add the count values to the end of each bar (offset = 2% of max)
    offset = top_values.max() * 0.02
    for index, value in enumerate(top_values.values):
        plt.text(value + offset, index, f"{value:,}", va='center', fontsize=10)

    # Remove top and right borders
    sns.despine()
    plt.tight_layout()
    plt.show()



# =============================================================================
# BIVARIATE ANALYSIS
# =============================================================================
# TODO: Add functions for:
#   - Scatter plots (e.g., Price vs. Rating)
#   - Box plots (e.g., Price by Property_Type)
#   - Correlation heatmap for numeric columns
#   - Cross-tabulations for categorical pairs


# =============================================================================
# MULTIVARIATE ANALYSIS
# =============================================================================
# TODO: Add functions for:
#   - Pair plots for key numeric variables
#   - Grouped analysis (e.g., Price by Theme × Property_Type)
#   - Parallel coordinates or radar charts for amenity profiles
