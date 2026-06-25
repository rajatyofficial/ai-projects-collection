"""
Analysis & Visualization Module for Airbnb Listings Deep Dive
==============================================================
Provides:
  1. prepare_analysis_df() — drops redundant raw columns, reorders the
     remaining columns into logical groups, and returns a clean DataFrame
     ready for graphing and statistical analysis.

  2. save_analysis_df() / load_analysis_df() — persist the cleaned DataFrame
     to a .parquet file so the full pipeline doesn't need to re-run every time.

  3. Univariate, bivariate, and multivariate analysis functions that operate
     on the cleaned DataFrame returned by prepare_analysis_df() (e.g.,
     univariate_bar_plot() for visualizing categorical column distributions).

Usage
-----
    from analysis import load_analysis_df

    # First time — run the full pipeline and save:
    from data_loader import load_data
    from feature_extraction import extract_all_features
    from data_cleaning import clean_all_columns
    from analysis import prepare_analysis_df, save_analysis_df, univariate_bar_plot

    df = load_data()
    df = extract_all_features(df)
    df = clean_all_columns(df)
    df = prepare_analysis_df(df)
    save_analysis_df(df)

    # Subsequent times — just load directly:
    df = load_analysis_df()   # reads from ../data/cleaned_and_processed_data.parquet
"""

import os
import numpy as np
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

    # Print summary statistics
    print("=" * 60)
    print(f"Summary Statistics for: {column}")
    print("=" * 60)
    print(df[column].describe())
    print("=" * 60)
    
    plt.figure(figsize=figsize)

    # Get the top N value counts
    top_values = df[column].value_counts().nlargest(top_n)

    # Create a horizontal bar plot
    sns.barplot(
        y=top_values.index,
        x=top_values.values,
        hue=top_values.index,
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


def univariate_pie_plot(df: pd.DataFrame, column: str, 
                        palette: str = "viridis", figsize: tuple = (8, 8)) -> None:
    """
    Plot the proportion of True vs False values of a boolean column as a pie chart.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    column : str
        Name of the boolean column to plot.
    palette : str, optional
        Seaborn color palette name. Defaults to "viridis".
    figsize : tuple, optional
        Figure size as (width, height). Defaults to (8, 8).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available: {list(df.columns)}")

    # Print summary statistics
    print("=" * 60)
    print(f"Summary Statistics for: {column}")
    print("=" * 60)
    print(df[column].describe())
    print("=" * 60)

    plt.figure(figsize=figsize)
    
    # Get value counts
    counts = df[column].value_counts()
    
    # Check if it has more than 2 values (in case a non-boolean is passed)
    if len(counts) > 2:
        print(f"Warning: '{column}' has more than 2 unique values. This pie chart might be cluttered.")
    
    # Extract colors from the seaborn palette
    colors = sns.color_palette(palette, len(counts))
    
    # Format the column name for display
    display_name = column.replace("_", " ")

    # Create the pie plot
    plt.pie(
        counts.values,
        labels=[str(idx) for idx in counts.index],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Add title
    plt.title(f"Proportion of {display_name}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.show()

def univariate_violin_plot(df: pd.DataFrame, column: str, 
                           color: str = "skyblue", figsize: tuple = (10, 6)) -> None:
    """
    Print summary statistics and plot the distribution of a numerical column using a violin plot.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    column : str
        Name of the numerical column to plot.
    color : str, optional
        Color for the violin plot. Defaults to "skyblue".
    figsize : tuple, optional
        Figure size as (width, height). Defaults to (10, 6).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available: {list(df.columns)}")

    # Print summary statistics
    print("=" * 60)
    print(f"Summary Statistics for: {column}")
    print("=" * 60)
    print(df[column].describe())
    print("=" * 60)

    # Plot
    plt.figure(figsize=figsize)
    display_name = column.replace("_", " ")

    # Create the violin plot
    sns.violinplot(x=df[column], color=color, inner="quartile")
    
    # Add labels and title
    plt.xlabel(display_name, fontsize=12)
    plt.title(f"Distribution of {display_name}", fontsize=14, fontweight="bold")
    
    # Remove top and right borders
    sns.despine()
    plt.tight_layout()
    plt.show()



# =============================================================================
# BIVARIATE ANALYSIS
# =============================================================================


# ── 1. Numerical vs. Numerical ──────────────────────────────────────────────

def bivariate_scatter(df: pd.DataFrame, x: str, y: str, *,
                      hue: str | None = None,
                      add_regression: bool = True,
                      palette: str = "viridis",
                      alpha: float = 0.6,
                      figsize: tuple = (12, 8)) -> None:
    """
    Scatter plot of two numerical columns with an optional regression line.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    x, y : str
        Column names for the horizontal and vertical axes.
    hue : str or None, optional
        Categorical column to colour-code the points by (e.g. "Theme").
    add_regression : bool, default True
        If True, overlay a linear-regression trend line via seaborn's
        regplot (drawn on the full dataset, ignoring *hue* groups).
    palette : str, optional
        Seaborn colour palette.  Defaults to "viridis".
    alpha : float, optional
        Point transparency (0 = invisible, 1 = opaque).  Defaults to 0.6.
    figsize : tuple, optional
        Figure size as (width, height).  Defaults to (12, 8).
    """
    for col in (x, y):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # --- Print quick summary ---
    print("=" * 60)
    print(f"Scatter — {x} vs {y}")
    print("=" * 60)
    subset = df[[x, y]].dropna()
    print(f"  Non-null pairs : {len(subset)}")
    corr = subset[x].corr(subset[y])
    print(f"  Pearson r      : {corr:.4f}")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=figsize)

    # Main scatter
    sns.scatterplot(
        data=df, x=x, y=y, hue=hue,
        palette=palette, alpha=alpha, edgecolor="w", linewidth=0.3, ax=ax
    )

    # Optional regression overlay (full data, ignoring hue)
    if add_regression:
        sns.regplot(
            data=df, x=x, y=y,
            scatter=False, color="crimson",
            line_kws={"linewidth": 2, "linestyle": "--", "label": f"r = {corr:.3f}"},
            ax=ax
        )
        ax.legend(loc="best")

    display_x = x.replace("_", " ")
    display_y = y.replace("_", " ")
    ax.set_xlabel(display_x, fontsize=12)
    ax.set_ylabel(display_y, fontsize=12)
    ax.set_title(f"{display_y} vs {display_x}", fontsize=14, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.show()


def correlation_heatmap(df: pd.DataFrame, *,
                        columns: list[str] | None = None,
                        method: str = "pearson",
                        cmap: str = "coolwarm",
                        annot_fontsize: int = 9,
                        figsize: tuple = (14, 10)) -> pd.DataFrame:
    """
    Plot a correlation heatmap for numeric columns and return the matrix.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    columns : list[str] or None, optional
        Specific numeric columns to include.  If None, every numeric
        column in the DataFrame is used.
    method : str, optional
        Correlation method — "pearson", "spearman", or "kendall".
        Defaults to "pearson".
    cmap : str, optional
        Matplotlib colourmap.  Defaults to "coolwarm".
    annot_fontsize : int, optional
        Font size for the annotation text inside each cell.
    figsize : tuple, optional
        Figure size.  Defaults to (14, 10).

    Returns
    -------
    pd.DataFrame
        The computed correlation matrix.
    """
    if columns is None:
        num_df = df.select_dtypes(include="number")
    else:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        num_df = df[columns].select_dtypes(include="number")

    if num_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for a correlation heatmap.")

    corr = num_df.corr(method=method)

    # --- Print notable correlations ---
    print("=" * 60)
    print(f"Correlation Heatmap  ({method.title()})")
    print("=" * 60)
    # Upper-triangle pairs sorted by absolute correlation
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = (
        corr.where(mask_upper)
            .stack()
            .rename("corr")
            .reset_index()
            .rename(columns={"level_0": "Var1", "level_1": "Var2"})
    )
    pairs["abs_corr"] = pairs["corr"].abs()
    pairs = pairs.sort_values("abs_corr", ascending=False)
    print("  Top 5 strongest correlations:")
    for _, row in pairs.head(5).iterrows():
        print(f"    {row['Var1']:<20s} × {row['Var2']:<20s}  r = {row['corr']:+.4f}")
    print("=" * 60)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f", annot_kws={"size": annot_fontsize},
        linewidths=0.5, square=True, cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title(f"{method.title()} Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return corr


# ── 2. Categorical vs. Numerical ────────────────────────────────────────────

def bivariate_boxplot(df: pd.DataFrame, cat_col: str, num_col: str, *,
                      top_n: int = 10,
                      order_by_median: bool = True,
                      palette: str = "viridis",
                      figsize: tuple = (12, 8)) -> None:
    """
    Box plot of a numerical column split by the top-N groups of a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    cat_col : str
        The categorical (grouping) column.
    num_col : str
        The numerical column whose distribution is compared.
    top_n : int, optional
        Number of most-frequent categories to include.  Defaults to 10.
    order_by_median : bool, default True
        If True, categories are sorted by descending median of *num_col*.
    palette : str, optional
        Seaborn colour palette.  Defaults to "viridis".
    figsize : tuple, optional
        Figure size.  Defaults to (12, 8).
    """
    for col in (cat_col, num_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Keep only the top_n most frequent categories
    top_cats = df[cat_col].value_counts().nlargest(top_n).index
    plot_df = df[df[cat_col].isin(top_cats)].dropna(subset=[num_col])

    # Determine order
    if order_by_median:
        order = (
            plot_df.groupby(cat_col)[num_col]
                   .median()
                   .sort_values(ascending=False)
                   .index
        )
    else:
        order = top_cats

    # --- Print group statistics ---
    print("=" * 60)
    print(f"Box Plot — {num_col} by {cat_col}  (top {top_n})")
    print("=" * 60)
    stats = (
        plot_df.groupby(cat_col)[num_col]
               .agg(["count", "median", "mean", "std"])
               .reindex(order)
    )
    print(stats.to_string())
    print("=" * 60)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=plot_df, x=num_col, y=cat_col,
        order=order, hue=cat_col,
        palette=palette, legend=False,
        flierprops={"marker": "o", "markerfacecolor": "grey", "markersize": 4, "alpha": 0.5},
        ax=ax
    )

    display_cat = cat_col.replace("_", " ")
    display_num = num_col.replace("_", " ")
    ax.set_xlabel(display_num, fontsize=12)
    ax.set_ylabel(display_cat, fontsize=12)
    ax.set_title(f"{display_num} by {display_cat}", fontsize=14, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.show()


def bivariate_bar_mean(df: pd.DataFrame, cat_col: str, num_col: str, *,
                       top_n: int = 10,
                       estimator: str = "mean",
                       palette: str = "viridis",
                       figsize: tuple = (12, 8)) -> None:
    """
    Horizontal bar chart showing the mean (or median) of a numerical column
    for each category, with 95% confidence-interval error bars.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    cat_col : str
        The categorical (grouping) column.
    num_col : str
        The numerical column to aggregate.
    top_n : int, optional
        Number of most-frequent categories to include.  Defaults to 10.
    estimator : str, optional
        "mean" or "median".  Defaults to "mean".
    palette : str, optional
        Seaborn colour palette.  Defaults to "viridis".
    figsize : tuple, optional
        Figure size.  Defaults to (12, 8).
    """
    for col in (cat_col, num_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    est_func = np.median if estimator == "median" else np.mean

    top_cats = df[cat_col].value_counts().nlargest(top_n).index
    plot_df = df[df[cat_col].isin(top_cats)].dropna(subset=[num_col])

    # Sort categories by the chosen estimator
    order = (
        plot_df.groupby(cat_col)[num_col]
               .agg(est_func)
               .sort_values(ascending=False)
               .index
    )

    # --- Print aggregated values ---
    print("=" * 60)
    print(f"Bar Chart — {estimator.title()} {num_col} by {cat_col}  (top {top_n})")
    print("=" * 60)
    agg = (
        plot_df.groupby(cat_col)[num_col]
               .agg(["count", "mean", "median"])
               .reindex(order)
    )
    print(agg.to_string())
    print("=" * 60)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=plot_df, x=num_col, y=cat_col,
        order=order, hue=cat_col,
        estimator=est_func, errorbar=("ci", 95),
        palette=palette, legend=False, ax=ax
    )

    display_cat = cat_col.replace("_", " ")
    display_num = num_col.replace("_", " ")
    ax.set_xlabel(f"{estimator.title()} {display_num}", fontsize=12)
    ax.set_ylabel(display_cat, fontsize=12)
    ax.set_title(
        f"{estimator.title()} {display_num} by {display_cat}",
        fontsize=14, fontweight="bold"
    )
    sns.despine()
    plt.tight_layout()
    plt.show()


# ── 3. Boolean Flag vs. Numerical ───────────────────────────────────────────

def bivariate_flag_comparison(df: pd.DataFrame, flag_col: str, num_col: str, *,
                              palette: list[str] | None = None,
                              figsize: tuple = (12, 6)) -> None:
    """
    Compare the distribution of a numerical column between True/False
    groups of a boolean flag using side-by-side KDE + box plots.

    Useful for answering questions like "Do properties with a pool
    charge more?".

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    flag_col : str
        Boolean column (e.g. "Has_Pool", "Has_Hot_Tub").
    num_col : str
        Numerical column to compare (e.g. "Price").
    palette : list[str] or None, optional
        Two colours for [False, True].  Defaults to ["#95a5a6", "#e74c3c"].
    figsize : tuple, optional
        Figure size.  Defaults to (12, 6).
    """
    for col in (flag_col, num_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    if palette is None:
        palette = ["#95a5a6", "#e74c3c"]  # grey for False, red for True

    plot_df = df[[flag_col, num_col]].dropna()

    group_false = plot_df.loc[plot_df[flag_col] == False, num_col]
    group_true  = plot_df.loc[plot_df[flag_col] == True,  num_col]

    # --- Print comparison stats ---
    print("=" * 60)
    print(f"Flag Comparison — {num_col} by {flag_col}")
    print("=" * 60)
    print(f"  {flag_col} = False  (n={len(group_false)})")
    print(f"    Mean   : {group_false.mean():.2f}")
    print(f"    Median : {group_false.median():.2f}")
    print(f"    Std    : {group_false.std():.2f}")
    print()
    print(f"  {flag_col} = True   (n={len(group_true)})")
    print(f"    Mean   : {group_true.mean():.2f}")
    print(f"    Median : {group_true.median():.2f}")
    print(f"    Std    : {group_true.std():.2f}")
    diff_pct = (
        (group_true.median() - group_false.median()) / group_false.median() * 100
        if group_false.median() != 0 else float("nan")
    )
    print(f"\n  Median difference : {group_true.median() - group_false.median():+.2f}  ({diff_pct:+.1f}%)")
    print("=" * 60)

    # --- Two-panel figure: KDE on the left, box plot on the right ---
    fig, (ax_kde, ax_box) = plt.subplots(1, 2, figsize=figsize)

    display_flag = flag_col.replace("_", " ")
    display_num  = num_col.replace("_", " ")

    # Left — KDE
    sns.kdeplot(group_false, color=palette[0], fill=True, alpha=0.35,
                label="False", ax=ax_kde)
    sns.kdeplot(group_true,  color=palette[1], fill=True, alpha=0.35,
                label="True",  ax=ax_kde)
    ax_kde.set_xlabel(display_num, fontsize=12)
    ax_kde.set_title(f"{display_num} — KDE by {display_flag}", fontsize=13, fontweight="bold")
    ax_kde.legend(title=display_flag)
    sns.despine(ax=ax_kde)

    # Right — Box plot
    sns.boxplot(
        data=plot_df, x=num_col, y=flag_col,
        hue=flag_col, palette={False: palette[0], True: palette[1]},
        legend=False, ax=ax_box
    )
    ax_box.set_xlabel(display_num, fontsize=12)
    ax_box.set_ylabel(display_flag, fontsize=12)
    ax_box.set_title(f"{display_num} — Box by {display_flag}", fontsize=13, fontweight="bold")
    sns.despine(ax=ax_box)

    plt.tight_layout()
    plt.show()


# ── 4. Categorical vs. Categorical ──────────────────────────────────────────

def categorical_crosstab_heatmap(df: pd.DataFrame, row_col: str, col_col: str, *,
                                  top_n_row: int = 10,
                                  top_n_col: int = 10,
                                  normalize: str = "row",
                                  cmap: str = "YlOrRd",
                                  figsize: tuple = (14, 8)) -> pd.DataFrame:
    """
    Heatmap of a cross-tabulation between two categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    row_col, col_col : str
        Categorical columns for the rows and columns of the cross-tab.
    top_n_row, top_n_col : int, optional
        Number of most-frequent categories to include for each axis.
        Defaults to 10 each.
    normalize : str, optional
        Normalisation axis — "row" (percentages across each row sum to 100),
        "col" (each column sums to 100), or "all" (grand total = 100).
        Pass None for raw counts.  Defaults to "row".
    cmap : str, optional
        Matplotlib colourmap.  Defaults to "YlOrRd".
    figsize : tuple, optional
        Figure size.  Defaults to (14, 8).

    Returns
    -------
    pd.DataFrame
        The computed cross-tabulation table (normalised or raw counts).
    """
    for col in (row_col, col_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Filter to top-N categories on each axis
    top_rows = df[row_col].value_counts().nlargest(top_n_row).index
    top_cols = df[col_col].value_counts().nlargest(top_n_col).index
    plot_df = df[df[row_col].isin(top_rows) & df[col_col].isin(top_cols)]

    # Build cross-tab
    ct_raw = pd.crosstab(plot_df[row_col], plot_df[col_col])

    # Normalise
    norm_map = {"row": "index", "col": "columns", "all": "all"}
    if normalize in norm_map:
        ct = pd.crosstab(
            plot_df[row_col], plot_df[col_col],
            normalize=norm_map[normalize]
        ) * 100  # express as percentages
        fmt = ".1f"
        label_suffix = " (%)"
    else:
        ct = ct_raw
        fmt = "d"
        label_suffix = " (count)"

    # --- Print summary ---
    print("=" * 60)
    print(f"Cross-tab — {row_col} × {col_col}{label_suffix}")
    print("=" * 60)
    print(ct.to_string())
    print("=" * 60)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        ct, annot=True, fmt=fmt, cmap=cmap,
        linewidths=0.5, cbar_kws={"label": f"{'Percentage' if normalize else 'Count'}"},
        ax=ax
    )

    display_row = row_col.replace("_", " ")
    display_col = col_col.replace("_", " ")
    ax.set_xlabel(display_col, fontsize=12)
    ax.set_ylabel(display_row, fontsize=12)
    norm_label = f" (normalised by {normalize})" if normalize else ""
    ax.set_title(
        f"{display_row} × {display_col}{norm_label}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    return ct


def bivariate_stacked_bar(df: pd.DataFrame, primary_col: str, secondary_col: str, *,
                          top_n_primary: int = 10,
                          top_n_secondary: int = 6,
                          palette: str = "Set2",
                          figsize: tuple = (14, 8)) -> None:
    """
    100% stacked horizontal bar chart showing the composition of a
    secondary categorical column within each primary category.

    Useful for questions like "What is the Bed_Type breakdown within
    each Theme?" or "What themes are most common per Property_Type?".

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    primary_col : str
        The grouping axis (plotted on the y-axis).
    secondary_col : str
        The composition variable (shown as stacked segments).
    top_n_primary : int, optional
        Number of most-frequent primary categories.  Defaults to 10.
    top_n_secondary : int, optional
        Number of most-frequent secondary categories; remaining values
        are collapsed into an "Other" segment.  Defaults to 6.
    palette : str, optional
        Seaborn colour palette.  Defaults to "Set2".
    figsize : tuple, optional
        Figure size.  Defaults to (14, 8).
    """
    for col in (primary_col, secondary_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Filter primary categories
    top_primary = df[primary_col].value_counts().nlargest(top_n_primary).index
    plot_df = df[df[primary_col].isin(top_primary)].copy()

    # Collapse rare secondary categories into "Other"
    top_secondary = plot_df[secondary_col].value_counts().nlargest(top_n_secondary).index
    plot_df["_sec"] = plot_df[secondary_col].where(
        plot_df[secondary_col].isin(top_secondary), other="Other"
    )

    # Build a normalised cross-tab (rows sum to 100%)
    ct = pd.crosstab(plot_df[primary_col], plot_df["_sec"], normalize="index") * 100

    # Sort rows by total listing count (most frequent first)
    row_order = (
        plot_df[primary_col].value_counts()
                            .reindex(ct.index)
                            .sort_values(ascending=True)
                            .index
    )
    ct = ct.reindex(row_order)

    # --- Print summary ---
    print("=" * 60)
    print(f"Stacked Bar — {secondary_col} composition within {primary_col}  (top {top_n_primary})")
    print("=" * 60)
    print(ct.round(1).to_string())
    print("=" * 60)

    # --- Plot ---
    colors = sns.color_palette(palette, ct.shape[1])
    fig, ax = plt.subplots(figsize=figsize)
    ct.plot.barh(stacked=True, color=colors, edgecolor="white", linewidth=0.5, ax=ax)

    display_primary   = primary_col.replace("_", " ")
    display_secondary = secondary_col.replace("_", " ")
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel(display_primary, fontsize=12)
    ax.set_title(
        f"{display_secondary} Composition by {display_primary}",
        fontsize=14, fontweight="bold"
    )
    ax.legend(
        title=display_secondary, bbox_to_anchor=(1.02, 1),
        loc="upper left", fontsize=9
    )
    sns.despine()
    plt.tight_layout()
    plt.show()


# =============================================================================
# MULTIVARIATE ANALYSIS
# =============================================================================
# TODO: Add functions for:
#   - Pair plots for key numeric variables
#   - Grouped analysis (e.g., Price by Theme × Property_Type)
#   - Parallel coordinates or radar charts for amenity profiles
