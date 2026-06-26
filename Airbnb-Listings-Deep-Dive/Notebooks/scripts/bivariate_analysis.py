import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# BIVARIATE ANALYSIS
# =============================================================================

# Global default figure sizes (adjust these to change all plots at once)
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_LARGE = (12, 8)
FIGSIZE_WIDE = (10, 5)


# ── 1. Numerical vs. Numerical ──────────────────────────────────────────────

def bivariate_scatter(df: pd.DataFrame, x: str, y: str, *,
                      hue: str | None = None,
                      add_regression: bool = True,
                      palette: str = "viridis",
                      alpha: float = 0.6,
                      figsize: tuple = FIGSIZE_STANDARD) -> None:
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
        Figure size as (width, height).  Defaults to FIGSIZE_STANDARD.
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
                        figsize: tuple = FIGSIZE_LARGE) -> pd.DataFrame:
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
        Figure size.  Defaults to FIGSIZE_LARGE.

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
                      figsize: tuple = FIGSIZE_STANDARD) -> None:
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
        Figure size.  Defaults to FIGSIZE_STANDARD.
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
                       figsize: tuple = FIGSIZE_STANDARD) -> None:
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
        Figure size.  Defaults to FIGSIZE_STANDARD.
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
                              figsize: tuple = FIGSIZE_WIDE) -> None:
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
        Figure size.  Defaults to FIGSIZE_WIDE.
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
                                  figsize: tuple = FIGSIZE_LARGE) -> pd.DataFrame:
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
        Figure size.  Defaults to FIGSIZE_LARGE.

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
                          figsize: tuple = FIGSIZE_LARGE) -> None:
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
        Figure size.  Defaults to FIGSIZE_LARGE.
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