import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# MULTIVARIATE ANALYSIS
# =============================================================================

# Global default figure sizes (adjust these to change all plots at once)
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_LARGE = (12, 8)
FIGSIZE_WIDE = (14, 8)
FIGSIZE_SQUARE = (10, 10)


# ── 1. Pivot Heatmap  (Cat × Cat → Num) ─────────────────────────────────────

def multivariate_pivot_heatmap(df: pd.DataFrame,
                                row_col: str,
                                col_col: str,
                                value_col: str, *,
                                aggfunc: str = "median",
                                top_n_row: int = 8,
                                top_n_col: int = 8,
                                cmap: str = "YlGnBu",
                                fmt: str = ".1f",
                                figsize: tuple = FIGSIZE_WIDE) -> pd.DataFrame:
    """
    Heatmap where rows and columns are categorical variables, and cell
    colours represent an aggregated numerical value (e.g. median Price).

    Answers questions like: "What is the median Price for each
    Country × Theme combination?"

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    row_col : str
        Categorical column for the rows (e.g. "Country").
    col_col : str
        Categorical column for the columns (e.g. "Theme").
    value_col : str
        Numerical column to aggregate (e.g. "Price").
    aggfunc : str, optional
        Aggregation function — "median", "mean", or "count".
        Defaults to "median".
    top_n_row, top_n_col : int, optional
        Number of most-frequent categories to include for each axis.
        Defaults to 8.
    cmap : str, optional
        Matplotlib colourmap.  Defaults to "YlGnBu".
    fmt : str, optional
        Number format for annotations.  Defaults to ".1f".
    figsize : tuple, optional
        Figure size.  Defaults to FIGSIZE_WIDE.

    Returns
    -------
    pd.DataFrame
        The computed pivot table.
    """
    for col in (row_col, col_col, value_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Filter to top-N categories on each axis
    top_rows = df[row_col].value_counts().nlargest(top_n_row).index
    top_cols = df[col_col].value_counts().nlargest(top_n_col).index
    plot_df = df[df[row_col].isin(top_rows) & df[col_col].isin(top_cols)].dropna(subset=[value_col])

    # Build pivot table
    pivot = plot_df.pivot_table(
        index=row_col, columns=col_col, values=value_col,
        aggfunc=aggfunc
    )

    # --- Print summary ---
    print("=" * 60)
    print(f"Pivot Heatmap — {aggfunc.title()} {value_col} by {row_col} × {col_col}")
    print("=" * 60)
    print(pivot.round(2).to_string())
    print(f"\n  Grand {aggfunc} : {plot_df[value_col].agg(aggfunc):.2f}")
    print(f"  Overall range  : {pivot.min().min():.2f} → {pivot.max().max():.2f}")
    print("=" * 60)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot, annot=True, fmt=fmt, cmap=cmap,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": f"{aggfunc.title()} {value_col}"},
        ax=ax
    )

    display_row = row_col.replace("_", " ")
    display_col = col_col.replace("_", " ")
    display_val = value_col.replace("_", " ")
    ax.set_xlabel(display_col, fontsize=12)
    ax.set_ylabel(display_row, fontsize=12)
    ax.set_title(
        f"{aggfunc.title()} {display_val} — {display_row} × {display_col}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    return pivot


# ── 2. Grouped Box Plot  (Cat × Bool → Num) ─────────────────────────────────

def multivariate_grouped_boxplot(df: pd.DataFrame,
                                  cat_col: str,
                                  hue_col: str,
                                  num_col: str, *,
                                  top_n: int = 8,
                                  palette: dict | str | None = None,
                                  figsize: tuple = FIGSIZE_LARGE) -> None:
    """
    Side-by-side box plots of a numerical column, grouped by a categorical
    column and split by a secondary hue variable (typically boolean).

    Answers questions like: "Does having a Pool add the same price
    premium to a Villa as it does to an Apartment?"

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    cat_col : str
        Primary categorical grouping column (y-axis, e.g. "Property_Type").
    hue_col : str
        Secondary split variable (e.g. "Has_Pool", "Is_Near_Beach",
        or another categorical like "Bed_Type").
    num_col : str
        Numerical column whose distribution is compared (e.g. "Price").
    top_n : int, optional
        Number of most-frequent primary categories.  Defaults to 8.
    palette : dict, str, or None, optional
        Colour palette for the hue variable.  If None, uses a sensible
        default (grey/red for boolean, "Set2" for categorical).
    figsize : tuple, optional
        Figure size.  Defaults to FIGSIZE_LARGE.
    """
    for col in (cat_col, hue_col, num_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Filter to top_n categories
    top_cats = df[cat_col].value_counts().nlargest(top_n).index
    plot_df = df[df[cat_col].isin(top_cats)].dropna(subset=[num_col])

    # Order by median of num_col
    order = (
        plot_df.groupby(cat_col)[num_col]
               .median()
               .sort_values(ascending=False)
               .index
    )

    # Auto-detect palette
    if palette is None:
        if plot_df[hue_col].dtype == bool or set(plot_df[hue_col].dropna().unique()) <= {True, False}:
            palette = {False: "#95a5a6", True: "#e74c3c"}
        else:
            palette = "Set2"

    # --- Print grouped statistics ---
    print("=" * 60)
    print(f"Grouped Box Plot — {num_col} by {cat_col}, split by {hue_col}")
    print("=" * 60)
    stats = (
        plot_df.groupby([cat_col, hue_col])[num_col]
               .agg(["count", "median", "mean"])
               .round(2)
    )
    print(stats.to_string())
    print("=" * 60)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=plot_df, x=num_col, y=cat_col, hue=hue_col,
        order=order, palette=palette,
        flierprops={"marker": "o", "markerfacecolor": "grey",
                    "markersize": 3, "alpha": 0.4},
        ax=ax
    )

    display_cat = cat_col.replace("_", " ")
    display_hue = hue_col.replace("_", " ")
    display_num = num_col.replace("_", " ")
    ax.set_xlabel(display_num, fontsize=12)
    ax.set_ylabel(display_cat, fontsize=12)
    ax.set_title(
        f"{display_num} by {display_cat} — split by {display_hue}",
        fontsize=14, fontweight="bold"
    )
    ax.legend(title=display_hue, loc="lower right")
    sns.despine()
    plt.tight_layout()
    plt.show()


# ── 3. Facet Scatter Grid  (Num × Num, faceted by Cat) ──────────────────────

def multivariate_facet_scatter(df: pd.DataFrame,
                                x: str,
                                y: str,
                                facet_col: str, *,
                                hue: str | None = None,
                                top_n_facets: int = 6,
                                col_wrap: int = 3,
                                add_regression: bool = True,
                                palette: str = "viridis",
                                alpha: float = 0.5,
                                height: float = 4,
                                aspect: float = 1.2) -> None:
    """
    Grid of scatter plots — one panel per category — showing how the
    relationship between two numerical columns changes across groups.

    Answers questions like: "Is the Price vs Rating relationship
    fundamentally different for Cabins vs Condos vs Villas?"

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    x, y : str
        Numerical columns for scatter axes.
    facet_col : str
        Categorical column to facet by (one panel per value).
    hue : str or None, optional
        Additional categorical variable for colour-coding within each panel.
    top_n_facets : int, optional
        Number of most-frequent categories to facet on.  Defaults to 6.
    col_wrap : int, optional
        Number of columns in the grid.  Defaults to 3.
    add_regression : bool, default True
        If True, overlay a per-panel regression line.
    palette : str, optional
        Seaborn colour palette for hue.  Defaults to "viridis".
    alpha : float, optional
        Point transparency.  Defaults to 0.5.
    height, aspect : float, optional
        Height of each facet and aspect ratio.  Defaults to 4 and 1.2.
    """
    for col in (x, y, facet_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Filter to top facets
    top_facets = df[facet_col].value_counts().nlargest(top_n_facets).index
    plot_df = df[df[facet_col].isin(top_facets)].dropna(subset=[x, y])

    # --- Print per-facet correlations ---
    print("=" * 60)
    print(f"Facet Scatter — {y} vs {x}, faceted by {facet_col}")
    print("=" * 60)
    for facet_val in top_facets:
        subset = plot_df[plot_df[facet_col] == facet_val]
        r = subset[x].corr(subset[y])
        print(f"  {facet_val:<20s}  n = {len(subset):>4d}   Pearson r = {r:+.4f}")
    print("=" * 60)

    # --- Plot ---
    g = sns.FacetGrid(
        plot_df, col=facet_col, col_wrap=col_wrap,
        height=height, aspect=aspect,
        sharex=True, sharey=True
    )

    g.map_dataframe(
        sns.scatterplot, x=x, y=y, hue=hue,
        palette=palette if hue else None,
        alpha=alpha, edgecolor="w", linewidth=0.3
    )

    if add_regression:
        g.map_dataframe(
            sns.regplot, x=x, y=y,
            scatter=False, color="crimson",
            line_kws={"linewidth": 1.5, "linestyle": "--"}
        )

    display_x = x.replace("_", " ")
    display_y = y.replace("_", " ")
    display_facet = facet_col.replace("_", " ")
    g.set_axis_labels(display_x, display_y, fontsize=11)
    g.set_titles(col_template=f"{{col_name}}", size=12, fontweight="bold")
    g.figure.suptitle(
        f"{display_y} vs {display_x} — by {display_facet}",
        fontsize=14, fontweight="bold", y=1.02
    )

    if hue:
        g.add_legend(title=hue.replace("_", " "))

    plt.tight_layout()
    plt.show()


# ── 4. Pairplot  (Multi-Num coloured by Cat) ────────────────────────────────

def multivariate_pairplot(df: pd.DataFrame,
                           num_cols: list[str],
                           hue: str, *,
                           top_n_hue: int = 4,
                           palette: str = "Set1",
                           alpha: float = 0.5,
                           diag_kind: str = "kde",
                           height: float = 2.5,
                           corner: bool = True) -> None:
    """
    Pairwise scatter matrix of several numerical columns, colour-coded
    by a categorical variable.

    Reveals whether different market segments (e.g. Theme, Property_Type)
    form visually distinct clusters in the multi-dimensional feature space.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    num_cols : list[str]
        List of numerical columns to include (typically 3–6).
    hue : str
        Categorical column to colour-code by (e.g. "Theme").
    top_n_hue : int, optional
        Number of most-frequent hue categories.  Defaults to 4.
    palette : str, optional
        Seaborn colour palette.  Defaults to "Set1".
    alpha : float, optional
        Point transparency.  Defaults to 0.5.
    diag_kind : str, optional
        Diagonal plot type — "kde" or "hist".  Defaults to "kde".
    height : float, optional
        Height of each facet cell.  Defaults to 2.5.
    corner : bool, default True
        If True, only shows the lower triangle (avoids redundancy).
    """
    for col in num_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
    if hue not in df.columns:
        raise ValueError(f"Column '{hue}' not found. Available: {list(df.columns)}")

    # Filter to top hue categories
    top_hue_vals = df[hue].value_counts().nlargest(top_n_hue).index
    plot_df = df[df[hue].isin(top_hue_vals)].dropna(subset=num_cols)

    # --- Print per-group summary ---
    print("=" * 60)
    print(f"Pairplot — {', '.join(num_cols)} coloured by {hue}")
    print("=" * 60)
    for hue_val in top_hue_vals:
        sub = plot_df[plot_df[hue] == hue_val]
        print(f"\n  {hue} = {hue_val}  (n = {len(sub)})")
        print(sub[num_cols].describe().loc[["mean", "50%", "std"]].round(2).to_string())
    print("\n" + "=" * 60)

    # --- Plot ---
    g = sns.pairplot(
        plot_df, vars=num_cols, hue=hue,
        palette=palette, diag_kind=diag_kind,
        plot_kws={"alpha": alpha, "edgecolor": "w", "linewidth": 0.2, "s": 30},
        diag_kws={"alpha": 0.4, "linewidth": 1.5},
        height=height, corner=corner
    )

    display_hue = hue.replace("_", " ")
    g.figure.suptitle(
        f"Pairplot — coloured by {display_hue}",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.show()


# ── 5. Amenity Radar / Profile Chart  (Bool flags across Cat groups) ────────

def multivariate_amenity_profile(df: pd.DataFrame,
                                  group_col: str,
                                  flag_cols: list[str], *,
                                  top_n_groups: int = 6,
                                  palette: str = "Set2",
                                  figsize: tuple = FIGSIZE_SQUARE) -> pd.DataFrame:
    """
    Radar (spider) chart comparing the prevalence of boolean amenity
    flags across different groups of a categorical column.

    Answers questions like: "What % of Luxury vs Nature vs Beach
    listings have a Pool / Hot Tub / Waterfront?"

    Parameters
    ----------
    df : pd.DataFrame
        The analysis-ready DataFrame.
    group_col : str
        Categorical column to compare (e.g. "Theme", "Property_Type").
    flag_cols : list[str]
        Boolean columns to profile (e.g. ["Has_Pool", "Has_Hot_Tub", ...]).
    top_n_groups : int, optional
        Number of most-frequent groups.  Defaults to 6.
    palette : str, optional
        Seaborn colour palette.  Defaults to "Set2".
    figsize : tuple, optional
        Figure size.  Defaults to FIGSIZE_SQUARE.

    Returns
    -------
    pd.DataFrame
        The prevalence table (percentage of True for each group × flag).
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found. Available: {list(df.columns)}")
    missing_flags = [f for f in flag_cols if f not in df.columns]
    if missing_flags:
        raise ValueError(f"Flag columns not found: {missing_flags}")

    # Filter to top groups
    top_groups = df[group_col].value_counts().nlargest(top_n_groups).index
    plot_df = df[df[group_col].isin(top_groups)]

    # Compute prevalence (% True) for each group × flag
    prevalence = (
        plot_df.groupby(group_col)[flag_cols]
               .mean() * 100  # convert to percentage
    )
    prevalence = prevalence.reindex(top_groups)  # keep frequency order

    # --- Print summary ---
    print("=" * 60)
    print(f"Amenity Profile — {group_col} × [{', '.join(flag_cols)}]")
    print("=" * 60)
    print(prevalence.round(1).to_string())
    print("=" * 60)

    # --- Radar chart ---
    categories = [c.replace("_", " ") for c in flag_cols]
    N = len(categories)

    # Compute angles for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors = sns.color_palette(palette, len(top_groups))

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for idx, (group_name, row) in enumerate(prevalence.iterrows()):
        values = row.values.tolist()
        values += values[:1]  # close the polygon
        ax.plot(angles, values, "o-", linewidth=2, label=str(group_name),
                color=colors[idx], markersize=5)
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Prevalence (%)", fontsize=10, labelpad=20)

    display_group = group_col.replace("_", " ")
    ax.set_title(
        f"Amenity Profile by {display_group}",
        fontsize=14, fontweight="bold", y=1.08
    )
    ax.legend(
        title=display_group, loc="upper right",
        bbox_to_anchor=(1.3, 1.1), fontsize=9
    )

    plt.tight_layout()
    plt.show()

    return prevalence
