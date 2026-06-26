import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# UNIVARIATE ANALYSIS
# =============================================================================

# Global default figure sizes (adjust these to change all plots at once)
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_LARGE = (12, 8)
FIGSIZE_SQUARE = (8, 8)
# TODO: Add functions for:
#   - Distribution plots (histograms, KDE) for numeric columns
#   - Count plots / bar charts for categorical columns
#   - Boolean flag frequency summary

def univariate_bar_plot(df: pd.DataFrame, column: str, top_n: int = 10,
                        palette: str = "viridis", figsize: tuple = FIGSIZE_STANDARD) -> None:
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
        Figure size as (width, height). Defaults to FIGSIZE_STANDARD.
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
                        palette: str = "viridis", figsize: tuple = FIGSIZE_SQUARE) -> None:
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
        Figure size as (width, height). Defaults to FIGSIZE_SQUARE.
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
                           color: str = "skyblue", figsize: tuple = FIGSIZE_STANDARD) -> None:
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
        Figure size as (width, height). Defaults to FIGSIZE_STANDARD.
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
