"""
Data Cleaning & Preprocessing Module
=====================================
Handles parsing and cleaning the raw Airbnb columns:
- Price(in dollar) → numeric float
- Offer price(in dollar) → numeric float
- Review and rating → separate Rating (float) and Review_Count (int)
- Number of bed → separate Bed_Count (int) and Bed_Type (str)
- Date → Start_Date, End_Date, Duration_Nights
"""

import re
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from typing import Optional


# =============================================================================
# PRICE CLEANING
# =============================================================================

def clean_price(price_str: str) -> Optional[float]:
    """
    Convert a price string like '306.00' to a float.
    Handles: currency symbols, commas, whitespace, non-numeric values.
    
    Returns None if the value cannot be parsed.
    """
    if not isinstance(price_str, str):
        return None
    # Remove currency symbols, commas, whitespace
    cleaned = re.sub(r'[^\d.]', '', price_str.strip())
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


# =============================================================================
# REVIEW & RATING PARSING
# =============================================================================

def extract_rating(review_str: str) -> Optional[float]:
    """
    Extract the numeric rating from a string like '4.85 (531)' or 'New'.
    Returns None if the string is 'New' or unparseable.
    """
    if not isinstance(review_str, str):
        return None
    review_str = review_str.strip()
    if review_str.lower() == 'new':
        return None
    match = re.match(r'([\d.]+)', review_str)
    if match:
        try:
            rating = float(match.group(1))
            # Sanity check: Airbnb ratings are between 0 and 5
            if 0 <= rating <= 5:
                return rating
        except ValueError:
            pass
    return None


def extract_review_count(review_str: str) -> Optional[int]:
    """
    Extract the review count from a string like '4.85 (531)' or 'New'.
    Returns 0 for 'New' listings, None for unparseable values.
    """
    if not isinstance(review_str, str):
        return None
    review_str = review_str.strip()
    if review_str.lower() == 'new':
        return 0
    match = re.search(r'\((\d+)\)', review_str)
    if match:
        return int(match.group(1))
    return None


# =============================================================================
# BED COUNT PARSING
# =============================================================================

def extract_bed_count(bed_str: str) -> Optional[int]:
    """
    Extract the numeric bed count from strings like:
    - '4 beds'
    - '1 queen bed'
    - '1 king bed'
    - '2 single beds'
    
    Returns None if unparseable.
    """
    if not isinstance(bed_str, str):
        return None
    match = re.search(r'(\d+)', bed_str.strip())
    if match:
        count = int(match.group(1))
        if 0 < count <= 50:  # sanity check
            return count
    return None


def extract_bed_type(bed_str: str) -> str:
    """
    Extract the bed type from strings like '1 queen bed', '4 beds'.
    Returns 'Standard' for generic '4 beds', otherwise the specific type.
    """
    if not isinstance(bed_str, str):
        return "Unknown"
    bed_str = bed_str.strip().lower()

    # Check for specific bed types
    bed_types = ['king', 'queen', 'single', 'double', 'twin', 'bunk', 'sofa', 'futon']
    for bt in bed_types:
        if bt in bed_str:
            return bt.capitalize()

    # Generic "X beds" → Standard
    if re.match(r'\d+\s*beds?$', bed_str):
        return "Standard"

    return "Other"


# =============================================================================
# DATE PARSING
# =============================================================================

_MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}


def extract_start_month(date_str: str) -> Optional[str]:
    """
    Extract the month name from a date range string like 'Jun 11 - 16'.
    Returns the full month name (e.g., 'June').
    """
    if not isinstance(date_str, str):
        return None
    match = re.match(r'([A-Za-z]+)', date_str.strip())
    if match:
        abbr = match.group(1).lower()[:3]
        month_num = _MONTH_MAP.get(abbr)
        if month_num:
            import calendar
            return calendar.month_name[month_num]
    return None


def extract_duration_nights(date_str: str) -> Optional[int]:
    """
    Extract the trip duration in nights from a date range like 'Jun 11 - 16'.
    Returns the number of nights (e.g., 5).
    """
    if not isinstance(date_str, str):
        return None
    match = re.search(r'(\d+)\s*-\s*(\d+)', date_str)
    if match:
        start_day = int(match.group(1))
        end_day = int(match.group(2))
        duration = end_day - start_day
        if duration > 0:
            return duration
        # Handle month boundary crossover (e.g., "Jan 29 - 3")
        if duration < 0:
            return None  # ambiguous without knowing the month
    return None


def extract_start_day(date_str: str) -> Optional[int]:
    """Extract the start day number from 'Jun 11 - 16' → 11."""
    if not isinstance(date_str, str):
        return None
    match = re.search(r'(\d+)\s*-\s*\d+', date_str)
    if match:
        return int(match.group(1))
    return None


def extract_end_day(date_str: str) -> Optional[int]:
    """Extract the end day number from 'Jun 11 - 16' → 16."""
    if not isinstance(date_str, str):
        return None
    match = re.search(r'\d+\s*-\s*(\d+)', date_str)
    if match:
        return int(match.group(1))
    return None


# =============================================================================
# DISCOUNT CALCULATION
# =============================================================================

def calculate_discount_pct(price: float, offer_price: float) -> Optional[float]:
    """
    Calculate the discount percentage between original price and offer price.
    Returns None if either value is missing or invalid.
    """
    if pd.isna(price) or pd.isna(offer_price) or price <= 0:
        return None
    discount = ((price - offer_price) / price) * 100
    return round(discount, 2)


# =============================================================================
# MASTER FUNCTION: Clean all columns
# =============================================================================

def clean_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ALL cleaning and parsing functions to the DataFrame.
    Converts raw string columns into properly typed, analysis-ready columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with original columns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with new cleaned columns added.
    """
    df = df.copy()

    # --- Price columns → float ---
    df["Price"] = df["Price(in dollar)"].apply(clean_price)
    df["Offer_Price"] = df["Offer price(in dollar)"].apply(clean_price)

    # --- Discount percentage ---
    df["Discount_Pct"] = df.apply(
        lambda row: calculate_discount_pct(row["Price"], row["Offer_Price"]),
        axis=1
    )

    # --- Review and rating → separate columns ---
    df["Rating"] = df["Review and rating"].apply(extract_rating)
    df["Review_Count"] = df["Review and rating"].apply(extract_review_count)
    df["Is_New_Listing"] = df["Review and rating"].apply(
        lambda x: isinstance(x, str) and x.strip().lower() == 'new'
    )

    # --- Number of bed → separate columns ---
    df["Bed_Count"] = df["Number of bed"].apply(extract_bed_count)
    df["Bed_Type"] = df["Number of bed"].apply(extract_bed_type)

    # --- Date → parsed components ---
    df["Month"] = df["Date"].apply(extract_start_month)
    df["Start_Day"] = df["Date"].apply(extract_start_day)
    df["End_Day"] = df["Date"].apply(extract_end_day)
    df["Duration_Nights"] = df["Date"].apply(extract_duration_nights)

    # --- Price per night ---
    df["Price_Per_Night"] = df.apply(
        lambda row: round(row["Price"] / row["Duration_Nights"], 2)
        if pd.notna(row["Price"]) and pd.notna(row["Duration_Nights"]) and row["Duration_Nights"] > 0
        else None,
        axis=1
    )

    print(f"[OK] Data cleaning complete: {df.shape[1]} columns total")
    return df
