"""
Feature Extraction Module for Airbnb Listings Deep Dive
========================================================
Extracts structured features from the unstructured 'Title' and 'Detail' columns
using keyword matching, regex patterns, and theme scoring.
"""

import re
import pandas as pd
from typing import Optional


# =============================================================================
# TITLE COLUMN EXTRACTORS
# =============================================================================

def extract_property_type(title: str) -> str:
    """Extract property type from Title (e.g., 'Cabin', 'Treehouse', 'Home')."""
    if not isinstance(title, str):
        return "Unknown"
    parts = title.split(" in ")
    if len(parts) >= 2:
        return parts[0].strip()
    return "Unknown"


def extract_city(title: str) -> str:
    """Extract city name from Title."""
    if not isinstance(title, str):
        return "Unknown"
    parts = title.split(" in ")
    if len(parts) >= 2:
        location = parts[1]
        city = location.split(",")[0].strip()
        return city if city else "Unknown"
    return "Unknown"


def extract_state(title: str) -> str:
    """Extract state/province from Title (available when title has 3+ comma-separated parts)."""
    if not isinstance(title, str):
        return "Unknown"
    parts = [p.strip() for p in title.split(",")]
    if len(parts) >= 3:
        return parts[-2]  # second-to-last is usually the state
    return "Unknown"


def extract_country(title: str) -> str:
    """Extract country from Title (last comma-separated part)."""
    if not isinstance(title, str):
        return "Unknown"
    parts = [p.strip() for p in title.split(",")]
    if len(parts) >= 2:
        return parts[-1]
    return "Unknown"


# =============================================================================
# AMENITY EXTRACTORS (Boolean flags from Detail column)
# =============================================================================

def _safe_search(text: str, pattern: str) -> bool:
    """Safely search for a regex pattern in text, handling non-string values."""
    if not isinstance(text, str):
        return False
    return bool(re.search(pattern, text, re.IGNORECASE))


def has_hot_tub(detail: str) -> bool:
    """Check if listing mentions a hot tub or jacuzzi."""
    return _safe_search(detail, r'hot\s*tub|jacuzzi')


def has_sauna(detail: str) -> bool:
    """Check if listing mentions a sauna."""
    return _safe_search(detail, r'sauna')


def has_pool(detail: str) -> bool:
    """Check if listing mentions a pool (excludes 'pool table')."""
    if not isinstance(detail, str):
        return False
    # Match 'pool' but exclude 'pool table'
    has_pool_mention = bool(re.search(r'pool', detail, re.IGNORECASE))
    is_pool_table = bool(re.search(r'pool\s*table', detail, re.IGNORECASE))
    return has_pool_mention and not is_pool_table


def has_bbq(detail: str) -> bool:
    """Check if listing mentions BBQ, grill, or barbecue."""
    return _safe_search(detail, r'bbq|grill|barbecue')


def has_fireplace(detail: str) -> bool:
    """Check if listing mentions a fireplace or fire pit."""
    return _safe_search(detail, r'fireplace|fire\s*place|fire\s*pit')


def has_parking(detail: str) -> bool:
    """Check if listing mentions parking or garage."""
    return _safe_search(detail, r'parking|garage')


def has_spa(detail: str) -> bool:
    """Check if listing mentions a spa."""
    return _safe_search(detail, r'\bspa\b')


# =============================================================================
# LOCATION FEATURE EXTRACTORS (Boolean flags from Detail column)
# =============================================================================

def has_waterfront(detail: str) -> bool:
    """Check if listing is on a waterfront (lake, ocean, sea, river)."""
    return _safe_search(detail, r'waterfront|lakefront|oceanfront|beachfront|seafront|riverfront')


def is_near_beach(detail: str) -> bool:
    """Check if listing is near or on a beach."""
    return _safe_search(detail, r'beach|ocean|surf|seaside|coast|shore')


def is_mountain(detail: str) -> bool:
    """Check if listing is in a mountain/highland area."""
    return _safe_search(detail, r'mountain|hilltop|highland|summit|alpine|ridge')


def is_rural(detail: str) -> bool:
    """Check if listing is in a rural or secluded area."""
    return _safe_search(detail, r'rural|farm|countryside|wilderness|secluded|off[\s-]*grid|remote|retreat')


def is_urban(detail: str) -> bool:
    """Check if listing is in an urban setting."""
    return _safe_search(detail, r'downtown|city\s+center|urban|metro|highrise|high[\s-]*rise')


# =============================================================================
# PROPERTY SUB-TYPE EXTRACTOR (from Detail column)
# =============================================================================

# Ordered from most specific to least specific to avoid mis-classification
_PROPERTY_SUBTYPES = [
    ("Treehouse", r'treehouse|tree\s*house'),
    ("A-Frame", r'a[\s-]*frame'),
    ("Dome", r'\bdome\b'),
    ("Yurt", r'\byurt\b'),
    ("Hut", r'\bhut\b'),
    ("Tiny Home", r'tiny\s*(home|house)'),
    ("Villa", r'\bvilla\b'),
    ("Cottage", r'\bcottage\b'),
    ("Cabin", r'\bcabin\b'),
    ("Suite", r'\bsuite\b'),
    ("Studio", r'\bstudio\b'),
    ("Loft", r'\bloft\b'),
    ("Bungalow", r'\bbungalow\b'),
    ("Penthouse", r'\bpenthouse\b'),
]


def extract_property_subtype(detail: str) -> str:
    """Extract a more specific property sub-type from the Detail column."""
    if not isinstance(detail, str):
        return "Other"
    for subtype, pattern in _PROPERTY_SUBTYPES:
        if re.search(pattern, detail, re.IGNORECASE):
            return subtype
    return "Other"


# =============================================================================
# THEME CLASSIFIER (Keyword scoring approach)
# =============================================================================

_THEME_KEYWORDS = {
    "Luxury": [
        (r'luxury|luxurious', 3),
        (r'deluxe|premium|executive', 2),
        (r'designer|elegant|upscale|resort|marble', 2),
        (r'private\s+pool|infinity\s+pool', 2),
        (r'penthouse|mansion|estate', 2),
    ],
    "Nature": [
        (r'treehouse|tree\s*house', 3),
        (r'cabin|wilderness|forest', 2),
        (r'eco|sustainable|off[\s-]*grid', 2),
        (r'dome|yurt|hut', 2),
        (r'mountain|creek|river|lake', 1),
        (r'nature|garden|botanical', 1),
    ],
    "Romantic": [
        (r'romantic', 3),
        (r'couples?\s*(getaway|retreat|escape)', 3),
        (r'honeymoon', 3),
        (r'cozy|intimate|charming', 1),
        (r'candle|sunset|wine', 1),
    ],
    "Adventure": [
        (r'a[\s-]*frame', 2),
        (r'tiny\s*(home|house)', 2),
        (r'off[\s-]*grid|secluded|remote', 2),
        (r'adventure|explore|hiking|trail', 2),
        (r'camping|glamping|tent', 2),
        (r'surf|kayak|dive|snorkel', 2),
    ],
    "Beach": [
        (r'beachfront|beach\s*front', 3),
        (r'beach|ocean|sea\b|seaside', 2),
        (r'oceanfront|waterfront|seafront', 3),
        (r'surf|coastal|shore|sand\b', 1),
        (r'island|tropical|palm', 1),
    ],
    "Family": [
        (r'family', 3),
        (r'kids|children|spacious', 2),
        (r'game\s*room|playground|yard', 2),
        (r'multiple\s*beds|bunk', 1),
    ],
    "Historic": [
        (r'historic|historical', 3),
        (r'colonial|heritage|traditional', 2),
        (r'antique|vintage|classic|century', 1),
        (r'castle|manor|chateau', 2),
    ],
    "Urban": [
        (r'studio\b', 1),
        (r'downtown|city\s*center', 2),
        (r'loft|highrise|high[\s-]*rise', 2),
        (r'metro|subway|transit', 1),
        (r'nightlife|restaurant|shopping', 1),
    ],
}


def classify_theme(detail: str) -> str:
    """
    Classify a listing into a theme using weighted keyword scoring.
    Returns the theme with the highest score, or 'General' if no keywords match.
    """
    if not isinstance(detail, str):
        return "General"

    scores = {}
    for theme, keyword_list in _THEME_KEYWORDS.items():
        score = 0
        for pattern, weight in keyword_list:
            if re.search(pattern, detail, re.IGNORECASE):
                score += weight
        if score > 0:
            scores[theme] = score

    if not scores:
        return "General"

    # Return the theme with the highest score
    return max(scores, key=scores.get)


# =============================================================================
# TEXT METRICS
# =============================================================================

def get_word_count(detail: str) -> int:
    """Get the word count of the detail text."""
    if not isinstance(detail, str):
        return 0
    return len(detail.split())


def get_char_count(detail: str) -> int:
    """Get the character count of the detail text."""
    if not isinstance(detail, str):
        return 0
    return len(detail)


def has_promo(detail: str) -> bool:
    """Check if listing mentions a promotional discount."""
    return _safe_search(detail, r'promo|discount|\d+%\s*off|\bsale\b|special\s*offer')


def extract_bedroom_count(detail: str) -> Optional[int]:
    """
    Try to extract number of bedrooms from detail text.
    Looks for patterns like '2BR', '3 bedroom', '2 bed room', '1 br'.
    Returns None if not found.
    """
    if not isinstance(detail, str):
        return None
    match = re.search(r'(\d+)\s*(?:br|bed\s*room|bedroom)', detail, re.IGNORECASE)
    if match:
        count = int(match.group(1))
        # Sanity check: bedrooms usually between 1-20
        if 1 <= count <= 20:
            return count
    return None


# =============================================================================
# MASTER FUNCTION: Apply all extractors to a DataFrame
# =============================================================================

def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ALL feature extraction functions to the DataFrame.
    Expects 'Title' and 'Detail' columns to exist.
    Returns the DataFrame with new columns added.
    """
    df = df.copy()

    # --- From Title column ---
    df["Property_Type"] = df["Title"].apply(extract_property_type)
    df["City"] = df["Title"].apply(extract_city)
    df["State"] = df["Title"].apply(extract_state)
    df["Country"] = df["Title"].apply(extract_country)

    # --- Amenities (from Detail column) ---
    df["Has_Hot_Tub"] = df["Detail"].apply(has_hot_tub)
    df["Has_Sauna"] = df["Detail"].apply(has_sauna)
    df["Has_Pool"] = df["Detail"].apply(has_pool)
    df["Has_BBQ"] = df["Detail"].apply(has_bbq)
    df["Has_Fireplace"] = df["Detail"].apply(has_fireplace)
    df["Has_Parking"] = df["Detail"].apply(has_parking)
    df["Has_Spa"] = df["Detail"].apply(has_spa)

    # --- Location Features (from Detail column) ---
    df["Has_Waterfront"] = df["Detail"].apply(has_waterfront)
    df["Is_Near_Beach"] = df["Detail"].apply(is_near_beach)
    df["Is_Mountain"] = df["Detail"].apply(is_mountain)
    df["Is_Rural"] = df["Detail"].apply(is_rural)
    df["Is_Urban"] = df["Detail"].apply(is_urban)

    # --- Property Sub-type (from Detail column) ---
    df["Property_Subtype"] = df["Detail"].apply(extract_property_subtype)

    # --- Theme Classification (from Detail column) ---
    df["Theme"] = df["Detail"].apply(classify_theme)

    # --- Text Metrics ---
    df["Detail_Word_Count"] = df["Detail"].apply(get_word_count)
    df["Detail_Char_Count"] = df["Detail"].apply(get_char_count)
    df["Has_Promo"] = df["Detail"].apply(has_promo)
    df["Bedroom_Count"] = df["Detail"].apply(extract_bedroom_count)

    return df
