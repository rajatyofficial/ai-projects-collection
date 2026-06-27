# 🏡 Airbnb Listings Deep Dive

An end-to-end exploratory data analysis project on **953 Airbnb listings** from around the globe. Starting from just 7 raw string columns, this project constructs a rich, 42-column analytical dataset through automated feature extraction, data cleaning, and multi-stage EDA.

---

## 📌 Project Overview

| Attribute          | Details                                      |
|--------------------|----------------------------------------------|
| **Dataset**        | `airnb.csv` — 953 Airbnb listings            |
| **Original Cols**  | 7 (all raw strings)                          |
| **Final Cols**     | 42 (after feature extraction & cleaning)     |
| **Analysis Stages**| 4 (Exploration → Univariate → Bivariate → Multivariate) |
| **Coverage**       | Global — US, Indonesia, Thailand, Canada, Mexico, and more |

---

## 📁 Project Structure

- **`data/`** — Raw and processed datasets
  - `airnb.csv` — Raw dataset (953 listings, 7 original string columns)
  - `cleaned_and_processed_data.parquet` — Final cleaned & feature-enriched dataset (42 columns)

- **`Notebooks/`** — Step-by-step Jupyter notebooks for each analysis phase
  - `01_Initial_Exploration_and_Data_Cleanup.ipynb` — Schema audit, feature extraction, type cleaning
  - `02_Univariate_Analysis.ipynb` — Per-column distributions and summary stats
  - `03_Bivariate_Analysis.ipynb` — Two-variable relationships and correlations
  - `04_Multivariate_Analysis.ipynb` — Complex cross-feature patterns and insights
  - **`scripts/`** — Reusable Python helper modules
    - `data_loader.py` — CSV loading utilities
    - `feature_extraction.py` — Regex-based feature engineering (7 → 29 columns)
    - `data_cleaning.py` — Type conversion and derived column generation (29 → 42 columns)
    - `univariate_analysis.py` — Single-variable analysis helpers
    - `bivariate_analysis.py` — Two-variable analysis helpers
    - `multivariate_analysis.py` — Multi-variable analysis helpers
    - `utils.py` — Shared utility functions

- **`archive/`** — Supporting documents
  - `Notes and Observations.txt` — Incremental analyst notes, decisions, and key findings

- **`Airbnb Listing Data Analysis Report.pdf`** — Final summarized report of all findings

---

## 🗂️ Dataset Description

The raw dataset (`airnb.csv`) contains **953 listings** across **7 string columns**: `Title`, `Detail`, `Date`, `Price(in dollar)`, `Offer price(in dollar)`, `Review and rating`, and `Number of bed`. Through automated feature extraction and cleaning, these expand to **42 analysis-ready columns**.

**Notable data quality issues:**
- `Offer price` is 82.6% missing (only 166 of 953 listings have a discount price)
- 34 duplicate rows (3.57%) were identified and removed
- 251 listings have `Unknown` as country due to inconsistent international title formatting

---

## ⚙️ Feature Engineering Pipeline

The raw 7-column dataset expands to **42 columns** through a 3-stage pipeline:

```
Raw Load (7 cols)
    ↓
feature_extraction.py  →  +22 columns  =  29 total
    ↓
data_cleaning.py       →  +13 columns  =  42 total
```


---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn pyarrow
```

### Run the Notebooks

Open the notebooks in order for a complete walkthrough:

```bash
# From the project root
jupyter notebook Notebooks/01_Initial_Exploration_and_Data_Cleanup.ipynb
```

Or jump directly to a specific analysis stage:

```bash
jupyter notebook Notebooks/02_Univariate_Analysis.ipynb
jupyter notebook Notebooks/03_Bivariate_Analysis.ipynb
jupyter notebook Notebooks/04_Multivariate_Analysis.ipynb
```

### Load the Cleaned Dataset Directly

```python
import pandas as pd

df = pd.read_parquet("data/cleaned_and_processed_data.parquet")
print(df.shape)   # (919, 42) — after duplicate removal
print(df.columns.tolist())
```

---

## 📄 Report

A summarized PDF report of findings is available at:
📎 `Airbnb Listing Data Analysis Report.pdf`

---

## 📝 Notes & Observations

Detailed incremental analyst notes are maintained in:
📎 `archive/Notes and Observations.txt`

This file documents every design decision, data quality issue, and key statistical insight encountered throughout the project.

---

## 🛠️ Tech Stack

| Tool         | Purpose                          |
|--------------|----------------------------------|
| Python 3.x   | Core language                    |
| Pandas       | Data manipulation                |
| NumPy        | Numerical operations             |
| Matplotlib   | Plotting                         |
| Seaborn      | Statistical visualizations       |
| PyArrow      | Parquet file I/O                 |
| Jupyter      | Interactive notebooks            |
| Regex (`re`) | Feature extraction & parsing     |

---

## 📂 Status

> 🟡 **In Progress** — Multivariate analysis stage ongoing.
> Last Updated: June 2026
