# 🏙️ Tokyo Rental Market Intelligence & Price Forecasting

A full end-to-end data analytics project covering web scraping, SQL feature engineering, exploratory data analysis, and predictive modeling — applied to Tokyo's rental housing market. Built to support practical housing decisions for incoming field staff relocating to the Nakai area.

> **v2 update:** This project now includes an automated production pipeline (`pipeline/`) that runs on a weekly cron schedule — scraping fresh listings, loading to Supabase (PostgreSQL), and regenerating an interactive HTML dashboard. See [v2 Pipeline](#v2-pipeline) for setup.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [v2 Pipeline](#v2-pipeline)
- [Business Context & Impact](#business-context--impact)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Key Metrics & Findings](#key-metrics--findings)
- [Market Overview](#market-overview)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Limitations & Future Work](#limitations--future-work)
- [Repository Structure](#repository-structure)

---

## Project Overview

This project scrapes, cleans, engineers, and models Tokyo rental listings from *SUUMO.jp* — Japan's largest real estate platform — with a focus on the Nakai neighborhood and surrounding stations. The pipeline collects ~5,000 listings, processes them through a multi-stage SQL view, and feeds a multivariate linear regression model capable of explaining **88% of rental price variance** on unseen data.

The project is structured as four sequential notebooks plus a companion Excel workbook (`TokyoRentalMarketOverview.xlsx`) containing a full listing table, station/floor plan market summary tables, and pivot chart data:

| Notebook | Description |
|---|---|
| `01_data_collection_and_cleaning` | Web scraping with `BeautifulSoup`, SQLite ingestion, SQL view with feature engineering |
| `02_exploratory_data_analysis` | Distribution analysis, Pearson correlations, confidence intervals, 8 EDA visualizations |
| `03_modeling_and_evaluation` | Linear regression (simple + multiple), one-hot encoding, cross-validation, residual analysis |
| `04_insights_limitations_and_conclusion` | Business-facing summary of findings, limitations, and next steps |

---

## v2 Pipeline

Built on top of the original analysis, the v2 pipeline automates the full data lifecycle on a weekly cron schedule.

### How it works

```
SUUMO.jp
    │
    ▼
pipeline/test_scraper.py        ← scrapes all listings with edge case handling
    ├── Loads raw data → Supabase (housing_data_raw table)
    ├── Creates SQL view → tokyo_housing (cleans + engineers features)
    └── Exports → tokyo_housing.csv
    │
    ▼
pipeline/generate_dashboard.py  ← detects CSV changes via mtime comparison
    └── Generates → reports/tokyo_rental_dashboard.html
```

### Interactive Dashboard

`reports/tokyo_rental_dashboard.html` is a self-contained HTML file — no server needed, open it in any browser.

- 4,600+ listings from West Tokyo
- Filters: floor plan, area (binned), station, building age
- KPI cards: total listings, median rent, mean rent, median area, median building age
- 6 charts: floor plan distribution, avg rent by floor plan, rent by area, building age distribution, top stations by rent, rent vs walk time
- Paginated listings table (20 per page) with clickable links to SUUMO listings

### Setup

**Requirements:**
```bash
pip install requests beautifulsoup4 lxml pandas numpy psycopg2-binary sqlalchemy
```

**Credentials** — create a `.env` file in the project root (never committed):
```
SUPABASE_HOST=db.xxxxxxxxxxxx.supabase.co
SUPABASE_PORT=5432
SUPABASE_DB=postgres
SUPABASE_USER=postgres
SUPABASE_PASSWORD=your-password-here
```

**Run the scraper:**
```bash
python3 pipeline/test_scraper.py          # full run with Supabase
python3 pipeline/test_scraper.py --skip-db  # CSV only, no database
```

**Regenerate the dashboard:**
```bash
python3 pipeline/generate_dashboard.py
```

**Recommended cron schedule** (scrape Sunday 5am, regenerate 6am):
```bash
0 5 * * 0 python3 /path/to/pipeline/test_scraper.py >> scraper.log 2>&1
0 6 * * 0 python3 /path/to/pipeline/generate_dashboard.py >> dashboard.log 2>&1
```

### v2 improvements over v1

| Area | v1 | v2 |
|------|----|-----|
| Storage | SQLite (local) | Supabase / PostgreSQL |
| Scraper | Basic pagination | Rate limiting, User-Agent, safe error handling |
| SQL | SQLite view | PostgreSQL-compatible view with safe CAST |
| Scheduling | Manual | Cron-ready with mtime-based change detection |
| Output | CSV + notebooks | CSV + interactive HTML dashboard |

---

## Business Context & Impact

Field staff relocating to Tokyo face a fragmented, Japanese-language rental market with limited visibility into what drives pricing. This project addresses that gap directly:

- **Establishes a market baseline** — average rent of ¥129,149/month with a 95% CI of ±¥1,941, giving staff a statistically grounded budget expectation
- **Quantifies what matters** — identifies `area` as the dominant pricing driver (Pearson r = 0.881), freeing staff from over-weighting factors like station proximity that have minimal real-world impact
- **Delivers a rent estimation tool** — the regression model predicts rent from key unit characteristics, supporting informed negotiation and housing prioritization before arrival
- **Surfaces non-obvious insights** — station proximity and specific station choice showed near-zero correlation with rent (r ≈ 0.09 and 0.00 respectively), countering a common assumption in Tokyo housing search
- **Provides station- and floor plan-level market tables** — the companion Excel workbook enables direct comparison of median rents, building ages, and unit types across 20+ stations and 15+ floor plan categories

---

## Tech Stack

| Category | Tools |
|---|---|
| **Data Collection** | Python, `requests`, `BeautifulSoup`, `re` |
| **Storage** | SQLite, SQL Magic (`%sql`) |
| **Data Processing** | `pandas`, `numpy` |
| **Feature Engineering** | Raw SQL (CTEs, window functions, `PARTITION BY`) |
| **EDA & Visualization** | `matplotlib`, `seaborn` |
| **Modeling** | `scikit-learn` — `LinearRegression`, `StandardScaler`, `OneHotEncoder`, `cross_val_score` |
| **Statistics** | `scipy.stats` — Pearson correlation, z-test, confidence intervals |
| **Reporting** | Excel (pivot tables, summary tables, chart data) |

---

## Project Architecture

```
SUUMO.jp (HTML)
      │
      ▼
TokyoHousingScraper (src/housing_scraper.py)
  ├── scrape_listings()          # Paginated HTML collection
  ├── parse_station_info()       # Station name + distance extraction (regex)
  ├── parse_sublistings()        # Unit-level metrics (rent, floor, area, etc.)
  └── build_housing_dataset()    # Structured DataFrame → SQLite (HOUSING_DATA_RAW)
      │
      ▼
SQL View: TOKYO_HOUSING (sql/data_cleaning_and_features.sql)
  ├── DEDUPLICATED_LISTINGS      # ROW_NUMBER() deduplication
  ├── STANDARDIZED_LISTINGS      # Type casting, unit normalization, floor plan mapping
  └── FEATURED_LISTINGS          # Window functions: avg_rent_by_station,
                                 #   avg_rent_by_floor_plan, count_listings_per_station
      │
      ├──► reports/TokyoRentalMarketOverview.xlsx
      │      ├── HOUSING_TABLE          # Full cleaned listing export
      │      ├── RENTAL_MARKET_OVERVIEW # Station + floor plan summary tables
      │      └── CHARTS                 # Pivot data: avg rent by building size & age
      │
      ▼
EDA → notebooks/02_exploratory_data_analysis.ipynb
      │
      ▼
Linear Regression → notebooks/03_modeling_and_evaluation.ipynb
  ├── Features: area, building_age, building_size, floor, floor_plan (OHE)
  ├── 5-fold cross-validation
  └── Train/test evaluation
```

---

## Key Metrics & Findings

### Rental Price Distribution
- **Mean rent:** ¥129,149/month
- **95% confidence interval:** ¥129,149 ± ¥1,941
- **Distribution shape:** Right-skewed — most listings fall below the mean, with a long tail of premium units pulling the average upward
- **Sample size:** 4,772 listings (post-deduplication)

### Feature Correlations with Rent

| Feature | Pearson r | Interpretation |
|---|---|---|
| `area` | **+0.881** | Strongest driver — larger units command significantly higher rents |
| `building_age` | **−0.400** | Newer buildings are meaningfully more expensive |
| `building_size` | **+0.340** | Taller buildings tend to price higher |
| `floor` | **+0.290** | Higher floors correlate with higher rents |
| `distance_to_nearest_station` | **+0.090** | Negligible — station proximity is not a meaningful price driver |
| `avg_distance_to_stations` | **+0.003** | No meaningful relationship with rent |

**Notable finding:** Station proximity — often cited as a top factor in Tokyo housing — showed near-zero correlation with rent in this dataset. Rent appears to be driven almost entirely by unit characteristics, not location within the Nakai area.

---

## Market Overview

Full station- and floor plan-level summary tables are available in [`reports/TokyoRentalMarketOverview.xlsx`](reports/TokyoRentalMarketOverview.xlsx). Key highlights below.

**By station** (top stations by listing volume, 100+ listings):

- **中井駅 (Nakai)** — the primary target area — has 335 listings at a median rent of ¥91,000 with a 5-minute walk and 20-year median building age; one of the most affordable well-connected stations in the dataset
- **沼袋駅 (Numabukuro)** offers the lowest median rent (¥75,000) among high-volume stations, though stock skews older (21-year median age) and smaller (1R dominant)
- **高田馬場駅 (Takadanobaba)** commands the highest median rent at ¥148,000 — newer stock, larger unit mix
- **新江古田駅 (Shin-Egota)** stands out for its newest median building age (1 year) at a moderate ¥115,500 median rent — a potential value option for staff prioritizing newer construction
- Despite a ¥75,000–¥148,000 spread across stations, the correlation between station and rent is near zero — unit characteristics dominate

**By floor plan:**

- **1K and 1R** account for 2,697 listings — over half the dataset — making them the most available options for single-person staff at median rents of ¥90,000 and ¥70,000 respectively
- The step from **1K → 1LDK** costs roughly ¥74,500/month more (+83%) for ~17 additional m² — a significant cost-per-area premium for adding a living room
- **2LDK and 3LDK** serve couples and families at median rents of ¥238,000 and ¥283,000, but supply is limited and price variability is high (IQR: ¥61,500–¥94,550)

**By building age:**

- Newly constructed buildings average **¥182,680** — more than 2.5× the ¥72,107 average for 61+ year-old buildings
- The steepest discount occurs in the first 30 years; after that, the curve flattens significantly

---

## Model Performance

### Simple Linear Regression (Area Only — Baseline)

| Metric | Value |
|---|---|
| 5-fold CV R² (avg) | **0.779** |

### Multiple Linear Regression (All Features)

| Metric | Value |
|---|---|
| 5-fold CV R² (avg) | **0.889** |
| Training set R² | **0.893** |
| Test set R² | **0.880** |

The jump from 77.9% to 88.9% R² demonstrates meaningful lift from adding `building_age`, `building_size`, `floor`, and `floor_plan` alongside `area`. The close alignment between training (89.3%) and test (88.0%) performance confirms the model generalizes well without overfitting.

### Top Coefficients (Standardized)

| Feature | Coefficient | Direction |
|---|---|---|
| `area` | 39,955 | ↑ Strongest positive driver |
| `building_age` | −16,628 | ↓ Older buildings significantly cheaper |
| `floor_plan_2LDK` | 10,428 | ↑ Large premium for 2LDK layouts |
| `floor_plan_3LDK` | 9,856 | ↑ Similar premium for 3LDK |
| `building_size` | 8,827 | ↑ Taller buildings price higher |
| `floor` | 2,751 | ↑ Higher floors modestly increase rent |
| `floor_plan_1K` | −3,930 | ↓ Discount relative to 1DK baseline |

The building age and size effects visible in the market summary tables are independently confirmed by the regression coefficients, and the floor plan premiums align with the median rent differentials observed in the data.

---

## Visualizations

### Distribution of Rental Prices
![rent_dist](figures/rent_dist.png)
> Right-skewed distribution with a mean of ¥129,149. The median sits below the mean, pulled upward by a small number of premium listings — a useful reminder that average rent overstates typical cost for most staff.

---

### Correlation of Rent with Floor Area
![rent_vs_area](figures/rent_vs_area.png)
> Pearson r = 0.881 — the strongest predictor in the dataset by a wide margin. The fan-shaped spread at larger areas reflects increasing price variability for bigger units, but the linear trend is clear throughout.

---

### Correlation of Rent with Building Age
![rent_vs_age](figures/rent_vs_age.png)
> Pearson r = −0.400. Newer buildings are consistently more expensive, with the steepest discount occurring in the first 30 years. Staff prioritizing budget over modernity have the most options in the 20–40 year age range.

---

### Distribution of Residuals
![residuals](figures/residuals.png)
> Residuals are centered near zero with an approximately symmetric, bell-shaped profile — consistent with linear regression assumptions. Extreme outliers on the left tail correspond to high-rent listings where the model under-predicts, suggesting that a budget-capped dataset would further improve accuracy for practical use.

---

## Limitations & Future Work

**Current limitations:**

- **Partial market coverage** — SUUMO's pagination structure prevented exhaustive scraping; ~5,000 listings represent a sample, not the full market
- **Static snapshot** — data reflects a single point in time; SUUMO updates listings frequently, limiting temporal validity
- **Assumed linearity** — scatter plots reveal some nonlinear patterns (particularly for `area` and `building_age`) that a linear model cannot fully capture

**Planned improvements:**

- Automated recurring scraping (daily/weekly) to track listing availability and price movement over time
- Polynomial feature transformations (`PolynomialFeatures`) with regularization (Ridge/Lasso) to better capture nonlinear relationships
- Budget-filtered modeling — removing listings above a practical rent ceiling to improve prediction accuracy for the target use case
- Station-level composition analysis (e.g., avg `building_age` and `floor_plan` mix per station) to explain the apparent price parity across locations

---

## Repository Structure

```
tokyo-housing-analysis/
├── notebooks/                              # v1 — original analysis
│   ├── 01_data_collection_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_modeling_and_evaluation.ipynb
│   └── 04_insights_limitations_and_conclusion.ipynb
├── src/                                    # v1 — original scraper
│   ├── housing_scraper.py
│   └── housing_scraper.ipynb
├── sql/
│   └── data_cleaning_and_features.sql      # Feature engineering view (SQLite)
├── pipeline/                               # v2 — automated pipeline
│   ├── test_scraper.py                     # Scrape → Supabase → CSV
│   └── generate_dashboard.py              # CSV → HTML dashboard
├── data/
│   ├── processed/
│   │   ├── coef_table.csv
│   │   ├── model_residuals.csv
│   │   └── tokyo_housing.csv
│   └── raw/
│       └── housing_data_raw.csv
├── figures/
│   ├── rent_dist.png
│   ├── rent_vs_area.png
│   ├── rent_vs_age.png
│   ├── residuals.png
│   └── ...
├── reports/
│   ├── tokyo_rental_dashboard.html         # v2 — interactive dashboard
│   ├── TokyoRentalMarketOverview.xlsx
│   └── ...
└── README.md
```

---

*Data sourced from SUUMO.jp. All analysis performed on listings in the Nakai neighborhood and surrounding area of Tokyo, Japan.*
