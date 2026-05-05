#!/usr/bin/env python3
"""
Scrapes SUUMO listings, loads raw data into Supabase (PostgreSQL),
creates a cleaned SQL view, then exports the final dataset to CSV.

Setup:
    1. pip install -r requirements.txt
    2. Copy .env.example to .env and fill in your Supabase credentials
    3. python3 pipeline/housing_scraper_pipeline.py

Usage:
    python3 pipeline/housing_scraper_pipeline.py                        # full run with Supabase
    python3 pipeline/housing_scraper_pipeline.py --output /path/to.csv  # custom CSV output path
    python3 pipeline/housing_scraper_pipeline.py --skip-db              # skip Supabase, CSV only
"""

import re
import sys
import time
import argparse
import numpy as np
import pandas as pd
import requests
import psycopg2
from sqlalchemy import create_engine, text
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

BASE_URL = "https://suumo.jp/"
STARTING_URL = (
    "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&pc=50&smk=&po1=25"
    "&po2=99&shkr1=03&shkr2=03&shkr3=03&shkr4=03&rn=0025&ek=002506940&rn=0350"
    "&ek=035017990&ek=035026830&ek=035001440&rn=0070&ek=007026830&ek=007006960"
    "&ra=013&ae=00251&ae=03501&cb=0.0&ct=9999999&md=01&md=02&md=03&md=04&md=05"
    "&md=06&md=07&md=08&md=09&md=10&md=11&md=12&md=13&et=9999999&mb=0&mt=9999999"
    "&cn=9999999&fw2="
)
OUTPUT_DIR = Path(__file__).parent.parent
ENV_FILE = OUTPUT_DIR / ".env"
REQUEST_DELAY = 1.5
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

# PostgreSQL table and view names
RAW_TABLE = "housing_data_raw"
VIEW_NAME = "tokyo_housing"

# SQL view — PostgreSQL-compatible adaptation of the original SQLite view.
# Key differences from SQLite:
#   - Subquery inside CTE requires an alias (sub)
#   - ROUND() requires NUMERIC type, not FLOAT
#   - REGEXP_REPLACE used for safer building_age parsing
#   - NULLIF guards empty strings before CAST to prevent runtime errors
CREATE_VIEW_SQL = f"""
DROP VIEW IF EXISTS {VIEW_NAME};

CREATE VIEW {VIEW_NAME} AS

WITH deduplicated_listings AS (
    SELECT * FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY title, floor, floor_plan, area, rent
                ORDER BY url
            ) AS rn
        FROM {RAW_TABLE}
    ) sub
    WHERE rn = 1
),

standardized_listings AS (
    SELECT
        url, title, address,

        CAST(NULLIF(REGEXP_REPLACE(REPLACE(rent, '万円', ''), '[^0-9.]', '', 'g'), '') AS FLOAT) * 10000 AS rent,
        CAST(NULLIF(REGEXP_REPLACE(REPLACE(management_fee, '円', ''), '[^0-9.]', '', 'g'), '') AS FLOAT) AS management_fee,
        CAST(NULLIF(REGEXP_REPLACE(REPLACE(deposit, '万円', ''), '[^0-9.]', '', 'g'), '') AS FLOAT) * 10000 AS deposit,
        CAST(NULLIF(REGEXP_REPLACE(REPLACE(key_money, '万円', ''), '[^0-9.]', '', 'g'), '') AS FLOAT) * 10000 AS key_money,

        RTRIM(floor, '階') AS floor,

        CASE
            WHEN floor_plan = 'ワンルーム' THEN '1R'
            ELSE floor_plan
        END AS floor_plan,

        CAST(NULLIF(REGEXP_REPLACE(REPLACE(area, 'm2', ''), '[^0-9.]', '', 'g'), '') AS FLOAT) AS area,

        CASE
            WHEN building_age LIKE '%新築%' THEN 0
            ELSE CAST(
                NULLIF(REGEXP_REPLACE(building_age, '[^0-9]', '', 'g'), '')
            AS INTEGER)
        END AS building_age,

        CASE
            WHEN building_size LIKE '%平屋%' THEN '1'
            ELSE RTRIM(building_size, '階建')
        END AS building_size,

        stations,
        nearest_station,
        distance_to_nearest_station,
        ROUND(CAST(avg_distance_to_stations AS NUMERIC), 2) AS avg_distance_to_stations
    FROM deduplicated_listings
),

featured_listings AS (
    SELECT
        url, title, address, rent,

        NULLIF(management_fee, 0.0) AS management_fee,
        NULLIF(deposit, 0.0) AS deposit,
        NULLIF(key_money, 0.0) AS key_money,
        floor, floor_plan, area, building_age,
        building_size, nearest_station,
        distance_to_nearest_station, avg_distance_to_stations,

        ROUND(CAST(AVG(rent) OVER (PARTITION BY nearest_station) AS NUMERIC), 2)
            AS avg_rent_by_station,
        ROUND(CAST(AVG(rent) OVER (PARTITION BY floor_plan) AS NUMERIC), 2)
            AS avg_rent_by_floor_plan,

        COUNT(title) OVER (PARTITION BY nearest_station) AS count_listings_per_station,
        COUNT(title) OVER (PARTITION BY floor_plan)      AS count_listings_per_floor_plan
    FROM standardized_listings
)

SELECT * FROM featured_listings;
"""


# ------------------------------------------------------------------
# Credentials + connection
# ------------------------------------------------------------------

def load_env() -> dict:
    """Read key=value pairs from the .env file. Raises if file or keys are missing."""
    if not ENV_FILE.exists():
        raise FileNotFoundError(
            f".env file not found at {ENV_FILE}\n"
            "Create it with your Supabase credentials. See .env for the template."
        )
    creds = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        creds[key.strip()] = val.strip()

    required = ["SUPABASE_HOST", "SUPABASE_PORT", "SUPABASE_DB", "SUPABASE_USER", "SUPABASE_PASSWORD"]
    missing = [k for k in required if not creds.get(k) or creds.get(k, "").startswith("your-")]
    if missing:
        raise ValueError(f"Missing or unfilled credentials in .env: {missing}")
    return creds


def get_engine(creds: dict):
    """Return a SQLAlchemy engine connected to Supabase via psycopg2."""
    url = (
        f"postgresql+psycopg2://{creds['SUPABASE_USER']}:{creds['SUPABASE_PASSWORD']}"
        f"@{creds['SUPABASE_HOST']}:{creds['SUPABASE_PORT']}/{creds['SUPABASE_DB']}"
        f"?sslmode=require"
    )
    return create_engine(url)


# ------------------------------------------------------------------
# Scraper class
# ------------------------------------------------------------------

class TestTokyoHousingScraper:

    def __init__(self, base_url: str, starting_url: str):
        self.base_url = base_url
        self.starting_url = starting_url
        self.listings: list = []

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def scrape_listings(self) -> None:
        """Paginate through all listing pages and collect raw HTML cassettes."""
        self.listings = []
        next_url: str | None = None

        while True:
            url = next_url if next_url else self.starting_url
            try:
                response = requests.get(url, headers=HEADERS, timeout=15)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"  [WARN] Request failed for {url}: {e} — stopping pagination.")
                break

            soup = BeautifulSoup(response.text, "lxml")
            cassettes = soup.select("div.cassetteitem")
            self.listings.extend(cassettes)
            print(f"  Page scraped — {len(cassettes)} listings found (total so far: {len(self.listings)})")

            next_url = self._get_next_page_url(soup)
            if not next_url:
                break

            time.sleep(REQUEST_DELAY)

        print(f"\n{len(self.listings)} properties gathered in total.")

    def _get_next_page_url(self, soup: BeautifulSoup) -> str | None:
        """Return the absolute URL of the next page, or None if on the last page."""
        current = soup.find("li", class_="pagination-current")
        if not current:
            return None
        siblings = current.find_next_siblings("li")
        if len(siblings) < 2:
            return None
        anchor = siblings[1].select_one("a")
        if not anchor or not anchor.get("href"):
            return None
        return self.base_url.rstrip("/") + anchor["href"]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_station_info(self, item) -> tuple:
        """
        Extract station names and walk distances from a listing block.
        Returns (stations_str, nearest_station, distance_to_nearest, avg_distance)
        or (None, None, None, None) if no valid station data is found.
        """
        raw_blocks = item.select("li.cassetteitem_detail-col2 div.cassetteitem_detail-text")

        stations, distances = [], []
        for block in raw_blocks:
            text = block.get_text().strip()
            station_match = re.findall(r"/(?P<station>.*?)\s*歩", text)
            distance_match = re.findall(r"\d+", text)
            if station_match and distance_match:
                stations.append(station_match[0])
                distances.append(int(distance_match[0]))

        if not stations or not distances:
            return (None, None, None, None)

        stations_str = ",".join(stations)
        nearest_dist = min(distances)
        nearest_idx = distances.index(nearest_dist)
        nearest_station = stations[nearest_idx]
        avg_distance = round(float(np.mean(distances)), 2)

        return (stations_str, nearest_station, nearest_dist, avg_distance)

    def parse_sublistings(self, sub, building_meta: dict) -> dict | None:
        """
        Parse a single sublisting <tr> element into a flat dict.
        Returns None if critical fields (rent, floor_plan, area) are all missing.
        """
        url_tag = sub.select_one("td.ui-text--midium.ui-text--bold a")
        rent_tag = sub.select_one("span.cassetteitem_price.cassetteitem_price--rent")
        mgmt_tag = sub.select_one("span.cassetteitem_price.cassetteitem_price--administration")
        deposit_tag = sub.select_one("span.cassetteitem_price.cassetteitem_price--deposit")
        key_tag = sub.select_one("span.cassetteitem_price.cassetteitem_price--gratuity")
        floor_cells = sub.select("tr.js-cassette_link td")
        fp_tag = sub.select_one("span.cassetteitem_madori")
        area_tag = sub.select_one("span.cassetteitem_menseki")

        rent = rent_tag.get_text().strip() if rent_tag else None
        floor_plan = fp_tag.get_text().strip() if fp_tag else None
        area = area_tag.get_text().strip() if area_tag else None

        if not any([rent, floor_plan, area]):
            return None

        return {
            **building_meta,
            "url": (
                self.base_url.rstrip("/") + url_tag["href"]
                if url_tag and url_tag.get("href")
                else None
            ),
            "rent": rent,
            "management_fee": mgmt_tag.get_text().strip() if mgmt_tag else None,
            "deposit": deposit_tag.get_text().strip() if deposit_tag else None,
            "key_money": key_tag.get_text().strip() if key_tag else None,
            "floor": (
                floor_cells[2].get_text().strip()
                if len(floor_cells) >= 3
                else None
            ),
            "floor_plan": floor_plan,
            "area": area,
        }

    # ------------------------------------------------------------------
    # Dataset building
    # ------------------------------------------------------------------

    def build_housing_dataset(self) -> pd.DataFrame:
        """Parse all scraped listings into a raw structured DataFrame."""
        rows = []
        for item in self.listings:
            title_tag = item.select_one("div.cassetteitem_content-title")
            address_tag = item.select_one("li.cassetteitem_detail-col1")
            building_cells = item.select("li.cassetteitem_detail-col3 div")
            stations_str, nearest_station, nearest_dist, avg_dist = self.parse_station_info(item)

            building_meta = {
                "title": title_tag.get_text().strip() if title_tag else None,
                "address": address_tag.get_text().strip() if address_tag else None,
                "building_age": (
                    building_cells[0].get_text().strip() if building_cells else None
                ),
                "building_size": (
                    building_cells[1].get_text().strip()
                    if len(building_cells) >= 2
                    else None
                ),
                "stations": stations_str,
                "nearest_station": nearest_station,
                "distance_to_nearest_station": nearest_dist,
                "avg_distance_to_stations": avg_dist,
            }

            for sub in item.select("tr.js-cassette_link"):
                parsed = self.parse_sublistings(sub, building_meta)
                if parsed:
                    rows.append(parsed)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Supabase — load raw table, create view, query cleaned result
    # ------------------------------------------------------------------

    def load_to_supabase(self, raw_df: pd.DataFrame, engine) -> pd.DataFrame:
        """
        1. Drop the view (so we can safely replace the underlying table)
        2. Load raw_df into housing_data_raw (replacing any existing data)
        3. Create the tokyo_housing view
        4. Query the view and return the cleaned DataFrame
        """
        with engine.connect() as conn:
            print(f"  Dropping view '{VIEW_NAME}' if it exists...")
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.commit()

        print(f"  Loading {len(raw_df):,} raw rows into '{RAW_TABLE}'...")
        raw_df.to_sql(
            name=RAW_TABLE,
            con=engine,
            if_exists="replace",
            index=False,
        )

        with engine.connect() as conn:
            print(f"  Creating view '{VIEW_NAME}'...")
            # Execute each statement separately (psycopg2 doesn't support multi-statement)
            for statement in CREATE_VIEW_SQL.strip().split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))
            conn.commit()

            print(f"  Querying '{VIEW_NAME}'...")
            clean_df = pd.read_sql(f"SELECT * FROM {VIEW_NAME}", conn)

        return clean_df

    # ------------------------------------------------------------------
    # Post-SQL pandas cleaning (floor expansion + building size)
    # These mirror what the original notebook did after querying the view
    # ------------------------------------------------------------------

    @staticmethod
    def apply_pandas_transforms(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations that can't be done cleanly in SQL:
          - Expand floor ranges into individual rows (e.g. "B1-2" → -1, 0, 1, 2)
          - Sum building size components (e.g. "地下1地上3" → 4)
        """
        def floor_parser(val):
            if pd.isna(val):
                return np.nan
            cleaned = re.sub(r"階", "", str(val)).strip()
            parts = cleaned.upper().split("-")
            if "" in parts or not parts:
                return np.nan
            if parts[0].startswith("B"):
                try:
                    parts[0] = str(-int(parts[0][1:]))
                except ValueError:
                    return np.nan
            if len(parts) == 1:
                try:
                    return np.array([int(parts[0])])
                except ValueError:
                    return np.nan
            try:
                lo, hi = int(parts[0]), int(parts[1])
                return np.arange(lo, hi + 1)
            except ValueError:
                return np.nan

        def building_size_parser(val):
            if pd.isna(val):
                return np.nan
            if "平屋" in str(val):
                return 1
            nums = re.findall(r"\d+", str(val))
            return sum(map(int, nums)) if nums else np.nan

        df = (
            df.assign(floor=df["floor"].apply(floor_parser))
              .explode("floor")
              .dropna(subset=["floor"])
        )
        df["floor"] = df["floor"].astype("int64")
        df["building_size"] = df["building_size"].apply(building_size_parser)
        df = df.dropna(subset=["rent"]).reset_index(drop=True)

        return df


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape SUUMO, load to Supabase, export cleaned CSV."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "tokyo_housing.csv"),
        help="Path for the output CSV file.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip Supabase entirely — clean data in pandas and write CSV only.",
    )
    args = parser.parse_args()
    output_path = Path(args.output)

    # --- Scrape ---
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting scrape...")
    scraper = TestTokyoHousingScraper(BASE_URL, STARTING_URL)
    scraper.scrape_listings()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Building raw dataset...")
    raw_df = scraper.build_housing_dataset()
    print(f"  Raw rows: {len(raw_df):,}")

    # --- Supabase path ---
    if not args.skip_db:
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connecting to Supabase...")
            creds = load_env()
            engine = get_engine(creds)
            clean_df = scraper.load_to_supabase(raw_df, engine)
            print(f"  Rows returned from view: {len(clean_df):,}")
        except (FileNotFoundError, ValueError) as e:
            print(f"\n[ERROR] {e}")
            print("Falling back to pandas cleaning...")
            args.skip_db = True

    # --- CSV-only fallback path ---
    if args.skip_db:
        from test_scraper import TestTokyoHousingScraper as _S
        # Reuse the pandas cleaning from the previous version inline
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cleaning data in pandas (no DB)...")
        clean_df = _pandas_clean(raw_df)
        print(f"  Rows after cleaning: {len(clean_df):,}")

    # --- Apply floor + building size transforms (both paths) ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Applying final transforms...")
    clean_df = TestTokyoHousingScraper.apply_pandas_transforms(clean_df)
    print(f"  Final rows: {len(clean_df):,}")

    # --- Save CSV ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving to {output_path}...")
    clean_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done — {len(clean_df):,} listings written to {output_path.name}")


def _pandas_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback cleaning in pandas when --skip-db is used."""
    df = df.sort_values("url").drop_duplicates(
        subset=["title", "floor", "floor_plan", "area", "rent"]
    ).reset_index(drop=True)

    df["rent"] = pd.to_numeric(df["rent"].str.replace("万円", "", regex=False), errors="coerce") * 10000
    df["management_fee"] = pd.to_numeric(df["management_fee"].str.replace("円", "", regex=False), errors="coerce")
    df["deposit"] = pd.to_numeric(df["deposit"].str.replace("万円", "", regex=False), errors="coerce") * 10000
    df["key_money"] = pd.to_numeric(df["key_money"].str.replace("万円", "", regex=False), errors="coerce") * 10000
    df["area"] = pd.to_numeric(df["area"].str.replace("m2", "", regex=False), errors="coerce")

    for col in ["management_fee", "deposit", "key_money"]:
        df[col] = df[col].replace(0.0, np.nan)

    df["floor_plan"] = df["floor_plan"].replace("ワンルーム", "1R")

    def parse_building_age(val):
        if pd.isna(val): return np.nan
        if "新築" in val: return 0
        nums = re.findall(r"\d+", val)
        return int(nums[0]) if nums else np.nan

    df["building_age"] = df["building_age"].apply(parse_building_age)
    df["avg_rent_by_station"] = df.groupby("nearest_station")["rent"].transform("mean").round(2)
    df["avg_rent_by_floor_plan"] = df.groupby("floor_plan")["rent"].transform("mean").round(2)
    df["count_listings_per_station"] = df.groupby("nearest_station")["title"].transform("count")
    df["count_listings_per_floor_plan"] = df.groupby("floor_plan")["title"].transform("count")

    return df


if __name__ == "__main__":
    main()
