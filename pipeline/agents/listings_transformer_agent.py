from typing import Dict, Any
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from datetime import datetime
import json, re
from listings_loader_agent import ListingsLoaderAgent
import numpy as np

# Resolves the .env file path relative to this file's location —
# walks up three levels: agents/ → pipeline/ → Tokyo Housing/ → .env
ENV_FILE = Path(__file__).parent.parent.parent / ".env"

class ListingsTransformerAgent:
    def __init__(self, VIEW_NAME: str, ENV_FILE = ENV_FILE):
        # Name of the PostgreSQL view to query (e.g. "tokyo_housing" or "test_tokyo_housing")
        self.VIEW_NAME = VIEW_NAME
        # Path to the .env file — passed through to ListingsLoaderAgent for credentials
        self.ENV_FILE = ENV_FILE

    def query_view(self) -> Dict[str, Any]:
        """Query the cleaned SQL view and return a structured result dict with the DataFrame."""

        # Reuse ListingsLoaderAgent's credential loading and engine creation —
        # passing an empty dict because we only need the DB utilities, not load_to_db
        loader_agent = ListingsLoaderAgent({}, ENV_FILE=self.ENV_FILE)
        creds = loader_agent.load_env()
        if creds["status"] != "VALID":
            results = {
                "status": "ERROR",
                "message": f"Cannot query view due to invalid credentials: {creds.get('message', 'No message')}"
            }
            self._write_log(results, step = "query")
            return results

        try:
            engine = loader_agent.get_engine(creds)
        except ConnectionError as e:
            # get_engine raises ConnectionError if the DB is unreachable —
            # catch it here so the caller gets a clean error dict instead of a crash
            results = {
                "status": "ERROR",
                "message": f"Failed to connect to the database: {e}"
            }
            self._write_log(results, step = "query")
            return results

        try:
            with engine.connect() as conn:
                # Fetch all rows from the view into a DataFrame —
                # result.keys() preserves the column names from the SQL view
                result = conn.execute(text(f"SELECT * FROM {self.VIEW_NAME}"))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            results = {
                "status": "VALID",
                "message": f"Query successful, {len(df):,} rows retrieved.",
                "rows_queried": len(df),
                # DataFrame nested under "data" key — consistent with all other agents
                "data": df
            }
            self._write_log(results, step = "query")
            return results
        except Exception as e:
            results = {
                "status": "ERROR",
                "message": f"Database query failed: {e}"
            }
            self._write_log(results, step = "query")
            return results

    def apply_pandas_transforms(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply transformations that can't be done cleanly in SQL:
          - Expand floor ranges into individual rows (e.g. "B1-2" → -1, 0, 1, 2)
          - Sum building size components (e.g. "地下1地上3" → 4)
        """

        def floor_parser(val):
            """
            Convert a raw SUUMO floor string into a numpy array of integer floor numbers.
            - Single floor  → array of one element:  "3階"   → [3]
            - Floor range   → array of all floors:   "2-4階" → [2, 3, 4]
            - Basement      → negative integer:      "B1階"  → [-1]
            - Unparseable   → np.nan (row dropped by explode + dropna downstream)
            """
            if pd.isna(val):
                return np.nan
            # Strip the Japanese floor suffix 階 before splitting on "-"
            cleaned = re.sub(r"階", "", str(val)).strip()
            parts = cleaned.upper().split("-")
            # Guard against empty strings produced by splitting edge cases (e.g. "-")
            if "" in parts or not parts:
                return np.nan
            # Convert basement notation "B1" → "-1" so it can be cast to int
            if parts[0].startswith("B"):
                try:
                    parts[0] = str(-int(parts[0][1:]))
                except ValueError:
                    return np.nan
            if len(parts) == 1:
                # Single floor — wrap in array so explode() handles it uniformly
                try:
                    return np.array([int(parts[0])])
                except ValueError:
                    return np.nan
            # Floor range — generate every integer between lo and hi inclusive
            try:
                lo, hi = int(parts[0]), int(parts[1])
                return np.arange(lo, hi + 1)
            except ValueError:
                return np.nan

        def building_size_parser(val):
            """
            Convert a raw SUUMO building size string into a total floor count integer.
            - "地下1地上14階建" → 1 + 14 = 15  (sums all numeric values found)
            - "平屋"           → 1              (single-story, no numeric content)
            - Unparseable      → np.nan
            """
            if pd.isna(val):
                return np.nan
            # 平屋 means "single-story" — no floor number in the string, return 1 directly
            if "平屋" in str(val):
                return 1
            # Extract all numeric substrings and sum them (handles "地下X地上Y" format)
            nums = re.findall(r"\d+", str(val))
            return sum(map(int, nums)) if nums else np.nan

        # Capture row count before any transforms — used in the log to show how many rows were dropped
        rows_queried = len(df)

        # Apply floor_parser to each cell, then explode arrays into separate rows
        # (a range like [2, 3, 4] becomes three rows), then drop any rows where
        # floor is still NaN after parsing
        df = (
            df.assign(floor=df["floor"].apply(floor_parser))
              .explode("floor")
              .dropna(subset=["floor"])
        )
        # explode() produces float64 — cast back to int64 for clean downstream use
        df["floor"] = df["floor"].astype("int64")

        # Replace raw building size strings with a clean integer total floor count
        df["building_size"] = df["building_size"].apply(building_size_parser)

        # Drop any rows still missing rent — these are not usable for analysis or modeling
        df = df.dropna(subset=["rent"]).reset_index(drop=True)

        results = {
            "status": "VALID",
            "message": "Transformations applied successfully.",
            "rows_queried": rows_queried,       # row count before transforms
            "rows_after_transform": len(df),    # row count after drops
            "data": df
        }
        self._write_log(results, step = "transform")
        return results

    def _write_log(self, results: Dict[str, Any], step: str) -> None:
        # Metadata-only payload — DataFrame is excluded to keep the log
        # small and human-readable (a full DataFrame could be thousands of rows)
        log = {
            "agent": "ListingsTransformerAgent",
            "timestamp": datetime.now().isoformat(),
            "status": results["status"],
            "message": results.get("message", ""),
            "rows_queried": results.get("rows_queried", 0),              # rows pulled from the view
            "rows_after_transform": results.get("rows_after_transform", 0)  # rows remaining after drops
        }
        # Timestamp in the filename gives each run its own log file — preserves history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(__file__).parent / "logs" / f"transformer_log_{step}_{timestamp}.json"
        # Create the logs/ directory if it doesn't exist yet
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False preserves Japanese characters in station/address fields
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Saved to {log_path}")


if __name__ == "__main__":
    # Use the test view to avoid touching production data during development
    VIEW_NAME = "test_tokyo_housing"

    transformer_agent = ListingsTransformerAgent(VIEW_NAME)

    # Step 1 — pull the cleaned view from the database
    query_results = transformer_agent.query_view()
    if query_results["status"] != "VALID":
        print(f"Query failed: {query_results['message']}")
        raise Exception("Query failed.")

    # Step 2 — apply pandas transforms that couldn't be done in SQL
    df = query_results["data"]
    transform_results = transformer_agent.apply_pandas_transforms(df)
    if transform_results["status"] != "VALID":
        print(f"Transform failed: {transform_results['message']}")
        raise Exception("Transform failed.")

    final_df = transform_results["data"]
    print(f"Transformations completed — final dataset has {len(final_df):,} rows.")

    final_df.to_csv(Path(__file__).parent.parent / "test_transformed_listings.csv", index = False)
    print(f"Transformed data saved to test_transformed_listings.csv")
