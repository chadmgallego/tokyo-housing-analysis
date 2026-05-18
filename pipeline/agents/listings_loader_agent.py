from sqlalchemy import create_engine, text
from typing import Dict, Any
import pandas as pd
from pathlib import Path
from listings_scraper_agent import ListingsScraperAgent
from listings_parser_agent import ListingsParserAgent
from datetime import datetime
import json

# Resolves the .env file path relative to this file's location —
# walks up three levels: agents/ → pipeline/ → Tokyo Housing/ → .env
ENV_FILE = Path(__file__).parent.parent.parent / ".env"

class ListingsLoaderAgent:
    def __init__(self, sublistings_object: Dict[str, Any], ENV_FILE = ENV_FILE):
        # The parsed result dict from ListingsParserAgent — contains the "dataset" key
        # with a list of flat unit-level dicts ready to be loaded into the database
        self.sublistings_object = sublistings_object
        # Path to the .env file containing Supabase credentials
        self.ENV_FILE = ENV_FILE

    def load_env(self) -> Dict[str, Any]:
        """Read key=value pairs from the .env file. Raises if file or keys are missing."""
        # Guard against missing .env file before attempting to read it
        if not self.ENV_FILE.exists():
            creds = {
                "status": "ERROR",
                "message": f".env file not found at {self.ENV_FILE}\n"
                           "Create it with your Supabase credentials. See .env for the template."
            }
            return creds

        creds = {}
        for line in self.ENV_FILE.read_text().splitlines():
            line = line.strip()
            # Skip blank lines and comments
            if not line or line.startswith("#"):
                continue
            # partition("=") splits on the first "=" only — safe for values that contain "="
            key, _, val = line.partition("=")
            creds[key.strip()] = val.strip()

        # All five keys are required to build the SQLAlchemy connection URL
        required = ["SUPABASE_HOST", "SUPABASE_PORT", "SUPABASE_DB", "SUPABASE_USER", "SUPABASE_PASSWORD"]
        # Also flag keys that are still set to the placeholder value from .env.example
        missing = [k for k in required if not creds.get(k) or creds.get(k, "").startswith("your-")]
        if missing:
            creds = {
                "status": "ERROR",
                "message": f"Missing or unfilled credentials in .env: {', '.join(missing)}"
            }
            return creds

        creds.update({
            "status": "VALID",
            "message": "Credentials loaded successfully."
        })
        return creds

    def get_engine(self, creds: Dict[str, Any]) -> Any:
        """Return a SQLAlchemy engine connected to Supabase via psycopg2."""
        # Defensive check — caller should have validated creds before calling this
        if creds["status"] != "VALID":
            raise ValueError(f"Invalid credentials: {creds.get('message', 'No message')}")

        # Build the PostgreSQL connection URL from .env credentials.
        # sslmode=require is mandatory for Supabase connections.
        url = (
            f"postgresql+psycopg2://{creds['SUPABASE_USER']}:{creds['SUPABASE_PASSWORD']}"
            f"@{creds['SUPABASE_HOST']}:{creds['SUPABASE_PORT']}/{creds['SUPABASE_DB']}"
            f"?sslmode=require"
        )
        try:
            engine = create_engine(url)
            # Eagerly test the connection with a lightweight query —
            # create_engine() is lazy and won't fail until a query is actually run
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as e:
            # Wrap any connection failure in a consistent exception type
            # so callers only need to catch ConnectionError
            raise ConnectionError(f"Failed to connect to the database: {e}")

    def load_to_db(self, RAW_TABLE, VIEW_NAME, CREATE_VIEW_SQL) -> Dict[str, Any]:
        """
        1. Drop the view (so we can safely replace the underlying table)
        2. Load raw_df into housing_data_raw (replacing any existing data)
        3. Create the tokyo_housing view
        4. Query the view and return the cleaned DataFrame
        """
        # Convert the list of flat unit dicts into a DataFrame for bulk loading
        raw_df = pd.DataFrame(self.sublistings_object.get("dataset"))
        if raw_df.empty:
            # Nothing to load — fail early before attempting any DB operations
            result = {
                "status": "ERROR",
                "message": "No data to load: the dataset is empty.",
                "sublistings_count": 0,
                "rows_loaded": 0
            }
            self._write_log(result)
            return result

        creds = self.load_env()
        if creds["status"] != "VALID":
            # Inject zero-count fields so _write_log can read them without KeyError
            creds.update({
                "sublistings_count": 0,
                "rows_loaded": 0})
            self._write_log(creds)
            return creds

        try:
            engine = self.get_engine(creds)
        except ConnectionError as e:
            result = {
                "status": "ERROR",
                "message": str(e),
                "rows_loaded": 0,
                "rows_in_view": 0
            }
            self._write_log(result)
            return result

        with engine.connect() as conn:
            print(f"  Dropping view '{VIEW_NAME}' if it exists...")
            # The view must be dropped before replacing the underlying table —
            # PostgreSQL raises a dependency error if you try to replace a table
            # that an existing view depends on
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.commit()

        print(f"  Loading {len(raw_df):,} raw rows into '{RAW_TABLE}'...")
        # if_exists="replace" drops and recreates the raw table on each run —
        # index=False prevents pandas from writing its 0-based integer index as a column
        raw_df.to_sql(
            name=RAW_TABLE,
            con=engine,
            if_exists="replace",
            index=False,
        )

        with engine.connect() as conn:
            print(f"  Creating view '{VIEW_NAME}'...")
            # psycopg2 doesn't support executing multiple SQL statements in a single call,
            # so we split on ";" and execute each statement individually.
            # Note: semicolons inside SQL comments would break this split —
            # the SQL file uses periods instead of semicolons in comments to avoid this.
            for statement in CREATE_VIEW_SQL.strip().split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))
            conn.commit()

            print(f"  Querying '{VIEW_NAME}'...")
            # Read the cleaned view back into a DataFrame so the caller
            # has the final, production-ready dataset immediately
            clean_df = pd.read_sql(f"SELECT * FROM {VIEW_NAME}", conn)

        result = {
            "status": "VALID",
            "message": f"Loaded {len(raw_df):,} rows into '{RAW_TABLE}', view '{VIEW_NAME}' created.",
            "rows_loaded": len(raw_df),
            "rows_in_view": len(clean_df),
            # DataFrame is nested under "dataframe" key — keeps the result dict
            # consistent with all other agents while still making data accessible
            "dataframe": clean_df
            }
        self._write_log(result)
        return result

    def _write_log(self, results: Dict[str, Any]) -> None:
        # Metadata-only payload — DataFrame is excluded to keep the log
        # small and human-readable (a full DataFrame could be thousands of rows)
        log = {
            "agent": "ListingsLoaderAgent",
            "timestamp": datetime.now().isoformat(),
            "status": results["status"],
            "message": results.get("message", ""),
            "rows_loaded": results.get("rows_loaded", 0),   # raw rows written to Supabase
            "rows_in_view": results.get("rows_in_view", 0) # rows after SQL cleaning/deduplication
        }
        # Timestamp in the filename gives each run its own log file — preserves history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(__file__).parent / "logs" / f"loader_log_{timestamp}.json"
        # Create the logs/ directory if it doesn't exist yet
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False preserves Japanese characters in any error messages
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Saved to {log_path}")

if __name__ == "__main__":
    # PostgreSQL table and view names — prefixed with "test_" to avoid
    # overwriting production data when running this agent in isolation
    RAW_TABLE = "test_housing_data_raw"
    VIEW_NAME = "test_tokyo_housing"

    # Navigate up from agents/ → pipeline/ → Tokyo Housing/ → sql/
    sql_path = Path(__file__).parent.parent.parent / "sql" / "data_cleaning_and_features_postgresql.sql"

    # Read the SQL file and substitute placeholder names with the test table/view names.
    # This lets us reuse the same SQL file for both test and production runs.
    CREATE_VIEW_SQL = (
        sql_path.read_text(encoding="utf-8")
        .replace("housing_data_raw", RAW_TABLE)  # swap raw table name
        .replace("tokyo_housing", VIEW_NAME)     # swap view name
    )

    BASE_URL = "https://suumo.jp/"
    STARTING_URL = (
        "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&pc=50&smk=&po1=25"
        "&po2=99&shkr1=03&shkr2=03&shkr3=03&shkr4=03&rn=0025&ek=002506940&rn=0350"
        "&ek=035017990&ek=035026830&ek=035001440&rn=0070&ek=007026830&ek=007006960"
        "&ra=013&ae=00251&ae=03501&cb=0.0&ct=9999999&md=01&md=02&md=03&md=04&md=05"
        "&md=06&md=07&md=08&md=09&md=10&md=11&md=12&md=13&et=9999999&mb=0&mt=9999999"
        "&cn=9999999&fw2="
    )
    REQUEST_DELAY = 1.5
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }

    scraper_agent = ListingsScraperAgent(BASE_URL, STARTING_URL, HEADERS, REQUEST_DELAY)
    scraper_results = scraper_agent.scrape_listings()

    # Fail fast — if scraping failed there's nothing to parse
    if scraper_results["status"] != "VALID":
        print(f"Scraping failed with message: {scraper_results.get('message', 'No message')}")
        raise Exception("Scraping failed, cannot proceed to parsing.")

    print(f"Scraping completed successfully with {scraper_results['listings_count']} listings.")

    parser_agent = ListingsParserAgent(scraper_results, BASE_URL)
    parser_results = parser_agent.build_housing_dataset()

    # Logging is handled inside build_housing_dataset — just confirm count here
    print(f"Parsing completed — {parser_results['sublistings_count']} sublistings extracted.")

    loader_agent = ListingsLoaderAgent(parser_results)
    loader_results = loader_agent.load_to_db(RAW_TABLE, VIEW_NAME, CREATE_VIEW_SQL)

    # Fail fast — loading failure means the view is not queryable downstream
    if loader_results["status"] != "VALID":
        print(f"Loading failed: {loader_results['message']}")
        raise Exception("Loading failed, cannot proceed to downstream analysis.")

    if loader_results["status"] == "VALID":
        print(f"Loading completed — {loader_results['rows_in_view']:,} rows in view.")
