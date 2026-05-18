from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json
 
from listings_scraper_agent import ListingsScraperAgent
from listings_parser_agent import ListingsParserAgent
from listings_loader_agent import ListingsLoaderAgent
from listings_transformer_agent import ListingsTransformerAgent
from anomaly_detection_agent import AnomalyDetectionAgent
 
# ── PIPELINE CONFIG ───────────────────────────────────────────────────────────
BASE_URL = "https://suumo.jp/"
STARTING_URL = (
    "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&pc=50&smk=&po1=25"
    "&po2=99&shkr1=03&shkr2=03&shkr3=03&shkr4=03&rn=0025&ek=002506940&rn=0350"
    "&ek=035017990&ek=035026830&ek=035001440&rn=0070&ek=007026830&ek=007006960"
    "&ra=013&ae=00251&ae=03501&cb=0.0&ct=9999999&md=01&md=02&md=03&md=04&md=05"
    "&md=06&md=07&md=08&md=09&md=10&md=11&md=12&md=13&et=9999999&mb=0&mt=9999999"
    "&cn=9999999&fw2="
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}
REQUEST_DELAY = 1.5
RAW_TABLE     = "housing_data_raw"
VIEW_NAME     = "tokyo_housing"
 
# SQL file lives at: Tokyo Housing/sql/data_cleaning_and_features_postgresql.sql
SQL_PATH = Path(__file__).parent.parent.parent / "sql" / "data_cleaning_and_features_postgresql.sql"
 
# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────
 
class AgentOrchestratorAgent:
    def __init__(
        self,
        base_url: str        = BASE_URL,
        starting_url: str    = STARTING_URL,
        headers: dict        = HEADERS,
        request_delay: float = REQUEST_DELAY,
        raw_table: str       = RAW_TABLE,
        view_name: str       = VIEW_NAME,
        sql_path: Path       = SQL_PATH,
        contamination: float = 0.05,
    ):
        self.base_url       = base_url
        self.starting_url   = starting_url
        self.headers        = headers
        self.request_delay  = request_delay
        self.raw_table      = raw_table
        self.view_name      = view_name
        self.sql_path       = sql_path
        self.contamination  = contamination
        self.timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    # ── STAGES ───────────────────────────────────────────────────────────────
 
    def _run_scraper(self) -> Dict[str, Any]:
        agent = ListingsScraperAgent(
            self.base_url, self.starting_url, self.headers, self.request_delay
        )
        return agent.scrape_listings()
 
    def _run_parser(self, scraper_results: Dict[str, Any]) -> Dict[str, Any]:
        agent = ListingsParserAgent(scraper_results, self.base_url)
        return agent.build_housing_dataset()
 
    def _run_loader(self, parser_results: Dict[str, Any]) -> Dict[str, Any]:
        create_view_sql = (
            self.sql_path.read_text(encoding="utf-8")
            .replace("housing_data_raw", self.raw_table)
            .replace("tokyo_housing", self.view_name)
        )
        agent = ListingsLoaderAgent(parser_results)
        return agent.load_to_db(self.raw_table, self.view_name, create_view_sql)
 
    def _run_transformer(self) -> Dict[str, Any]:
        # Transformer queries the DB itself — no data passed in from previous stage
        agent = ListingsTransformerAgent(self.view_name)
        query_results = agent.query_view()
        if query_results["status"] != "VALID":
            return query_results
        return agent.apply_pandas_transforms(query_results["data"])
 
    def _run_anomaly(self, df) -> Dict[str, Any]:
        agent = AnomalyDetectionAgent(df)
        return agent.run(contamination=self.contamination)
 
    # ── MAIN ENTRY POINT ─────────────────────────────────────────────────────
 
    def run(self) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"  AgentOrchestratorAgent — {self.timestamp}")
        print(f"{'='*60}")
 
        # ── STAGE 1: Scraping ─────────────────────────────────
        print("\n── STAGE 1: Scraping ────────────────────────────────────────")
        scraper_results = self._run_scraper()
        if scraper_results["status"] != "VALID":
            print(f"\n[FAIL] Pipeline halted at scraper: {scraper_results['message']}")
            self._write_log(scraper_results, failed_at="scraper")
            return scraper_results
        print(f"  → {scraper_results['listings_count']:,} listings scraped")
 
        # ── STAGE 2: Parsing ──────────────────────────────────
        print("\n── STAGE 2: Parsing ─────────────────────────────────────────")
        parser_results = self._run_parser(scraper_results)
        if parser_results["status"] != "VALID":
            print(f"\n[FAIL] Pipeline halted at parser: {parser_results['message']}")
            self._write_log(parser_results, failed_at="parser")
            return parser_results
        print(f"  → {parser_results['sublistings_count']:,} sublistings parsed")
 
        # ── STAGE 3: Loading ──────────────────────────────────
        print("\n── STAGE 3: Loading ─────────────────────────────────────────")
        loader_results = self._run_loader(parser_results)
        if loader_results["status"] != "VALID":
            print(f"\n[FAIL] Pipeline halted at loader: {loader_results['message']}")
            self._write_log(loader_results, failed_at="loader")
            return loader_results
        print(f"  → {loader_results['rows_in_view']:,} rows in view after loading")
 
        # ── STAGE 4: Transforming ─────────────────────────────
        print("\n── STAGE 4: Transforming ────────────────────────────────────")
        transformer_results = self._run_transformer()
        if transformer_results["status"] != "VALID":
            print(f"\n[FAIL] Pipeline halted at transformer: {transformer_results['message']}")
            self._write_log(transformer_results, failed_at="transformer")
            return transformer_results
        print(f"  → {transformer_results['rows_after_transform']:,} rows after transforms")
 
        # ── STAGE 5: Anomaly Detection ────────────────────────
        print("\n── STAGE 5: Anomaly Detection ───────────────────────────────")
        anomaly_results = self._run_anomaly(transformer_results["data"])
        if anomaly_results["status"] != "VALID":
            print(f"\n[FAIL] Pipeline halted at anomaly: {anomaly_results['message']}")
            self._write_log(anomaly_results, failed_at="anomaly")
            return anomaly_results
        print(f"  → {anomaly_results['number_of_anomalies']:,} anomalies flagged")
        
        final_df = anomaly_results["data"]
        path_to_results = Path(__file__).parent.parent / "dataset_results"
        path_to_results.mkdir(exist_ok=True)
        final_df.to_csv(path_to_results / f"final_housing_results_{self.timestamp}.csv", index=False)
        print(f"  → Final results saved to final_housing_results_{self.timestamp}.csv")
 
        # ── DONE ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  Pipeline completed successfully.")
        print(f"{'='*60}\n")
        self._write_log(anomaly_results)
        return anomaly_results
 
    # ── LOGGING ──────────────────────────────────────────────────────────────
 
    def _write_log(self, results: Dict[str, Any], failed_at: str = None) -> None:
        # Metadata-only payload — DataFrame is excluded to keep the log
        # small and human-readable (a full DataFrame could be thousands of rows)
        log = {
            "agent": "AgentOrchestratorAgent",
            "timestamp": self.timestamp,
            "status": results.get("status"),
            "message": results.get("message", ""),
            "failed_at": failed_at,                              # None on success
            "number_of_anomalies": results.get("number_of_anomalies"),
            "contamination": self.contamination,
        }
        # Timestamp in the filename gives each run its own log file — preserves history
        log_path = Path(__file__).parent / "logs" / f"orchestrator_log_{self.timestamp}.json"
        # Create the logs/ directory if it doesn't exist yet
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False preserves Japanese characters in any error messages
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Saved to {log_path}")
 
 
# ── CRON ENTRY POINT ─────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    orchestrator = AgentOrchestratorAgent()
    orchestrator.run()
 