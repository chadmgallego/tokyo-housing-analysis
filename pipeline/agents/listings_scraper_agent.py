
from typing import Dict, Any
import requests, time
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import json


class ListingsScraperAgent:
    def __init__(self, base_url, starting_url, HEADERS, REQUEST_DELAY):
        # Root domain used to construct absolute URLs from relative hrefs
        self.base_url = base_url
        # The first page URL to begin pagination from
        self.starting_url = starting_url
        # HTTP headers sent with every request to mimic a real browser
        self.HEADERS = HEADERS
        # Seconds to wait between page requests to avoid rate limiting
        self.REQUEST_DELAY = REQUEST_DELAY

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def scrape_listings(self) -> Dict[str, Any]:
        """Paginate through all listing pages and collect raw HTML cassettes."""

        # Accumulates BeautifulSoup Tag objects — one per property cassette
        self.listings = []

        # Tracks the URL for the next page; None on the first iteration
        # so the loop falls back to self.starting_url
        next_url: str | None = None

        while True:
            # First iteration uses the starting URL; subsequent iterations
            # use the next page URL returned by _get_next_page_url()
            url = next_url if next_url else self.starting_url

            try:
                response = requests.get(url, headers=self.HEADERS, timeout=15)
                # Raises an HTTPError for 4xx/5xx status codes (e.g. 403 blocked,
                # 404 not found) so they're caught below instead of silently failing
                response.raise_for_status()
            except requests.RequestException as e:
                # Any network or HTTP error stops pagination immediately.
                # We return whatever listings were collected before the failure
                # so the result is still usable for partial data if needed.
                self.results = {
                    "status": "ERROR",
                    "message": f"Request failed for {url}: {e}",
                    "listings_count": len(self.listings) if self.listings else 0,
                    "listings_html": self.listings if self.listings else []
                }
                self._write_log(self.results)
                return self.results

            # Parse the full page HTML into a navigable BeautifulSoup tree
            soup = BeautifulSoup(response.text, "lxml")

            # Each property listing on SUUMO is wrapped in a div.cassetteitem —
            # this selects all of them from the current page
            cassettes = soup.select("div.cassetteitem")

            if not cassettes:
                # Zero cassettes usually means SUUMO returned a blocked/empty page.
                # We warn but don't crash — the pipeline continues to the next page.
                print(f"  [WARN] No cassettes found on {url} — possible block or layout change.")
            else:
                # Append this page's Tag objects to the running list
                self.listings.extend(cassettes)

            print(f"  Page scraped — {len(cassettes)} listings found (total so far: {len(self.listings)})")

            # Check if there's a next page; returns None when we've hit the last page
            next_url = self._get_next_page_url(soup)

            if not next_url:
                # Pagination is complete — build the success result and exit
                print(f"\n{len(self.listings)} properties gathered in total.")
                self.results = {
                    "status": "VALID",
                    "message": "All pages scraped successfully.",
                    "listings_count": len(self.listings),
                    # Raw BeautifulSoup Tag objects passed directly to the next
                    # agent in memory — no serialization needed at this stage
                    "listings": self.listings
                }
                self._write_log(self.results)
                return self.results

            # Polite delay between requests — prevents triggering SUUMO's
            # rate limiter and mimics human browsing behavior
            time.sleep(self.REQUEST_DELAY)


    def _get_next_page_url(self, soup: BeautifulSoup) -> str | None:
        """Return the absolute URL of the next page, or None if on the last page."""

        # The currently active page in SUUMO's pagination is marked with
        # the class 'pagination-current'
        current = soup.find("li", class_="pagination-current")
        if not current:
            # No pagination element found — single page result or layout change
            return None

        # Get all <li> elements that follow the current page marker
        siblings = current.find_next_siblings("li")

        # Requires at least 2 siblings — SUUMO's pagination structure places
        # the next page link at siblings[1]; fewer than 2 means we're at or
        # near the last page
        if len(siblings) < 2:
            return None

        # siblings[1] contains the anchor tag linking to the next page
        anchor = siblings[1].select_one("a")
        if not anchor or not anchor.get("href"):
            # Anchor missing or has no href — treat as last page
            return None

        # SUUMO hrefs are relative (e.g. /jj/chintai/...) — prepend the base
        # domain to construct a valid absolute URL
        return self.base_url.rstrip("/") + anchor["href"]


    def _write_log(self, results: Dict[str, Any]) -> None:
        # Build a lightweight log payload — metadata only, no raw HTML.
        # Keeping HTML out of the log prevents multi-MB log files.
        log = {
            "agent": "ListingsScraperAgent",
            "timestamp": datetime.now().isoformat(),
            "status": results["status"],
            "message": results.get("message", ""),
            "listings_count": len(results.get("listings", []))
        }

        # Timestamp in the filename ensures each run gets its own log file
        # instead of overwriting the previous one — preserves full run history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(__file__).parent / "logs" / f"scraper_log_{timestamp}.json"

        # Create the logs/ directory if it doesn't exist yet
        log_path.parent.mkdir(exist_ok=True)

        with open(log_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False preserves Japanese characters in station/address fields
            json.dump(log, f, ensure_ascii=False, indent=2)

        print(f"[LOG] Saved to {log_path}")


if __name__ == "__main__":
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

    agent = ListingsScraperAgent(BASE_URL, STARTING_URL, HEADERS, REQUEST_DELAY)
    scraper_object = agent.scrape_listings()

    # Only confirm success here — logging is handled inside scrape_listings()
    if scraper_object["status"] == "VALID":
        print(f"Scraping completed successfully with {scraper_object['listings_count']} listings.")
