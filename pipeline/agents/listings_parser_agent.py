import statistics as stats
from pathlib import Path
from datetime import datetime
import re, json
from typing import Dict, Tuple, Any
from listings_scraper_agent import ListingsScraperAgent


class ListingsParserAgent:
    def __init__(self, listings_object, base_url):
        # The full result dict passed in from ListingsScraperAgent —
        # contains status, listings_count, and the raw BeautifulSoup Tag objects
        self.listings_object = listings_object
        # Root domain used to construct absolute listing URLs from relative hrefs
        self.base_url = base_url
        # Cache the status so other methods can check it without re-fetching
        self.status = listings_object.get("status")

    def parse_station_info(self, item) -> Tuple[str | None, str | None, int | None, float | None]:
        """
        Extract station names and walk distances from a listing block.
        Returns (stations_str, nearest_station, distance_to_nearest, avg_distance)
        or (None, None, None, None) if no valid station data is found.
        """
        # Each station is listed in its own div inside the second detail column
        raw_blocks = item.select("li.cassetteitem_detail-col2 div.cassetteitem_detail-text")

        stations, distances = [], []
        for block in raw_blocks:
            text = block.get_text().strip() if block else ""

            # SUUMO formats station info as: "線名 / 駅名 歩X分"
            # The regex captures everything between "/" and "歩" (walk) as the station name
            station_match = re.findall(r"/(?P<station>.*?)\s*歩", text)

            # Extract all numeric values from the text — the first number is the walk time in minutes
            distance_match = [int(n) for n in re.findall(r"\d+", text)]

            # Only record a station if both a name and a distance were found
            if station_match and distance_match:
                stations.append(station_match[0])
                distances.append(distance_match[0])

        # If no valid station data was found, return nulls for all four fields
        if not stations or not distances:
            return (None, None, None, None)

        # Join all station names into a single comma-separated string for storage
        stations_str = ",".join(stations)

        # Identify the nearest station by finding the minimum walk distance
        nearest_dist = min(distances)
        nearest_idx = distances.index(nearest_dist)
        nearest_station = stations[nearest_idx]

        # Average walk distance across all stations — rounded to 2 decimal places
        avg_distance = round(float(stats.mean(distances)), 2)

        return (stations_str, nearest_station, nearest_dist, avg_distance)

    def parse_sublistings(self, sub, building_meta: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Parse a single sublisting <tr> element into a flat dict.
        Returns None if critical fields (rent, floor_plan, area) are all missing.
        """
        # Each sublisting row contains unit-level data — these selectors target
        # the specific SUUMO CSS classes for each field
        url_tag = sub.select_one("td.ui-text--midium.ui-text--bold a")
        rent_tag = sub.select_one("span.cassetteitem_price.cassetteitem_price--rent")
        mgmt_tag = sub.select_one("span.cassetteitem_price.cassetteitem_price--administration")
        deposit_tag = sub.select_one("span.cassetteitem_price.cassetteitem_price--deposit")
        key_tag = sub.select_one("span.cassetteitem_price.cassetteitem_price--gratuity")

        # Floor info lives in the third <td> cell of the sublisting row
        floor_cells = sub.select("tr.js-cassette_link td")
        fp_tag = sub.select_one("span.cassetteitem_madori")
        area_tag = sub.select_one("span.cassetteitem_menseki")

        # Extract the three most critical fields — used below to decide whether
        # this sublisting is worth keeping at all
        rent = rent_tag.get_text().strip() if rent_tag else None
        floor_plan = fp_tag.get_text().strip() if fp_tag else None
        area = area_tag.get_text().strip() if area_tag else None

        # Drop the sublisting entirely if all three critical fields are missing —
        # a row with no rent, floor plan, or area is not usable downstream
        if not any([rent, floor_plan, area]):
            return None

        return {
            # Spread building-level fields (title, address, age, stations, etc.)
            # so each sublisting row is fully self-contained — no join needed later
            **building_meta,
            # SUUMO hrefs are relative — prepend base domain to make absolute
            "url": (
                self.base_url.rstrip("/") + url_tag["href"]
                if url_tag and url_tag.get("href")
                else None
            ),
            "rent": rent,
            "management_fee": mgmt_tag.get_text().strip() if mgmt_tag else None,
            "deposit": deposit_tag.get_text().strip() if deposit_tag else None,
            "key_money": key_tag.get_text().strip() if key_tag else None,
            # Floor is at index 2 in the row's <td> cells — guard against
            # rows with fewer cells than expected
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

    def build_housing_dataset(self) -> Dict[str, Any]:
        """Parse all scraped listings and return the enriched result dict."""

        # Accumulates one flat dict per individual rental unit
        rows = []

        for item in self.listings_object.get("listings", []):
            # Each cassette item represents one building — extract building-level fields first
            title_tag = item.select_one("div.cassetteitem_content-title")
            address_tag = item.select_one("li.cassetteitem_detail-col1")

            # Building age is at index 0, building size (floors) at index 1
            building_cells = item.select("li.cassetteitem_detail-col3 div")

            # Station info is shared across all units in this building
            stations_str, nearest_station, nearest_dist, avg_dist = self.parse_station_info(item)

            # Bundle all building-level fields into a dict that gets spread into
            # each sublisting row — avoids repeating extraction inside the inner loop
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

            # Each building can have multiple rental units — iterate over each
            # sublisting row and parse it into a flat dict
            for sub in item.select("tr.js-cassette_link"):
                parsed = self.parse_sublistings(sub, building_meta)
                # parse_sublistings returns None if critical fields are missing — skip those
                if parsed:
                    rows.append(parsed)

        # Build the result dict without mutating the input — spread the scraper's
        # result and add parser-specific fields on top
        results = {
            **self.listings_object,
            "sublistings_count": len(rows),
            # Downstream agents consume "dataset" — a list of flat unit-level dicts
            "dataset": rows
        }
        self._write_log(results)
        return results

    def _write_log(self, results: Dict[str, Any]) -> None:
        # Metadata-only payload — raw HTML and parsed rows are excluded
        # to keep the log file small and human-readable
        log = {
            "agent": "ListingsParserAgent",
            "timestamp": datetime.now().isoformat(),
            "status": results["status"],
            "message": results.get("message", ""),
            "listings_count": results.get("listings_count", 0),       # cassettes received from scraper
            "sublistings_count": results.get("sublistings_count", 0)  # individual units parsed
        }
        # Timestamp in filename gives each run its own log file — preserves history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(__file__).parent / "logs" / f"parser_log_{timestamp}.json"
        # Create logs/ directory if it doesn't exist yet
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False preserves Japanese station and address names
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
