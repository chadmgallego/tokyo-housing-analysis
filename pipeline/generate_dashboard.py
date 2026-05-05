#!/usr/bin/env python3
"""
Cron-friendly dashboard generator for tokyo_housing.csv.

On each run it:
  1. Reads the stored CSV mtime from .dashboard_state.env
  2. Compares it to the current mtime of tokyo_housing.csv
  3. If the CSV is newer (or state file is missing), regenerates the dashboard
     and updates the stored mtime
  4. Otherwise exits immediately — no work needed

Cron example (run every day at 6am):
    0 6 * * * /usr/bin/python3 /path/to/generate_dashboard.py >> /path/to/dashboard.log 2>&1
"""

import csv
import json
import sys
import statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime

WATCH_DIR = Path(__file__).parent.parent  # repo root, one level above pipeline/
CSV_FILE = WATCH_DIR / "tokyo_housing.csv"
OUTPUT_FILE = WATCH_DIR / "tokyo_rental_dashboard.html"
STATE_FILE = WATCH_DIR / ".dashboard_state.env"
STATE_KEY = "LAST_CSV_MTIME"

FP_ORDER = ["1R", "1K", "1DK", "1LDK", "2K", "2DK", "2LDK", "3LDK"]
FP_COLORS = {
    "1R": "#AFA9EC", "1K": "#378ADD", "1DK": "#5DCAA5", "1LDK": "#1D9E75",
    "2K": "#EF9F27", "2DK": "#FAC775", "2LDK": "#E24B4A", "3LDK": "#534AB7",
}


def parse_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rent = float(row["rent"]) if row.get("rent") else None
                area = float(row["area"]) if row.get("area") else None
                age = int(float(row["building_age"])) if row.get("building_age") else None
                walk = int(float(row["distance_to_nearest_station"])) if row.get("distance_to_nearest_station") else None
                fp = row.get("floor_plan", "").strip() or None
                station = row.get("nearest_station", "").strip() or None
                if None in (rent, area, age, walk, fp, station):
                    continue
                if rent <= 0 or area <= 0:
                    continue
                rows.append({
                    "fp": fp, "rent": rent, "area": area, "age": age, "walk": walk, "station": station,
                    "url": row.get("url", ""),
                    "title": row.get("title", ""),
                    "address": row.get("address", ""),
                    "floor": int(float(row["floor"])) if row.get("floor") else None,
                    "mgmt": float(row["management_fee"]) if row.get("management_fee") else None,
                })
            except (ValueError, KeyError):
                continue
    return rows


def compute_stats(rows: list[dict]) -> dict:
    fp_rents: dict[str, list] = defaultdict(list)
    area_buckets = {"<20": [], "20-25": [], "25-30": [], "30-40": [], "40-50": [], "50+": []}
    age_dist = {"<5": 0, "5-10": 0, "11-20": 0, "21-30": 0, "31-40": 0, "41-50": 0, "50+": 0}
    station_rents: dict[str, list] = defaultdict(list)
    walk_rents: dict[int, list] = defaultdict(list)

    for r in rows:
        fp_rents[r["fp"]].append(r["rent"])

        a = r["area"]
        if a < 20:
            area_buckets["<20"].append(r["rent"])
        elif a < 25:
            area_buckets["20-25"].append(r["rent"])
        elif a < 30:
            area_buckets["25-30"].append(r["rent"])
        elif a < 40:
            area_buckets["30-40"].append(r["rent"])
        elif a < 50:
            area_buckets["40-50"].append(r["rent"])
        else:
            area_buckets["50+"].append(r["rent"])

        age = r["age"]
        if age < 5:
            age_dist["<5"] += 1
        elif age <= 10:
            age_dist["5-10"] += 1
        elif age <= 20:
            age_dist["11-20"] += 1
        elif age <= 30:
            age_dist["21-30"] += 1
        elif age <= 40:
            age_dist["31-40"] += 1
        elif age <= 50:
            age_dist["41-50"] += 1
        else:
            age_dist["50+"] += 1

        station_rents[r["station"]].append(r["rent"])
        if 1 <= r["walk"] <= 20:
            walk_rents[r["walk"]].append(r["rent"])

    fp_counts = {fp: len(v) for fp, v in fp_rents.items()}
    fp_rent_avg = {fp: round(statistics.mean(v)) for fp, v in fp_rents.items()}
    area_rent_avg = {k: round(statistics.mean(v)) for k, v in area_buckets.items() if v}

    stations_list = [
        {"name": s, "rent": round(statistics.mean(rents))}
        for s, rents in station_rents.items()
        if len(rents) >= 50
    ]
    stations_list.sort(key=lambda x: x["rent"], reverse=True)
    stations_list = stations_list[:8]

    walk_rent_avg = {k: round(statistics.mean(v)) for k, v in sorted(walk_rents.items())}

    all_rents = [r["rent"] for r in rows]
    all_areas = [r["area"] for r in rows]
    all_ages = [r["age"] for r in rows]

    return {
        "total": len(rows),
        "median_rent": statistics.median(all_rents),
        "mean_rent": statistics.mean(all_rents),
        "median_area": statistics.median(all_areas),
        "median_age": statistics.median(all_ages),
        "fp_counts": fp_counts,
        "fp_rent": fp_rent_avg,
        "area_rent": area_rent_avg,
        "age_dist": age_dist,
        "stations": stations_list,
        "walk_rent": walk_rent_avg,
    }


def fmt_yen(n: float) -> str:
    man = round(n) // 10000
    remainder = round(n) % 10000
    if man > 0 and remainder == 0:
        return f"¥{man}万"
    if man > 0:
        return f"¥{man}.{remainder // 1000}万"
    return f"¥{round(n):,}"


def generate_html(rows: list[dict], stats: dict) -> str:
    raw_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))

    # Ordered fp_counts and fp_rent aligned to FP_ORDER
    fp_counts_ordered = {fp: stats["fp_counts"].get(fp, 0) for fp in FP_ORDER if fp in stats["fp_counts"]}
    fp_rent_ordered = {fp: stats["fp_rent"].get(fp, 0) for fp in FP_ORDER if fp in stats["fp_rent"]}

    # Known floor plans in data (for filter options)
    known_fps = [fp for fp in FP_ORDER if fp in stats["fp_counts"]]
    fp_options = "\n".join(
        f'    <option value="{fp}">{fp}</option>' for fp in known_fps
    )

    # Unique stations sorted by name (for station filter)
    all_stations_sorted = sorted({r["station"] for r in rows if r.get("station")})
    station_options = "\n".join(
        f'    <option value="{s}">{s}</option>' for s in all_stations_sorted
    )

    total_fmt = f"{stats['total']:,}"
    median_rent_fmt = fmt_yen(stats["median_rent"])
    mean_rent_fmt = fmt_yen(stats["mean_rent"])
    median_area_fmt = f"{stats['median_area']:.1f} m²"
    median_age_fmt = f"{round(stats['median_age'])} yrs"

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tokyo Rental Market Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
:root {{
  --color-background-primary: #ffffff;
  --color-background-secondary: #f4f2ed;
  --color-background-tertiary: #eceae5;
  --color-text-primary: #1a1a18;
  --color-text-secondary: #5f5e5a;
  --color-text-tertiary: #888780;
  --color-border-tertiary: rgba(0,0,0,0.12);
  --color-border-secondary: rgba(0,0,0,0.22);
  --font-sans: system-ui, -apple-system, sans-serif;
  --border-radius-md: 8px;
  --border-radius-lg: 12px;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --color-background-primary: #1e1e1c;
    --color-background-secondary: #2a2a27;
    --color-background-tertiary: #333330;
    --color-text-primary: #f0ede8;
    --color-text-secondary: #b4b2a9;
    --color-text-tertiary: #888780;
    --color-border-tertiary: rgba(255,255,255,0.1);
    --color-border-secondary: rgba(255,255,255,0.2);
  }}
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: var(--font-sans);
  background: var(--color-background-tertiary);
  color: var(--color-text-primary);
  min-height: 100vh;
  padding: 2rem;
}}
.page-header {{ margin-bottom: 1.5rem; }}
.page-title {{ font-size: 20px; font-weight: 500; color: var(--color-text-primary); margin-bottom: 4px; }}
.page-sub {{ font-size: 13px; color: var(--color-text-tertiary); }}
.section-label {{
  font-size: 10px; font-weight: 500; color: var(--color-text-tertiary);
  letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px;
}}
.filter-row {{
  display: flex; align-items: center; gap: 10px; margin-bottom: 1.25rem; flex-wrap: wrap;
}}
.filter-row label {{ font-size: 12px; color: var(--color-text-secondary); }}
.filter-row select {{
  font-size: 12px; padding: 5px 10px;
  border-radius: var(--border-radius-md);
  border: 0.5px solid var(--color-border-secondary);
  background: var(--color-background-primary);
  color: var(--color-text-primary); cursor: pointer;
}}
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px; margin-bottom: 1.25rem;
}}
.kpi {{
  background: var(--color-background-secondary);
  border-radius: var(--border-radius-md); padding: 14px 16px;
}}
.kpi-label {{ font-size: 11px; color: var(--color-text-tertiary); margin-bottom: 6px; font-weight: 500; letter-spacing: 0.04em; }}
.kpi-value {{ font-size: 22px; font-weight: 500; color: var(--color-text-primary); line-height: 1; }}
.kpi-sub {{ font-size: 11px; color: var(--color-text-tertiary); margin-top: 5px; }}
.charts-grid {{
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px; margin-bottom: 12px;
}}
.charts-grid-2 {{
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}}
.chart-card {{
  background: var(--color-background-primary);
  border: 0.5px solid var(--color-border-tertiary);
  border-radius: var(--border-radius-lg); padding: 16px;
}}
.chart-title {{ font-size: 13px; font-weight: 500; color: var(--color-text-primary); margin-bottom: 3px; }}
.chart-sub {{ font-size: 11px; color: var(--color-text-tertiary); margin-bottom: 12px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--color-text-secondary); }}
.legend-dot {{ width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }}
@media (max-width: 900px) {{
  .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }}
  .charts-grid, .charts-grid-2 {{ grid-template-columns: 1fr; }}
}}
.table-card {{
  background: var(--color-background-primary);
  border: 0.5px solid var(--color-border-tertiary);
  border-radius: var(--border-radius-lg);
  padding: 16px; margin-top: 12px;
}}
.tbl-wrap {{ overflow-x: auto; }}
.tbl-wrap table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.tbl-th {{
  text-align: left; padding: 8px 10px;
  font-size: 11px; font-weight: 500; color: var(--color-text-tertiary);
  letter-spacing: 0.04em; text-transform: uppercase;
  border-bottom: 0.5px solid var(--color-border-tertiary);
  cursor: pointer; white-space: nowrap; user-select: none;
}}
.tbl-th:hover {{ color: var(--color-text-primary); }}
.tbl-th.sorted {{ color: var(--color-text-primary); }}
.tbl-td {{
  padding: 7px 10px; border-bottom: 0.5px solid var(--color-border-tertiary);
  color: var(--color-text-secondary); white-space: nowrap;
}}
tr:last-child .tbl-td {{ border-bottom: none; }}
tr:hover .tbl-td {{ background: var(--color-background-secondary); }}
.listing-link {{ color: var(--color-text-primary); text-decoration: none; font-weight: 500; }}
.listing-link:hover {{ text-decoration: underline; }}
.tbl-pagination {{
  display: flex; align-items: center; justify-content: space-between;
  margin-top: 12px; font-size: 12px; color: var(--color-text-tertiary);
}}
.tbl-btn {{
  font-size: 12px; padding: 5px 12px;
  border-radius: var(--border-radius-md);
  border: 0.5px solid var(--color-border-secondary);
  background: var(--color-background-primary);
  color: var(--color-text-primary); cursor: pointer;
}}
.tbl-btn:disabled {{ opacity: 0.35; cursor: not-allowed; }}
</style>
</head>
<body>

<div class="page-header">
  <div class="page-title">Tokyo Rental Market Intelligence</div>
  <div class="page-sub">{total_fmt} listings · Suumo scrape · Generated {generated_at}</div>
</div>

<div class="section-label">filters</div>
<div class="filter-row">
  <label>Floor plan:</label>
  <select id="fpFilter" onchange="applyFilter()">
    <option value="all">All floor plans</option>
{fp_options}
  </select>
  <label style="margin-left:8px;">Area:</label>
  <select id="areaFilter" onchange="applyFilter()">
    <option value="all">All sizes</option>
    <option value="u20">&lt;20 m²</option>
    <option value="20-25">20–25 m²</option>
    <option value="25-30">25–30 m²</option>
    <option value="30-40">30–40 m²</option>
    <option value="40-50">40–50 m²</option>
    <option value="50p">50+ m²</option>
  </select>
  <label style="margin-left:8px;">Station:</label>
  <select id="stationFilter" onchange="applyFilter()">
    <option value="all">All stations</option>
{station_options}
  </select>
  <label style="margin-left:8px;">Building age:</label>
  <select id="ageFilter" onchange="applyFilter()">
    <option value="all">All ages</option>
    <option value="new">New (&lt;5 yrs)</option>
    <option value="mid">Mid (5–20 yrs)</option>
    <option value="old">Older (20+ yrs)</option>
  </select>
</div>

<div class="section-label">key metrics</div>
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-label">Total listings</div><div class="kpi-value" id="k-total">—</div><div class="kpi-sub">in filtered set</div></div>
  <div class="kpi"><div class="kpi-label">Median rent</div><div class="kpi-value" id="k-median">—</div><div class="kpi-sub">¥ / month</div></div>
  <div class="kpi"><div class="kpi-label">Mean rent</div><div class="kpi-value" id="k-mean">—</div><div class="kpi-sub">¥ / month</div></div>
  <div class="kpi"><div class="kpi-label">Median area</div><div class="kpi-value" id="k-area">—</div><div class="kpi-sub">m² per unit</div></div>
  <div class="kpi"><div class="kpi-label">Median age</div><div class="kpi-value" id="k-age">—</div><div class="kpi-sub">building years</div></div>
</div>

<div class="section-label">visualizations</div>
<div class="charts-grid">
  <div class="chart-card">
    <div class="chart-title">Listings by floor plan</div>
    <div class="chart-sub">Unit type distribution across all listings</div>
    <div class="legend" id="fp-legend"></div>
    <div style="position:relative;height:170px;"><canvas id="c1" role="img" aria-label="Listings by floor plan donut chart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Avg rent by floor plan</div>
    <div class="chart-sub">Mean monthly rent (¥) per unit type</div>
    <div style="position:relative;height:210px;"><canvas id="c2" role="img" aria-label="Average rent by floor plan bar chart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Rent by area size</div>
    <div class="chart-sub">Mean rent by floor area bucket (m²)</div>
    <div style="position:relative;height:210px;"><canvas id="c3" role="img" aria-label="Rent by area size bar chart"></canvas></div>
  </div>
</div>

<div class="charts-grid-2" style="margin-top:12px;">
  <div class="chart-card">
    <div class="chart-title">Building age distribution</div>
    <div class="chart-sub">Number of listings by building age (years)</div>
    <div style="position:relative;height:210px;"><canvas id="c4" role="img" aria-label="Building age distribution bar chart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Top stations by avg rent</div>
    <div class="chart-sub">Stations with 50+ listings, ranked by price</div>
    <div style="position:relative;height:210px;"><canvas id="c5" role="img" aria-label="Top stations by average rent horizontal bar chart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Rent vs walk time</div>
    <div class="chart-sub">Mean rent by minutes to nearest station</div>
    <div style="position:relative;height:210px;"><canvas id="c6" role="img" aria-label="Rent by walk time to station line chart"></canvas></div>
  </div>
</div>

<div class="section-label" style="margin-top:1.5rem;">listings</div>
<div class="table-card">
  <div class="tbl-wrap">
    <table>
      <thead id="tbl-head"></thead>
      <tbody id="tbl-body"></tbody>
    </table>
  </div>
  <div class="tbl-pagination">
    <button class="tbl-btn" id="tbl-prev" onclick="prevPage()">← Prev</button>
    <span id="tbl-info"></span>
    <button class="tbl-btn" id="tbl-next" onclick="nextPage()">Next →</button>
  </div>
</div>

<script>
const RAW = {raw_json};

const FP_ORDER = {json.dumps(FP_ORDER)};
const FP_COLORS = {json.dumps(FP_COLORS)};

const ALL_FP_COUNTS = {json.dumps(fp_counts_ordered, ensure_ascii=False)};
const ALL_FP_RENT   = {json.dumps(fp_rent_ordered, ensure_ascii=False)};
const ALL_AREA_RENT = {json.dumps(stats["area_rent"], ensure_ascii=False)};
const ALL_AGE_DIST  = {json.dumps(stats["age_dist"], ensure_ascii=False)};
const ALL_STATIONS  = {json.dumps(stats["stations"], ensure_ascii=False)};
const ALL_WALK_RENT = {json.dumps({str(k): v for k, v in stats["walk_rent"].items()}, ensure_ascii=False)};

const DEFAULTS = {{
  total: "{total_fmt}",
  median: "{median_rent_fmt}",
  mean: "{mean_rent_fmt}",
  area: "{median_area_fmt}",
  age: "{median_age_fmt}",
}};

let charts = {{}};

function applyFilter() {{
  const fp  = document.getElementById("fpFilter").value;
  const ar  = document.getElementById("areaFilter").value;
  const st  = document.getElementById("stationFilter").value;
  const ag  = document.getElementById("ageFilter").value;

  let sub = RAW.filter(d => {{
    const fpOk = fp === "all" || d.fp === fp;
    const arOk = ar === "all"
      || (ar === "u20"  && d.area < 20)
      || (ar === "20-25" && d.area >= 20 && d.area < 25)
      || (ar === "25-30" && d.area >= 25 && d.area < 30)
      || (ar === "30-40" && d.area >= 30 && d.area < 40)
      || (ar === "40-50" && d.area >= 40 && d.area < 50)
      || (ar === "50p"  && d.area >= 50);
    const stOk = st === "all" || d.station === st;
    const agOk = ag === "all"
      || (ag === "new" && d.age < 5)
      || (ag === "mid" && d.age >= 5 && d.age <= 20)
      || (ag === "old" && d.age > 20);
    return fpOk && arOk && stOk && agOk;
  }});

  const hasFilter = fp !== "all" || ar !== "all" || st !== "all" || ag !== "all";
  updateKPIs(sub, hasFilter);
  updateCharts(sub, fp, ag, ag, hasFilter);
  renderTable(hasFilter ? sub : RAW);
}}

function median(arr) {{
  if (!arr.length) return 0;
  const s = [...arr].sort((a,b)=>a-b);
  const m = Math.floor(s.length/2);
  return s.length % 2 ? s[m] : (s[m-1]+s[m])/2;
}}
function mean(arr) {{ return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0; }}
function fmtY(n) {{ return n >= 10000 ? "¥"+(n/10000).toFixed(0)+"万" : "¥"+Math.round(n).toLocaleString(); }}
function fmtYfull(n) {{ return "¥"+Math.round(n).toLocaleString(); }}

function updateKPIs(sub, hasFilter) {{
  if (!hasFilter) {{
    document.getElementById("k-total").textContent = DEFAULTS.total;
    document.getElementById("k-median").textContent = DEFAULTS.median;
    document.getElementById("k-mean").textContent = DEFAULTS.mean;
    document.getElementById("k-area").textContent = DEFAULTS.area;
    document.getElementById("k-age").textContent = DEFAULTS.age;
    return;
  }}
  const rents = sub.map(d=>d.rent);
  const areas = sub.map(d=>d.area);
  const ages = sub.map(d=>d.age);
  document.getElementById("k-total").textContent = sub.length;
  document.getElementById("k-median").textContent = fmtY(median(rents));
  document.getElementById("k-mean").textContent = fmtY(mean(rents));
  document.getElementById("k-area").textContent = median(areas).toFixed(1)+" m²";
  document.getElementById("k-age").textContent = median(ages).toFixed(0)+" yrs";
}}

function destroyChart(id) {{ if(charts[id]){{charts[id].destroy();delete charts[id];}} }}

function buildFPDonut(fpCounts) {{
  destroyChart("c1");
  const fps = FP_ORDER.filter(f => fpCounts[f] > 0);
  document.getElementById("fp-legend").innerHTML = fps.map(f =>
    `<span class="legend-item"><span class="legend-dot" style="background:${{FP_COLORS[f]}}"></span>${{f}} ${{fpCounts[f]}}</span>`
  ).join("");
  charts["c1"] = new Chart(document.getElementById("c1"),{{
    type:"doughnut",
    data:{{labels:fps,datasets:[{{data:fps.map(f=>fpCounts[f]),backgroundColor:fps.map(f=>FP_COLORS[f]),borderWidth:0}}]}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>" "+ctx.label+": "+ctx.raw}}}}}}}}
  }});
}}

function buildFPRentBar(fpRent) {{
  destroyChart("c2");
  const fps = FP_ORDER.filter(f => fpRent[f] !== undefined);
  charts["c2"] = new Chart(document.getElementById("c2"),{{
    type:"bar",
    data:{{labels:fps,datasets:[{{data:fps.map(f=>Math.round(fpRent[f])),backgroundColor:fps.map(f=>FP_COLORS[f]),borderWidth:0}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>" "+fmtYfull(ctx.raw)}}}}}},
      scales:{{x:{{ticks:{{font:{{size:10}}}},grid:{{display:false}}}},y:{{ticks:{{font:{{size:10}},callback:v=>"¥"+(v/10000).toFixed(0)+"万"}},grid:{{color:"rgba(128,128,128,0.1)"}}}}}}}}
  }});
}}

function buildAreaRentBar(areaRent) {{
  destroyChart("c3");
  const labels = Object.keys(areaRent);
  const vals = Object.values(areaRent).map(v=>Math.round(v));
  charts["c3"] = new Chart(document.getElementById("c3"),{{
    type:"bar",
    data:{{labels,datasets:[{{data:vals,backgroundColor:"#378ADD",borderWidth:0}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>" "+fmtYfull(ctx.raw)}}}}}},
      scales:{{x:{{ticks:{{font:{{size:10}}}},grid:{{display:false}}}},y:{{ticks:{{font:{{size:10}},callback:v=>"¥"+(v/10000).toFixed(0)+"万"}},grid:{{color:"rgba(128,128,128,0.1)"}}}}}}}}
  }});
}}

function buildAgeDist(ageDist) {{
  destroyChart("c4");
  const labels = Object.keys(ageDist);
  const vals = Object.values(ageDist);
  charts["c4"] = new Chart(document.getElementById("c4"),{{
    type:"bar",
    data:{{labels,datasets:[{{data:vals,backgroundColor:"#1D9E75",borderWidth:0}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}}}},
      scales:{{x:{{ticks:{{font:{{size:10}}}},grid:{{display:false}}}},y:{{ticks:{{font:{{size:10}}}},grid:{{color:"rgba(128,128,128,0.1)"}}}}}}}}
  }});
}}

function buildStationBar(stations) {{
  destroyChart("c5");
  const sorted = [...stations].sort((a,b)=>b.rent-a.rent);
  charts["c5"] = new Chart(document.getElementById("c5"),{{
    type:"bar",
    data:{{labels:sorted.map(s=>s.name),datasets:[{{data:sorted.map(s=>Math.round(s.rent)),backgroundColor:"#534AB7",borderWidth:0}}]}},
    options:{{responsive:true,maintainAspectRatio:false,indexAxis:"y",
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>" "+fmtYfull(ctx.raw)}}}}}},
      scales:{{x:{{ticks:{{font:{{size:10}},callback:v=>"¥"+(v/10000).toFixed(0)+"万"}},grid:{{color:"rgba(128,128,128,0.1)"}}}},y:{{ticks:{{font:{{size:10}}}},grid:{{display:false}}}}}}}}
  }});
}}

function buildWalkLine(walkRent) {{
  destroyChart("c6");
  const keys = Object.keys(walkRent).map(Number).sort((a,b)=>a-b);
  const labels = keys.map(k=>k+"min");
  const vals = keys.map(k=>Math.round(walkRent[k]));
  charts["c6"] = new Chart(document.getElementById("c6"),{{
    type:"line",
    data:{{labels,datasets:[{{data:vals,borderColor:"#E24B4A",backgroundColor:"rgba(226,75,74,0.08)",tension:0.35,pointRadius:4,pointBackgroundColor:"#E24B4A",fill:true,borderWidth:2}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>" "+fmtYfull(ctx.raw)}}}}}},
      scales:{{x:{{ticks:{{font:{{size:10}}}},grid:{{display:false}}}},y:{{ticks:{{font:{{size:10}},callback:v=>"¥"+(v/10000).toFixed(0)+"万"}},grid:{{color:"rgba(128,128,128,0.1)"}}}}}}}}
  }});
}}

function updateCharts(sub, fp, ag, wk, hasFilter) {{
  if (!hasFilter) {{
    buildFPDonut(ALL_FP_COUNTS);
    buildFPRentBar(ALL_FP_RENT);
    buildAreaRentBar(ALL_AREA_RENT);
    buildAgeDist(ALL_AGE_DIST);
    buildStationBar(ALL_STATIONS);
    buildWalkLine(ALL_WALK_RENT);
    return;
  }}

  const fpCounts = {{}};
  const fpRents = {{}};
  const areaRents = {{"<20":[],"20-25":[],"25-30":[],"30-40":[],"40-50":[],"50+":[]}};
  const ageDist = {{"<5":0,"5-10":0,"11-20":0,"21-30":0,"31-40":0,"41-50":0,"50+":0}};
  const walkMap = {{}};

  sub.forEach(d => {{
    fpCounts[d.fp] = (fpCounts[d.fp]||0)+1;
    if (!fpRents[d.fp]) fpRents[d.fp] = [];
    fpRents[d.fp].push(d.rent);

    if (d.area < 20) areaRents["<20"].push(d.rent);
    else if (d.area < 25) areaRents["20-25"].push(d.rent);
    else if (d.area < 30) areaRents["25-30"].push(d.rent);
    else if (d.area < 40) areaRents["30-40"].push(d.rent);
    else if (d.area < 50) areaRents["40-50"].push(d.rent);
    else areaRents["50+"].push(d.rent);

    if (d.age < 5) ageDist["<5"]++;
    else if (d.age <= 10) ageDist["5-10"]++;
    else if (d.age <= 20) ageDist["11-20"]++;
    else if (d.age <= 30) ageDist["21-30"]++;
    else if (d.age <= 40) ageDist["31-40"]++;
    else if (d.age <= 50) ageDist["41-50"]++;
    else ageDist["50+"]++;

    if (!walkMap[d.walk]) walkMap[d.walk] = [];
    walkMap[d.walk].push(d.rent);
  }});

  const fpRentAvg = {{}};
  Object.keys(fpRents).forEach(f => {{ fpRentAvg[f] = mean(fpRents[f]); }});

  const areaRentAvg = {{}};
  Object.keys(areaRents).forEach(k => {{
    if (areaRents[k].length > 0) areaRentAvg[k] = mean(areaRents[k]);
  }});

  const stationMap = {{}};
  sub.forEach(d => {{ if(!stationMap[d.station]) stationMap[d.station]=[]; stationMap[d.station].push(d.rent); }});
  const stationArr = Object.entries(stationMap).map(([name,rents])=>{{return {{name,rent:mean(rents)}}}}).sort((a,b)=>b.rent-a.rent).slice(0,8);

  const walkKeys = Object.keys(walkMap).map(Number).sort((a,b)=>a-b);
  const walkRentAvg = {{}};
  walkKeys.forEach(k => {{ walkRentAvg[k] = mean(walkMap[k]); }});

  buildFPDonut(fpCounts);
  buildFPRentBar(fpRentAvg);
  buildAreaRentBar(Object.keys(areaRents).reduce((acc,k)=>{{ if(areaRents[k].length) acc[k]=mean(areaRents[k]); return acc; }},{{}}));
  buildAgeDist(ageDist);
  buildStationBar(stationArr.length ? stationArr : ALL_STATIONS);
  buildWalkLine(walkRentAvg);
}}

// ------------------------------------------------------------------
// Listings table
// ------------------------------------------------------------------

const PAGE_SIZE = 20;
let tablePage = 0;
let sortCol = "rent";
let sortDir = 1;
let tableRows = [];

const TABLE_COLS = [
  {{key:"title",   label:"Listing",   fmt:(v,d)=>d.url ? `<a href="${{d.url}}" target="_blank" class="listing-link">${{v||"—"}}</a>` : (v||"—")}},
  {{key:"address", label:"Address",   fmt:v=>v||"—"}},
  {{key:"rent",    label:"Rent",      fmt:v=>v!=null ? fmtYfull(v) : "—"}},
  {{key:"mgmt",    label:"Mgmt Fee",  fmt:v=>v!=null ? fmtYfull(v) : "—"}},
  {{key:"fp",      label:"Plan",      fmt:v=>v||"—"}},
  {{key:"area",    label:"Area",      fmt:v=>v!=null ? v.toFixed(1)+" m²" : "—"}},
  {{key:"floor",   label:"Floor",     fmt:v=>v!=null ? v : "—"}},
  {{key:"age",     label:"Bldg Age",  fmt:v=>v!=null ? v+" yrs" : "—"}},
  {{key:"station", label:"Station",   fmt:v=>v||"—"}},
  {{key:"walk",    label:"Walk",      fmt:v=>v!=null ? v+" min" : "—"}},
];

function renderTable(rows) {{
  tableRows = [...rows];
  tablePage = 0;
  sortAndRender();
}}

function sortAndRender() {{
  tableRows.sort((a, b) => {{
    const av = a[sortCol], bv = b[sortCol];
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string") return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  }});
  renderPage();
}}

function renderPage() {{
  const start = tablePage * PAGE_SIZE;
  const pageRows = tableRows.slice(start, start + PAGE_SIZE);
  const total = tableRows.length;

  document.getElementById("tbl-head").innerHTML = "<tr>" +
    TABLE_COLS.map(c => {{
      const active = c.key === sortCol;
      const arrow = active ? (sortDir === 1 ? " ▲" : " ▼") : "";
      return `<th class="tbl-th${{active ? " sorted" : ""}}" onclick="setSort('${{c.key}}')">${{c.label}}${{arrow}}</th>`;
    }}).join("") + "</tr>";

  document.getElementById("tbl-body").innerHTML = pageRows.map(d =>
    "<tr>" + TABLE_COLS.map(c => `<td class="tbl-td">${{c.fmt(d[c.key], d)}}</td>`).join("") + "</tr>"
  ).join("");

  document.getElementById("tbl-info").textContent = total
    ? `${{start + 1}}–${{Math.min(start + PAGE_SIZE, total)}} of ${{total.toLocaleString()}} listings`
    : "No listings match the current filters";

  document.getElementById("tbl-prev").disabled = tablePage === 0;
  document.getElementById("tbl-next").disabled = start + PAGE_SIZE >= total;
}}

function setSort(col) {{
  sortDir = sortCol === col ? sortDir * -1 : 1;
  sortCol = col;
  sortAndRender();
}}

function prevPage() {{ if (tablePage > 0) {{ tablePage--; renderPage(); }} }}
function nextPage() {{ if ((tablePage + 1) * PAGE_SIZE < tableRows.length) {{ tablePage++; renderPage(); }} }}

updateKPIs([], false);
updateCharts([], "all","all","all", false);
renderTable(RAW);
</script>
</body>
</html>"""


def read_stored_mtime() -> float | None:
    """Read the last-processed mtime from the state file. Returns None if not set."""
    if not STATE_FILE.exists():
        return None
    for line in STATE_FILE.read_text().splitlines():
        if line.startswith(f"{STATE_KEY}="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def write_stored_mtime(mtime: float) -> None:
    """Persist the mtime to the state file."""
    STATE_FILE.write_text(f"{STATE_KEY}={mtime}\n")


def build_dashboard() -> None:
    """Parse CSV, compute stats, write HTML."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Parsing {CSV_FILE.name}...")
    rows = parse_csv(CSV_FILE)
    if not rows:
        print("[ERROR] No valid rows parsed from CSV.")
        sys.exit(1)
    stats = compute_stats(rows)
    html = generate_html(rows, stats)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generated {OUTPUT_FILE.name} — {stats['total']:,} listings")


if __name__ == "__main__":
    if not CSV_FILE.exists():
        print(f"[ERROR] {CSV_FILE} not found.")
        sys.exit(1)

    current_mtime = CSV_FILE.stat().st_mtime
    stored_mtime = read_stored_mtime()

    if stored_mtime is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] No prior state found — running for the first time.")
        build_dashboard()
        write_stored_mtime(current_mtime)
    elif current_mtime > stored_mtime:
        prev = datetime.fromtimestamp(stored_mtime).strftime("%Y-%m-%d %H:%M:%S")
        curr = datetime.fromtimestamp(current_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] CSV updated ({prev} → {curr}) — regenerating dashboard.")
        build_dashboard()
        write_stored_mtime(current_mtime)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] CSV unchanged (mtime: {datetime.fromtimestamp(current_mtime).strftime('%Y-%m-%d %H:%M:%S')}) — nothing to do.")
