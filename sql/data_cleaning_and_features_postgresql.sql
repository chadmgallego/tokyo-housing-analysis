-- PostgreSQL version (used by pipeline/scraper.py via Supabase)
-- Drops and recreates the tokyo_housing view from housing_data_raw.
-- Run automatically by scraper.py — can also be run manually in Supabase SQL editor.

DROP VIEW IF EXISTS tokyo_housing;

CREATE VIEW tokyo_housing AS

-- Deduplicate listings sharing the same title, floor, floor plan, area, and rent.
-- PostgreSQL requires a subquery alias (here: sub) — unlike SQLite.
WITH deduplicated_listings AS (
    SELECT * FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY title, floor, floor_plan, area, rent
                ORDER BY url
            ) AS rn
        FROM housing_data_raw
    ) sub
    WHERE rn = 1
),

standardized_listings AS (
    SELECT
        url, title, address,

        -- SUUMO uses '-' as a null placeholder. REGEXP_REPLACE strips non-numeric chars
        -- before casting so PostgreSQL's strict CAST doesn't error on those values.
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

        -- CAST to NUMERIC required for ROUND() in PostgreSQL (unlike SQLite)
        ROUND(CAST(AVG(rent) OVER (PARTITION BY nearest_station) AS NUMERIC), 2)
            AS avg_rent_by_station,
        ROUND(CAST(AVG(rent) OVER (PARTITION BY floor_plan) AS NUMERIC), 2)
            AS avg_rent_by_floor_plan,

        COUNT(title) OVER (PARTITION BY nearest_station) AS count_listings_per_station,
        COUNT(title) OVER (PARTITION BY floor_plan)      AS count_listings_per_floor_plan
    FROM standardized_listings
)

SELECT * FROM featured_listings;
