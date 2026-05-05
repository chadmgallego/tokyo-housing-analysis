
-- Remove the view if it already exists
DROP VIEW IF EXISTS TOKYO_HOUSING;

-- Create a cleaned + feature-engineered housing view
CREATE VIEW TOKYO_HOUSING AS

-- Deduplicate listings that appear multiple times due to scraping artifacts.
-- Listings are considered duplicates if they share the same title, address,
-- rent, floor plan, and floor.

-- ROW_NUMBER() is used to retain a single representative row per duplicate group.
-- Ordering by `url` provides a stable and unique tie-breaker
WITH DEDUPLICATED_LISTINGS AS (
    SELECT * 
    FROM  (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY title, floor, floor_plan, area, rent
                ORDER BY url
                ) 
                AS rn
        FROM HOUSING_DATA_RAW
    )
    WHERE rn = 1
),

STANDARDIZED_LISTINGS AS (
    SELECT 
        url, title, address, 
        
        CAST(REPLACE(rent, '万円', '') AS FLOAT) * 10000 AS rent,
        CAST(REPLACE(management_fee, '円', '') AS FLOAT) AS management_fee,
        CAST(REPLACE(deposit, '万円', '') AS FLOAT) * 10000 AS deposit,
        CAST(REPLACE(key_money, '万円', '') AS FLOAT) * 10000 AS key_money,
        
        RTRIM(floor, '階') AS floor,
        
        CASE
            WHEN floor_plan = 'ワンルーム' THEN '1R'
            ELSE floor_plan
        END AS floor_plan,
        
        CAST(REPLACE(area, 'm2', '') AS FLOAT) AS area,
        
        CASE 
            WHEN building_age LIKE '%新築%' THEN 0
            WHEN building_age LIKE '%以上'
                THEN CAST(REPLACE(REPLACE(building_age, '築', ''), '年以上', '') AS INTEGER)
            ELSE CAST(REPLACE(REPLACE(building_age, '築', ''), '年', '') AS INTEGER)
        END AS building_age,
        
        CASE
            WHEN building_size LIKE '%平屋%' THEN '1'
            ELSE RTRIM(building_size, '階建')
        END AS building_size,
        
        stations,
        nearest_station,
        distance_to_nearest_station,
        ROUND(avg_distance_to_stations, 2) AS avg_distance_to_stations
    FROM DEDUPLICATED_LISTINGS
),

FEATURED_LISTINGS AS (
    SELECT 
        url, title, address, rent, 
        
        -- Replace 0 values with NULLs
        NULLIF(management_fee, 0.0) AS management_fee,
        NULLIF(deposit, 0.0) AS deposit,
        NULLIF(key_money, 0.0) AS key_money,
        floor, floor_plan, area, building_age,
        building_size, nearest_station,
        distance_to_nearest_station, avg_distance_to_stations,
        
        -- Average rent by station, floor plan
        ROUND(AVG(rent) 
            OVER (PARTITION BY nearest_station), 2) 
            AS avg_rent_by_station, 
        ROUND(AVG(rent)
            OVER (PARTITION BY floor_plan), 2) 
            AS avg_rent_by_floor_plan,
        
        -- Number of listings per station, floor plan
        COUNT(title)
            OVER (PARTITION BY nearest_station)
            AS count_listings_per_station,
        COUNT(title)
            OVER (PARTITION BY floor_plan) 
            AS count_listings_per_floor_plan
    FROM STANDARDIZED_LISTINGS
)

-- Final output 
SELECT * FROM FEATURED_LISTINGS




