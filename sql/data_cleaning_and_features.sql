
-- Remove the view if it already exists
DROP VIEW IF EXISTS TOKYO_HOUSING;

-- Create a cleaned + feature-engineered housing view
CREATE VIEW TOKYO_HOUSING AS

-- Deduplicate listings that appear multiple times due to scraping artifacts.
-- Listings are considered duplicates if they share the same title, address,
-- rent, floor plan, and floor.

-- ROW_NUMBER() is used to retain a single representative row per duplicate group.
-- Ordering by `img` provides a stable (though not necessarily unique) tie-breaker
WITH DEDUPLICATED_LISTINGS AS (
    SELECT * 
    FROM  (
        SELECT 
            *,
            ROW_NUMBER() 
                OVER (
                    PARTITION BY title, address, rent, floor_plan, floor
                    ORDER BY img
                )
                AS rn
        FROM HOUSING_DATA 
        )
    WHERE rn = 1
),

STANDARDIZED_LISTINGS AS (
    SELECT 
        -- Basic identifiers
        img, title, address, 
        
        -- Convert rent/deposit/key money into numeric
        CAST(RTRIM(rent, '万円') AS FLOAT) * 10000 AS rent,
        CAST(RTRIM(management_fee, '円') AS FLOAT) AS management_fee,
        CAST(RTRIM(deposit, '万円') AS FLOAT) * 10000 AS deposit,
        CAST(RTRIM(key_money, '万円') AS FLOAT) * 10000 AS key_money,
        
        -- Remove floor label
        RTRIM(floor, '階') AS floor,
        
        -- Normalize floor plan categories 
        CASE
            WHEN floor_plan = 'ワンルーム' THEN '1R'
            ELSE floor_plan
        END AS floor_plan,
        
        -- Convert area to numeric (square meters)
        CAST(RTRIM(area, 'm2') AS FLOAT) AS area,
        
        -- Extract building age in years
        CAST(LTRIM(RTRIM(building_age, '年'), '築') AS INTEGER) AS building_age,
        
        -- Remove building size label 
        RTRIM(building_size, '階建') AS building_size,

        -- Station-related features
        stations,
        nearest_station,
        distance_to_nearest_station,
        ROUND(avg_distance_to_stations, 2) AS avg_distance_to_stations
    FROM DEDUPLICATED_LISTINGS
),

FEATURED_LISTINGS AS (
    SELECT 
        img, title, address, rent, 
        
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





