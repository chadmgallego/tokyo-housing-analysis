#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests, sqlite3, sys, re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


# ## Data Collection Class: `TokyoHousingScraper`
# > This class handles all **data ingestion** tasks, including:
# > - Initializing local SQLite database connection
# > - Scraping Tokyo listings HTML from *SUUMO.jp*
# > - Parsing station information and other housing metrics from raw HTML
# > - Building a robust housing dataset which includes features such as: `title`, `floor`, `area`, `rent`, `deposit`, etc.
# > - Storing the resulting dataset in local SQLite table

# In[ ]:


# Define TokyoHousingScraper to:
# - Scrape Tokyo housing listings from Suumo.jp
# - Collect listings, parse property details
# - Store the results in SQLite database

class TokyoHousingScraper:
    def __init__(self, db, base_url, url):
        # Initialize DB connection
        self.db = db
        self.conn = sqlite3.connect(self.db)
        self.cursor = self.conn.cursor()

        # Base URL and starting page
        self.base_url = base_url
        self.url = url

    def scrape_listings(self):
        # Define list for storing HTML
        self.listings = list()

        # Iterate through all pages of listings
        while True:
            try:
                response = requests.get(next_page) #this will only work after the first page
            except:
                response = requests.get(self.url) # starting url 
            soup = BeautifulSoup(response.text, 'lxml')

            # Each listing = cassetteitem div
            cassettes = soup.select('div.cassetteitem')
            self.listings.extend(cassettes)
            #print(len(self.listings))

            # Find next page link (pagination)
            try:
                current_page = soup.find('li', class_ = 'pagination-current')
                next_page_path = current_page.find_next_siblings('li')[1]
                next_page = self.base_url + next_page_path.select_one('a').get('href')
            except: break #no more pages to comb through

        #print(f'{len(self.listings)} listings were successfully gathered!')

    def parse_station_info(self, item):

        # Extract station information (names, distances, nearest, average).
        # Returns tuple: 
            # (stations_str, nearest_station, distance_to_nearest_station, avg_distance).

        # Get raw station blocks
        stations_list = item.select('li.cassetteitem_detail-col2 div.cassetteitem_detail-text')

        # If there is no station information, return None
        if not stations_list:
            return (None, None, None, None)
        else: pass

        # Remove empty tags
        stations_list = [s for s in stations_list if s != '']

        # All stations as a single string (for DB storage)
        self.stations_str = ",".join([station.get_text().strip() for station in stations_list])

        # Extract stations and distances with regex
        stations_dict = {
            # All listed stations
            'stations': [
                re.findall(r'/(?P<station>.*?)\s*歩', station.get_text().strip())[0] 
                for station in stations_list
                if re.findall(r'/(?P<station>.*?)\s*歩', station.get_text().strip())
            ],

            'distances': [
                re.findall(r'\d+', station.get_text().strip())[0]
                for station in stations_list
                if re.findall(r'\d+', station.get_text().strip())
            ]
        }

        # Compute distance to nearest station
        self.distance_to_nearest_station = min([int(dist) for dist in stations_dict['distances']])
        nearest_idx = stations_dict['distances'].index(str(self.distance_to_nearest_station))
        self.nearest_station = stations_dict['stations'][nearest_idx]

        # Compute average distance to surrounding stations
        self.avg_distance = np.mean([float(dist) for dist in stations_dict['distances']])

        return self.stations_str, self.nearest_station, self.distance_to_nearest_station, self.avg_distance

    def build_housing_dataset(self):

        # Parse all previously scraped listings into a structured dataset
        # and load the results to SQLite.

        # First pass: collect all relevant BeautifulSoup tags for each listing.
        # We separate "tag extraction" from "text/value extraction" so:
        #   1) Each CSS selector runs only once per listing
        #   2) Missing fields can be handled consistently later
        # select_one() returns a single value
        # select() returns a list (empty if no tags exist)
        listing_tags = [
            {
                # Core listing info
                'img_tag': item.select_one('div.cassetteitem_object img'),
                'title_tag': item.select_one('div.cassetteitem_content-title'),
                'address_tag': item.select_one('li.cassetteitem_detail-col1'),

                # Pricing information
                'rent_tag': item.select_one('span.cassetteitem_price.cassetteitem_price--rent'),
                'management_fee_tag': item.select_one('span.cassetteitem_price.cassetteitem_price--administration'),
                'deposit_tag': item.select_one('span.cassetteitem_price.cassetteitem_price--deposit'),
                'key_money_tag': item.select_one('span.cassetteitem_price.cassetteitem_price--gratuity'),

                # Property details
                'floor_cells': item.select('div.cassetteitem-item tr.js-cassette_link td'),
                'floor_plan_tag': item.select_one('span.cassetteitem_madori'),
                'area_tag': item.select_one('span.cassetteitem_menseki'),
                'building_cells': item.select('li.cassetteitem_detail-col3 div'),

                # Pre-computed station information
                # (returns tuple: stations_str, nearest_station, nearest_dist, avg_dist)              
                'stations_info': self.parse_station_info(item)
            }
            for item in self.listings
        ]

        # Second pass: convert tags into clean, normalized Python values
        # All conditional checks rely on truthiness:
        #   - BeautifulSoup tag → truthy
        #   - None or empty list → falsy
        housing_data = [
            {
                # Image URL (some listings omit images)
                'img': listing['img_tag'].get('rel') if listing['img_tag'] else None,

                # Text-based fields
                'title': listing['title_tag'].get_text().strip() if listing['title_tag'] else None,
                'address': listing['address_tag'].get_text().strip() if listing['address_tag'] else None,

                # Pricing fields
                'rent': listing['rent_tag'].get_text().strip() if listing['rent_tag'] else None,
                'management_fee': listing['management_fee_tag'].get_text().strip() if listing['management_fee_tag'] else None,
                'deposit': listing['deposit_tag'].get_text().strip() if listing['deposit_tag'] else None,
                'key_money': listing['key_money_tag'].get_text().strip() if listing['key_money_tag'] else None,

                # Floor information is stored in a fixed table layout:
                # index 2 corresponds to the floor number when present
                'floor': (
                    listing['floor_cells'][2].get_text().strip() 
                    if len(listing['floor_cells']) >= 3 
                    else None
                ),

                'floor_plan': listing['floor_plan_tag'].get_text().strip() if listing['floor_plan_tag'] else None,
                'area': listing['area_tag'].get_text().strip() if listing['area_tag'] else None,

                # Building metadata
                'building_age': listing['building_cells'][0].get_text().strip() if listing['building_cells'] else None,
                'building_size': (
                    listing['building_cells'][1].get_text().strip() 
                    if len(listing['building_cells']) >= 2 
                    else None
                ),

                # Station-related features (already parsed and computed)
                'stations': listing['stations_info'][0],
                'nearest_station': listing['stations_info'][1],
                'distance_to_nearest_station': listing['stations_info'][2],
                'avg_distance_to_stations': listing['stations_info'][3]
            }
            for listing in listing_tags
        ]

        # Convert to DataFrame and load to SQLite
        housing_data_df = pd.DataFrame(housing_data)
        housing_data_df.to_sql(name = 'HOUSING_DATA', con = self.conn, if_exists = 'replace', index = False)

        # Close the DB connection
        self.conn.close()

