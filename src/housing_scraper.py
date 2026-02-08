#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[80]:


# Define TokyoHousingScraper to:
# - Scrape Tokyo housing listings from SUUMO.jp
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

        print(f'{len(self.listings)} properties were successfully gathered!')

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

    def parse_sublistings(self, sub):
        # Parse a single sublisting HTML element and extract key property details.

        # Parameters 
            # sub : bs4.element.Tag
            # A BeautifulSoup <tr> or sublisting element containing rent, fees, floor, 
            # floor plan, and area information.

        # Returns
            # A pandas Series with the following fields:
            # - url: full URL for the sublisting
            # - rent: rent amount (string)
            # - management_fee: management fee (string)
            # - deposit: deposit amount (string)
            # - key_money: key money / deposit (string)
            # - floor: floor number or description (string)
            # - floor_plan: floor plan (string, e.g., 1R, 2LDK)
            # - area: area in square meters (string)

        sublisting_tags = {
            # Get URL tags for each sublisting
            'url_tag': sub.select_one('td.ui-text--midium.ui-text--bold a'),

            # Pricing fields
            'rent_tag': sub.select_one('span.cassetteitem_price.cassetteitem_price--rent'),
            'management_fee_tag': sub.select_one('span.cassetteitem_price.cassetteitem_price--administration'),
            'deposit_tag': sub.select_one('span.cassetteitem_price.cassetteitem_price--deposit'),
            'key_money_tag': sub.select_one('span.cassetteitem_price.cassetteitem_price--gratuity'),

            # Property details
            'floor_cells': sub.select('tr.js-cassette_link td'),
            'floor_plan_tag': sub.select_one('span.cassetteitem_madori'),
            'area_tag': sub.select_one('span.cassetteitem_menseki')
        }

        sublisting_data = {
            # Combine base_url with endpoints for complete URLs
            'url': self.base_url[:-1] + sublisting_tags['url_tag'].get('href') if sublisting_tags['url_tag'] else None,

            # Pricing fields
            'rent': sublisting_tags['rent_tag'].get_text().strip() if sublisting_tags['rent_tag'] else None,
            'management_fee': sublisting_tags['management_fee_tag'].get_text().strip() if sublisting_tags['management_fee_tag'] else None,
            'deposit': sublisting_tags['deposit_tag'].get_text().strip() if sublisting_tags['deposit_tag'] else None,
            'key_money': sublisting_tags['key_money_tag'].get_text().strip() if sublisting_tags['key_money_tag'] else None,

            # Floor information is in the third <td> cell of the sublisting row.
            # If fewer than 3 cells exist, set to None
            'floor': (
                sublisting_tags['floor_cells'][2].get_text().strip() 
                if len(sublisting_tags['floor_cells']) >= 3 
                else None
            ),

            'floor_plan': sublisting_tags['floor_plan_tag'].get_text().strip() if sublisting_tags['floor_plan_tag'] else None,
            'area': sublisting_tags['area_tag'].get_text().strip() if sublisting_tags['area_tag'] else None
        }

        # Return pandas DataFrame for added columns
        return pd.Series(sublisting_data)

    def build_housing_dataset(self):

        # Parse all previously scraped listings into a structured dataset
        # and load the results to SQLite.

        # First pass: collect all relevant BeautifulSoup tags for each listing.
        # We separate "tag extraction" from "text/value extraction" so:
        #   1) Each CSS selector runs only once per listing
        #   2) Missing fields can be handled consistently later
        # select_one() returns a single value
        # select() returns a list (empty if no tags exist)
        building_tags = [
            {
                # Core building info
                'title_tag': item.select_one('div.cassetteitem_content-title'),
                'address_tag': item.select_one('li.cassetteitem_detail-col1'),
                'building_cells': item.select('li.cassetteitem_detail-col3 div'),

                # Pre-computed station information
                # (returns tuple: stations_str, nearest_station, nearest_dist, avg_dist)              
                'stations_info': self.parse_station_info(item),

                # Building sublistings
                'sublisting_tags': item.select('tr.js-cassette_link')
            }
            for item in self.listings
        ]

        # Second pass: convert tags into clean, normalized Python values
        # All conditional checks rely on truthiness:
        #   - BeautifulSoup tag → truthy
        #   - None or empty list → falsy
        building_data = [
            {
                # Text-based fields
                'title': building['title_tag'].get_text().strip() if building['title_tag'] else None,
                'address': building['address_tag'].get_text().strip() if building['address_tag'] else None,

                # Building metadata
                'building_age': building['building_cells'][0].get_text().strip() if building['building_cells'] else None,
                'building_size': (
                    building['building_cells'][1].get_text().strip() 
                    if len(building['building_cells']) >= 2 
                    else None
                ),

                # Station-related features (already parsed and computed)
                'stations': building['stations_info'][0],
                'nearest_station': building['stations_info'][1],
                'distance_to_nearest_station': building['stations_info'][2],
                'avg_distance_to_stations': building['stations_info'][3],

                # Building sublistings (which will be parsed with pandas)
                'sublistings': building['sublisting_tags']
            }
            for building in building_tags
        ]

        # Convert to DataFrame
        housing_data_df = pd.DataFrame(building_data)

        # Explode sublistings into unique rows
        housing_data_df = housing_data_df.explode('sublistings')

        # Parse sublistings and retrieve listing-level metrics
        housing_data_df[['url', 'rent', 'management_fee', 'deposit', 'key_money', 'floor', 'floor_plan', 'area']] = (
            housing_data_df['sublistings'].apply(self.parse_sublistings)
        )

        # Drop sublistings column after parsing 
        housing_data_df.drop(columns = ['sublistings'], inplace = True)

        # Load raw, structured housing data to SQLite table
        housing_data_df.to_sql(name = 'HOUSING_DATA_RAW', con = self.conn, if_exists = 'replace', index = False)

        # Close the DB connection
        self.conn.close()

