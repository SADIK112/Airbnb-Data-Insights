"""
Data processing module for Airbnb NYC data.

This module provides functionality to load, clean, and preprocess Airbnb listing data.
It includes functions for handling missing values, converting data types, and preparing
the dataset for feature engineering and modeling.
"""
# Import the DataProcessor class from the script
import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import logging
from pathlib import Path
import requests
import time
from config import GEO_API_TOKEN, RAW_DATA_DIR, PROCESSED_DATA_DIR
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class to handle loading and preprocessing of Airbnb listing data.
    
    Attributes:
        data (pd.DataFrame): The processed dataset
        raw_data_path (str): Path to the raw data file
        processed_data_path (str): Path to save processed data
    """

    def __init__(self, input_path: str, output_path: str, geo_api_token: str):
        """ Initialize with input and output file paths. """
        self.input_path = input_path
        self.output_path = output_path
        self.geo_api_token = geo_api_token
        self.df = None
        logger.info("Data Processor initialized with input: %s, output: %s", input_path, output_path)

    def load_data(self) -> None:
        """ Load the airbnb dataset from a csv file. """
        try:
            logger.info(f"Loading data from {self.input_path}")
            self.df = pd.read_csv(self.input_path, low_memory = False)
            logger.info(f"Successfully loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        except FileNotFoundError:
            logger.error(f"File not found: {self.input_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def rename_columns(self) -> None:
        """ Rename columns for consistancy. """
        logger.info("Renaiming Columns")
        cols = {
            "NAME": "name",
            "host id": "host_id",
            "host name": "host_name",
            "neighbourhood group": "neighbourhood_group",
            "country code": "country_code",
            "room type": "room_type",
            "Construction year": "construction_year",
            "service fee": "service_fee",
            "minimum nights": "minimum_nights",
            "number of reviews": "number_of_reviews",
            "last review": "last_review",
            "reviews per month": "reviews_per_month",
            "review rate number": "review_rate_number",
            "calculated host listings count": "calculated_host_listings_count",
            "availability 365": "availability_365",
        }
        self.df = self.df.rename(columns=cols)
        logger.info("Columns renamed successfully")

    def handle_duplicates_and_dates(self) -> None:
        """ Convert last_review to date, drop null ids and handle duplicates """
        logger.info("Handling duplicates and converting date formats")
        self.df["last_review"] = pd.to_datetime(self.df["last_review"], errors = "coerce").dt.date
        self.df["construction_year"] = pd.to_datetime(self.df["construction_year"], format='%Y', errors='coerce').dt.year

        self.df = (
            self.df.dropna(subset=["id"])
            .sort_values(["id", "last_review"], ascending=[True, False])
            .groupby("id", as_index=False)
            .first()
        )
        logger.info(f"After deduplication, {self.df.shape[0]} rows remain")

    def clean_text_columns(self) -> None:
        """ Clean text columns: name, host_identity_verified, host_name and country name"""
        logger.info("Cleaning text columns")
        self.df["name"] = self.df["name"].fillna("N/A").replace("#NAME", "N/A").str.strip()
        self.df["host_identity_verified"] = self.df["host_identity_verified"].fillna("unconfirmed").str.strip()
        self.df["host_name"] = self.df["host_name"].fillna("N/A").str.strip()
        self.df["neighbourhood_group"] = self.df["neighbourhood_group"].str.strip()
        self.df["neighbourhood"] = self.df["neighbourhood"].str.strip()
        # Update the null country name as we only dealing with one country data
        self.df.loc[self.df["country"].isna(), "country"] = "United States"
        self.df["country"] = self.df["country"].str.strip()
        # Since there is only one country which is United States, we will assign same for null countries code
        self.df.loc[self.df["country_code"].isna(), "country_code"] = "US"
        self.df["country_code"] = self.df["country_code"].str.strip()
        logger.info("Text columns cleaned")

    def get_coordinates_info(self, latitude: str, longitude: str) -> pd.Series:
        """Fetch neighborhood information from LocationIQ API using coordinates."""
        url = f"https://us1.locationiq.com/v1/reverse?key={self.geo_api_token}&lat={latitude}&lon={longitude}&format=json"
        try:
            headers = {"accept": "application/json"}
            response = requests.get(url, headers = headers)
            if response.status_code == 200:
                data = response.json()
                neigh = data.get("address", {}).get("neighbourhood")
                neigh_group = data.get("address", {}).get("suburb")
                return pd.Series([neigh_group, neigh])
            else:
                logger.warning(f"API request failed for lat={latitude}, lon={longitude}: Status {response.status_code}")
                return pd.Series([None, None])
        except Exception as e:
            logger.error(f"Error in API request for lat={latitude}, lon={longitude}: {str(e)}")
            return pd.Series([None, None])
        
    def fill_missing_neighbourhoods(self) -> None:
        """Fill missing neighbourhood_group and neighbourhood using LocationIQ API."""
        logger.info("Filling missing neighborhood data using LocationIQ API")
        null_neighbourhood_df = self.df[
            (self.df["neighbourhood_group"].isna()) | (self.df["neighbourhood"].isna())
        ][["lat", "long"]].drop_duplicates()

        results = []

        for _, row in null_neighbourhood_df.iterrows():
            _lat = row["lat"]
            _long = row["long"]
            result = self.get_coordinates_info(_lat, _long)
            results.append(result.to_list())
            time.sleep(1)

        # combine the result
        neigh_df = pd.DataFrame(results, columns = ["neighbourhood_group", "neighbourhood"])
        null_neighbourhood_df = null_neighbourhood_df.reset_index(drop = True)
        null_neighbourhood_df[["neighbourhood_group", "neighbourhood"]] = neigh_df
        
        for _, row in null_neighbourhood_df.iterrows():
            _lat = row['lat']
            _long = row['long']
            new_group = row['neighbourhood_group']
            new_neigh = row['neighbourhood']

            # Find the index of matching row in airbnb_df
            mask = (self.df['lat'] == _lat) & (self.df['long'] == _long)

            # Only fill nulls
            if new_group and self.df.loc[mask, 'neighbourhood_group'].isna().any():
                self.df.loc[mask, 'neighbourhood_group'] = new_group

            if new_neigh and self.df.loc[mask, 'neighbourhood'].isna().any():
                self.df.loc[mask, 'neighbourhood'] = new_neigh

        logger.info("Missing neighborhoods filled")

    def correct_rows(self) -> None:
        """Fix the wrongly named rows"""

        replace_neighgroup_name = {"brookln": "brooklyn", "manhatan": "manhattan"}
        self.df["neighbourhood_group"] = (
            self.df["neighbourhood_group"]
            .str.lower()
            .replace(replace_neighgroup_name)
            .str.title()
    )

    def fill_missing_coordinates(self) -> None:
        """Fill missing lat/long with median values per neighbourhood."""
        logger.info("Filling missing latitude and longitude values")
        # Calculate median lat/long per neighbourhood
        median_coords = self.df.groupby('neighbourhood')[['lat', 'long']].median()
        
        def fill_coords(row):
            if pd.isna(row['lat']) or pd.isna(row['long']):
                if row['neighbourhood'] in median_coords.index:
                    row['lat'] = median_coords.loc[row['neighbourhood'], 'lat']
                    row['long'] = median_coords.loc[row['neighbourhood'], 'long']
            return row
        
        self.df = self.df.apply(fill_coords, axis=1)
        logger.info("Missing coordinates filled with neighbourhood medians")

    def fill_missing_instant_bookable(self) -> None:
        """Fill missing instant_bookable based on most frequent value per host_id."""
        logger.info("Filling missing instant_bookable values")
        host_group = (
            self.df.groupby(["host_id", "instant_bookable"])
            .size()
            .reset_index(name="count")
            .sort_values(["host_id", "count"], ascending=[True, False])
        )
        host_group["rank"] = (
            host_group.groupby("host_id")["count"]
            .rank(method="first", ascending=False).astype(int)
        )
        host_group = host_group[host_group["rank"] == 1].set_index("host_id")

        def fill_instant_bookable(row):
            if pd.isna(row['instant_bookable']):
                host_id = row['host_id']
                if host_id in host_group.index:
                    row['instant_bookable'] = host_group.loc[host_id, 'instant_bookable']
                else:
                    row['instant_bookable'] = False
            return row

        self.df = self.df.apply(fill_instant_bookable, axis=1)

        logger.info("Missing instant_bookable values filled")

    def fill_missing_cancellation_policy(self) -> None:
        """Fill missing cancellation_policy based on most frequent value per host_id."""
        logger.info("Filling missing cancellation_policy values")
        host_group_cancel_policy = (
            self.df.groupby(["host_id", "cancellation_policy"])
            .size()
            .reset_index(name="count")
            .sort_values(["host_id", "count"], ascending=[True, False])
        )
        host_group_cancel_policy["rank"] = (
            host_group_cancel_policy.groupby("host_id")["count"]
            .rank(method="first", ascending=False).astype(int)
        )
        host_group_cancel_policy = host_group_cancel_policy[
            host_group_cancel_policy["rank"] == 1
        ].set_index("host_id")

        def fill_cancel_policy(row):
            if pd.isna(row['cancellation_policy']):
                host_id = row['host_id']
                if host_id in host_group_cancel_policy.index:
                    row['cancellation_policy'] = host_group_cancel_policy.loc[host_id, 'cancellation_policy']
                else:
                    row['cancellation_policy'] = "moderate"
            return row

        self.df = self.df.apply(fill_cancel_policy, axis=1)
        self.df["cancellation_policy"] = self.df["cancellation_policy"].str.strip()

        logger.info("Missing cancellation_policy values filled")

    
    def convert_categorical_columns(self) -> None:
        """Convert specified columns to categorical type."""
        logger.info("Converting columns to categorical type")
        categorical_columns = ["instant_bookable", "cancellation_policy"]
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")
        
        logger.info("Categorical columns converted")


    def clean_numeric_columns(self) -> None:
        """Clean price and service_fee, impute missing values, and handle outliers."""
        logger.info("Cleaning numeric columns")
        # Clean price and service_fee
        for col in ["price", "service_fee"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(r'[\$,]', '', regex=True).astype(float)
        
        # Impute missing values
        numeric_columns = [
            "price", "service_fee", "minimum_nights", "number_of_reviews",
            "reviews_per_month", "review_rate_number", "calculated_host_listings_count",
            "availability_365"
        ]
        for col in numeric_columns:
            if col in self.df.columns:
                if col in ["price", "service_fee"]:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif col in ["calculated_host_listings_count"]:
                    self.df[col] = self.df[col].fillna(1)
                else:
                    self.df[col] = self.df[col].fillna(0)
        
        # Handle outliers
        self.df.loc[self.df["minimum_nights"] < 0, "minimum_nights"] = 0
        self.df.loc[self.df["minimum_nights"] > 365, "minimum_nights"] = 365
        self.df.loc[self.df["availability_365"] < 0, "availability_365"] = 0
        self.df.loc[self.df["availability_365"] > 365, "availability_365"] = 365

        logger.info("Numeric columns cleaned and outliers handled")

    
    def drop_and_fill_columns(self) -> None:
        """fill NaNs in house_rules and license """
        logger.info("Dropping license column and filling house_rules")

        self.df["license"] = self.df["license"].fillna("N/A")
        self.df["house_rules"] = self.df["house_rules"].fillna("N/A")

        logger.info("License and house_rules filled")

    def save_data(self) -> None:
        """Save the cleaned DataFrame to a CSV file."""
        try:
            logger.info(f"Saving cleaned data to {self.output_path}")
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def preprocess(self) -> pd.DataFrame:
        """Execute all preprocessing steps and return the cleaned DataFrame."""
        logger.info("Starting data preprocessing")
        self.load_data()
        self.rename_columns()
        self.handle_duplicates_and_dates()
        self.clean_text_columns()
        self.fill_missing_neighbourhoods()
        self.correct_rows()
        self.fill_missing_coordinates()
        self.fill_missing_instant_bookable()
        self.fill_missing_cancellation_policy()
        self.convert_categorical_columns()
        self.clean_numeric_columns()
        self.drop_and_fill_columns()
        self.save_data()
        logger.info("Data preprocessing completed")
        return self.df
    
if __name__ == "__main__":
    input_file = f"{RAW_DATA_DIR}/airbnb_listings.csv"
    output_file = f"{PROCESSED_DATA_DIR}/airbnb_listings.csv"
    geo_api_token = GEO_API_TOKEN
    processor = DataProcessor(input_file, output_file, geo_api_token)
    cleaned_df = processor.preprocess()