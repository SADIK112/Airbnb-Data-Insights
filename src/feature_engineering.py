"""
Feature Engineering module for Airbnb NYC data.

This module provides functionality to transform features
(make new meaningful feature from existing features) of Airbnb listing data.
It includes functions to transform categorial columns to numeric and make new columns from existing colum data
"""
# Import the DataProcessor class from the script
import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from config import PROCESSED_DATA_DIR, TRANSFORMED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler()
    ] 
)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
        A class to handle feature engineering Airbnb listing data
        Attributes:
            data (pd.DataFrame): The transformed dataset
            processed_data_path (str): Path to the processed data file
            transformed_data_path (str): Path to save transformed data
    """

    def __init__(self, input_path: str, output_path: str, n_clusters: int = 10):
        """ Initialize with input and output file paths """
        self.input_path = input_path
        self.output_path = output_path
        self.n_clusters = n_clusters
        self.df = None
        logger.info(f"Feature Engineering initialized with input: {input_path}, output: {output_path}")

    def load_data(self) -> None:
        """ Load the Airbnb processed/cleaned dataset from a csv file. """
        try:
            logger.info(f"Loading cleaned data from {self.input_path}")
            self.df = pd.read_csv(self.input_path, low_memory = False)
            logger.info(f"Successfully loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        except FileNotFoundError:
            logger.error(f"File not found: {self.input_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_name_length(self) -> None:
        """Create name_length feature and drop name column."""
        logger.info("Creating name_length feature")
        self.df["name_length"] = self.df["name"].fillna("").apply(len)
        self.df.drop(columns=["name"], inplace=True)
        logger.info("name_length created and name column dropped")
    
    def encode_categorical_to_binary(self) -> None:
        """Convert instant_bookable and host_identity_verified to binary (1 and 0)."""
        self.df["host_identity_verified"] = self.df["host_identity_verified"].map({"verified": 1, "unconfirmed": 0})
        logger.info("host_identity_verified encoded")
        self.df["instant_bookable"] = self.df["instant_bookable"].map({True: 1, False: 0})
        logger.info("instant_bookable encoded")
    
    def perform_one_hot_encoding(self) -> None:
        """ Convert categorial feature to numeric """
        self.df = pd.get_dummies(self.df, columns=["cancellation_policy"], prefix="policy", dtype=int)
        logger.info("cancellation_policy one-hot encoded")
        self.df = pd.get_dummies(self.df, columns=["room_type"], prefix="", prefix_sep="", dtype=int)
        logger.info("room_type one-hot encoded")
        self.df = pd.get_dummies(self.df, columns=["neighbourhood_group"], prefix="neighbourhood_group", dtype=int)
        logger.info("neighbourhood_group one-hot encoded")

    def target_encode_neighbourhood(self) -> None:
        """Create neigh_encoded feature with mean price per neighbourhood."""
        logger.info("Target encoding neighbourhood")
        self.df["neigh_encoded"] = self.df.groupby("neighbourhood")["price"].transform("mean")
        logger.info("neigh_encoded feature created")

    def cluster_locations(self) -> None:
        """Group listings into clusters based on coordinates."""
        logger.info("Clustering locations with KMeans")
        coords = self.df[["lat", "long"]]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.df["location_cluster"] = kmeans.fit_predict(coords)
        logger.info("location_cluster feature created")

    def extract_temporal_features(self) -> None:
        """Extract temporal features from last_review."""
        logger.info("Extracting temporal features from last_review")
        self.df["last_review"] = pd.to_datetime(self.df["last_review"], errors="coerce")
        self.df["days_since_last_review"] = (pd.Timestamp.today() - self.df["last_review"]).dt.days
        self.df["days_since_last_review"] = self.df["days_since_last_review"].fillna(-1)
        self.df["last_review_year"] = self.df["last_review"].dt.year.fillna(-1)
        self.df["last_review_month"] = self.df["last_review"].dt.month.fillna(-1)
        self.df["last_review_dayofweek"] = self.df["last_review"].dt.dayofweek.fillna(-1)
        logger.info("Temporal features extracted")

    def create_domain_features(self) -> None:
        """Create domain-specific features."""
        logger.info("Creating domain-specific features")
        self.df["availability_ratio"] = self.df["availability_365"] / 365
        self.df["price_per_min_stay"] = self.df["price"] / self.df["minimum_nights"].replace(0, 1)
        self.df["property_age"] = 2025 - self.df["construction_year"]
        self.df["property_age"] = self.df["property_age"].fillna(-1)
        self.df["has_house_rules"] = self.df["house_rules"].notna().astype(int)
        self.df["has_license"] = self.df["license"].notna().astype(int)
        self.df["total_cost"] = self.df["price"] + self.df["service_fee"]
        self.df["popularity_score"] = self.df["review_rate_number"] * self.df["number_of_reviews"]
        logger.info("Domain-specific features created")

    def create_advanced_features(self) -> None:
        """Create advanced features."""
        logger.info("Creating advanced features")
        self.df["price_relative_to_neighbourhood"] = self.df.groupby("neighbourhood")["price"].transform(lambda pr: pr / pr.median())
        self.df["avg_reviews_per_listing"] = self.df["number_of_reviews"] / (self.df["calculated_host_listings_count"] + 1)
        logger.info("Advanced features created")

    def drop_columns(self) -> None:
        """Drop unnecessary columns."""
        logger.info("Dropping unnecessary columns")
        columns_to_drop = ["id", "host_id", "neighbourhood", "host_name", "lat", "long", "country", "country_code", "house_rules", "license"]
        self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], inplace=True)
        logger.info("Unnecessary columns dropped")

    def save_data(self) -> None:
        """Save the transformed DataFrame to a CSV file."""
        try:
            logger.info(f"Saving transformed data to {self.output_path}")
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def engineer_features(self) -> pd.DataFrame:
        """Execute all feature engineering steps and return the transformed DataFrame."""
        logger.info("Starting feature engineering")
        self.load_data()
        self.create_name_length()
        self.encode_categorical_to_binary()
        self.perform_one_hot_encoding()
        self.target_encode_neighbourhood()
        self.cluster_locations()
        self.extract_temporal_features()
        self.create_domain_features()
        self.create_advanced_features()
        self.drop_columns()
        self.save_data()
        logger.info("Feature engineering completed")
        return self.df

if __name__ == "__main__":
    input_file = f"{PROCESSED_DATA_DIR}/airbnb_listings.csv"
    output_file = f"{TRANSFORMED_DATA_DIR}/airbnb_listings.csv"
    processor = FeatureEngineering(input_file, output_file, n_clusters=10)
    transformed_df = processor.engineer_features()