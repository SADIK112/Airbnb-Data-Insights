# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRANSFORMED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# File names
LISTINGS_FILE = os.getenv("LISTINGS_FILE")
MODEL_FILE = os.getenv("MODEL_FILE")

# Model settings
RANDOM_STATE = os.getenv("RANDOM_STATE")
TEST_SIZE = os.getenv("TEST_SIZE")

# Geolocation Api Token
GEO_API_TOKEN = os.getenv("GEO_API_TOKEN")
# API settings
API_HOST = os.getenv("API_HOST")
API_PORT = os.getenv("API_PORT")