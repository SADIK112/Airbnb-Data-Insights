import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TRANSFORMED_DATA_DIR, MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle model training, evaluation, and saving for Airbnb price prediction."""

    def __init__(
        self,
        input_path: str,
        model_path: str,
        plot_dir: str = "../plots",
        target_col: str = "price",
    ):
        """Initialize with input file path, model save path, plot directory, and target column."""
        self.input_path = input_path
        self.model_path = model_path
        self.plot_dir = Path(plot_dir)
        self.target_col = target_col
        self.df = None
        self.model_features = [
            "host_identity_verified",
            "instant_bookable",
            "service_fee",
            "minimum_nights",
            "number_of_reviews",
            "reviews_per_month",
            "review_rate_number",
            "calculated_host_listings_count",
            "availability_365",
            "policy_flexible",
            "policy_moderate",
            "policy_strict",
            "Entire home/apt",
            "Hotel room",
            "Private room",
            "Shared room",
            "neighbourhood_group_Bronx",
            "neighbourhood_group_Brooklyn",
            "neighbourhood_group_Manhattan",
            "neighbourhood_group_Queens",
            "neighbourhood_group_Staten Island",
            "neighbourhood_group_Williamsburg",
            "location_cluster",
            "days_since_last_review",
            "availability_ratio",
            "property_age",
            "has_house_rules",
            "has_license",
            "popularity_score",
            "avg_reviews_per_listing",
        ]
        self.models = {
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42),
                "param_grid": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                },
            },
            "XGBoost": {
                "model": XGBRegressor(
                    objective="reg:squarederror", eval_metric="rmse", random_state=42
                ),
                "param_grid": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },
            },
            "Neural Network": {
                "model": MLPRegressor(max_iter=1000, random_state=42),
                "param_grid": {
                    "hidden_layer_sizes": [(50,), (100,)],
                    "activation": ["relu"],
                    "learning_rate_init": [0.001, 0.01],
                    "early_stopping": [True],
                },
            },
        }
        self.best_models = {}
        self.model_r2_scores = []
        self.model_names = []
        self.best_model_name = None
        self.best_r2 = -np.inf
        # Initialize transformers
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        logger.info(
            "ModelTrainer initialized with input: %s, model_path: %s",
            input_path,
            model_path,
        )

    def load_data(self) -> None:
        """Load the transformed Airbnb dataset from a CSV file."""
        try:
            logger.info(f"Loading data from {self.input_path}")
            self.df = pd.read_csv(self.input_path, low_memory=False)
            logger.info(
                f"Successfully loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns"
            )
        except FileNotFoundError:
            logger.error(f"File not found: {self.input_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def prepare_data(
        self,
        test_size: float = 0.3,
        val_size: float = 0.5,
        random_state: int = 42,
    ) -> None:
        """Prepare data: split into train/val/test, apply PCA."""
        logger.info("Preparing data: splitting and applying PCA")
        X = self.df[self.model_features]
        y = self.df[self.target_col]

        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Split train_val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )

        # Fit and transform training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_val_scaled shape: {X_val_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
        # Apply PCA - fit on training data and transform all sets
        self.X_train_pca = self.pca.fit_transform(X_train_scaled)
        self.X_val_pca = self.pca.transform(X_val_scaled)
        self.X_test_pca = self.pca.transform(X_test_scaled)
        print(f"X_train_pca shape: {self.X_train_pca.shape}")
        print(f"X_val_pca shape: {self.X_val_pca.shape}")
        print(f"X_test_pca shape: {self.X_test_pca.shape}")
        print(f"pca components: {self.pca.components_.shape}")
        
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        logger.info(
            f"Data prepared: Train shape {self.X_train_pca.shape}, Val shape {self.X_val_pca.shape}, Test shape {self.X_test_pca.shape}"
        )

    def train_models(self) -> None:
        """Train multiple models using GridSearchCV."""
        logger.info("Starting model training")
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        for name, config in self.models.items():
            logger.info(f"Training {name}")
            grid_search = GridSearchCV(
                config["model"],
                config["param_grid"],
                cv=cv,
                scoring="r2",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(self.X_train_pca, self.y_train)
            self.best_models[name] = grid_search.best_estimator_

            logger.info(f"Best Parameters for {name}: {grid_search.best_params_}")
            logger.info(
                f"Best Cross-validation R² for {name}: {grid_search.best_score_ * 100:.2f}%"
            )

    def evaluate_models(self) -> None:
        """Evaluate models on train, val, test sets."""
        logger.info("Evaluating models")
        for name, model in self.best_models.items():
            y_train_pred = model.predict(self.X_train_pca)
            y_val_pred = model.predict(self.X_val_pca)
            y_test_pred = model.predict(self.X_test_pca)

            train_r2 = r2_score(self.y_train, y_train_pred) * 100
            val_r2 = r2_score(self.y_val, y_val_pred) * 100
            test_r2 = r2_score(self.y_test, y_test_pred) * 100

            self.model_names.append(name)
            self.model_r2_scores.append(test_r2)

            if test_r2 > self.best_r2:
                self.best_r2 = test_r2
                self.best_model_name = name

            logger.info(f"Train R² for {name}: {train_r2:.2f}%")
            logger.info(f"Validation R² for {name}: {val_r2:.2f}%")
            logger.info(f"Test R² for {name}: {test_r2:.2f}%")

    def save_best_model(self) -> None:
        """Save the best model along with preprocessing transformers."""
        if self.best_model_name:
            try:
                logger.info(
                    f"Saving best model {self.best_model_name} to {self.model_path}"
                )
                Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

                # Save model, scaler, and PCA together
                model_package = {
                    "model": self.best_models[self.best_model_name],
                    "scaler": self.scaler,
                    "pca": self.pca,
                    "features": self.model_features,    
                    "model_name": self.best_model_name,
                    "model_version": "1.0.0",
                }

                joblib.dump(model_package, self.model_path)
                logger.info("Best model and transformers saved successfully")
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                raise

    def train_and_evaluate(self) -> dict:
        """Execute full training and evaluation pipeline."""
        logger.info("Starting model training and evaluation")
        self.load_data()
        self.prepare_data()
        self.train_models()
        self.evaluate_models()
        self.save_best_model()
        logger.info("Training and evaluation completed")
        return {
            "best_model_name": self.best_model_name,
            "best_test_r2": self.best_r2,
            "model_r2_scores": dict(zip(self.model_names, self.model_r2_scores)),
        }


if __name__ == "__main__":
    input_file = f"{TRANSFORMED_DATA_DIR}/airbnb_listings.csv"
    model_file = f"{MODELS_DIR}/price_prediction.joblib"
    trainer = ModelTrainer(input_file, model_file)
    results = trainer.train_and_evaluate()
    print(results)
