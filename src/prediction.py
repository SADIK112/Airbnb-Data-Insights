import joblib
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODELS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictor:
    """
    A class to handle price prediction using a trained model.
    
    Attributes:
        model: The loaded machine learning model
        model_version: Version of the model
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the PricePredictor with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Match the training configuration
        self.model_version = "1.0.0"
        # These are the exact features in the order used during training
        self.features = [
            'host_identity_verified', 'instant_bookable', 'service_fee', 'minimum_nights',
            'number_of_reviews', 'reviews_per_month', 'review_rate_number',
            'calculated_host_listings_count', 'availability_365', 'policy_flexible',
            'policy_moderate', 'policy_strict', 'Entire home/apt', 'Hotel room',
            'Private room', 'Shared room', 'neighbourhood_group_Bronx',
            'neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan',
            'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island',
            'neighbourhood_group_Williamsburg', 'location_cluster', 'days_since_last_review',
            'availability_ratio', 'property_age', 'has_house_rules', 'has_license',
            'popularity_score', 'avg_reviews_per_listing'
        ]
        self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """
        Load the trained model from the specified path.
        
        Args:
            model_path: Path to the trained model file
            
        Raises:
            FileNotFoundError: If the model file is not found
            Exception: For any other errors during model loading
        """
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            model_data = joblib.load(model_path)
            logger.info(f"Successfully loaded model from {model_path}")

            if isinstance(model_data, dict):
                # Look for the model in common keys
                if 'model' in model_data:
                    self.model = model_data['model']
                if 'pca' in model_data:
                    self.pca = model_data['pca']
                if 'scaler' in model_data:
                    self.scaler = model_data['scaler']
                if 'model_name' in model_data:
                    self.model_name = model_data['model_name']
            else:
                self.model = model_data
                
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            

    
    def validate_features(self, features: Dict[str, Union[float, int]]) -> np.ndarray:
        """Validate and convert input features to the format expected by the model."""
        if not self.features:
            raise ValueError("Model features not available - model may not have been loaded properly")
            
        # Check for missing features
        missing = set(self.features) - set(features.keys())
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
            
        # Create feature vector in the correct order
        try:
            feature_vector = np.array([[features[feature] for feature in self.features]])
            logger.info(f"Created feature vector with shape: {feature_vector.shape}")
            return feature_vector
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"Invalid feature values: {str(e)}")

    def predict(self, features: Dict[str, Union[float, int]]) -> Dict[str, Any]:
        """Make a prediction using the loaded model."""
        if not all([self.model, self.scaler, self.pca]):
            raise ValueError("Model, scaler, or PCA not loaded properly")
            
        try:
            # Validate and convert features
            feature_vector = self.validate_features(features)
            print(f"Feature vector shape: {feature_vector.shape}")
            # Apply the same transformations as in training
            # 1. Scale the features using the fitted scaler
            features_scaled = self.scaler.transform(feature_vector)
            print(f"Features scaled shape: {features_scaled.shape}")
            
            # 2. Apply PCA transformation using the fitted PCA
            features_pca = self.pca.transform(features_scaled)
            print(f"Features PCA shape: {features_pca.shape}")
            # Make prediction
            prediction = self.model.predict(features_pca)[0]
            print(f"Prediction: {prediction}")
            return {
                "prediction": float(prediction),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "features_used": len(self.features),
                "original_features_shape": feature_vector.shape,
                "pca_features_shape": features_pca.shape
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_version": self.model_version,
            "model_loaded": self.model is not None,
            "model_type": str(type(self.model).__name__) if self.model else None,
            "features_expected": len(self.features),
            "features_list": self.features
        }
        
    @classmethod
    def get_example_features(cls) -> Dict[str, Union[float, int]]:
        """
        Get an example set of feature values for the model.
        
        Returns:
            Dictionary containing example feature values (30 features)
        """
        return {
            'host_identity_verified': 1,                    # 1 = verified, 0 = not verified
            'instant_bookable': 1,                          # 1 = yes, 0 = no
            'service_fee': 75.50,                           # in local currency
            'minimum_nights': 3,                            # minimum nights to book
            'number_of_reviews': 24,                        # total number of reviews
            'reviews_per_month': 2.5,                       # average reviews per month
            'review_rate_number': 4.8,                      # rating out of 5
            'calculated_host_listings_count': 2,            # number of listings by this host
            'availability_365': 200,                        # days available in next year
            'policy_flexible': 0,                           # 1 = flexible cancellation
            'policy_moderate': 1,                           # 1 = moderate cancellation
            'policy_strict': 0,                             # 1 = strict cancellation
            'Entire home/apt': 1,                           # 1 = property type is entire home
            'Hotel room': 0,                                # 1 = property type is hotel room
            'Private room': 0,                              # 1 = property type is private room
            'Shared room': 0,                               # 1 = property type is shared room
            'neighbourhood_group_Bronx': 0,                 # 1 = located in the Bronx
            'neighbourhood_group_Brooklyn': 0,              # 1 = located in Brooklyn
            'neighbourhood_group_Manhattan': 1,             # 1 = located in Manhattan
            'neighbourhood_group_Queens': 0,                # 1 = located in Queens
            'neighbourhood_group_Staten Island': 0,         # 1 = located in Staten Island
            'neighbourhood_group_Williamsburg': 0,          # 1 = located in Williamsburg
            'location_cluster': 3,                          # numerical cluster ID
            'days_since_last_review': 45,                   # days since last review
            'availability_ratio': 0.8,                      # available_days/365
            'property_age': 5,                              # years since construction
            'has_house_rules': 1,                           # 1 = has house rules
            'has_license': 1,                               # 1 = has valid license
            'popularity_score': 7.5,                        # custom score (1-10)
            'avg_reviews_per_listing': 3.2                  # average reviews per listing
        }

if __name__ == "__main__":
    # Initialize predictor
    predictor = PricePredictor(
        model_path=f"{MODELS_DIR}/price_prediction.joblib"
    )
    features = predictor.get_example_features()
    prediction = predictor.predict(features)
    print(prediction)