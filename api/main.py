from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Union
from config import MODELS_DIR
from fastapi import HTTPException
from src.prediction import PricePredictor
import sys
import os
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application")
    global price_predictor
    price_predictor = PricePredictor(model_path=f"{MODELS_DIR}/price_prediction.joblib")
    yield
    # Shutdown (cleanup if needed)

app = FastAPI(lifespan=lifespan)

class PredictionInput(BaseModel):
    features: Dict[str, Union[float, int]]

class PredictionOutput(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    features_used: int

@app.post("/price-predict", response_model=PredictionOutput)
async def price_predict(input: PredictionInput):
    try:
        logger.info("Received prediction request")
        features = input.features
        result = price_predictor.predict(features)
        logger.info("Prediction result: %s", result)
        return {
            "prediction": result['prediction'],
            "model_name": result['model_name'],
            "model_version": result['model_version'],
            "features_used": result['features_used'],
            "original_features_shape": result['original_features_shape'],
            "pca_features_shape": result['pca_features_shape']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok!"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)