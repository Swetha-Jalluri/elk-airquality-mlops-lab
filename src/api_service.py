"""
FastAPI Prediction Service for Air Quality Model - FIXED
Includes all 13 sensor features to match training data
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()

log_dir = Path('logstash')
log_dir.mkdir(exist_ok=True)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'log_level': record.levelname,
            'service': 'api_service',
            'message': record.getMessage()
        }
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        return json.dumps(log_data)

file_handler = logging.FileHandler('logstash/predictions.log')
file_handler.setFormatter(JSONFormatter())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

app = FastAPI(
    title="Air Quality Prediction API",
    description="REST API for real-time air quality predictions",
    version="1.0.0"
)

try:
    kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    kafka_producer = KafkaProducer(
        bootstrap_servers=kafka_bootstrap_servers.split(','),
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        max_block_ms=5000
    )
    kafka_enabled = True
    logger.info(f"Kafka producer connected: {kafka_bootstrap_servers}")
except Exception as e:
    logger.warning(f"Kafka unavailable: {e}")
    kafka_enabled = False
    kafka_producer = None


class PredictionRequest(BaseModel):
    """Request model - ALL 13 sensor features"""
    co_gt: float = Field(..., ge=0, le=100)
    pt08_s1_co: float = Field(..., ge=0)
    nmhc_gt: float = Field(..., ge=0, le=1000)
    c6h6_gt: float = Field(..., ge=0, le=100)
    pt08_s2_nmhc: float = Field(..., ge=0)
    nox_gt: float = Field(..., ge=0, le=1000)
    pt08_s3_nox: float = Field(..., ge=0)
    no2_gt: float = Field(..., ge=0, le=500)
    pt08_s4_no2: float = Field(..., ge=0)
    pt08_s5_o3: float = Field(..., ge=0)
    temperature: float = Field(..., alias='t', ge=-50, le=60)
    relative_humidity: float = Field(..., alias='rh', ge=0, le=100)
    absolute_humidity: float = Field(..., alias='ah', ge=0, le=5)
    hour: int = Field(default=12, ge=0, le=23)
    day_of_week: int = Field(default=1, ge=0, le=6)
    month: int = Field(default=3, ge=1, le=12)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "co_gt": 2.6,
                "pt08_s1_co": 1360,
                "nmhc_gt": 150,
                "c6h6_gt": 11.9,
                "pt08_s2_nmhc": 1046,
                "nox_gt": 166,
                "pt08_s3_nox": 1056,
                "no2_gt": 113,
                "pt08_s4_no2": 1692,
                "pt08_s5_o3": 1268,
                "t": 13.6,
                "rh": 48.9,
                "ah": 0.7578
            }
        },
        "populate_by_name": True
    }


class PredictionResponse(BaseModel):
    prediction: str
    prediction_label: int
    confidence: float
    probabilities: Dict[str, float]
    model_version_info: str = Field(alias='model_version')
    timestamp: str
    latency_ms: float
    
    model_config = {"populate_by_name": True}


class ModelService:
    def __init__(self, model_path: str = 'models/best_model.pkl'):
        self.model_path = Path(model_path)
        self.model = None
        self.model_version_info = os.getenv('MODEL_VERSION', '1.0.0')
        self.class_names = ['Good', 'Moderate', 'Unhealthy']
        self.load_model()
    
    def load_model(self):
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded: {type(self.model).__name__}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_features(self, request: PredictionRequest) -> np.ndarray:
        """Prepare features matching training data order"""
        hour_sin = np.sin(2 * np.pi * request.hour / 24)
        hour_cos = np.cos(2 * np.pi * request.hour / 24)
        dow_sin = np.sin(2 * np.pi * request.day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * request.day_of_week / 7)
        is_weekend = 1 if request.day_of_week >= 5 else 0
        is_rush_hour = 1 if (7 <= request.hour <= 9) or (17 <= request.hour <= 19) else 0
        
        features = np.array([
            request.co_gt,
            request.pt08_s1_co,
            request.nmhc_gt,
            request.c6h6_gt,
            request.pt08_s2_nmhc,
            request.nox_gt,
            request.pt08_s3_nox,
            request.no2_gt,
            request.pt08_s4_no2,
            request.pt08_s5_o3,
            request.temperature,
            request.relative_humidity,
            request.absolute_humidity,
            request.hour,
            request.day_of_week,
            request.month,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            is_weekend,
            is_rush_hour
        ]).reshape(1, -1)
        
        return features
    
    def predict(self, request: PredictionRequest) -> Dict:
        start_time = time.time()
        features = self.prepare_features(request)
        prediction_label = int(self.model.predict(features)[0])
        prediction_proba = self.model.predict_proba(features)[0]
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'prediction': self.class_names[prediction_label],
            'prediction_label': prediction_label,
            'confidence': float(prediction_proba[prediction_label]),
            'probabilities': {
                name: float(prob) 
                for name, prob in zip(self.class_names, prediction_proba)
            },
            'model_version': self.model_version_info,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'latency_ms': round(latency_ms, 2)
        }


model_service = ModelService()


@app.get("/")
async def root():
    return {
        "service": "Air Quality Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "kafka_enabled": kafka_enabled
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        result = model_service.predict(request)
        
        log_entry = logging.LogRecord(
            name=logger.name, level=logging.INFO, pathname='', lineno=0,
            msg="Prediction made", args=(), exc_info=None
        )
        log_entry.extra_data = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'latency_ms': result['latency_ms']
        }
        file_handler.emit(log_entry)
        
        if kafka_enabled and kafka_producer:
            try:
                kafka_message = {**result, 'input_features': request.dict()}
                kafka_producer.send('air-quality-predictions', value=kafka_message)
            except Exception as e:
                logger.warning(f"Kafka send failed: {e}")
        
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server on 0.0.0.0:8000")
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=False)