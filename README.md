# Enhanced ELK Stack Lab: Real-Time Air Quality Prediction with Streaming MLOps

## Project Overview

This project demonstrates a production-grade MLOps pipeline using the ELK stack (Elasticsearch, Logstash, Kibana) combined with real-time streaming via Apache Kafka. The system predicts air quality levels from sensor data using machine learning models, with comprehensive monitoring, logging, and drift detection.

### Key Features

1. **Real-Time Data Streaming**: Apache Kafka for event-driven architecture
2. **Multi-Model Training**: Logistic Regression, Random Forest, XGBoost with MLflow tracking
3. **REST API Service**: FastAPI for real-time predictions
4. **Drift Detection**: Statistical monitoring for data quality
5. **Comprehensive Logging**: Structured JSON logs for ELK ingestion
6. **Advanced Visualization**: Interactive Kibana dashboards
7. **Feature Caching**: Redis for performance optimization
8. **Containerized Deployment**: Docker Compose for one-command setup

## Architecture
```
Data Simulator → Kafka → [Training Pipeline] → MLflow
                   ↓            ↓
              API Service   Model Registry
                   ↓            ↓
           Kafka Predictions   Redis Cache
                   ↓
           Logstash Consumer
                   ↓
           Elasticsearch
                   ↓
              Kibana Dashboards
```

## Technology Stack

- **Data Processing**: Python 3.9, Pandas, NumPy, Scikit-learn, XGBoost
- **ML Tracking**: MLflow
- **API**: FastAPI, Uvicorn
- **Streaming**: Apache Kafka, Zookeeper
- **Logging**: ELK Stack (Elasticsearch 7.16.2, Logstash 7.16.2, Kibana 7.16.2)
- **Monitoring**: Metricbeat
- **Caching**: Redis
- **Containerization**: Docker, Docker Compose

## Prerequisites

- macOS or Linux
- Docker Desktop installed and running
- Python 3.8+
- 8GB RAM minimum
- 15GB disk space

## Quick Start

### 1. Start Docker Services
```bash
# Navigate to project directory
cd "/Users/Admin/OneDrive - Northeastern University/Desktop/MLOps-1/Labs/ELK_Labs/Lab_New"

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

**Wait 2-3 minutes for all services to initialize**

### 2. Verify Services
```bash
# Check Elasticsearch
curl http://localhost:9200

# Check Kibana (in browser)
open http://localhost:5601

# Check MLflow (in browser)
open http://localhost:5000

# Check Kafka
docker-compose logs kafka | grep "started"
```

### 3. Install Python Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 4. Run Training Pipeline
```bash
# Train all models with MLflow tracking
python src/train_model.py

# This will:
# - Preprocess UCI Air Quality data
# - Train 3 models (LR, RF, XGBoost)
# - Log experiments to MLflow
# - Save best model to models/
# - Generate training logs for ELK
```

**Expected output:**
```
Starting preprocessing pipeline
Loaded reference data: (9357, 15)
Training Logistic Regression...
Training Random Forest...
Training XGBoost...
Best model: xgboost (F1: 0.89)
```

### 5. Start API Service
```bash
# In a new terminal
source venv/bin/activate
python src/api_service.py

# API will start on http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

### 6. Start Data Simulator (Optional)
```bash
# In another terminal
source venv/bin/activate
python src/data_simulator.py

# This streams synthetic data to Kafka
```

### 7. Start Drift Detection (Optional)
```bash
# In another terminal
source venv/bin/activate
python src/drift_detection.py

# This monitors Kafka stream for drift
```

## Usage

### Making Predictions via API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "co_gt": 2.6,
    "pt08_s1_co": 1360,
    "c6h6_gt": 11.9,
    "pt08_s2_nmhc": 1046,
    "nox_gt": 166,
    "pt08_s3_nox": 1056,
    "no2_gt": 113,
    "pt08_s4_no2": 1692,
    "pt08_s5_o3": 1268,
    "temperature": 13.6,
    "relative_humidity": 48.9,
    "absolute_humidity": 0.7578
  }'
```

**Response:**
```json
{
  "prediction": "Good",
  "prediction_label": 0,
  "confidence": 0.87,
  "probabilities": {
    "Good": 0.87,
    "Moderate": 0.11,
    "Unhealthy": 0.02
  },
  "model_version": "1.0.0",
  "timestamp": "2024-11-30T19:15:00Z",
  "latency_ms": 12.5
}
```

### Viewing Logs in Kibana

1. Open Kibana: http://localhost:5601
2. Go to **Management** → **Stack Management** → **Data Views**
3. Click **Create data view**
4. Name: `airquality-logs`
5. Index pattern: `airquality-*`
6. Timestamp field: `@timestamp`
7. Click **Save**
8. Go to **Analytics** → **Discover**
9. Select `airquality-logs` data view

You should see:
- Training logs with model metrics
- Prediction logs with confidence scores
- Drift detection alerts

### Viewing MLflow Experiments

1. Open MLflow: http://localhost:5000
2. Click on `air-quality-prediction` experiment
3. Compare model runs
4. View metrics, parameters, and artifacts
5. Download trained models

## Project Structure
```
Lab_New/
├── README.md                    # This file
├── docker-compose.yml           # All services configuration
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── raw/
│   │   └── AirQualityUCI.csv   # UCI dataset
│   └── processed/              # Preprocessed data (generated)
│
├── src/
│   ├── data_preprocessing.py   # Data cleaning and feature engineering
│   ├── train_model.py          # Multi-model training with MLflow
│   ├── api_service.py          # FastAPI prediction service
│   ├── drift_detection.py      # Real-time drift monitoring
│   └── data_simulator.py       # Kafka data producer
│
├── logstash/
│   ├── logstash.conf          # Logstash pipeline configuration
│   ├── training.log           # Training logs (generated)
│   ├── predictions.log        # Prediction logs (generated)
│   └── drift_detection.log    # Drift logs (generated)
│
├── config/
│   └── metricbeat.yml         # Metricbeat configuration
│
├── models/
│   └── best_model.pkl         # Saved model (generated)
│
├── mlruns/                     # MLflow experiments (generated)
├── mlartifacts/               # MLflow artifacts (generated)
│
├── tests/                     # Test scripts
└── docs/                      # Additional documentation
```

## Key Differences from Original Lab

| Aspect | Original Lab | Our Enhanced Lab |
|--------|-------------|------------------|
| Dataset | Iris (toy) | UCI Air Quality (real-world) |
| Models | Single Logistic Regression | 3 models with comparison |
| Deployment | None | FastAPI REST API |
| Streaming | File-based | Kafka real-time |
| Tracking | Basic logs | MLflow + structured JSON |
| Monitoring | Static Kibana | Live dashboards + alerts |
| Drift Detection | None | Statistical tests + alerts |
| Caching | None | Redis feature cache |
| Scalability | Single machine | Distributed architecture |

## Troubleshooting

### Docker Services Not Starting
```bash
# Check Docker is running
docker ps

# Restart services
docker-compose down
docker-compose up -d

# Check logs
docker-compose logs elasticsearch
docker-compose logs kafka
```

### Kafka Connection Errors
```bash
# Wait for Kafka to be ready
docker-compose logs kafka | grep "started"

# Restart dependent services
docker-compose restart api_service
```

### Model File Not Found
```bash
# Train models first
python src/train_model.py

# Verify model exists
ls -lh models/best_model.pkl
```

### Port Already in Use
```bash
# Check what's using the port
lsof -i :9200  # Elasticsearch
lsof -i :5601  # Kibana
lsof -i :9092  # Kafka
lsof -i :8000  # API

# Kill process or change port in .env
```

## Performance Metrics

- **Training Time**: ~2-3 minutes for all 3 models
- **API Latency**: <50ms average
- **Throughput**: ~100 predictions/second
- **Drift Detection**: Batch of 100 samples every ~2 minutes

## Future Enhancements

- Add CI/CD pipeline with GitHub Actions
- Implement automated retraining on drift detection
- Add Prometheus + Grafana for advanced metrics
- Implement A/B testing framework
- Add model explainability (SHAP values)
- Deploy to cloud (AWS/GCP/Azure)

## References

- UCI Air Quality Dataset: https://archive.ics.uci.edu/dataset/360/air+quality
- Original ELK Lab: https://www.youtube.com/watch?v=sBqWEH5VKJY
- MLflow Documentation: https://mlflow.org/docs/latest/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Apache Kafka: https://kafka.apache.org/

--