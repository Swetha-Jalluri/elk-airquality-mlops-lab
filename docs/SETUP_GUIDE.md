# Quick Setup Guide

## Step-by-Step Installation

### Prerequisites Check
```bash
# Check Docker
docker --version
# Expected: Docker version 28.x.x

# Check Python
python3 --version
# Expected: Python 3.9.x or higher

# Check available RAM
# Need at least 8GB free
```

### Step 1: Start Docker Services (5 minutes)
```bash
# Navigate to project
cd "/Users/Admin/OneDrive - Northeastern University/Desktop/MLOps-1/Labs/ELK_Labs/Lab_New"

# Start all services
docker-compose up -d

# Wait for services to initialize (2-3 minutes)
# Watch the logs
docker-compose logs -f
```

**Look for these messages:**
- Elasticsearch: "started"
- Kafka: "Kafka Server started"
- Kibana: "Kibana is now available"
- MLflow: "Listening at: http://0.0.0.0:5000"

Press `Ctrl+C` to stop watching logs.

### Step 2: Verify Services (2 minutes)
```bash
# Test Elasticsearch
curl http://localhost:9200
# Should return JSON with cluster info

# Test Kibana (open in browser)
open http://localhost:5601
# Should show Kibana home page

# Test MLflow (open in browser)
open http://localhost:5000
# Should show MLflow UI
```

### Step 3: Setup Python Environment (3 minutes)
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# This will take 2-3 minutes
```

### Step 4: Train Models (5 minutes)
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run training
python src/train_model.py
```

**Expected output:**
```
Starting preprocessing pipeline
Loaded reference data: (9357, 15)
Clean dataset: 9357 rows
Created 21 total features

Training Model: LOGISTIC_REGRESSION
Training completed in 2.15 seconds
Test Accuracy: 0.8523
Test F1 Score: 0.8456

Training Model: RANDOM_FOREST
Training completed in 8.42 seconds
Test Accuracy: 0.8876
Test F1 Score: 0.8823

Training Model: XGBOOST
Training completed in 12.34 seconds
Test Accuracy: 0.9012
Test F1 Score: 0.8967

BEST MODEL: XGBOOST
Test F1 Score: 0.8967
```

### Step 5: Start API Service (1 minute)
```bash
# Open NEW terminal
cd "/Users/Admin/OneDrive - Northeastern University/Desktop/MLOps-1/Labs/ELK_Labs/Lab_New"

# Activate environment
source venv/bin/activate

# Start API
python src/api_service.py
```

**Expected output:**
```
Model loaded successfully
Kafka producer connected
API ready to serve predictions
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Test Prediction (1 minute)
```bash
# Open ANOTHER NEW terminal

# Test health endpoint
curl http://localhost:8000/health

# Make prediction
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

**Expected response:**
```json
{
  "prediction": "Good",
  "confidence": 0.87,
  "model_version": "1.0.0"
}
```

### Step 7: View Logs in Kibana (5 minutes)

1. Open: http://localhost:5601
2. Click **Management** (gear icon in left menu)
3. Click **Stack Management**
4. Click **Data Views** (under Kibana section)
5. Click **Create data view**
6. Fill in:
   - Name: `airquality-logs`
   - Index pattern: `airquality-*`
   - Timestamp field: `@timestamp`
7. Click **Save data view to Kibana**
8. Go to **Analytics** → **Discover**
9. Select `airquality-logs` from dropdown
10. You should see training and prediction logs!

### Step 8: View MLflow Experiments (2 minutes)

1. Open: http://localhost:5000
2. Click on `air-quality-prediction` experiment
3. You'll see 3 runs (one for each model)
4. Click on any run to see:
   - Parameters (hyperparameters)
   - Metrics (accuracy, F1, etc.)
   - Artifacts (saved model)

## Optional: Streaming Components

### Start Data Simulator
```bash
# New terminal
cd "/Users/Admin/OneDrive - Northeastern University/Desktop/MLOps-1/Labs/ELK_Labs/Lab_New"
source venv/bin/activate

python src/data_simulator.py
# This streams synthetic data to Kafka
```

### Start Drift Detection
```bash
# New terminal
cd "/Users/Admin/OneDrive - Northeastern University/Desktop/MLOps-1/Labs/ELK_Labs/Lab_New"
source venv/bin/activate

python src/drift_detection.py
# This monitors for data drift
```

## Stopping Everything
```bash
# Stop Python scripts: Press Ctrl+C in each terminal

# Stop Docker services
docker-compose down

# To also remove volumes (data)
docker-compose down -v
```

## Total Setup Time

- Docker services: 5 minutes
- Python environment: 3 minutes
- Model training: 5 minutes
- API setup: 1 minute
- Kibana configuration: 5 minutes

**Total: ~20 minutes**

## Common Issues

### "Port already in use"
```bash
# Find what's using the port
lsof -i :9200

# Kill the process
kill -9 <PID>
```

### "Docker daemon not running"
```bash
# Open Docker Desktop application
# Wait for it to start
```

### "Module not found"
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### "Kafka connection failed"
```bash
# Check Kafka is running
docker-compose ps kafka

# Check logs
docker-compose logs kafka

# Restart if needed
docker-compose restart kafka
```

## Next Steps

After setup:
1. Explore MLflow experiments
2. Create custom Kibana visualizations
3. Test drift detection
4. Review the code in `src/` directory
5. Read main README.md for detailed documentation
