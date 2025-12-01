"""
Real-Time Data Drift Detection for Air Quality Data
Monitors incoming data for statistical drift and sends alerts
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure structured logging
log_dir = Path('logstash')
log_dir.mkdir(exist_ok=True)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'log_level': record.levelname,
            'service': 'drift_detection',
            'message': record.getMessage()
        }
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        return json.dumps(log_data)

# Setup logger
file_handler = logging.FileHandler('logstash/drift_detection.log')
file_handler.setFormatter(JSONFormatter())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
logger.addHandler(console_handler)


class DriftDetector:
    """
    Detects data drift using statistical tests
    Compares incoming data distributions with reference (training) data
    """
    
    def __init__(self, reference_data_path: str = 'data/processed/X_train.csv'):
        """
        Initialize drift detector with reference data
        
        Args:
            reference_data_path: Path to reference (training) data
        """
        self.reference_data_path = Path(reference_data_path)
        self.reference_data = None
        self.feature_stats = {}
        self.drift_threshold = 0.05  # p-value threshold for statistical tests
        
        self.load_reference_data()
    
    def load_reference_data(self):
        """Load reference data and calculate statistics"""
        try:
            if not self.reference_data_path.exists():
                logger.warning(f"Reference data not found: {self.reference_data_path}")
                logger.warning("Please run training pipeline first to generate reference data")
                return
            
            self.reference_data = pd.read_csv(self.reference_data_path)
            logger.info(f"Loaded reference data: {self.reference_data.shape}")
            
            # Calculate statistics for each feature
            for column in self.reference_data.columns:
                self.feature_stats[column] = {
                    'mean': self.reference_data[column].mean(),
                    'std': self.reference_data[column].std(),
                    'min': self.reference_data[column].min(),
                    'max': self.reference_data[column].max(),
                    'median': self.reference_data[column].median()
                }
            
            logger.info(f"Calculated statistics for {len(self.feature_stats)} features")
            
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
    
    def kolmogorov_smirnov_test(self, reference_sample: np.ndarray, current_sample: np.ndarray) -> tuple:
        """
        Perform Kolmogorov-Smirnov test to detect distribution drift
        
        Args:
            reference_sample: Sample from reference distribution
            current_sample: Sample from current distribution
            
        Returns:
            Tuple of (statistic, p_value)
        """
        try:
            statistic, p_value = stats.ks_2samp(reference_sample, current_sample)
            return float(statistic), float(p_value)
        except Exception as e:
            logger.error(f"KS test error: {e}")
            return 0.0, 1.0
    
    def detect_feature_drift(self, feature_name: str, current_values: np.ndarray) -> dict:
        """
        Detect drift for a single feature
        
        Args:
            feature_name: Name of the feature
            current_values: Current values for the feature
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None or feature_name not in self.reference_data.columns:
            return {
                'feature': feature_name,
                'drift_detected': False,
                'reason': 'No reference data available'
            }
        
        # Get reference values
        reference_values = self.reference_data[feature_name].values
        
        # Perform KS test
        ks_statistic, p_value = self.kolmogorov_smirnov_test(
            reference_values,
            current_values
        )
        
        # Calculate statistical measures
        current_mean = np.mean(current_values)
        current_std = np.std(current_values)
        reference_mean = self.feature_stats[feature_name]['mean']
        reference_std = self.feature_stats[feature_name]['std']
        
        # Calculate percentage change in mean
        mean_change_pct = abs(current_mean - reference_mean) / reference_mean * 100 if reference_mean != 0 else 0
        
        # Detect drift
        drift_detected = p_value < self.drift_threshold
        
        result = {
            'feature': feature_name,
            'drift_detected': drift_detected,
            'ks_statistic': round(ks_statistic, 4),
            'p_value': round(p_value, 4),
            'threshold': self.drift_threshold,
            'current_mean': round(current_mean, 4),
            'reference_mean': round(reference_mean, 4),
            'mean_change_pct': round(mean_change_pct, 2),
            'current_std': round(current_std, 4),
            'reference_std': round(reference_std, 4)
        }
        
        return result
    
    def detect_batch_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Detect drift across all features in a batch
        
        Args:
            current_data: DataFrame with current data batch
            
        Returns:
            Dictionary with comprehensive drift analysis
        """
        logger.info(f"Analyzing drift for batch of {len(current_data)} samples")
        
        feature_results = []
        features_with_drift = []
        
        for feature_name in current_data.columns:
            if feature_name in self.reference_data.columns:
                result = self.detect_feature_drift(
                    feature_name,
                    current_data[feature_name].values
                )
                feature_results.append(result)
                
                if result['drift_detected']:
                    features_with_drift.append(feature_name)
        
        # Calculate overall drift score (proportion of features with drift)
        drift_score = len(features_with_drift) / len(feature_results) if feature_results else 0
        
        # Determine drift severity
        if drift_score >= 0.5:
            severity = 'critical'
        elif drift_score >= 0.3:
            severity = 'high'
        elif drift_score >= 0.15:
            severity = 'moderate'
        else:
            severity = 'low'
        
        batch_result = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'batch_size': len(current_data),
            'features_analyzed': len(feature_results),
            'features_with_drift': len(features_with_drift),
            'drift_score': round(drift_score, 4),
            'drift_severity': severity,
            'drift_detected': len(features_with_drift) > 0,
            'drifted_features': features_with_drift,
            'feature_results': feature_results
        }
        
        return batch_result
    
    def log_drift_alert(self, drift_result: dict):
        """
        Log drift detection results
        
        Args:
            drift_result: Dictionary with drift analysis results
        """
        if drift_result['drift_detected']:
            logger.warning(
                f"DRIFT ALERT: {drift_result['features_with_drift']} features drifting "
                f"(severity: {drift_result['drift_severity']})"
            )
            
            # Log detailed results
            log_entry = logging.LogRecord(
                name=logger.name,
                level=logging.WARNING,
                pathname='',
                lineno=0,
                msg="Data drift detected",
                args=(),
                exc_info=None
            )
            log_entry.extra_data = drift_result
            file_handler.emit(log_entry)
        else:
            logger.info("No drift detected in current batch")


class StreamingDriftMonitor:
    """
    Monitor for data drift from Kafka stream
    Continuously analyzes incoming predictions for drift
    """
    
    def __init__(self):
        """Initialize streaming drift monitor"""
        self.detector = DriftDetector()
        self.batch_size = 100  # Analyze every 100 predictions
        self.current_batch = []
        
        # Initialize Kafka consumer and producer
        kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        
        try:
            self.consumer = KafkaConsumer(
                os.getenv('KAFKA_TOPIC_PREDICTIONS', 'air-quality-predictions'),
                bootstrap_servers=kafka_servers.split(','),
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='drift_detection_group'
            )
            
            self.producer = KafkaProducer(
                bootstrap_servers=kafka_servers.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            self.kafka_enabled = True
            logger.info("Kafka connections established")
            
        except Exception as e:
            logger.warning(f"Kafka unavailable: {e}")
            self.kafka_enabled = False
            self.consumer = None
            self.producer = None
    
    def extract_features_from_prediction(self, prediction_data: dict) -> dict:
        """
        Extract feature values from prediction message
        
        Args:
            prediction_data: Prediction message from Kafka
            
        Returns:
            Dictionary of feature values
        """
        if 'input_features' in prediction_data:
            return prediction_data['input_features']
        return {}
    
    def monitor_stream(self):
        """
        Continuously monitor Kafka stream for drift
        """
        if not self.kafka_enabled:
            logger.error("Cannot monitor stream: Kafka not available")
            return
        
        logger.info("=" * 70)
        logger.info("STARTING DRIFT DETECTION MONITORING")
        logger.info("=" * 70)
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Drift threshold: {self.detector.drift_threshold}")
        logger.info("Listening for predictions...")
        
        try:
            for message in self.consumer:
                prediction_data = message.value
                
                # Extract features
                features = self.extract_features_from_prediction(prediction_data)
                
                if features:
                    self.current_batch.append(features)
                    
                    # Analyze batch when full
                    if len(self.current_batch) >= self.batch_size:
                        batch_df = pd.DataFrame(self.current_batch)
                        
                        # Detect drift
                        drift_result = self.detector.detect_batch_drift(batch_df)
                        
                        # Log results
                        self.detector.log_drift_alert(drift_result)
                        
                        # Send alert to Kafka if drift detected
                        if drift_result['drift_detected']:
                            self.producer.send(
                                os.getenv('KAFKA_TOPIC_DRIFT_ALERTS', 'drift-alerts'),
                                value=drift_result
                            )
                            logger.info("Drift alert sent to Kafka")
                        
                        # Reset batch
                        self.current_batch = []
                        
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")
        finally:
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()
            logger.info("Drift monitoring shut down")


def main():
    """Main function to run drift detection"""
    monitor = StreamingDriftMonitor()
    
    if monitor.kafka_enabled:
        monitor.monitor_stream()
    else:
        logger.error("Cannot start monitoring: Kafka not available")
        logger.info("Please ensure Kafka is running in Docker")


if __name__ == "__main__":
    main()