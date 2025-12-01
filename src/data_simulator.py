Perfect! Drift detection script created (375 lines, 13KB). Now let's create the final script - the data simulator.

6.5 Create Data Simulator (Kafka Producer)
bashcd "/Users/Admin/OneDrive - Northeastern University/Desktop/MLOps-1/Labs/ELK_Labs/Lab_New"

cat > src/data_simulator.py << 'ENDOFFILE'
"""
Data Simulator - Generates synthetic air quality data and streams to Kafka
Simulates real-time sensor readings for testing the streaming pipeline
"""

import os
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AirQualityDataSimulator:
    """
    Simulates realistic air quality sensor readings
    Streams data to Kafka topic for real-time processing
    """
    
    def __init__(self):
        """Initialize data simulator with Kafka producer"""
        self.kafka_enabled = False
        self.producer = None
        
        # Load reference data for realistic simulation
        self.reference_data = None
        self.load_reference_data()
        
        # Initialize Kafka producer
        self.initialize_kafka()
    
    def load_reference_data(self):
        """Load real data to base simulations on actual patterns"""
        try:
            data_path = Path('data/raw/AirQualityUCI.csv')
            if data_path.exists():
                self.reference_data = pd.read_csv(
                    data_path,
                    sep=';',
                    decimal=',',
                    na_values=-200,
                    nrows=1000  # Load subset for statistics
                )
                self.reference_data = self.reference_data.dropna(axis=1, how='all')
                logger.info(f"Loaded reference data: {self.reference_data.shape}")
            else:
                logger.warning("Reference data not found, using default ranges")
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
    
    def initialize_kafka(self):
        """Initialize Kafka producer connection"""
        kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=kafka_servers.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                max_block_ms=5000
            )
            self.kafka_enabled = True
            logger.info(f"Kafka producer connected: {kafka_servers}")
        except Exception as e:
            logger.warning(f"Kafka unavailable: {e}")
            self.kafka_enabled = False
    
    def generate_realistic_sample(self, add_noise: bool = True, drift_factor: float = 0.0):
        """
        Generate realistic air quality sample based on reference data
        
        Args:
            add_noise: Whether to add random noise
            drift_factor: Factor to simulate data drift (0.0 to 1.0)
        
        Returns:
            Dictionary with sensor readings
        """
        if self.reference_data is not None and len(self.reference_data) > 0:
            # Select random row from reference data as base
            base_sample = self.reference_data.sample(n=1).iloc[0]
            
            # Extract values with optional noise and drift
            noise_scale = 0.1 if add_noise else 0.0
            
            sample = {
                'co_gt': float(base_sample.get('CO(GT)', 2.0) * (1 + drift_factor) * (1 + np.random.normal(0, noise_scale))),
                'pt08_s1_co': float(base_sample.get('PT08.S1(CO)', 1300) * (1 + drift_factor * 0.5) * (1 + np.random.normal(0, noise_scale))),
                'c6h6_gt': float(base_sample.get('C6H6(GT)', 10.0) * (1 + drift_factor) * (1 + np.random.normal(0, noise_scale))),
                'pt08_s2_nmhc': float(base_sample.get('PT08.S2(NMHC)', 900) * (1 + drift_factor * 0.5) * (1 + np.random.normal(0, noise_scale))),
                'nox_gt': float(base_sample.get('NOx(GT)', 150) * (1 + drift_factor) * (1 + np.random.normal(0, noise_scale))),
                'pt08_s3_nox': float(base_sample.get('PT08.S3(NOx)', 1100) * (1 + drift_factor * 0.5) * (1 + np.random.normal(0, noise_scale))),
                'no2_gt': float(base_sample.get('NO2(GT)', 100) * (1 + drift_factor) * (1 + np.random.normal(0, noise_scale))),
                'pt08_s4_no2': float(base_sample.get('PT08.S4(NO2)', 1500) * (1 + drift_factor * 0.5) * (1 + np.random.normal(0, noise_scale))),
                'pt08_s5_o3': float(base_sample.get('PT08.S5(O3)', 1000) * (1 + drift_factor * 0.5) * (1 + np.random.normal(0, noise_scale))),
                'temperature': float(base_sample.get('T', 15.0) * (1 + np.random.normal(0, noise_scale * 0.5))),
                'relative_humidity': float(base_sample.get('RH', 50.0) * (1 + np.random.normal(0, noise_scale * 0.5))),
                'absolute_humidity': float(base_sample.get('AH', 0.8) * (1 + np.random.normal(0, noise_scale * 0.5)))
            }
        else:
            # Default ranges if no reference data
            sample = {
                'co_gt': random.uniform(0.5, 5.0) * (1 + drift_factor),
                'pt08_s1_co': random.uniform(800, 2000) * (1 + drift_factor * 0.5),
                'c6h6_gt': random.uniform(2.0, 20.0) * (1 + drift_factor),
                'pt08_s2_nmhc': random.uniform(600, 1500) * (1 + drift_factor * 0.5),
                'nox_gt': random.uniform(50, 300) * (1 + drift_factor),
                'pt08_s3_nox': random.uniform(700, 2000) * (1 + drift_factor * 0.5),
                'no2_gt': random.uniform(40, 200) * (1 + drift_factor),
                'pt08_s4_no2': random.uniform(1000, 2500) * (1 + drift_factor * 0.5),
                'pt08_s5_o3': random.uniform(500, 2000) * (1 + drift_factor * 0.5),
                'temperature': random.uniform(5.0, 35.0),
                'relative_humidity': random.uniform(20.0, 90.0),
                'absolute_humidity': random.uniform(0.3, 2.0)
            }
        
        # Add temporal features
        now = datetime.now()
        sample['hour'] = now.hour
        sample['day_of_week'] = now.weekday()
        sample['month'] = now.month
        
        # Add metadata
        sample['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        sample['source'] = 'simulator'
        
        # Ensure all values are non-negative and within reasonable bounds
        for key, value in sample.items():
            if isinstance(value, (int, float)):
                sample[key] = max(0, value)
        
        return sample
    
    def stream_data(self, 
                    num_samples: int = 1000,
                    interval_seconds: float = 1.0,
                    drift_probability: float = 0.1,
                    drift_magnitude: float = 0.3):
        """
        Stream simulated data to Kafka
        
        Args:
            num_samples: Number of samples to generate
            interval_seconds: Time between samples
            drift_probability: Probability of introducing drift
            drift_magnitude: Magnitude of drift when introduced
        """
        if not self.kafka_enabled:
            logger.error("Cannot stream: Kafka not available")
            return
        
        logger.info("=" * 70)
        logger.info("STARTING DATA SIMULATION")
        logger.info("=" * 70)
        logger.info(f"Samples to generate: {num_samples}")
        logger.info(f"Interval: {interval_seconds}s")
        logger.info(f"Drift probability: {drift_probability}")
        logger.info("Streaming to Kafka topic: air-quality-raw")
        
        samples_sent = 0
        drift_active = False
        drift_factor = 0.0
        
        try:
            for i in range(num_samples):
                # Randomly introduce drift
                if random.random() < drift_probability and not drift_active:
                    drift_active = True
                    drift_factor = drift_magnitude
                    logger.warning(f"Introducing data drift (factor: {drift_factor})")
                
                # Generate sample
                sample = self.generate_realistic_sample(
                    add_noise=True,
                    drift_factor=drift_factor
                )
                
                # Send to Kafka
                topic = os.getenv('KAFKA_TOPIC_RAW_DATA', 'air-quality-raw')
                self.producer.send(topic, value=sample)
                samples_sent += 1
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Sent {i + 1}/{num_samples} samples")
                
                # Wait before next sample
                time.sleep(interval_seconds)
                
                # Reset drift after some time
                if drift_active and random.random() < 0.05:
                    drift_active = False
                    drift_factor = 0.0
                    logger.info("Drift period ended, returning to normal")
            
            logger.info("=" * 70)
            logger.info(f"SIMULATION COMPLETE: {samples_sent} samples sent")
            logger.info("=" * 70)
            
        except KeyboardInterrupt:
            logger.info(f"\nSimulation stopped by user. Sent {samples_sent} samples")
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
        finally:
            if self.producer:
                self.producer.flush()
                self.producer.close()
            logger.info("Data simulator shut down")


def main():
    """Main function to run data simulator"""
    simulator = AirQualityDataSimulator()
    
    if simulator.kafka_enabled:
        # Stream 500 samples at 1 second intervals
        # With 10% chance of drift, magnitude 0.3
        simulator.stream_data(
            num_samples=500,
            interval_seconds=1.0,
            drift_probability=0.1,
            drift_magnitude=0.3
        )
    else:
        logger.error("Cannot start simulator: Kafka not available")
        logger.info("Please ensure Kafka is running in Docker")


if __name__ == "__main__":
    main()