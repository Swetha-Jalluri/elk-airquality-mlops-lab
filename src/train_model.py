"""
Multi-Model Training Pipeline with MLflow Tracking
Trains Logistic Regression, Random Forest, and XGBoost models
Logs all experiments to MLflow and structured logs to ELK
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from data_preprocessing import AirQualityPreprocessor

# Load environment variables
load_dotenv()

# Configure structured logging for ELK
log_dir = Path('logstash')
log_dir.mkdir(exist_ok=True)

# Create JSON formatter for structured logging
class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format
    This makes it easier for Logstash to parse
    """
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'log_level': record.levelname,
            'logger_name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'model_name'):
            log_data['model_name'] = record.model_name
        if hasattr(record, 'metrics'):
            log_data.update(record.metrics)
            
        return json.dumps(log_data)

# Setup file handler with JSON formatter
file_handler = logging.FileHandler('logstash/training.log')
file_handler.setFormatter(JSONFormatter())

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Also add console handler for debugging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(console_handler)


class MultiModelTrainer:
    """
    Trainer class that handles multiple model types
    Logs all experiments to MLflow and structured logs to ELK
    """
    
    def __init__(self, experiment_name: str = 'air-quality-prediction'):
        """
        Initialize trainer with MLflow experiment
        
        Args:
            experiment_name: Name for MLflow experiment
        """
        self.experiment_name = experiment_name
        
        # Set MLflow tracking URI
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
            
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            logger.info("Continuing without MLflow tracking")
    
    def get_model_config(self, model_type: str):
        """
        Get model configuration and hyperparameters
        
        Args:
            model_type: Type of model ('logistic_regression', 'random_forest', 'xgboost')
            
        Returns:
            Dictionary with model instance and parameters
        """
        configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    solver='lbfgs',
                    multi_class='multinomial'
                ),
                'params': {
                    'max_iter': 1000,
                    'solver': 'lbfgs',
                    'multi_class': 'multinomial',
                    'random_state': 42
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'xgboost': {
                'model': XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='mlogloss',
                    use_label_encoder=False
                ),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            }
        }
        
        return configs.get(model_type)
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive metrics for model evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # Calculate ROC AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(
                    y_true, y_pred_proba,
                    multi_class='ovr', average='weighted'
                ))
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        # Calculate confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        
        # For multi-class, calculate per-class metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (fp + fn + tp)
        
        metrics['true_positives'] = int(tp.sum())
        metrics['false_positives'] = int(fp.sum())
        metrics['true_negatives'] = int(tn.sum())
        metrics['false_negatives'] = int(fn.sum())
        
        return metrics
    
    def train_model(self, model_type: str, X_train, X_test, y_train, y_test):
        """
        Train a single model and log results
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Trained model and metrics
        """
        logger.info(f"Starting training for {model_type}")
        
        # Get model configuration
        config = self.get_model_config(model_type)
        if config is None:
            logger.error(f"Unknown model type: {model_type}")
            return None, None
        
        model = config['model']
        params = config['params']
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param('model_type', model_type)
            mlflow.log_param('n_train_samples', len(X_train))
            mlflow.log_param('n_test_samples', len(X_test))
            mlflow.log_param('n_features', X_train.shape[1])
            
            # Train model and measure time
            start_time = time.time()
            
            logger.info(f"Training {model_type} with {len(X_train)} samples...")
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Get prediction probabilities
            y_pred_proba_train = model.predict_proba(X_train)
            y_pred_proba_test = model.predict_proba(X_test)
            
            # Calculate metrics for train set
            train_metrics = self.calculate_metrics(y_train, y_pred_train, y_pred_proba_train)
            
            # Calculate metrics for test set
            test_metrics = self.calculate_metrics(y_test, y_pred_test, y_pred_proba_test)
            
            # Log all metrics to MLflow
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f'train_{metric_name}', value)
            
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f'test_{metric_name}', value)
            
            mlflow.log_metric('training_time_seconds', training_time)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                model,
                artifact_path='model',
                registered_model_name=f'air_quality_{model_type}'
            )
            
            # Create structured log entry for ELK
            log_entry = {
                'model_name': model_type,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'training_time_seconds': round(training_time, 3),
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'test_{k}': v for k, v in test_metrics.items()}
            }
            
            # Log to file for ELK ingestion
            extra_log = logging.LogRecord(
                name=logger.name,
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=f"Model training complete: {model_type}",
                args=(),
                exc_info=None
            )
            extra_log.model_name = model_type
            extra_log.metrics = log_entry
            
            file_handler.emit(extra_log)
            
            # Print summary
            logger.info(f"\n{model_type.upper()} Results:")
            logger.info(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  Test F1 Score: {test_metrics['f1_score']:.4f}")
            logger.info(f"  Training Time: {training_time:.2f}s")
            
            return model, test_metrics
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """
        Train all model types and compare results
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary of trained models and their metrics
        """
        logger.info("=" * 70)
        logger.info("STARTING MULTI-MODEL TRAINING PIPELINE")
        logger.info("=" * 70)
        
        model_types = ['logistic_regression', 'random_forest', 'xgboost']
        results = {}
        
        for model_type in model_types:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Training Model: {model_type.upper()}")
            logger.info(f"{'=' * 70}")
            
            model, metrics = self.train_model(
                model_type, X_train, X_test, y_train, y_test
            )
            
            if model is not None:
                results[model_type] = {
                    'model': model,
                    'metrics': metrics
                }
        
        # Find best model based on F1 score
        best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
        best_model = results[best_model_name]['model']
        best_metrics = results[best_model_name]['metrics']
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"BEST MODEL: {best_model_name.upper()}")
        logger.info(f"Test F1 Score: {best_metrics['f1_score']:.4f}")
        logger.info(f"{'=' * 70}")
        
        # Save best model
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        
        import joblib
        model_path = model_dir / 'best_model.pkl'
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved to {model_path}")
        
        return results


def main():
    """Main training pipeline"""
    
    # Step 1: Preprocess data
    logger.info("Step 1: Data Preprocessing")
    preprocessor = AirQualityPreprocessor('data/raw/AirQualityUCI.csv')
    data = preprocessor.preprocess_pipeline(save_processed=True)
    
    # Step 2: Train models
    logger.info("\nStep 2: Model Training")
    trainer = MultiModelTrainer()
    results = trainer.train_all_models(
        data['X_train'],
        data['X_test'],
        data['y_train'],
        data['y_test']
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total models trained: {len(results)}")
    logger.info("Check MLflow UI at http://localhost:5000 for experiment tracking")
    logger.info("Check Kibana at http://localhost:5601 for log visualization")


if __name__ == "__main__":
    main()