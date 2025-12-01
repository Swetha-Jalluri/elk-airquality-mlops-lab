"""
Data Preprocessing Module for UCI Air Quality Dataset
Handles data loading, cleaning, feature engineering, and train-test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AirQualityPreprocessor:
    """
    Preprocessor for UCI Air Quality Dataset
    
    The dataset uses semicolon separators and European decimal format (comma).
    This class handles all data cleaning, feature engineering, and preparation.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize preprocessor with data path
        
        Args:
            data_path: Path to the AirQualityUCI.csv file
        """
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load UCI Air Quality dataset from CSV
        
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Load data with semicolon separator
        # The dataset uses European decimal format (comma instead of dot)
        df = pd.read_csv(
            self.data_path,
            sep=';',
            decimal=',',
            na_values=-200  # Missing values are marked as -200
        )
        
        # Remove empty columns (last two columns are empty)
        df = df.dropna(axis=1, how='all')
        
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and invalid data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning")
        
        initial_rows = len(df)
        
        # Combine Date and Time into single datetime column
        df['datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'],
            format='%d/%m/%Y %H.%M.%S',
            errors='coerce'
        )
        
        # Drop original Date and Time columns
        df = df.drop(['Date', 'Time'], axis=1)
        
        # Remove rows with missing datetime
        df = df.dropna(subset=['datetime'])
        
        # Select numeric columns for analysis (exclude datetime)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove rows where ALL numeric values are missing
        df = df.dropna(subset=numeric_cols, how='all')
        
        # For remaining missing values, use forward fill then backward fill
        # This assumes air quality changes gradually over time
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining rows with missing values
        df = df.dropna()
        
        rows_removed = initial_rows - len(df)
        logger.info(f"Removed {rows_removed} rows with missing/invalid data")
        logger.info(f"Clean dataset: {len(df)} rows")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features")
        
        # Extract temporal features from datetime
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        # Create cyclical features for hour (24-hour cycle)
        # This preserves the circular nature of time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Create cyclical features for day of week (7-day cycle)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Create is_weekend feature (1 for weekend, 0 for weekday)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Create is_rush_hour feature (morning and evening rush hours)
        df['is_rush_hour'] = (
            ((df['hour'] >= 7) & (df['hour'] <= 9)) |  # Morning rush
            ((df['hour'] >= 17) & (df['hour'] <= 19))  # Evening rush
        ).astype(int)
        
        # Create Air Quality Index (AQI) categories as target
        # Based on CO(GT) concentration levels
        # CO levels: Good < 2.0, Moderate 2.0-4.0, Unhealthy > 4.0
        if 'CO(GT)' in df.columns:
            df['AQI_Category'] = pd.cut(
                df['CO(GT)'],
                bins=[-np.inf, 2.0, 4.0, np.inf],
                labels=['Good', 'Moderate', 'Unhealthy']
            )
            
            # Convert categories to numeric labels for classification
            df['AQI_Label'] = df['AQI_Category'].map({
                'Good': 0,
                'Moderate': 1,
                'Unhealthy': 2
            })
        
        logger.info(f"Created {df.shape[1]} total features")
        
        return df
    
    def prepare_features_target(self, df: pd.DataFrame):
        """
        Separate features and target, and prepare for modeling
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        logger.info("Preparing features and target")
        
        # Define target column
        self.target_column = 'AQI_Label'
        
        # Define feature columns (exclude datetime, categories, and target)
        exclude_cols = ['datetime', 'AQI_Category', 'AQI_Label']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Separate features and target
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        logger.info(f"Features: {len(self.feature_columns)} columns")
        logger.info(f"Target: {self.target_column}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """
        Split data into train and test sets
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data: {test_size*100}% for testing")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Maintain class distribution in splits
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Scale features using StandardScaler
        Fit on training data only to prevent data leakage
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        logger.info("Scaling features")
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform test data using fitted scaler
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames with column names
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Feature scaling complete")
        
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self, save_processed: bool = True):
        """
        Complete preprocessing pipeline
        
        Args:
            save_processed: Whether to save processed data to disk
            
        Returns:
            Dictionary containing all processed data splits
        """
        logger.info("=" * 50)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 50)
        
        # Load data
        df = self.load_data()
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Save processed data if requested
        if save_processed:
            processed_dir = Path('data/processed')
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            X_train_scaled.to_csv(processed_dir / 'X_train.csv', index=False)
            X_test_scaled.to_csv(processed_dir / 'X_test.csv', index=False)
            y_train.to_csv(processed_dir / 'y_train.csv', index=False, header=True)
            y_test.to_csv(processed_dir / 'y_test.csv', index=False, header=True)
            
            logger.info(f"Processed data saved to {processed_dir}")
        
        logger.info("=" * 50)
        logger.info("Preprocessing pipeline complete")
        logger.info("=" * 50)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }


if __name__ == "__main__":
    # Example usage
    preprocessor = AirQualityPreprocessor('data/raw/AirQualityUCI.csv')
    data = preprocessor.preprocess_pipeline(save_processed=True)
    
    print(f"\nPreprocessing complete!")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Test samples: {len(data['X_test'])}")
    print(f"Number of features: {len(data['feature_columns'])}")