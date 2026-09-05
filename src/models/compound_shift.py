"""
Model to predict how environmental origin shifts compound concentrations.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. Model training unavailable.")
    SKLEARN_AVAILABLE = False


class CompoundShiftPredictor:
    """
    Predict compound concentration shifts based on environmental features.
    
    Model: Origin features → Δ Compound concentration
    """
    
    def __init__(self, model_type: str = "gradient_boosting"):
        """
        Args:
            model_type: Type of model ('gradient_boosting', 'random_forest', 'neural_net')
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for model training")
        
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        
        logger.info(f"Initialized CompoundShiftPredictor with {model_type}")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.01
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Feature matrix (origin features)
            y: Target variable (compound concentration or log-fold change)
            test_size: Fraction for test set
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            
        Returns:
            Dict with training metrics
        """
        logger.info(f"Training model on {len(X)} samples...")
        
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Initialize model
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'n_train': len(X_train),
            'n_test': len(X_test),
        }
        
        logger.info(f"Training complete: R²={metrics['test_r2']:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict compound shifts for new origin features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted shifts
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances from trained model.
        
        Returns:
            DataFrame with features and importances
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained or doesn't support feature importances")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, path: Path) -> None:
        """Save trained model to disk."""
        import joblib
        
        if self.model is None:
            raise ValueError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load trained model from disk."""
        import joblib
        
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        
        logger.info(f"Model loaded from {path}")
