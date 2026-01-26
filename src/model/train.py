"""Train regression model: salary -> expected production."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import pickle
import os
from .. import config


def create_model(model_type):
    """
    Create regression model.

    Args:
        model_type: Type of model ('gradient_boosting', 'spline', etc.)

    Returns:
        Sklearn model
    """
    if model_type == 'gradient_boosting':
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_model(historical_df):
    """
    Train regression model on historical data using config settings.

    Args:
        historical_df: DataFrame with historical rookies (must have 'salary' and 'production')

    Returns:
        Trained pipeline (scaler + model)
    """
    print("\n=== Training Model ===")
    print(f"Model type: {config.MODEL_TYPE}")
    print(f"CV folds: {config.CROSS_VALIDATION_FOLDS}")

    # Prepare features and target
    X = historical_df[['salary']].values
    y = historical_df['production'].values

    print(f"Training data: {len(X)} rookies")
    print(f"  Average salary range: ${X.min():,.0f} - ${X.max():,.0f}")
    print(f"  Production range: {y.min():.2f} - {y.max():.2f}")

    # Create pipeline with scaling
    model = create_model(config.MODEL_TYPE)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])

    # Cross-validation
    print(f"\nPerforming {config.CROSS_VALIDATION_FOLDS}-fold cross-validation...")
    cv_scores = cross_val_score(
        pipeline, X, y,
        cv=config.CROSS_VALIDATION_FOLDS,
        scoring='r2',
        n_jobs=-1
    )

    print(f"  CV R² scores: {cv_scores}")
    print(f"  Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Train final model on all data
    print("\nTraining final model on all historical data...")
    pipeline.fit(X, y)

    # Report feature importance if available
    if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
        importance = pipeline.named_steps['regressor'].feature_importances_
        print(f"  Feature importance (salary): {importance[0]:.3f}")

    print("  Model training complete")

    return pipeline


def save_model(pipeline, filepath='outputs/model.pkl'):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath='outputs/model.pkl'):
    """Load trained model from disk."""
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline
