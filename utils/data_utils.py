"""
Data and Metric Utilities for House Price Predictor
===================================================

This module provides common utilities, constants, and helper functions
used across the training pipeline, testing suite, and web application.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Canonical feature names and order expected by all models and endpoints
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET_NAME = "MedHouseVal"

# The target variable (MedHouseVal) is measured in units of $100,000 USD
TARGET_MULTIPLIER = 100_000

# Feature descriptions and standard units for documentation & UI
FEATURE_DESCRIPTIONS = {
    "MedInc": "Median income in block group (in tens of thousands of USD, e.g., 3.5 = $35,000)",
    "HouseAge": "Median house age in block group (in years)",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Block group population (number of people)",
    "AveOccup": "Average number of household members (occupancy)",
    "Latitude": "Block group latitude (approx. 32.5° to 42° N)",
    "Longitude": "Block group longitude (approx. -124.5° to -114° W)",
}


def load_california_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Safely load the California Housing dataset using scikit-learn.
    
    Returns:
        X (pd.DataFrame): 8 explanatory features.
        y (pd.Series): Target variable (MedHouseVal in $100,000 units).
        
    Raises:
        RuntimeError: If dataset fetching fails due to network or dependency issues.
    """
    try:
        data = fetch_california_housing(as_frame=True)
        X = data.data.copy()
        y = data.target.copy()
        return X[FEATURE_NAMES], y
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load California Housing dataset from scikit-learn: {exc}. "
            "Please verify your internet connection or scikit-learn installation."
        ) from exc


def format_currency(value_in_hundred_thousands: float) -> str:
    """
    Convert a value in $100,000 units to a formatted USD string ($XX,XXX).
    
    Args:
        value_in_hundred_thousands (float): e.g. 2.45
        
    Returns:
        str: e.g. "$245,000"
    """
    dollar_val = value_in_hundred_thousands * TARGET_MULTIPLIER
    return f"${dollar_val:,.2f}"


def format_metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a metrics dictionary into a readable pandas DataFrame.
    """
    rows = [
        {
            "Metric": "Mean Absolute Error (MAE)",
            "Dataset Units ($100k)": f"{metrics['mae']:.4f}",
            "Dollar Value (USD)": f"${metrics['mae_usd']:,.2f}",
            "Interpretation": "Average absolute difference between actual & predicted price",
        },
        {
            "Metric": "Root Mean Squared Error (RMSE)",
            "Dataset Units ($100k)": f"{metrics['rmse']:.4f}",
            "Dollar Value (USD)": f"${metrics['rmse_usd']:,.2f}",
            "Interpretation": "Square root of mean squared error (penalizes large errors more)",
        },
        {
            "Metric": "R² (Coefficient of Determination)",
            "Dataset Units ($100k)": f"{metrics['r2']:.4f}",
            "Dollar Value (USD)": f"{metrics['r2']*100:.2f}%",
            "Interpretation": "Proportion of target variance explained by the linear model",
        },
    ]
    return pd.DataFrame(rows)
