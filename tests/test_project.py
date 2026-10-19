"""
=============================================================================
Automated Test Suite for California House Price Predictor
=============================================================================
Tests dataset loading, model integrity, prediction accuracy, evaluation
metrics, and artifact generation.
"""

import os
import re
import json
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from utils.data_utils import (
    load_california_data,
    FEATURE_NAMES,
    TARGET_NAME,
    TARGET_MULTIPLIER,
    format_currency
)

MODEL_PATH = os.path.join("models", "house_price_model.pkl")
METRICS_PATH = os.path.join("reports", "metrics.json")
REPORT_PATH = os.path.join("reports", "house_price_prediction_report.pdf")
NOTEBOOK_PATH = "task1_ml_linear_regression.ipynb"
REQUIRED_IMAGES = [
    "house_value_distribution.png",
    "feature_correlations.png",
    "income_vs_house_value.png",
    "actual_vs_predicted.png",
    "residual_distribution.png",
]


def test_data_loading():
    """Verify that California Housing dataset loads with expected shape and columns."""
    X, y = load_california_data()
    assert X.shape[0] == 20640, f"Expected 20,640 rows, got {X.shape[0]}"
    assert X.shape[1] == 8, f"Expected 8 features, got {X.shape[1]}"
    assert list(X.columns) == FEATURE_NAMES, "Feature column names do not match expected order"
    assert len(y) == 20640, "Target length does not match features length"
    assert X.isnull().sum().sum() == 0, "Missing values found in features"
    assert y.isnull().sum() == 0, "Missing values found in target"


def test_model_artifact():
    """Verify that the model file exists, can be unpickled, and contains all required metadata."""
    assert os.path.exists(MODEL_PATH), f"Model file missing at {MODEL_PATH}"
    bundle = joblib.load(MODEL_PATH)
    assert "model" in bundle, "Missing 'model' object in bundle"
    assert "feature_names" in bundle, "Missing 'feature_names' in bundle"
    assert bundle["feature_names"] == FEATURE_NAMES, "Feature names mismatch in model artifact"
    assert "metrics" in bundle, "Missing metrics dictionary in bundle"
    assert "feature_stats" in bundle, "Missing feature_stats in bundle"


def test_model_prediction():
    """Verify that model generates correct prediction shapes and realistic values."""
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    
    # Single sample prediction with realistic median values
    sample_data = pd.DataFrame([{
        "MedInc": 3.87,
        "HouseAge": 28.6,
        "AveRooms": 5.4,
        "AveBedrms": 1.1,
        "Population": 1425.0,
        "AveOccup": 3.0,
        "Latitude": 35.6,
        "Longitude": -119.5,
    }])[FEATURE_NAMES]

    pred = model.predict(sample_data)
    assert len(pred) == 1, "Expected 1 prediction output"
    # Target values are in $100k units, typical values between 0.5 and 5.0
    assert 0.0 < pred[0] < 10.0, f"Predicted value {pred[0]} is out of reasonable housing range"


def test_evaluation_metrics():
    """Verify that test metrics are calculated correctly and saved properly."""
    assert os.path.exists(METRICS_PATH), f"Metrics JSON missing at {METRICS_PATH}"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    test = metrics["test"]
    assert "mae" in test and "rmse" in test and "r2" in test
    assert 0.4 < test["mae"] < 0.7, f"Unusual test MAE: {test['mae']}"
    assert 0.6 < test["rmse"] < 0.9, f"Unusual test RMSE: {test['rmse']}"
    assert 0.50 < test["r2"] < 0.65, f"Unusual test R2: {test['r2']}"
    assert test["mae_usd"] == pytest.approx(test["mae"] * TARGET_MULTIPLIER, rel=1e-4)


def test_images_generated():
    """Verify that all five required visualizations exist in images/ directory."""
    for img_name in REQUIRED_IMAGES:
        img_path = os.path.join("images", img_name)
        assert os.path.exists(img_path), f"Required image missing: {img_path}"
        assert os.path.getsize(img_path) > 1000, f"Image {img_name} is suspiciously small (< 1KB)"


def test_pdf_report_exists_and_pages():
    """Verify that the PDF report exists and contains between 2 and 4 pages."""
    assert os.path.exists(REPORT_PATH), f"PDF report missing at {REPORT_PATH}"
    assert os.path.getsize(REPORT_PATH) > 5000, "PDF report file is empty or corrupted"

    # Count pages via regex on PDF stream
    with open(REPORT_PATH, "rb") as f:
        content = f.read()
    page_matches = re.findall(rb"/Type\s*/Page\b", content)
    page_count = len(page_matches)
    assert 2 <= page_count <= 4, f"Report page count ({page_count}) is not between 2 and 4 pages"


def test_notebook_validity():
    """Verify that the Jupyter notebook exists, is valid JSON, and has executed code cells."""
    assert os.path.exists(NOTEBOOK_PATH), f"Notebook missing at {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb, "Notebook missing 'cells' property"
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 15, f"Too few code cells: {len(code_cells)}"
    
    # Ensure every code cell has executed outputs
    executed_cells = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    assert len(executed_cells) == len(code_cells), "Found unexecuted code cells in notebook"
