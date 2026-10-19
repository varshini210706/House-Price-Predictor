"""
=============================================================================
California House Price Prediction - Model Training Pipeline
=============================================================================
This script trains a Multiple Linear Regression model on the California Housing
dataset, computes evaluation metrics (MAE, RMSE, R²), produces high-resolution
exploratory and diagnostic visualizations, and serializes the trained model
with comprehensive metadata.

Author: AI/ML Engineering Team
Date: October 2026
=============================================================================
"""

import os
import json
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Import shared constants from utils package
from utils.data_utils import (
    FEATURE_NAMES,
    TARGET_NAME,
    TARGET_MULTIPLIER,
    FEATURE_DESCRIPTIONS,
    format_currency
)

# Set random seed for complete reproducibility
RANDOM_SEED = 42

# Paths
MODELS_DIR = "models"
REPORTS_DIR = "reports"
IMAGES_DIR = "images"
MODEL_PATH = os.path.join(MODELS_DIR, "house_price_model.pkl")
METRICS_JSON_PATH = os.path.join(REPORTS_DIR, "metrics.json")


def load_and_validate_data() -> pd.DataFrame:
    """
    Fetch the California Housing dataset and validate its integrity.
    
    Returns:
        pd.DataFrame: Validated dataframe with 8 features and the target.
    """
    print("[1/6] Loading California Housing dataset from scikit-learn...")
    try:
        raw_data = fetch_california_housing(as_frame=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch California housing data: {exc}. "
            "Check internet connection or scikit-learn installation."
        ) from exc

    df = raw_data.frame.copy()

    # Validation checks
    assert df.shape[0] > 0, "Dataset is empty!"
    assert all(col in df.columns for col in FEATURE_NAMES), "Missing expected features!"
    assert TARGET_NAME in df.columns, f"Target column '{TARGET_NAME}' not found!"

    # Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"  Warning: Found missing values:\n{null_counts[null_counts > 0]}")
    else:
        print("  - Integrity check: 0 missing values across all features.")

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"  - Integrity check: {duplicates} duplicate rows detected.")

    # Check for infinite values
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"  - Integrity check: {inf_counts} infinite values detected.")

    print(f"  - Loaded {df.shape[0]:,} records and {df.shape[1]} columns successfully.")
    return df


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training (80%) and testing (20%) sets.
    
    Args:
        df: Input DataFrame containing features and target.
        test_size: Proportion of dataset allocated for testing (default: 0.2).
        random_state: Seed for reproducible pseudorandom splitting.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"[2/6] Splitting data into train ({(1-test_size)*100:.0f}%) and test ({test_size*100:.0f}%) sets (seed={random_state})...")
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_NAME].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"  - Training samples: {X_train.shape[0]:,}")
    print(f"  - Testing samples:  {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Fit an Ordinary Least Squares (OLS) Linear Regression model strictly on training data.
    
    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        
    Returns:
        LinearRegression: Fitted scikit-learn model.
    """
    print("[3/6] Training Ordinary Least Squares (OLS) Linear Regression model...")
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    print(f"  - Model learned intercept: {model.intercept_:.6f}")
    print("  - Learned feature coefficients:")
    for feature, coef in zip(FEATURE_NAMES, model.coef_):
        sign = "+" if coef >= 0 else "-"
        print(f"      * {feature:10s}: {sign} {abs(coef):.6f}")

    return model


def evaluate_model(
    model: LinearRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics (MAE, RMSE, R²) on both training and test sets.
    
    Converts dataset unit metrics ($100k) into actual US Dollar equivalents.
    """
    print("[4/6] Evaluating model performance...")
    # Train predictions
    y_pred_train = model.predict(X_train)
    train_mae = float(mean_absolute_error(y_train, y_pred_train))
    train_rmse = float(root_mean_squared_error(y_train, y_pred_train))
    train_r2 = float(r2_score(y_train, y_pred_train))

    # Test predictions
    y_pred_test = model.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_rmse = float(root_mean_squared_error(y_test, y_pred_test))
    test_r2 = float(r2_score(y_test, y_pred_test))

    metrics = {
        "train": {
            "mae": train_mae,
            "mae_usd": train_mae * TARGET_MULTIPLIER,
            "rmse": train_rmse,
            "rmse_usd": train_rmse * TARGET_MULTIPLIER,
            "r2": train_r2,
        },
        "test": {
            "mae": test_mae,
            "mae_usd": test_mae * TARGET_MULTIPLIER,
            "rmse": test_rmse,
            "rmse_usd": test_rmse * TARGET_MULTIPLIER,
            "r2": test_r2,
        },
        "intercept": float(model.intercept_),
        "coefficients": {
            feat: float(coef) for feat, coef in zip(FEATURE_NAMES, model.coef_)
        },
    }

    print("\n" + "=" * 68)
    print("                 MODEL PERFORMANCE EVALUATION SUMMARY")
    print("=" * 68)
    print(f"{'Metric':<18} | {'Training Set':<20} | {'Test Set (Held-Out)':<20}")
    print("-" * 68)
    print(f"{'MAE':<18} | {train_mae:.4f} ({format_currency(train_mae)}) | {test_mae:.4f} ({format_currency(test_mae)})")
    print(f"{'RMSE':<18} | {train_rmse:.4f} ({format_currency(train_rmse)}) | {test_rmse:.4f} ({format_currency(test_rmse)})")
    print(f"{'R-squared':<18} | {train_r2*100:.2f}%               | {test_r2*100:.2f}%")
    print("=" * 68 + "\n")

    return metrics, y_pred_test


def generate_visualizations(
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_test: np.ndarray,
    output_dir: str = IMAGES_DIR,
) -> None:
    """
    Generate and save the 5 required high-resolution plots.
    """
    print(f"[5/6] Generating and saving 5 high-resolution diagnostic visualizations to '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # -------------------------------------------------------------
    # 1. House Value Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(
        df[TARGET_NAME],
        kde=True,
        color="#1f77b4",
        bins=40,
        edgecolor="black",
        alpha=0.6,
        ax=ax,
    )
    ax.axvline(
        df[TARGET_NAME].median(),
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Median: {df[TARGET_NAME].median():.2f} (${df[TARGET_NAME].median()*100_000:,.0f})",
    )
    ax.axvline(
        5.0,
        color="#ff7f0e",
        linestyle=":",
        linewidth=2,
        label="Capped Upper Ceiling ($500,000)",
    )
    ax.set_title("Distribution of Median House Value (Target Variable)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Median House Value ($100,000 USD)", fontsize=11)
    ax.set_ylabel("Frequency (Block Groups)", fontsize=11)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "house_value_distribution.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved: {fig1_path}")

    # -------------------------------------------------------------
    # 2. Feature Correlation Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Features vs Median House Value", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "feature_correlations.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved: {fig2_path}")

    # -------------------------------------------------------------
    # 3. Median Income vs Median House Value
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    # Subsample 2500 points for a crisp, legible scatter without heavy overplotting
    sample_indices = np.random.RandomState(RANDOM_SEED).choice(len(df), size=min(2500, len(df)), replace=False)
    sample_df = df.iloc[sample_indices]
    
    sns.regplot(
        x="MedInc",
        y=TARGET_NAME,
        data=sample_df,
        scatter_kws={"alpha": 0.35, "color": "#1f77b4", "s": 18},
        line_kws={"color": "#d62728", "linewidth": 2, "label": "Linear Fit Trendline"},
        ax=ax,
    )
    ax.set_title("Median Income vs Median House Value", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Median Income ($10,000 USD)", fontsize=11)
    ax.set_ylabel("Median House Value ($100,000 USD)", fontsize=11)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "income_vs_house_value.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved: {fig3_path}")

    # -------------------------------------------------------------
    # 4. Actual vs Predicted Values
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    # Subsample test set points for clean plotting
    test_sub_idx = np.random.RandomState(RANDOM_SEED).choice(len(y_test), size=min(2000, len(y_test)), replace=False)
    y_test_sub = y_test.iloc[test_sub_idx]
    y_pred_sub = y_pred_test[test_sub_idx]

    ax.scatter(y_test_sub, y_pred_sub, alpha=0.3, color="#2ca02c", edgecolors="none", s=22, label="Test Observations")
    
    # 45-degree reference line
    ax.plot([0, 5.5], [0, 5.5], color="#d62728", linestyle="--", linewidth=2, label="Perfect Prediction (y = x)")

    ax.set_title("Actual vs Predicted House Values (Test Set)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Median House Value ($100,000 USD)", fontsize=11)
    ax.set_ylabel("Predicted Median House Value ($100,000 USD)", fontsize=11)
    ax.set_xlim(-0.2, 5.5)
    ax.set_ylim(-0.5, 5.5)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "actual_vs_predicted.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved: {fig4_path}")

    # -------------------------------------------------------------
    # 5. Residual Distribution (Errors: y_actual - y_pred)
    # -------------------------------------------------------------
    residuals = y_test - y_pred_test
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot A: Histogram of residuals
    sns.histplot(residuals, kde=True, color="#9467bd", bins=40, edgecolor="black", alpha=0.6, ax=ax_hist)
    ax_hist.axvline(0, color="#d62728", linestyle="--", linewidth=2, label="Zero Error Line")
    ax_hist.set_title("Residual Distribution (Error = Actual - Pred)", fontsize=12, fontweight="bold")
    ax_hist.set_xlabel("Residual Error ($100,000 USD)", fontsize=10)
    ax_hist.set_ylabel("Frequency", fontsize=10)
    ax_hist.legend(frameon=True)

    # Subplot B: Residuals vs Predicted Values (Homoscedasticity check)
    ax_scatter.scatter(y_pred_sub, (y_test_sub - y_pred_sub), alpha=0.3, color="#9467bd", s=20)
    ax_scatter.axhline(0, color="#d62728", linestyle="--", linewidth=2)
    ax_scatter.set_title("Residuals vs Predicted Values", fontsize=12, fontweight="bold")
    ax_scatter.set_xlabel("Predicted Values ($100,000 USD)", fontsize=10)
    ax_scatter.set_ylabel("Residuals ($100,000 USD)", fontsize=10)

    plt.suptitle("Residual Analysis: Normality and Variance Diagnostics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "residual_distribution.png")
    fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {fig5_path}")


def save_model_and_artifacts(
    model: LinearRegression,
    metrics: Dict[str, Any],
    X_train: pd.DataFrame,
    model_path: str = MODEL_PATH,
    metrics_path: str = METRICS_JSON_PATH,
) -> None:
    """
    Save the trained model and associated operational metadata using joblib.
    Also save evaluation metrics to a readable JSON file.
    """
    print(f"[6/6] Serializing model and metadata to '{model_path}'...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # Calculate training feature statistics to empower downstream input validation
    feature_stats = {}
    for col in FEATURE_NAMES:
        feature_stats[col] = {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
            "median": float(X_train[col].median()),
            "q25": float(X_train[col].quantile(0.25)),
            "q75": float(X_train[col].quantile(0.75)),
        }

    artifact_bundle = {
        "model": model,
        "model_type": "LinearRegression",
        "feature_names": FEATURE_NAMES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "target_name": TARGET_NAME,
        "target_units": "Units of $100,000 USD",
        "target_multiplier": TARGET_MULTIPLIER,
        "metrics": metrics,
        "feature_stats": feature_stats,
        "random_seed": RANDOM_SEED,
        "train_samples": int(X_train.shape[0]),
    }

    joblib.dump(artifact_bundle, model_path)
    print(f"  [OK] Trained model bundle saved successfully: {model_path}")

    # Save metrics JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"  [OK] Metrics JSON saved successfully: {metrics_path}")


def main() -> None:
    """End-to-end execution pipeline."""
    print("=" * 68)
    print("  CALIFORNIA HOUSING PRICE PREDICTOR - TRAINING PIPELINE")
    print("=" * 68)

    # 1. Load data
    df = load_and_validate_data()

    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=RANDOM_SEED)

    # 3. Train model
    model = train_linear_regression(X_train, y_train)

    # 4. Evaluate model
    metrics, y_pred_test = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 5. Generate and save diagnostic visualizations
    generate_visualizations(df, X_test, y_test, y_pred_test, output_dir=IMAGES_DIR)

    # 6. Save model and metrics
    save_model_and_artifacts(model, metrics, X_train, model_path=MODEL_PATH, metrics_path=METRICS_JSON_PATH)

    print("\n[SUCCESS] Model training and evaluation pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
