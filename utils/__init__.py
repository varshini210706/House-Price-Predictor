"""
Utility package for California House Price Prediction project.
Provides helper modules for data loading, validation, metrics formatting, and plotting.
"""

from .data_utils import (
    load_california_data,
    FEATURE_NAMES,
    TARGET_NAME,
    TARGET_MULTIPLIER,
    FEATURE_DESCRIPTIONS,
    format_currency,
    format_metrics_table
)

__all__ = [
    "load_california_data",
    "FEATURE_NAMES",
    "TARGET_NAME",
    "TARGET_MULTIPLIER",
    "FEATURE_DESCRIPTIONS",
    "format_currency",
    "format_metrics_table"
]
