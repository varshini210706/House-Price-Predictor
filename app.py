"""
=============================================================================
California House Price Predictor - Interactive Web Application
=============================================================================
Streamlit web application for interactive property value estimation using
the trained Multiple Linear Regression model on California census data.

Author: AI/ML Engineering Team
Date: October 2026
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join("models", "house_price_model.pkl")


@st.cache_resource
def load_trained_artifact():
    """Load serialized model bundle and metadata."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle
    except Exception as exc:
        st.error(f"Error loading model artifact: {exc}")
        return None


def main():
    # -------------------------------------------------------------------------
    # Application Header & Concept Explanation
    # -------------------------------------------------------------------------
    st.title("🏠 California House Price Predictor")
    st.markdown(
        """
        Welcome to the **California House Price Predictor**! This interactive machine learning application
        uses an **Ordinary Least Squares (OLS) Multiple Linear Regression** model trained on the benchmark
        California Housing dataset from scikit-learn (1990 U.S. Census data across 20,640 block groups).
        
        Adjust the neighborhood census characteristics below to generate a real-time price prediction.
        """
    )

    # -------------------------------------------------------------------------
    # Model Loading & Safety Check
    # -------------------------------------------------------------------------
    bundle = load_trained_artifact()

    if bundle is None:
        st.warning("⚠️ **Trained Model Artifact Not Found!**")
        st.info(
            f"The model file `{MODEL_PATH}` could not be located. "
            "Please train the model first by executing the following command in your terminal:\n\n"
            "```bash\npython train_model.py\n```\n"
            "Once training completes, refresh this page."
        )
        st.stop()

    model = bundle["model"]
    feature_names = bundle["feature_names"]
    target_multiplier = bundle.get("target_multiplier", 100_000)
    stats = bundle.get("feature_stats", {})
    metrics = bundle.get("metrics", {})
    test_metrics = metrics.get("test", {})

    # -------------------------------------------------------------------------
    # Sidebar: Model Performance & Information
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("📊 Model Overview")
        st.markdown(
            """
            **Algorithm:** Multiple Linear Regression  
            **Library:** `scikit-learn`  
            **Training Samples:** 16,512 (80%)  
            **Test Samples:** 4,128 (20%)  
            """
        )
        st.divider()

        st.subheader("🎯 Test-Set Evaluation")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label="Test R²",
                value=f"{test_metrics.get('r2', 0.5758)*100:.1f}%",
                help="Coefficient of determination: variance in house prices explained by the 8 features.",
            )
        with col_m2:
            st.metric(
                label="Test MAE",
                value=f"${test_metrics.get('mae_usd', 53320):,.0f}",
                help="Mean Absolute Error in US Dollars on unseen test data.",
            )

        st.metric(
            label="Test RMSE",
            value=f"${test_metrics.get('rmse_usd', 74558):,.0f}",
            help="Root Mean Squared Error (penalizes larger errors).",
        )
        st.divider()

        st.subheader("💡 Learned Intercept & Slope")
        st.caption(f"**Baseline Intercept (β₀):** {bundle.get('metrics', {}).get('intercept', -37.02):.3f}")
        st.markdown(
            """
            * **MedInc (+0.449):** High income drives property values upward.
            * **AveBedrms (+0.783):** Dedicated bedrooms add value.
            * **AveRooms (-0.123):** Collinear adjustment with AveBedrms.
            * **Lat/Long (-0.42 / -0.43):** Coastal southern areas command premium.
            """
        )

        st.divider()
        st.caption("AI & ML Internship Task 1 — Antigravity Workspace")

    # -------------------------------------------------------------------------
    # Input Features Form
    # -------------------------------------------------------------------------
    st.subheader("📝 Input Census Block Characteristics")
    st.markdown("Enter property and neighborhood features below. Default values reflect the **median** of the training dataset:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 👤 Socioeconomic & Household Metrics")

        # MedInc
        med_inc_default = float(stats.get("MedInc", {}).get("median", 3.53))
        med_inc = st.number_input(
            "Median Income (MedInc)",
            min_value=0.1,
            max_value=25.0,
            value=med_inc_default,
            step=0.1,
            help="Median income in block group in tens of thousands of USD (e.g., 3.53 = $35,300/year).",
        )
        st.caption(f"Equivalent annual income: **${med_inc * 10_000:,.0f} USD**")

        # HouseAge
        house_age_default = float(stats.get("HouseAge", {}).get("median", 29.0))
        house_age = st.slider(
            "Median House Age in Block (years)",
            min_value=1.0,
            max_value=60.0,
            value=house_age_default,
            step=1.0,
            help="Median age of houses in this census block (1 to 52 years in original census data).",
        )

        # AveRooms
        ave_rooms_default = float(stats.get("AveRooms", {}).get("median", 5.23))
        ave_rooms = st.number_input(
            "Average Rooms per Household (AveRooms)",
            min_value=1.0,
            max_value=30.0,
            value=ave_rooms_default,
            step=0.1,
            help="Average total rooms per household in the block (living rooms, bedrooms, kitchen, etc.).",
        )

        # AveBedrms
        ave_bedrms_default = float(stats.get("AveBedrms", {}).get("median", 1.05))
        ave_bedrms = st.number_input(
            "Average Bedrooms per Household (AveBedrms)",
            min_value=0.5,
            max_value=10.0,
            value=ave_bedrms_default,
            step=0.05,
            help="Average number of dedicated bedrooms per residential unit.",
        )

    with col2:
        st.markdown("##### 📍 Population & Geographic Coordinates")

        # Population
        population_default = float(stats.get("Population", {}).get("median", 1166.0))
        population = st.number_input(
            "Block Group Population",
            min_value=10.0,
            max_value=40000.0,
            value=population_default,
            step=50.0,
            help="Total number of residents living within this census block group.",
        )

        # AveOccup
        ave_occup_default = float(stats.get("AveOccup", {}).get("median", 2.82))
        ave_occup = st.number_input(
            "Average Household Occupancy (AveOccup)",
            min_value=0.5,
            max_value=25.0,
            value=ave_occup_default,
            step=0.1,
            help="Average number of household members residing together.",
        )

        # Latitude
        lat_default = float(stats.get("Latitude", {}).get("median", 34.26))
        latitude = st.slider(
            "Latitude (°N)",
            min_value=32.0,
            max_value=42.5,
            value=lat_default,
            step=0.05,
            help="Geographic latitude coordinate within California (approx. 32.5° to 42° N).",
        )

        # Longitude
        long_default = float(stats.get("Longitude", {}).get("median", -118.49))
        longitude = st.slider(
            "Longitude (°W)",
            min_value=-125.0,
            max_value=-114.0,
            value=long_default,
            step=0.05,
            help="Geographic longitude coordinate within California (approx. -124.5° to -114° W).",
        )

    # -------------------------------------------------------------------------
    # Input Validation Warnings
    # -------------------------------------------------------------------------
    warnings = []
    if stats:
        if med_inc > stats["MedInc"]["max"] or med_inc < stats["MedInc"]["min"]:
            warnings.append(f"Median Income ({med_inc:.2f}) falls outside the observed training dataset range [{stats['MedInc']['min']:.2f}, {stats['MedInc']['max']:.2f}].")
        if ave_rooms < ave_bedrms:
            warnings.append(f"Average Rooms ({ave_rooms:.1f}) is less than Average Bedrooms ({ave_bedrms:.2f}), which is physically unlikely.")
        if ave_rooms > 20.0:
            warnings.append(f"Average Rooms ({ave_rooms:.1f}) is unusually high compared to typical residential dwellings.")
        if ave_occup > 10.0:
            warnings.append(f"Average Occupancy ({ave_occup:.1f}) indicates dense multi-person occupancy uncommon in suburban blocks.")

    if warnings:
        for warn_msg in warnings:
            st.warning(f"⚠️ **Data Range Notice:** {warn_msg}")

    # Map preview of selected coordinate
    with st.expander("🗺️ Preview Selected California Location on Map", expanded=False):
        coord_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        st.map(coord_df, zoom=6)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Prediction Trigger & Output Card
    # -------------------------------------------------------------------------
    if st.button("🔮 Predict Median House Value", type="primary", use_container_width=True):
        # Assemble feature dictionary strictly matching training schema
        input_data = {
            "MedInc": [med_inc],
            "HouseAge": [house_age],
            "AveRooms": [ave_rooms],
            "AveBedrms": [ave_bedrms],
            "Population": [population],
            "AveOccup": [ave_occup],
            "Latitude": [latitude],
            "Longitude": [longitude],
        }

        # Build DataFrame with exact feature order
        input_df = pd.DataFrame(input_data)[feature_names]

        # Execute prediction
        raw_pred = float(model.predict(input_df)[0])
        dollar_pred = raw_pred * target_multiplier

        st.success("✅ **Prediction Generated Successfully!**")

        st.markdown("### 🏷️ Estimated Property Value")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="Estimated Dollar Value",
                value=f"${dollar_pred:,.2f} USD" if dollar_pred > 0 else "$0.00 USD (Floor)",
                delta=f"${(dollar_pred - 206855):+,.0f} vs CA Census Mean ($206,855)",
            )

        with res_col2:
            st.metric(
                label="Raw Dataset Units",
                value=f"{raw_pred:.4f}",
                help="Target variable MedHouseVal measured in units of $100,000 USD.",
            )

        with res_col3:
            # Benchmark percentile
            benchmark_text = "Mid-tier range"
            if raw_pred > 4.5:
                benchmark_text = "Top Tier / Luxury block (> $450k)"
            elif raw_pred < 1.2:
                benchmark_text = "Affordable / Entry block (< $120k)"
            st.metric(
                label="Market Segment",
                value=benchmark_text,
            )

        # Non-negative disclaimer notice if linear model produces negative or capped value
        if raw_pred < 0:
            st.error(
                "ℹ️ **Boundary Notice:** The unconstrained linear regression equation extrapolated to a negative value. "
                "In real estate, property values cannot fall below zero. This illustrates a key limitation of purely linear models."
            )
        elif raw_pred > 5.0:
            st.info(
                "ℹ️ **Ceiling Notice:** The predicted value exceeds $500,000. Recall that the training dataset's target "
                "variable was artificially capped at 5.0 ($500,000), meaning the training data lacked visibility into ultra-luxury price dynamics."
            )

        # Visual indicator bar
        pct_progress = min(max(raw_pred / 5.0, 0.0), 1.0)
        st.caption(f"Estimated Price Position on California 1990 Scale ($0 to $500,000):")
        st.progress(pct_progress)

    # -------------------------------------------------------------------------
    # Educational Disclaimer & Transparency
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown(
        """
        #### ⚖️ Educational & Analytical Disclaimer
        * **Census Block Aggregation:** Predictions generated by this tool represent the **median value of all homes** in a census 
          block group (typically 600–3,000 residents) based on 1990 U.S. Census statistics.
        * **Not an Appraisal Tool:** This model does not incorporate modern individual parcel attributes such as square footage, 
          architectural condition, interior renovations, school zone ratings, or current inflation rates. It must **not** be used for 
          real-world commercial appraisals, mortgage underwriting, or financial investments.
        * **Educational Purpose:** Developed for Artificial Intelligence & Machine Learning Internship — Task 1.
        """
    )


if __name__ == "__main__":
    main()
