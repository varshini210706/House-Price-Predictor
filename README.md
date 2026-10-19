# 🏠 California House Price Predictor
## Artificial Intelligence & Machine Learning Internship — Task 1

An end-to-end Machine Learning project implementing an **Ordinary Least Squares (OLS) Multiple Linear Regression** model on the **California Housing Dataset** from `scikit-learn`. 

This project covers the full lifecycle of a predictive modeling workflow: automated data ingestion, exploratory data analysis (EDA), data integrity checks, leakage-free train/test splitting, model training, coefficient interpretation, rigorous dual-unit metric evaluation ($100k & USD), diagnostic visualizations, automated 4-page PDF technical reporting, and an interactive Streamlit web application.

---

## 📑 Table of Contents
1. [Project Overview & Objective](#-project-overview--objective)
2. [Dataset Architecture & Features](#-dataset-architecture--features)
3. [Project Directory Structure](#-project-directory-structure)
4. [Machine Learning Workflow](#-machine-learning-workflow)
5. [Model Evaluation & Genuine Results](#-model-evaluation--genuine-results)
6. [Prerequisites & System Setup (Windows)](#-prerequisites--system-setup-windows)
7. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
   - [A. Train the Model & Generate Visualizations](#a-run-the-model-training-pipeline)
   - [B. Generate the 4-Page PDF Report](#b-generate-the-technical-pdf-report)
   - [C. Run the Jupyter Notebook](#c-open-and-execute-the-jupyter-notebook)
   - [D. Launch the Streamlit Web Application](#d-launch-the-streamlit-web-application)
   - [E. Run the Automated Test Suite](#e-run-the-automated-test-suite)
8. [Generated Artifacts & Output Files](#-generated-artifacts--output-files)
9. [Troubleshooting & Common Issues](#-troubleshooting--common-issues)
10. [Model Limitations & Future Roadmap](#-model-limitations--future-roadmap)

---

## 🎯 Project Overview & Objective

* **Title:** Build & Evaluate a Linear Regression Model (House Price Predictor)
* **Target Audience:** AI/ML Internship Mentors, Peer Students, and Practitioners.
* **Core Goal:** Predict median house values for California census districts using 8 continuous demographic and geographic variables.
* **Key Learning Outcomes:**
  - Understand the mathematics and assumptions of Ordinary Least Squares regression.
  - Diagnose multicollinearity between predictors (`AveRooms` vs `AveBedrms`).
  - Calculate and interpret **MAE**, **RMSE**, and **R²** in both normalized dataset units and real-world currency values.
  - Detect non-linear ceilings and homoscedasticity violations via residual analysis.
  - Package models with operational metadata for production-style serving.

---

## 📊 Dataset Architecture & Features

The dataset is obtained directly via `sklearn.datasets.fetch_california_housing(as_frame=True)` (originating from the 1990 U.S. Census). It contains **20,640 census block group observations** (clusters of roughly 600 to 3,000 residents).

### Feature Catalog

| Column | Type | Unit & Description | Observed Range | Correlation with Target ($r$) |
| :--- | :--- | :--- | :--- | :--- |
| **MedInc** | `float64` | Median income in block group (in tens of thousands USD, e.g., 3.5 = $35,000) | 0.50 – 15.00 | **+0.6881** (Strongest driver) |
| **HouseAge** | `float64` | Median age of housing units in years | 1.0 – 52.0 | +0.1056 |
| **AveRooms** | `float64` | Average total rooms per household | 0.85 – 141.9 | +0.1519 |
| **AveBedrms**| `float64` | Average bedrooms per household | 0.38 – 34.1 | -0.0467 |
| **Population**| `float64` | Total block group population | 3 – 35,682 | -0.0246 |
| **AveOccup** | `float64` | Average number of household members (occupancy) | 0.69 – 1,243.3 | -0.0237 |
| **Latitude** | `float64` | Block group centroid latitude | 32.54°N – 41.95°N | -0.1442 |
| **Longitude**| `float64` | Block group centroid longitude | -124.35°W – -114.31°W | -0.0459 |
| **MedHouseVal**| `float64` | **TARGET:** Median house value (units of $100,000 USD) | 0.15 – 5.00 | 1.0000 |

> **Important Unit Conversion:** Target values are scaled in units of **$100,000**.  
> To convert a prediction into US Dollars: $\text{Price (USD)} = \hat{y} \times 100{,}000$.

---

## 📁 Project Directory Structure

```text
House_Price_Predictor/
│
├── task1_ml_linear_regression.ipynb   # Complete, executed 21-section Jupyter Notebook
├── train_model.py                     # Modular training, evaluation & plotting pipeline
├── generate_report.py                 # 4-Page publication-quality PDF report builder
├── app.py                             # Interactive Streamlit prediction web application
├── requirements.txt                   # Production package dependencies
├── README.md                          # Comprehensive documentation (this file)
├── .gitignore                         # Standard Python, Jupyter, and virtualenv exclusions
│
├── models/
│   └── house_price_model.pkl          # Serialized model artifact bundle with metadata
│
├── reports/
│   ├── house_price_prediction_report.pdf # Formal 4-page academic/technical report
│   └── metrics.json                   # Machine-readable performance metrics
│
├── images/
│   ├── house_value_distribution.png   # Target distribution histogram & $500k ceiling
│   ├── feature_correlations.png       # Correlation matrix heatmap
│   ├── income_vs_house_value.png      # Income vs Price scatter with OLS trendline
│   ├── actual_vs_predicted.png        # Actual vs Predicted scatter with y=x diagonal
│   └── residual_distribution.png      # Residual normality & homoscedasticity checks
│
├── tests/
│   └── test_project.py                # Automated pytest test suite (7 tests)
│
└── utils/
    ├── __init__.py                    # Package exports
    └── data_utils.py                  # Shared constants, helpers & currency formatters
```

---

## 🔄 Machine Learning Workflow

```text
[ Data Ingestion ] ──────> [ Data Integrity Check ] ──────> [ Exploratory Data Analysis ]
(fetch_california_housing)  (0 Nulls, 0 Duplicates)          (Correlations, Histograms)
                                                                       │
                                                                       ▼
[ Web Serving / UI ] <───── [ Model Serialization ] <───── [ Model Evaluation & Residuals ]
(Streamlit: app.py)        (Joblib: house_price_model.pkl)   (MAE, RMSE, R² in $ & Units)
                                                                       ▲
                                                                       │
                                                            [ OLS Linear Regression ]
                                                            (fit strictly on 80% Train)
```

1. **Integrity Validation:** Zero null values, zero duplicate entries, zero infinite values.
2. **Leakage Prevention:** 80% training set (16,512 samples) and 20% test set (4,128 samples) partitioned using `random_state=42`.
3. **Model Estimation:** OLS analytical solver minimizing $\sum (y_i - \hat{y}_i)^2$.
4. **Dual-Unit Metrics:** Errors reported both in raw units ($100k) and US Dollars.

---

## 📈 Model Evaluation & Genuine Results

All metrics below are computed directly from the held-out test set ($N = 4,128$):

| Evaluation Metric | Mathematical Definition | Test Set (Dataset Units) | Test Set (US Dollars) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **MAE** | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | **0.5332** | **$53,320.01** | Typical average prediction deviation |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | **0.7456** | **$74,558.14** | Root mean squared error (penalizes outliers) |
| **$R^2$ Score** | $1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ | **0.5758** | **57.58%** | Variance explained by the 8 linear features |

### Generalization Stability
* **Training $R^2$:** 61.26%
* **Testing $R^2$:** 57.58%
* **Generalization Gap:** 3.68% (Indicating healthy generalization without overfitting).

### Learned Model Coefficients

$$\hat{y} = -37.0233 + 0.4487(\text{MedInc}) + 0.0097(\text{HouseAge}) - 0.1233(\text{AveRooms}) + 0.7831(\text{AveBedrms}) - 0.000002(\text{Population}) - 0.0035(\text{AveOccup}) - 0.4198(\text{Latitude}) - 0.4337(\text{Longitude})$$

* **Median Income ($\beta = +0.4487$):** Every $10,000 rise in median income raises house price by **$44,867**.
* **Bedrooms ($\beta = +0.7831$) vs Rooms ($\beta = -0.1233$):** Collinearity between rooms and bedrooms ($r = 0.85$) causes total rooms to have a negative sign when bedrooms is held constant.
* **Latitude ($\beta = -0.4198$) & Longitude ($\beta = -0.4337$):** Moving north or eastward away from the California coast lowers property value.

---

## 💻 Prerequisites & System Setup (Windows)

### Requirements
* Python 3.10, 3.11, or 3.12 (Tested on Python 3.11.5 64-bit on Windows).
* Windows Command Prompt (`cmd.exe`) or PowerShell.

### Step 1: Clone or Navigate to the Workspace
Open Windows Command Prompt:
```cmd
cd /d d:\House_Price_Predictor
```

### Step 2: Create and Activate a Virtual Environment
```cmd
python -m venv venv
call venv\Scripts\activate.bat
```
*(In PowerShell, run: `.\venv\Scripts\Activate.ps1`)*

### Step 3: Install Required Dependencies
```cmd
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Execution Guide

### A. Run the Model Training Pipeline
Trains the linear regression model, computes metrics, saves the model artifact, and outputs all 5 diagnostic figures:
```cmd
python train_model.py
```
**Expected Console Output:**
```text
====================================================================
  CALIFORNIA HOUSING PRICE PREDICTOR - TRAINING PIPELINE
====================================================================
[1/6] Loading California Housing dataset from scikit-learn...
  - Integrity check: 0 missing values across all features.
  - Integrity check: 0 duplicate rows detected.
  - Integrity check: 0 infinite values detected.
  - Loaded 20,640 records and 9 columns successfully.
[2/6] Splitting data into train (80%) and test (20%) sets (seed=42)...
[3/6] Training Ordinary Least Squares (OLS) Linear Regression model...
[4/6] Evaluating model performance...
[5/6] Generating and saving 5 high-resolution diagnostic visualizations to 'images/'...
  [OK] Saved: images\house_value_distribution.png
  [OK] Saved: images\feature_correlations.png
  [OK] Saved: images\income_vs_house_value.png
  [OK] Saved: images\actual_vs_predicted.png
  [OK] Saved: images\residual_distribution.png
[6/6] Serializing model and metadata to 'models\house_price_model.pkl'...
  [OK] Trained model bundle saved successfully: models\house_price_model.pkl
  [OK] Metrics JSON saved successfully: reports\metrics.json
[SUCCESS] Model training and evaluation pipeline completed successfully!
```

---

### B. Generate the Technical PDF Report
Generates the publication-grade 4-page technical report using ReportLab:
```cmd
python generate_report.py
```
The resulting PDF report will be located at:
`reports/house_price_prediction_report.pdf`

---

### C. Open and Execute the Jupyter Notebook
You can open `task1_ml_linear_regression.ipynb` directly in VS Code, Jupyter Notebook, or JupyterLab:
```cmd
jupyter notebook task1_ml_linear_regression.ipynb
```
*(Or simply double-click `task1_ml_linear_regression.ipynb` in the VS Code editor).*

To programmatically re-execute the notebook from top to bottom and re-render all cell outputs:
```cmd
python build_notebook.py
```

---

### D. Launch the Streamlit Web Application
To run the interactive web application:
```cmd
python -m streamlit run app.py
```
Once launched, open your web browser at:
`http://localhost:8501`

**Application Capabilities:**
* Interactive sliders and number inputs for all 8 features.
* Range validation warnings when inputs fall outside observed census training limits.
* Real-time property valuation in both raw units ($100k) and US Dollars ($).
* Interactive California coordinate map preview.
* Model performance summary cards in the sidebar.

---

### E. Run the Automated Test Suite
To run all 7 automated unit and regression tests:
```cmd
python -m pytest -v tests/test_project.py
```
**Expected Test Output:**
```text
tests/test_project.py::test_data_loading PASSED                          [ 14%]
tests/test_project.py::test_model_artifact PASSED                        [ 28%]
tests/test_project.py::test_model_prediction PASSED                      [ 42%]
tests/test_project.py::test_evaluation_metrics PASSED                    [ 57%]
tests/test_project.py::test_images_generated PASSED                      [ 71%]
tests/test_project.py::test_pdf_report_exists_and_pages PASSED           [ 85%]
tests/test_project.py::test_notebook_validity PASSED                     [100%]

============================== 7 passed in 2.70s ==============================
```

---

## 📦 Generated Artifacts & Output Files

| File / Folder | Purpose |
| :--- | :--- |
| `models/house_price_model.pkl` | Serialized model dictionary containing fitted weights, feature names, training statistics, and test metrics. |
| `reports/house_price_prediction_report.pdf` | Formal 4-page technical report with executive summary, tables, and embedded figures. |
| `reports/metrics.json` | JSON file containing train & test MAE, RMSE, and R² for automated verification. |
| `images/house_value_distribution.png` | Histogram and KDE showing target distribution and the $500k ceiling truncation. |
| `images/feature_correlations.png` | Correlation matrix heatmap between all numerical features and the target. |
| `images/income_vs_house_value.png` | Scatter plot of Median Income vs Median House Value with regression trendline. |
| `images/actual_vs_predicted.png` | Test-set scatter comparing actual prices against predictions with $y = x$ reference. |
| `images/residual_distribution.png` | Dual-panel diagnostic: residual error histogram and residual vs predicted homoscedasticity plot. |

---

## 🛠️ Troubleshooting & Common Issues

### 1. `ModuleNotFoundError: No module named 'streamlit'` (or `reportlab`, `seaborn`)
* **Cause:** The virtual environment is not active or dependencies were installed globally.
* **Solution:** Run `python -m pip install -r requirements.txt` using the active Python executable.

### 2. `'streamlit' is not recognized as an internal or external command`
* **Cause:** Windows Scripts directory is not on system PATH.
* **Solution:** Always run Streamlit using the Python module syntax:
  ```cmd
  python -m streamlit run app.py
  ```

### 3. `UnicodeEncodeError: 'charmap' codec can't encode character`
* **Cause:** Windows Command Prompt default code page (`cp1252`) cannot print special unicode characters (like `✓` or `²`).
* **Solution:** Our scripts are engineered to use ASCII-safe logging (`[OK]`, `R-squared`). If running other commands, run `chcp 65001` to enable UTF-8 support in Command Prompt.

### 4. Scikit-Learn California Dataset Download Timeout
* **Cause:** Network firewall or temporary downtime fetching from GitHub/scikit-learn mirror.
* **Solution:** Verify your internet connection. Scikit-learn caches the dataset automatically to `~/scikit_learn_data/`.

---

## ⚠️ Model Limitations & Future Roadmap

### Technical Limitations
1. **Linearity Assumption:** Real estate pricing is heavily governed by non-linear spatial interactions (e.g. proximity to ocean shores, specific school boundaries) that cannot be captured by linear planes.
2. **Census Block Aggregation:** The model estimates block-level medians (600–3,000 residents), not individual house attributes (e.g., lot size, interior condition, number of bathrooms, swimming pool).
3. **Upper Ceiling Truncation:** Properties valued at or above $500,000 were artificially capped at 5.0 in the 1990 census, introducing an underprediction bias at the luxury end.
4. **Historical Baseline:** Data reflects 1990 economic conditions and requires inflation adjustments for modern commercial use.

### Recommended Future Roadmap
* **Non-Linear Tree Ensembles:** Deploy **Random Forest**, **XGBoost**, or **LightGBM** to capture non-linear feature interactions and reduce RMSE.
* **Spatial Feature Engineering:** Compute Euclidean distance to major coastal economic hubs (San Francisco, Silicon Valley, Los Angeles, San Diego).
* **Feature Ratio Creation:** Create domain features such as `Bedrooms_per_Room = AveBedrms / AveRooms` and `Rooms_per_Person = AveRooms / AveOccup`.
* **Outlier Regularization:** Apply **Huber Regression** or `RobustScaler` to dampen the influence of extreme rural occupancy outliers.

---

## 🎓 Academic Integrity & Authorship
This project was implemented as part of the **Artificial Intelligence & Machine Learning Internship — Task 1**. All code, metrics, diagnostic plots, reports, and web interfaces have been verified and validated inside the Google Antigravity development workspace.
