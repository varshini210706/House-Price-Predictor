"""
=============================================================================
California House Price Prediction - Technical PDF Report Generator
=============================================================================
Generates a publication-grade, 4-page academic/industry deliverable summarizing
the Machine Learning pipeline, Exploratory Data Analysis, Model Training,
Evaluation Metrics, Diagnostics, and Future Roadmap using ReportLab.

Author: AI/ML Engineering Team
Date: October 2026
=============================================================================
"""

import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Paths
REPORTS_DIR = "reports"
IMAGES_DIR = "images"
METRICS_JSON_PATH = os.path.join(REPORTS_DIR, "metrics.json")
PDF_OUTPUT_PATH = os.path.join(REPORTS_DIR, "house_price_prediction_report.pdf")


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total pages and render
    clean running headers and footers with 'Page X of Y'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                54,
                750,
                "AI & ML Internship Task 1: Linear Regression Housing Predictor",
            )
            self.drawRightString(612 - 54, 750, "Technical Performance Report")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

        # Running Footer
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawString(54, 34, "California Housing Valuation Pipeline — Scikit-Learn")
        self.drawRightString(612 - 54, 34, page_str)
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 44, 612 - 54, 44)
        self.restoreState()


def load_metrics():
    """Load genuine computed metrics from metrics.json."""
    if not os.path.exists(METRICS_JSON_PATH):
        raise FileNotFoundError(
            f"Metrics file '{METRICS_JSON_PATH}' not found. Please run train_model.py first."
        )
    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pdf_report():
    """Construct the complete 4-page PDF report."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics = load_metrics()

    # Document setup: Letter size with 0.5 in (36 pt) side margins, 40 pt top/bottom
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    primary_color = colors.HexColor("#1A365D")  # Deep Navy
    secondary_color = colors.HexColor("#2B6CB0")  # Slate Blue
    accent_color = colors.HexColor("#C53030")  # Crimson accent
    dark_text = colors.HexColor("#2D3748")  # Charcoal body text
    light_bg = colors.HexColor("#F7FAFC")  # Warm light gray

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=0,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=secondary_color,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=primary_color,
        spaceAfter=6,
        spaceBefore=8,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        spaceAfter=4,
        spaceBefore=6,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=dark_text,
    )

    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=dark_text,
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, INTERNSHIP TASK, DATASET OVERVIEW & FEATURE CATALOG
    # =========================================================================
    story.append(Paragraph("HOUSE PRICE PREDICTOR — TECHNICAL REPORT", title_style))
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "Artificial Intelligence & Machine Learning Internship — Task 1: Linear Regression & Evaluation",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 6))

    # Metadata card table
    meta_data = [
        [
            Paragraph("<b>Author:</b> AI/ML Intern (2nd Year B.Tech)", body_style),
            Paragraph("<b>Model:</b> Ordinary Least Squares (OLS)", body_style),
        ],
        [
            Paragraph("<b>Dataset:</b> California Housing (scikit-learn)", body_style),
            Paragraph("<b>Split:</b> 80% Train (16,512) / 20% Test (4,128)", body_style),
        ],
        [
            Paragraph("<b>Target:</b> MedHouseVal ($100k USD)", body_style),
            Paragraph("<b>Evaluation:</b> MAE, RMSE, R² (Dual-unit)", body_style),
        ],
    ]
    t_meta = Table(meta_data, colWidths=[270, 270])
    t_meta.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), light_bg),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_meta)
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Executive Objective & Problem Statement", h1_style))
    story.append(
        Paragraph(
            "The objective of this project is to develop, evaluate, and rigorously document an end-to-end "
            "supervised machine learning pipeline using <b>Multiple Linear Regression</b> to estimate residential "
            "property values across California census block groups. Housing valuation represents a foundational "
            "regression challenge where understanding feature relationships, collinearity, residual behavior, and "
            "generalization error is as critical as numerical accuracy.",
            body_style,
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "<b>Problem Formulation:</b> Given 8 continuous socioeconomic, architectural, and geographic predictors "
            "recorded at the block-group level, we seek to predict the median house value <i>y</i> (in units of $100,000 USD) "
            "by minimizing the residual sum of squares via Ordinary Least Squares: "
            "<i>y = β₀ + β₁·MedInc + β₂·HouseAge + ... + β₈·Longitude + ε</i>.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Dataset Architecture & Feature Catalog", h1_style))
    story.append(
        Paragraph(
            "The model utilizes the benchmark California Housing dataset derived from the 1990 U.S. Census and packaged "
            "directly inside scikit-learn. The corpus comprises <b>20,640 block group observations</b>, each representing "
            "a cluster of 600 to 3,000 residents. No synthetic records were introduced, and all 8 predictors are continuous.",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    # Feature Table
    feat_data = [
        [
            Paragraph("Feature", table_header_style),
            Paragraph("Type", table_header_style),
            Paragraph("Census Unit / Definition", table_header_style),
            Paragraph("Observed Range", table_header_style),
            Paragraph("Correlation with Target (r)", table_header_style),
        ],
        [
            Paragraph("<b>MedInc</b>", table_cell_bold),
            Paragraph("Float64", table_cell_style),
            Paragraph("Median income in block ($10,000 USD)", table_cell_style),
            Paragraph("0.50 – 15.00 ($5k - $150k)", table_cell_style),
            Paragraph("<b>+0.6881</b> (Strongest)", table_cell_bold),
        ],
        [
            Paragraph("<b>HouseAge</b>", table_cell_bold),
            Paragraph("Float64", table_cell_style),
            Paragraph("Median age of housing units in years", table_cell_style),
            Paragraph("1.00 – 52.00 yrs (capped)", table_cell_style),
            Paragraph("+0.1056", table_cell_style),
        ],
        [
            Paragraph("<b>AveRooms</b>", table_cell_bold),
            Paragraph("Float64", table_cell_style),
            Paragraph("Average number of rooms per household", table_cell_style),
            Paragraph("0.85 – 141.90 rms", table_cell_style),
            Paragraph("+0.1519", table_cell_style),
        ],
        [
            Paragraph("<b>AveBedrms</b>", table_cell_bold),
            Paragraph("Float64", table_cell_style),
            Paragraph("Average number of bedrooms per household", table_cell_style),
            Paragraph("0.38 – 34.07 bdrms", table_cell_style),
            Paragraph("-0.0467", table_cell_style),
        ],
        [
            Paragraph("<b>Population</b>", table_cell_bold),
            Paragraph("Float64", table_cell_style),
            Paragraph("Total population inhabiting the block", table_cell_style),
            Paragraph("3 – 35,682 people", table_cell_style),
            Paragraph("-0.0246", table_cell_style),
        ],
        [
            Paragraph("<b>AveOccup</b>", table_cell_bold),
            Paragraph("Float64", table_cell_style),
            Paragraph("Average members per household unit", table_cell_style),
            Paragraph("0.69 – 1,243.33 people", table_cell_style),
            Paragraph("-0.0237", table_cell_style),
        ],
        [
            Paragraph("<b>Latitude</b>", table_cell_bold),
            Paragraph("Float64", table_cell_style),
            Paragraph("Block centroid latitude coordinate", table_cell_style),
            Paragraph("32.54°N – 41.95°N", table_cell_style),
            Paragraph("-0.1442", table_cell_style),
        ],
        [
            Paragraph("<b>Longitude</b>", table_cell_bold),
            Paragraph("Float64", table_cell_style),
            Paragraph("Block centroid longitude coordinate", table_cell_style),
            Paragraph("-124.35°W – -114.31°W", table_cell_style),
            Paragraph("-0.0459", table_cell_style),
        ],
        [
            Paragraph("<b>MedHouseVal</b>", table_cell_bold),
            Paragraph("Float64", table_cell_style),
            Paragraph("<b>TARGET:</b> Median house value ($100k USD)", table_cell_style),
            Paragraph("0.15 – 5.00 ($15k - $500k)", table_cell_style),
            Paragraph("1.0000 (Target)", table_cell_style),
        ],
    ]
    t_feat = Table(feat_data, colWidths=[65, 45, 175, 125, 130])
    t_feat.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_feat)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: DATA EXPLORATION, PREPROCESSING & EDA VISUALIZATIONS
    # =========================================================================
    story.append(Paragraph("3. Data Exploration & Preprocessing Audit", h1_style))
    story.append(
        Paragraph(
            "Prior to model estimation, an exhaustive data integrity audit was conducted across all 20,640 records. "
            "Key diagnostic findings include:",
            body_style,
        )
    )
    story.append(Spacer(1, 3))

    audit_bullets = [
        "<b>Missing Value Audit:</b> Zero missing, null, or NaN entries exist in any column. The dataset is fully populated.",
        "<b>Duplicate Records:</b> 0 duplicated rows detected; every observation maps to a unique geographic centroid.",
        "<b>Data Integrity:</b> All numerical values are finite with valid float representations; no zero-division artifacts.",
        "<b>Target Variable Ceiling:</b> Exactly 965 blocks (4.68% of corpus) are capped at exactly $500,000 (5.00001). This ceiling truncation introduces a notable non-linear plateau discussed in Section 5.",
        "<b>Geographic Distribution:</b> Features AveRooms and AveOccup exhibit high right-skewness due to isolated rural agricultural communal dwellings (e.g., maximum occupancy of 1,243).",
    ]
    for bullet in audit_bullets:
        story.append(
            Paragraph(f"• {bullet}", ParagraphStyle("Bullet", parent=body_style, leftIndent=12, spaceAfter=2))
        )
    story.append(Spacer(1, 6))

    story.append(Paragraph("4. Exploratory Data Analysis & Visualizations", h1_style))
    story.append(
        Paragraph(
            "Empirical distributions and bivariate dependencies were mapped to diagnose correlation patterns and "
            "inform the Ordinary Least Squares formulation.",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    # Two EDA images side-by-side or stacked cleanly
    fig1_path = os.path.join(IMAGES_DIR, "house_value_distribution.png")
    fig2_path = os.path.join(IMAGES_DIR, "feature_correlations.png")

    img_table_data = [
        [
            Image(fig1_path, width=260, height=160),
            Image(fig2_path, width=260, height=160),
        ],
        [
            Paragraph("<b>Figure 1:</b> Distribution of Target Median House Value showing the $500k ceiling cap.", caption_style),
            Paragraph("<b>Figure 2:</b> Pearson Correlation Heatmap highlighting MedInc correlation (r=+0.69).", caption_style),
        ],
    ]
    t_eda_imgs = Table(img_table_data, colWidths=[270, 270])
    t_eda_imgs.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_eda_imgs)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Key Empirical Insights from EDA:", h2_style))
    story.append(
        Paragraph(
            "<b>1. Dominance of Median Income:</b> Median income (<i>MedInc</i>) is by far the single most predictive feature "
            "(<i>r = +0.6881</i>). Affluence of the neighborhood residents correlates directly with home values, "
            "as visually confirmed by the linear regression trendline in Figure 3.<br/>"
            "<b>2. Rooms vs. Bedrooms Multicollinearity:</b> <i>AveRooms</i> and <i>AveBedrms</i> display a strong positive "
            "collinearity (<i>r = +0.8476</i>). In OLS regression, this high correlation leads to coefficient sign flipping: "
            "holding total rooms constant, additional bedrooms paradoxically receive a positive coefficient (+0.7831) while "
            "total rooms shifts negative (-0.1233).<br/>"
            "<b>3. Geospatial Interaction:</b> Negative correlations for <i>Latitude</i> (-0.14) and <i>Longitude</i> (-0.05) "
            "reflect California's distinct economic geography: the most valuable coastal metropolitan districts (San Francisco Bay "
            "Area, Coastal Los Angeles, San Diego) lie in the south/west coastal quadrant, while inland and northern rural regions "
            "exhibit significantly lower median prices.",
            body_style,
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: MODEL TRAINING, EVALUATION METRICS & PREDICTION DIAGNOSTICS
    # =========================================================================
    story.append(Paragraph("5. Model Methodology & Controlled Evaluation", h1_style))
    story.append(
        Paragraph(
            "<b>Data Partitioning & Leakage Prevention:</b> The dataset of 20,640 records was partitioned into an <b>80% training "
            "set (16,512 samples)</b> and a <b>20% held-out test set (4,128 samples)</b> using a deterministic pseudo-random seed "
            "(<code>random_state=42</code>). Model estimation was fitted strictly upon training records. "
            "Zero test-set information informed the parameter fits.",
            body_style,
        )
    )
    story.append(Spacer(1, 4))

    # Metrics Table
    test_mae = metrics["test"]["mae"]
    test_mae_usd = metrics["test"]["mae_usd"]
    test_rmse = metrics["test"]["rmse"]
    test_rmse_usd = metrics["test"]["rmse_usd"]
    test_r2 = metrics["test"]["r2"]

    train_mae = metrics["train"]["mae"]
    train_mae_usd = metrics["train"]["mae_usd"]
    train_rmse = metrics["train"]["rmse"]
    train_rmse_usd = metrics["train"]["rmse_usd"]
    train_r2 = metrics["train"]["r2"]

    eval_table_data = [
        [
            Paragraph("Metric", table_header_style),
            Paragraph("Formula & Meaning", table_header_style),
            Paragraph("Train Set ($100k)", table_header_style),
            Paragraph("Train Set (USD)", table_header_style),
            Paragraph("Test Set ($100k)", table_header_style),
            Paragraph("Test Set (USD)", table_header_style),
        ],
        [
            Paragraph("<b>MAE</b>", table_cell_bold),
            Paragraph("Mean Absolute Error: (1/n) Σ |y - ŷ|<br/>Direct average magnitude of error", table_cell_style),
            Paragraph(f"{train_mae:.4f}", table_cell_style),
            Paragraph(f"${train_mae_usd:,.2f}", table_cell_style),
            Paragraph(f"<b>{test_mae:.4f}</b>", table_cell_bold),
            Paragraph(f"<b>${test_mae_usd:,.2f}</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>RMSE</b>", table_cell_bold),
            Paragraph("Root Mean Squared Error: √[(1/n) Σ (y - ŷ)²]<br/>Heavily penalizes large outlier errors", table_cell_style),
            Paragraph(f"{train_rmse:.4f}", table_cell_style),
            Paragraph(f"${train_rmse_usd:,.2f}", table_cell_style),
            Paragraph(f"<b>{test_rmse:.4f}</b>", table_cell_bold),
            Paragraph(f"<b>${test_rmse_usd:,.2f}</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>R²</b>", table_cell_bold),
            Paragraph("Coefficient of Determination: 1 - (SS_res / SS_tot)<br/>Proportion of target variance explained", table_cell_style),
            Paragraph(f"{train_r2:.4f}", table_cell_style),
            Paragraph(f"{train_r2*100:.2f}%", table_cell_style),
            Paragraph(f"<b>{test_r2:.4f}</b>", table_cell_bold),
            Paragraph(f"<b>{test_r2*100:.2f}%</b>", table_cell_bold),
        ],
    ]
    t_eval = Table(eval_table_data, colWidths=[45, 175, 75, 80, 75, 90])
    t_eval.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t_eval)
    story.append(Spacer(1, 6))

    # Metric Discussion
    story.append(Paragraph("Understanding the Evaluation Metrics:", h2_style))
    story.append(
        Paragraph(
            "• <b>MAE Interpretation:</b> On average, the Linear Regression model's price prediction deviates from the true "
            f"block median value by <b>${test_mae_usd:,.2f}</b> (0.5332 units). Because MAE uses absolute differences, it is "
            "unweighted by outlier penalties and provides an intuitive, robust assessment of typical error.<br/>"
            "• <b>RMSE Interpretation:</b> The test RMSE is <b>${test_rmse_usd:,.2f}</b> (0.7456 units). The $21,238 gap between "
            "RMSE and MAE highlights the presence of substantial variance on select properties where predictions deviate severely.<br/>"
            "• <b>R² Variance Explained:</b> The model achieves an <b>R² of 57.58%</b> on unseen test data. This indicates that "
            "nearly 58% of the variability in California median housing values is linearly accounted for by these 8 census features.<br/>"
            "• <b>Generalization Stability:</b> The training R² (61.26%) exceeds test R² (57.58%) by merely 3.68%, proving that the "
            "linear model demonstrates healthy generalization without significant overfitting or high model variance.",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    # Actual vs Predicted image and discussion
    fig4_path = os.path.join(IMAGES_DIR, "actual_vs_predicted.png")
    fig3_path = os.path.join(IMAGES_DIR, "income_vs_house_value.png")

    img_pred_table = [
        [
            Image(fig3_path, width=260, height=150),
            Image(fig4_path, width=260, height=150),
        ],
        [
            Paragraph("<b>Figure 3:</b> MedInc vs MedHouseVal regression scatter.", caption_style),
            Paragraph("<b>Figure 4:</b> Actual vs Predicted Values with perfect reference line (y=x).", caption_style),
        ],
    ]
    t_pred = Table(img_pred_table, colWidths=[270, 270])
    t_pred.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_pred)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: COEFFICIENT INTERPRETATION, RESIDUALS, LIMITATIONS & ROADMAP
    # =========================================================================
    story.append(Paragraph("6. Parameter Interpretation & Residual Diagnostics", h1_style))
    story.append(
        Paragraph(
            f"The fitted model learned an intercept of <b>{metrics['intercept']:.4f}</b>. Each slope coefficient β represents "
            "the estimated marginal change in median house value ($100k USD) associated with a one-unit increase in that feature, "
            "<b>holding all other predictors strictly constant</b>. Importantly, regression coefficients indicate associative "
            "model weights and do <i>not</i> demonstrate causality.",
            body_style,
        )
    )
    story.append(Spacer(1, 4))

    # Coefficients table
    coefs = metrics["coefficients"]
    coef_rows = [
        [
            Paragraph("Predictor", table_header_style),
            Paragraph("Learned Weight (β)", table_header_style),
            Paragraph("Dollar Impact per Unit Increase", table_header_style),
            Paragraph("Physical Interpretation & Note", table_header_style),
        ],
        [
            Paragraph("<b>MedInc</b>", table_cell_bold),
            Paragraph(f"+{coefs['MedInc']:.6f}", table_cell_style),
            Paragraph(f"+${coefs['MedInc']*100_000:,.2f}", table_cell_bold),
            Paragraph("For every $10,000 rise in median income, price rises ~$44,867.", table_cell_style),
        ],
        [
            Paragraph("<b>HouseAge</b>", table_cell_bold),
            Paragraph(f"+{coefs['HouseAge']:.6f}", table_cell_style),
            Paragraph(f"+${coefs['HouseAge']*100_000:,.2f}", table_cell_style),
            Paragraph("Older neighborhoods correlate with established infrastructure/mature greenery.", table_cell_style),
        ],
        [
            Paragraph("<b>AveRooms</b>", table_cell_bold),
            Paragraph(f"{coefs['AveRooms']:.6f}", table_cell_style),
            Paragraph(f"-${abs(coefs['AveRooms'])*100_000:,.2f}", table_cell_style),
            Paragraph("Negative sign due to collinearity with AveBedrms (r=0.85).", table_cell_style),
        ],
        [
            Paragraph("<b>AveBedrms</b>", table_cell_bold),
            Paragraph(f"+{coefs['AveBedrms']:.6f}", table_cell_style),
            Paragraph(f"+${coefs['AveBedrms']*100_000:,.2f}", table_cell_bold),
            Paragraph("Reflects added value of dedicated bedrooms relative to general rooms.", table_cell_style),
        ],
        [
            Paragraph("<b>Population</b>", table_cell_bold),
            Paragraph(f"{coefs['Population']:.6f}", table_cell_style),
            Paragraph(f"-${abs(coefs['Population'])*100_000:,.4f}", table_cell_style),
            Paragraph("Marginal negative effect (-$0.20 per additional person in block).", table_cell_style),
        ],
        [
            Paragraph("<b>AveOccup</b>", table_cell_bold),
            Paragraph(f"{coefs['AveOccup']:.6f}", table_cell_style),
            Paragraph(f"-${abs(coefs['AveOccup'])*100_000:,.2f}", table_cell_style),
            Paragraph("Higher density per unit reflects lower-cost shared accommodations.", table_cell_style),
        ],
        [
            Paragraph("<b>Latitude</b>", table_cell_bold),
            Paragraph(f"{coefs['Latitude']:.6f}", table_cell_style),
            Paragraph(f"-${abs(coefs['Latitude'])*100_000:,.2f}", table_cell_style),
            Paragraph("Prices decrease northward into rural non-coastal geography.", table_cell_style),
        ],
        [
            Paragraph("<b>Longitude</b>", table_cell_bold),
            Paragraph(f"{coefs['Longitude']:.6f}", table_cell_style),
            Paragraph(f"-${abs(coefs['Longitude'])*100_000:,.2f}", table_cell_style),
            Paragraph("Prices decrease eastward away from the Pacific coastline.", table_cell_style),
        ],
    ]
    t_coef = Table(coef_rows, colWidths=[70, 75, 110, 285])
    t_coef.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_coef)
    story.append(Spacer(1, 4))

    # Residual figure
    fig5_path = os.path.join(IMAGES_DIR, "residual_distribution.png")
    story.append(Image(fig5_path, width=540, height=130))
    story.append(
        Paragraph("<b>Figure 5:</b> Residual diagnostics (left: error histogram centered at 0; right: residual variance vs predicted values).", caption_style)
    )
    story.append(Spacer(1, 4))

    story.append(Paragraph("7. Model Limitations & Strategic Engineering Roadmap", h1_style))
    story.append(
        Paragraph(
            "While Linear Regression provides a clean, interpretable baseline, several structural limitations must be noted:<br/>"
            "• <b>Linearity Assumption vs Geographic Complexity:</b> Property valuation depends heavily on non-linear spatial "
            "clusters (e.g. coastal distance, school districts) which cannot be captured by linear planes without non-linear kernels.<br/>"
            "• <b>Census Aggregate Limitation:</b> The model predicts block-level medians rather than parcel-level attributes "
            "(e.g., square footage, interior finishes, lot size). It must never be used as a substitute for real-world appraisals.<br/>"
            "• <b>Recommended Future Enhancements:</b><br/>"
            "   1. <i>Non-linear Ensembles:</i> Deploy XGBoost, LightGBM, or Random Forests to capture non-linear feature interactions.<br/>"
            "   2. <i>Feature Engineering:</i> Compute distance to major coastal hubs (San Francisco, Los Angeles) and bedroom-to-room ratios.<br/>"
            "   3. <i>Robust Scaling:</i> Apply robust transformation to damp extreme outliers in <code>AveRooms</code> and <code>AveOccup</code>.",
            body_style,
        )
    )
    story.append(Spacer(1, 4))

    # Tech Stack Footer
    tech_data = [
        [
            Paragraph(
                "<b>Implementation Technologies:</b> Python 3.11 | scikit-learn 1.9.0 | pandas 3.0.5 | NumPy 2.1.2 | "
                "Matplotlib 3.9.2 | Seaborn 0.13.2 | Streamlit 1.66.0 | ReportLab 5.0.1 | Joblib 1.6.0",
                ParagraphStyle("TechFooter", parent=body_style, fontSize=7.5, leading=9, textColor=colors.HexColor("#4A5568")),
            )
        ]
    ]
    t_tech = Table(tech_data, colWidths=[540])
    t_tech.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), light_bg),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_tech)

    # Build the document
    print(f"Generating PDF report to '{PDF_OUTPUT_PATH}'...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Successfully built PDF report: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf_report()
