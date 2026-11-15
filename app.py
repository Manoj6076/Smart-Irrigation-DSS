
# ================================================================
# SMART IRRIGATION DECISION SUPPORT SYSTEM
# M.TECH LEVEL PROFESSIONAL DASHBOARD
# Model: Gradient Boosting Classifier
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from datetime import datetime

# ================================================================
# 1. PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Agri Intelligence | Smart Irrigation DSS",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_gradient_boosting_model.pkl"
DATA_PATH = BASE_DIR / "data" / "cropdata_clean.csv"

# ================================================================
# 2. PROFESSIONAL CUSTOM CSS
# ================================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F4F7F5;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102B25 0%, #173F32 100%);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stSelectbox div,
[data-testid="stSidebar"] .stNumberInput input {
    color: #173F32 !important;
}

.main-header {
    background: linear-gradient(115deg, #102B25, #23634A);
    padding: 30px 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 6px 22px rgba(16,43,37,0.12);
}

.main-header h1 {
    font-size: 30px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.main-header p {
    color: #D0E7D8;
    font-size: 14px;
    margin-bottom: 0;
}

.section-title {
    color: #173F32;
    font-size: 21px;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 15px;
}

.kpi-card {
    background: white;
    padding: 21px;
    border-radius: 14px;
    border: 1px solid #E2EBE5;
    box-shadow: 0 3px 12px rgba(20,60,40,0.045);
    min-height: 125px;
}

.kpi-label {
    color: #718078;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.kpi-value {
    color: #173F32;
    font-size: 27px;
    font-weight: 800;
    margin-top: 12px;
}

.kpi-caption {
    color: #87948C;
    font-size: 12px;
    margin-top: 5px;
}

.result-card {
    background: linear-gradient(120deg, #E4F4E9, #F5FBF6);
    border: 1px solid #B9DEC5;
    padding: 25px;
    border-radius: 15px;
}

.result-heading {
    color: #17643C;
    font-size: 25px;
    font-weight: 800;
}

.info-card {
    background: white;
    padding: 20px;
    border: 1px solid #E2EBE5;
    border-radius: 14px;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #85928A;
    font-size: 12px;
}

div.stButton > button {
    background: #1F704B;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 700;
    min-height: 45px;
}

div.stButton > button:hover {
    background: #155638;
    color: white;
    border: none;
}

[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #E2EBE5;
}

</style>
""", unsafe_allow_html=True)


# ================================================================
# 3. LOAD MODEL AND DATASET
# ================================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    data = load_dataset()

except Exception as error:
    st.error(f"Application initialization failed: {error}")
    st.stop()


# ================================================================
# 4. DATA VALIDATION
# ================================================================

REQUIRED_COLUMNS = [
    "crop ID",
    "soil_type",
    "Seedling Stage",
    "MOI",
    "temp",
    "humidity",
    "result"
]

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in data.columns
]

if missing_columns:
    st.error(
        f"Missing dataset columns: {missing_columns}"
    )
    st.stop()

FEATURE_COLUMNS = [
    "crop ID",
    "soil_type",
    "Seedling Stage",
    "MOI",
    "temp",
    "humidity"
]

if not all(
    col in data.columns
    for col in FEATURE_COLUMNS
):
    st.error("Dataset feature validation failed.")
    st.stop()


# ================================================================
# 5. SESSION STATE
# ================================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# ================================================================
# 6. HELPER FUNCTIONS
# ================================================================

def get_class_names():
    if hasattr(model, "classes_"):
        return model.classes_

    if hasattr(model, "named_steps"):
        final_model = model.named_steps["model"]
        return final_model.classes_

    return np.sort(data["result"].dropna().unique())


def get_confidence_level(confidence):

    if confidence >= 0.80:
        return "High Confidence"

    elif confidence >= 0.60:
        return "Moderate Confidence"

    return "Low Confidence"


def get_feature_importance():

    try:
        if not hasattr(model, "named_steps"):
            return None

        estimator = model.named_steps["model"]

        if not hasattr(estimator, "feature_importances_"):
            return None

        preprocessor = model.named_steps["preprocessor"]

        feature_names = preprocessor.get_feature_names_out()

        importance = estimator.feature_importances_

        result = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        # Aggregate encoded categorical features
        # into their original feature groups.

        def original_feature(name):
            name = str(name)

            for feature in FEATURE_COLUMNS:
                if name.startswith("categorical__" + feature):
                    return feature

                if name.startswith("numerical__" + feature):
                    return feature

            return name

        result["Feature"] = result["Feature"].apply(
            original_feature
        )

        result = (
            result.groupby("Feature", as_index=False)
            ["Importance"].sum()
            .sort_values("Importance", ascending=False)
        )

        return result

    except Exception:
        return None


def make_prediction(
    crop,
    soil,
    stage,
    moi,
    temperature,
    humidity
):

    input_data = pd.DataFrame({
        "crop ID": [crop],
        "soil_type": [soil],
        "Seedling Stage": [stage],
        "MOI": [moi],
        "temp": [temperature],
        "humidity": [humidity]
    })

    predicted_class = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = get_class_names()

        probability_data = {
            str(cls): float(prob)
            for cls, prob in zip(classes, probabilities)
        }

        confidence = float(np.max(probabilities))

    else:
        probability_data = {}
        confidence = None

    return {
        "Crop": crop,
        "Soil Type": soil,
        "Growth Stage": stage,
        "MOI": moi,
        "Temperature": temperature,
        "Humidity": humidity,
        "Predicted Class": predicted_class,
        "Decision Level": f"Decision Level {predicted_class}",
        "Confidence": confidence,
        "Confidence Level": (
            get_confidence_level(confidence)
            if confidence is not None
            else "Not available"
        ),
        "Class Probabilities": probability_data,
        "Timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }


# ================================================================
# 7. SIDEBAR NAVIGATION
# ================================================================

st.sidebar.markdown("""
<div style="padding:15px 5px 25px 5px;">

<div style="font-size:35px;">🌱</div>

<div style="font-size:21px;font-weight:800;">
AGRI INTELLIGENCE
</div>

<div style="font-size:11px;color:#B8D6C3;">
SMART AGRICULTURE ANALYTICS
</div>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard Overview",
        "Irrigation Prediction",
        "Dataset Analytics",
        "Model Insights",
        "Prediction History"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 🌾 Field Conditions"
)

st.sidebar.caption(
    "Enter the current agricultural field parameters."
)

with st.sidebar.form("farm_input_form"):

    crop = st.selectbox(
        "Crop ID",
        sorted(data["crop ID"].dropna().unique())
    )

    soil = st.selectbox(
        "Soil Type",
        sorted(data["soil_type"].dropna().unique())
    )

    stage = st.selectbox(
        "Growth Stage",
        sorted(data["Seedling Stage"].dropna().unique())
    )

    moi = st.number_input(
        "Moisture / MOI",
        min_value=float(data["MOI"].min()),
        max_value=float(data["MOI"].max()),
        value=float(data["MOI"].median()),
        step=0.5
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=float(data["temp"].min()),
        max_value=float(data["temp"].max()),
        value=float(data["temp"].median()),
        step=0.5
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=float(data["humidity"].min()),
        max_value=float(data["humidity"].max()),
        value=float(data["humidity"].median()),
        step=1.0
    )

    submitted = st.form_submit_button(
        "⚡ Generate Irrigation Decision",
        use_container_width=True
    )


if submitted:

    try:

        result = make_prediction(
            crop,
            soil,
            stage,
            moi,
            temperature,
            humidity
        )

        st.session_state.latest_result = result

        st.session_state.prediction_history.insert(
            0, result
        )

        st.sidebar.success(
            "Prediction generated successfully!"
        )

    except Exception as error:

        st.sidebar.error(
            f"Prediction failed: {error}"
        )


# ================================================================
# 8. HEADER
# ================================================================

st.markdown("""
<div class="main-header">

<h1>🌱 Smart Irrigation Decision Support System</h1>

<p>
Machine Learning-Based Agricultural Water Management
and Intelligent Irrigation Decision Analytics
</p>

</div>
""", unsafe_allow_html=True)


# ================================================================
# 9. DASHBOARD OVERVIEW
# ================================================================

if page == "Dashboard Overview":

    st.markdown(
        '<div class="section-title">01 | System Overview</div>',
        unsafe_allow_html=True
    )

    total_records = len(data)

    crop_count = data["crop ID"].nunique()

    soil_count = data["soil_type"].nunique()

    class_count = data["result"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-label">Dataset Records</div>
        <div class="kpi-value">{total_records:,}</div>
        <div class="kpi-caption">Available observations</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-label">Crop Categories</div>
        <div class="kpi-value">{crop_count}</div>
        <div class="kpi-caption">Unique crop types</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-label">Soil Categories</div>
        <div class="kpi-value">{soil_count}</div>
        <div class="kpi-caption">Available soil types</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-label">Decision Classes</div>
        <div class="kpi-value">{class_count}</div>
        <div class="kpi-caption">Target output categories</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">02 | Agricultural Data Analytics</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        crop_counts = (
            data["crop ID"]
            .value_counts()
            .reset_index()
        )

        crop_counts.columns = ["Crop", "Observations"]

        fig = px.bar(
            crop_counts,
            x="Crop",
            y="Observations",
            title="Crop Distribution",
            color="Observations",
            color_continuous_scale=[
                "#B8DCC5", "#1F704B"
            ]
        )

        fig.update_layout(
            template="plotly_white",
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        class_counts = (
            data["result"]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        class_counts.columns = ["Class", "Count"]

        fig = px.pie(
            class_counts,
            names="Class",
            values="Count",
            hole=0.55,
            title="Decision Class Distribution",
            color_discrete_sequence=[
                "#1F704B",
                "#6AB58A",
                "#B8DCC5",
                "#E7B85C"
            ]
        )

        fig.update_layout(
            template="plotly_white",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">03 | Recent Prediction</div>',
        unsafe_allow_html=True
    )

    latest = st.session_state.latest_result

    if latest:

        a, b, c = st.columns(3)

        a.metric(
            "Predicted Class",
            str(latest["Predicted Class"])
        )

        b.metric(
            "Model Confidence",
            (
                f'{latest["Confidence"]:.2%}'
                if latest["Confidence"] is not None
                else "N/A"
            )
        )

        c.metric(
            "Confidence Category",
            latest["Confidence Level"]
        )

    else:

        st.info(
            "Enter your field conditions in the sidebar "
            "and generate a prediction to see the latest result."
        )


# ================================================================
# 10. IRRIGATION PREDICTION PAGE
# ================================================================

elif page == "Irrigation Prediction":

    st.markdown(
        '<div class="section-title">02 | AI-Based Irrigation Prediction</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.latest_result

    if result is None:

        st.info(
            "Enter field conditions in the sidebar and click "
            "'Generate Irrigation Decision'."
        )

    else:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-heading">
            🌱 {result["Decision Level"]}
            </div>
            <p style="color:#426B50;">
            Predicted target class:
            <b>{result["Predicted Class"]}</b>
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Prediction Confidence",
            (
                f'{result["Confidence"]:.2%}'
                if result["Confidence"] is not None
                else "N/A"
            )
        )

        c2.metric(
            "Confidence Level",
            result["Confidence Level"]
        )

        c3.metric(
            "Selected Crop",
            result["Crop"]
        )

        st.markdown("### Field Input Summary")

        summary = pd.DataFrame({
            "Parameter": [
                "Crop",
                "Soil Type",
                "Growth Stage",
                "Moisture / MOI",
                "Temperature",
                "Humidity"
            ],
            "Input Value": [
                result["Crop"],
                result["Soil Type"],
                result["Growth Stage"],
                result["MOI"],
                f'{result["Temperature"]} °C',
                f'{result["Humidity"]} %'
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Class Probability Analysis")

        probabilities = result["Class Probabilities"]

        if probabilities:

            probability_df = pd.DataFrame({
                "Decision Class": [
                    f"Class {key}"
                    for key in probabilities
                ],
                "Probability": list(
                    probabilities.values()
                )
            })

            fig = px.bar(
                probability_df,
                x="Decision Class",
                y="Probability",
                text=probability_df["Probability"].map(
                    lambda x: f"{x:.1%}"
                ),
                color="Probability",
                color_continuous_scale=[
                    "#B8DCC5", "#1F704B"
                ],
                title="Predicted Probability by Class"
            )

            fig.update_layout(
                template="plotly_white",
                height=380,
                yaxis_tickformat=".0%"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.warning(
            "Interpretation: Decision Level 0, 1 and 2 are "
            "the original target classes. Their exact irrigation "
            "actions must be mapped using the dataset's documented "
            "class definitions. Model confidence is not a measure "
            "of irrigation severity."
        )

        st.download_button(
            "⬇ Download Prediction Report",
            data=pd.DataFrame([{
                k: v for k, v in result.items()
                if k != "Class Probabilities"
            }]).to_csv(index=False),
            file_name="irrigation_prediction.csv",
            mime="text/csv"
        )


# ================================================================
# 11. DATASET ANALYTICS
# ================================================================

elif page == "Dataset Analytics":

    st.markdown(
        '<div class="section-title">03 | Dataset Analytics</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Exploratory analysis of the agricultural "
        "dataset used for model development."
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            data,
            x="MOI",
            color="result",
            title="Moisture Distribution by Decision Class",
            barmode="overlay",
            opacity=0.75
        )

        fig.update_layout(
            template="plotly_white",
            height=370
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            data,
            x="temp",
            y="humidity",
            color=data["result"].astype(str),
            size="MOI",
            hover_data=["crop ID", "soil_type"],
            title="Temperature vs Humidity Analysis"
        )

        fig.update_layout(
            template="plotly_white",
            height=370,
            legend_title="Decision Class"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### Numerical Statistical Summary")

    st.dataframe(
        data[
            ["MOI", "temp", "humidity"]
        ].describe().round(2),
        use_container_width=True
    )

    st.markdown("### Dataset Preview")

    st.dataframe(
        data.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇ Download Dataset",
        data=data.to_csv(index=False),
        file_name="cropdata_analytics.csv",
        mime="text/csv"
    )


# ================================================================
# 12. MODEL INSIGHTS
# ================================================================

elif page == "Model Insights":

    st.markdown(
        '<div class="section-title">04 | Machine Learning Model Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">

    <h3 style="color:#173F32;">
    Gradient Boosting Classifier
    </h3>

    <p style="color:#65756B;">
    The final classification model uses an ensemble of
    sequential decision trees to predict irrigation-related
    decision classes from agricultural field conditions.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Algorithm", "Gradient Boosting")

    m2.metric("Estimators", "120")

    m3.metric("Learning Rate", "0.05")

    m4.metric("Max Depth", "3")

    st.markdown("### Model Configuration")

    config = pd.DataFrame({
        "Parameter": [
            "Algorithm",
            "Number of Estimators",
            "Learning Rate",
            "Maximum Depth",
            "Minimum Samples Split",
            "Minimum Samples Leaf",
            "Random State",
            "Training/Test Ratio"
        ],
        "Configuration": [
            "Gradient Boosting Classifier",
            "120",
            "0.05",
            "3",
            "15",
            "5",
            "42",
            "80:20"
        ]
    })

    st.dataframe(
        config,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Feature Importance Analysis")

    importance_df = get_feature_importance()

    if importance_df is not None:

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text=importance_df["Importance"].map(
                lambda x: f"{x:.3f}"
            ),
            color="Importance",
            color_continuous_scale=[
                "#B8DCC5", "#1F704B"
            ],
            title="Relative Feature Importance"
        )

        fig.update_layout(
            template="plotly_white",
            height=420,
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance is unavailable for the loaded model."
        )

    st.info(
        "Feature importance represents the model's relative "
        "predictive contribution. It does not establish causal "
        "relationships between agricultural variables and "
        "irrigation decisions."
    )


# ================================================================
# 13. PREDICTION HISTORY
# ================================================================

elif page == "Prediction History":

    st.markdown(
        '<div class="section-title">05 | Prediction History</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.prediction_history

    if history:

        history_df = pd.DataFrame([
            {
                k: v for k, v in item.items()
                if k != "Class Probabilities"
            }
            for item in history
        ])

        st.metric(
            "Total Predictions",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Export Prediction History",
            data=history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button(
            "Clear Prediction History",
            type="secondary"
        ):

            st.session_state.prediction_history = []

            st.session_state.latest_result = None

            st.rerun()

    else:

        st.info(
            "No prediction history available. "
            "Generate a prediction to create records."
        )


# ================================================================
# 14. FOOTER
# ================================================================

st.markdown("---")

st.markdown("""
<div class="footer">

<b>SMART IRRIGATION DECISION SUPPORT SYSTEM</b>

<br>

M.Tech Research Project | Machine Learning-Based Agriculture

<br>

Developed using Python, Streamlit, Pandas and Scikit-learn

<br>

© 2026 | Agricultural Intelligence and Decision Analytics

</div>
""", unsafe_allow_html=True)
