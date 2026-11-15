from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Irrigation DSS",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "final_gradient_boosting_model.pkl"
DATA_PATH = BASE_DIR / "cropdata_clean.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.3);
        text-align: center;
        margin-top: 15px;
    }

    .result-value {
        font-size: 34px;
        font-weight: 700;
        margin: 10px 0;
    }

    .confidence {
        font-size: 22px;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# FILE VALIDATION
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please check "
        "final_gradient_boosting_model.pkl in the repository."
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        "Dataset file not found. Please check "
        "cropdata_clean.csv in the repository."
    )
    st.stop()


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

try:
    model = load_model()
    data = load_data()
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.error(
        "Please make sure the Streamlit Python and package "
        "versions match the model training environment."
    )
    st.exception(e)
    st.stop()


# ============================================================
# GET INPUT VALUES
# ============================================================

crop_values = sorted(data["crop ID"].dropna().unique().tolist())
soil_values = sorted(data["soil_type"].dropna().unique().tolist())
stage_values = sorted(data["Seedling Stage"].dropna().unique().tolist())


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌱 Smart Irrigation Decision Support System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    "Machine Learning-Based Irrigation Decision Support"
    "</div>",
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 Irrigation Inputs")

    st.markdown(
        "Enter the current crop and environmental conditions."
    )

    crop = st.selectbox(
        "Crop",
        crop_values
    )

    soil = st.selectbox(
        "Soil Type",
        soil_values
    )

    stage = st.selectbox(
        "Growth Stage",
        stage_values
    )

    moi = st.number_input(
        "Moisture / MOI",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=24.0,
        step=0.5
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    st.divider()

    predict_button = st.button(
        "💧 Generate Irrigation Decision",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# INPUT SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">Current Field Conditions</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Crop", crop)
    st.metric("Soil Type", soil)

with col2:
    st.metric("Growth Stage", stage)
    st.metric("MOI", f"{moi:.1f}")

with col3:
    st.metric("Temperature", f"{temperature:.1f} °C")
    st.metric("Humidity", f"{humidity:.1f}%")


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame(
        {
            "crop ID": [crop],
            "soil_type": [soil],
            "Seedling Stage": [stage],
            "MOI": [moi],
            "temp": [temperature],
            "humidity": [humidity]
        }
    )

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = float(np.max(probabilities)) * 100

        predicted_class = int(prediction)

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Irrigation Decision Result</div>',
            unsafe_allow_html=True
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.markdown(
                '<div class="result-card">'
                '<div>Predicted Class</div>'
                f'<div class="result-value">{predicted_class}</div>'
                "</div>",
                unsafe_allow_html=True
            )

        with result_col2:

            st.markdown(
                '<div class="result-card">'
                '<div>Prediction Confidence</div>'
                f'<div class="result-value">{confidence:.2f}%</div>'
                "</div>",
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Class Probabilities</div>',
            unsafe_allow_html=True
        )

        probability_data = pd.DataFrame(
            {
                "Decision Class": [
                    f"Class {i}" for i in range(len(probabilities))
                ],
                "Probability (%)": [
                    round(float(p) * 100, 2)
                    for p in probabilities
                ]
            }
        )

        st.dataframe(
            probability_data,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # PROBABILITY CHART
        # ----------------------------------------------------

        chart_data = probability_data.set_index(
            "Decision Class"
        )

        st.bar_chart(chart_data)


        # ----------------------------------------------------
        # INPUT DATA USED
        # ----------------------------------------------------

        with st.expander("View Prediction Input"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "Prediction could not be generated."
        )

        st.exception(e)


else:

    # ========================================================
    # INITIAL INFORMATION
    # ========================================================

    st.info(
        "Select the field conditions from the sidebar and "
        "click **Generate Irrigation Decision**."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write(
        "**Machine Learning Model:** "
        "Gradient Boosting Classifier"
    )

    st.write(
        "**Purpose:** Predict irrigation decision class "
        "from crop, soil, growth-stage and environmental features."
    )

    st.write(
        "**Input Features:** "
        "Crop ID, Soil Type, Growth Stage, MOI, Temperature, Humidity"
    )

    st.write(
        "**Model Accuracy:** 94.57%"
    )

    st.write(
        "**Macro F1-Score:** 82.91%"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    "Smart Irrigation Decision Support System | "
    "Machine Learning-Based Agricultural Decision Support"
    "</div>",
    unsafe_allow_html=True
)
