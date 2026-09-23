import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "final_gradient_boosting_model.pkl"
DATA_PATH = BASE_DIR / "cropdata_clean.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_data()

st.title('Smart Irrigation Decision Support System')
st.write('Machine Learning-Based Irrigation Decision Support Dashboard')
st.divider()

st.subheader('Enter Farm Conditions')

col1, col2, col3 = st.columns(3)

with col1:
    crop = st.selectbox('Crop ID', sorted(data['crop ID'].dropna().unique()))

with col2:
    soil = st.selectbox('Soil Type', sorted(data['soil_type'].dropna().unique()))

with col3:
    stage = st.selectbox('Growth Stage', sorted(data['Seedling Stage'].dropna().unique()))

col4, col5, col6 = st.columns(3)

with col4:
    moi = st.number_input('MOI', min_value=float(data['MOI'].min()), max_value=float(data['MOI'].max()), value=float(data['MOI'].median()))

with col5:
    temp = st.number_input('Temperature', min_value=float(data['temp'].min()), max_value=float(data['temp'].max()), value=float(data['temp'].median()))

with col6:
    humidity = st.number_input('Humidity', min_value=float(data['humidity'].min()), max_value=float(data['humidity'].max()), value=float(data['humidity'].median()))

st.divider()

if st.button('Predict Irrigation Decision', type='primary', use_container_width=True):

    input_data = pd.DataFrame({
        'crop ID': [crop],
        'soil_type': [soil],
        'Seedling Stage': [stage],
        'MOI': [moi],
        'temp': [temp],
        'humidity': [humidity]
    })

    prediction = model.predict(input_data)[0]

    probabilities = None
    confidence = None

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(np.max(probabilities)) * 100

    st.subheader('Prediction Result')

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric('Predicted Class', str(prediction))

    with result_col2:
        if confidence is not None:
            st.metric('Model Confidence', f'{confidence:.2f}%')

    st.info('Decision Level represents the predicted target class from the dataset. Specific irrigation actions should only be assigned when the dataset documentation defines the class meanings.')

    if probabilities is not None:
        st.subheader('Class Probabilities')

        classes = getattr(model, 'classes_', None)

        if classes is None and hasattr(model, 'named_steps'):
            final_step = list(model.named_steps.values())[-1]
            classes = getattr(final_step, 'classes_', None)

        if classes is not None:
            probability_df = pd.DataFrame({
                'Class': classes,
                'Probability (%)': probabilities * 100
            })

            probability_df['Probability (%)'] = probability_df['Probability (%)'].round(2)
            st.dataframe(probability_df, use_container_width=True, hide_index=True)

    st.subheader('Input Summary')
    st.dataframe(input_data, use_container_width=True, hide_index=True)
