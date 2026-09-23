# Smart Irrigation Decision Support System

## Project Overview

Smart Irrigation Decision Support System is a machine-learning-based application for predicting irrigation-related decision classes using crop, soil, growth stage, moisture, temperature, and humidity information.

## Input Features

- Crop ID
- Soil Type
- Seedling/Growth Stage
- MOI
- Temperature
- Humidity

## Machine Learning Model

The final model is a Gradient Boosting Classifier.

- Number of estimators: 120
- Learning rate: 0.05
- Maximum depth: 3
- Minimum samples split: 15
- Minimum samples leaf: 5
- Random state: 42

The trained model is stored in:

`models/final_gradient_boosting_model.pkl`

## Project Structure

```text
Smart-Irrigation-DSS/
|
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- models/
|   `-- final_gradient_boosting_model.pkl
|
`-- data/
    `-- cropdata_clean.csv
```

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

## Prediction

The application accepts crop, soil, growth stage, MOI, temperature, and humidity values.

The application produces:

- Predicted class
- Model confidence
- Class probabilities

The target classes are represented as Decision Level 0, 1, and 2. Their exact irrigation meaning should follow the documentation of the original dataset.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Google Colab
- GitHub

## Model Performance

The final Gradient Boosting model achieved approximately 94.57% test accuracy on the project dataset.

## Author

M.Tech Project - Smart Irrigation Decision Support System