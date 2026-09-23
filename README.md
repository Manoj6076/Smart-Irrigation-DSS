# 🌱 Smart Irrigation Decision Support System

### Machine Learning-Based Agricultural Decision Support System

A machine learning-based **Smart Irrigation Decision Support System (DSS)** that predicts an irrigation decision class using crop, soil, growth-stage, moisture, temperature, and humidity information.

The system provides a simple web-based interface where users can enter field conditions and obtain a machine-learning prediction with confidence and class probabilities.

---

## 🚀 Live Application

### 🌐 Streamlit Dashboard

**[Open Smart Irrigation DSS](https://smart-irrigation-dss-2jxmvezvyzjerae2ctobzb.streamlit.app/)**

The application is deployed using **Streamlit Community Cloud**.

---

## 📌 Project Overview

Traditional irrigation decisions can depend heavily on manual observation and fixed practices. This project applies machine learning to support irrigation-related decision-making using environmental and crop-condition data.

The system takes the following inputs:

* Crop
* Soil Type
* Growth Stage
* Moisture / MOI
* Temperature
* Humidity

The trained machine learning model predicts one of three decision classes:

* Class 0
* Class 1
* Class 2

> The class labels are kept as Class 0, Class 1, and Class 2 because the dataset does not provide a sufficiently verified semantic definition for assigning names such as "Irrigate", "Do Not Irrigate", or similar labels.

---

## 🎯 Objectives

* Develop a machine learning-based irrigation decision support system.
* Use crop and environmental parameters for prediction.
* Compare machine learning approaches during model development.
* Develop a reliable classification model.
* Provide prediction confidence and class probabilities.
* Deploy the final system as an interactive web application.
* Provide a practical interface for demonstrating ML-based agricultural decision support.

---

## 🧠 Machine Learning Model

### Final Model

**Gradient Boosting Classifier**

Model configuration:

```text
n_estimators       = 120
learning_rate      = 0.05
max_depth          = 3
min_samples_split  = 15
min_samples_leaf   = 5
random_state       = 42
```

### Preprocessing

The model uses a preprocessing pipeline consisting of:

* One-Hot Encoding for categorical variables
* Standard Scaling for numerical variables
* Gradient Boosting Classification

Categorical features:

```text
crop ID
soil_type
Seedling Stage
```

Numerical features:

```text
MOI
temp
humidity
```

---

## 📊 Model Performance

The final Gradient Boosting model achieved the following results on the evaluated test set:

| Metric            |      Score |
| ----------------- | ---------: |
| Accuracy          | **94.57%** |
| Macro F1-Score    | **82.91%** |
| Weighted F1-Score | **94.02%** |
| Class 0 F1-Score  | **97.83%** |
| Class 1 F1-Score  | **95.51%** |
| Class 2 F1-Score  | **55.40%** |

The difference between macro and weighted F1 reflects the varying performance across the three classes.

---

## 🌾 Input Parameters

| Feature        | Description                     |
| -------------- | ------------------------------- |
| Crop ID        | Selected crop                   |
| Soil Type      | Type of soil                    |
| Seedling Stage | Current crop growth stage       |
| MOI            | Moisture-related input          |
| Temperature    | Environmental temperature in °C |
| Humidity       | Environmental humidity in %     |

---

## 💻 Dashboard Features

The Streamlit application provides:

* 🌱 Crop selection
* 🌍 Soil type selection
* 🌿 Growth-stage selection
* 💧 MOI input
* 🌡️ Temperature input
* 💦 Humidity input
* 🤖 Machine learning prediction
* 📈 Prediction confidence
* 📊 Class probability distribution
* 📋 Prediction input summary
* 🧠 Model information
* 📱 Interactive web-based interface

---

## 🧪 Example Prediction

Example test input:

```text
Crop            : Wheat
Soil Type       : Loam Soil
Growth Stage    : Germination
MOI             : 25
Temperature     : 24 °C
Humidity        : 70 %
```

Example output from the evaluated model:

```text
Predicted Class : 0
Confidence      : approximately 99.91%
```

Example class probabilities:

```text
Class 0 : approximately 99.91%
Class 1 : approximately 0.07%
Class 2 : approximately 0.03%
```

---

## 🏗️ System Workflow

```text
User Input
    │
    ▼
Crop / Soil / Growth Stage
MOI / Temperature / Humidity
    │
    ▼
Data Preprocessing
    │
    ▼
One-Hot Encoding
    │
    ▼
Feature Scaling
    │
    ▼
Gradient Boosting Classifier
    │
    ▼
Predicted Class
    │
    ├──► Confidence
    │
    └──► Class Probabilities
```

---

## 📁 Project Structure

```text
Smart-Irrigation-DSS/
│
├── app.py
├── README.md
├── requirements.txt
├── cropdata_clean.csv
└── final_gradient_boosting_model.pkl
```

### File Description

| File                                | Purpose                         |
| ----------------------------------- | ------------------------------- |
| `app.py`                            | Streamlit dashboard application |
| `cropdata_clean.csv`                | Dataset used by the application |
| `final_gradient_boosting_model.pkl` | Trained Gradient Boosting model |
| `requirements.txt`                  | Python dependencies             |
| `README.md`                         | Project documentation           |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Google Colab
* GitHub
* Streamlit Community Cloud

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Manoj6076/Smart-Irrigation-DSS.git
```

Move into the project directory:

```bash
cd Smart-Irrigation-DSS
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📦 Python Environment

The model was developed and saved using the following environment:

```text
Python       : 3.13.15
scikit-learn: 1.6.1
NumPy        : 2.1.3
Pandas       : 2.2.3
Joblib       : 1.6.0
```

The deployment environment should use compatible package versions for reliable model loading.

---

## ☁️ Deployment

The application is deployed using:

**Streamlit Community Cloud**

Live application:

**https://smart-irrigation-dss-2jxmvezvyzjerae2ctobzb.streamlit.app/**

Source repository:

**https://github.com/Manoj6076/Smart-Irrigation-DSS**

---

## 🔬 Project Scope

This project demonstrates the use of machine learning for agricultural decision support.

The current implementation focuses on classification using crop, soil, growth-stage, moisture, temperature, and humidity attributes. The predicted classes represent the labels present in the underlying dataset.

Future work can incorporate additional real-world agricultural information such as:

* Soil moisture sensors
* Weather forecasts
* Rainfall information
* Evapotranspiration
* Soil nutrient information
* IoT-based field monitoring
* Real-time sensor data
* Historical irrigation records
* Crop-specific water requirements

---

## 👨‍🎓 Academic Project

**Project:** Smart Irrigation Decision Support System

**Domain:** Machine Learning / Agriculture / Decision Support Systems

**Platform:** Google Colab → GitHub → Streamlit Community Cloud

**Application:** Interactive ML-based irrigation decision support

---

## 📜 Disclaimer

This application is an academic machine learning project intended for demonstration and decision-support research. Predictions depend on the training dataset and model characteristics and should not be treated as a substitute for professional agricultural advice or direct field measurements.

---

## 🔗 Project Links

🌐 **Live Dashboard:**
https://smart-irrigation-dss-2jxmvezvyzjerae2ctobzb.streamlit.app/

💻 **GitHub Repository:**
https://github.com/Manoj6076/Smart-Irrigation-DSS
