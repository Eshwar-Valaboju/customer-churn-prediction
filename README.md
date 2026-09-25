# 📊 Customer Churn Prediction & Explainable Retention System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.0-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-ff69b4.svg)

## 📌 Problem Statement
Customer churn is a critical challenge in the telecommunications industry. Acquiring a new customer is significantly more expensive than retaining an existing one. Without proactive identification of at-risk customers, companies lose substantial recurring revenue. 

## 🎯 Project Objective
This project implements an end-to-end Machine Learning pipeline to predict whether a telecom customer is likely to churn. Beyond simply predicting a binary outcome, the system utilizes **SHAP (SHapley Additive exPlanations)** to provide transparent, feature-level reasons for the risk, and translates these insights into actionable business retention strategies.

## 🏗️ Architecture & Workflow

1. **Data Preprocessing Pipeline**: Automated handling of missing values, numerical scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`) using Scikit-Learn's `ColumnTransformer` to strictly prevent data leakage.
2. **Model Training & Evaluation**: Comparison of Logistic Regression, Random Forest, and XGBoost. Models are evaluated on **Recall, F1-Score, and ROC-AUC** to properly account for the imbalanced nature of churn data.
3. **API Layer**: A robust REST API built with **FastAPI** and **Pydantic** for strict request validation and model serving.
4. **Interactive Dashboard**: A **Streamlit** frontend allowing stakeholders to input hypothetical customer data, view probability metrics, and understand risk factors dynamically.
5. **Prediction Logging**: An integrated **SQLite** database tracks a history of predictions for future auditing and model drift monitoring.

## 📂 Project Structure

customer-churn-prediction/
├── data/                      # Raw IBM Telco dataset
├── notebooks/                 # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── src/                       # Core ML scripts (preprocessing, training, evaluation, inference)
├── models/                    # Serialized Joblib pipelines
├── api/                       # FastAPI application backend
├── app/                       # Streamlit interactive dashboard
├── database/                  # SQLite database and logging utilities
├── outputs/                   # Evaluation metrics (CSV) and Confusion Matrices (PNG)
├── docs/                      # Interview preparation and project Q&A
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
 Dataset
The model is trained on the IBM Telco Customer Churn dataset, which contains ~7,043 customer records detailing demographics, account information, subscribed services, and historical churn status.

🚀 Installation & Setup
1. Clone the Repository
bash


git clone https://github.com/YOUR_GITHUB_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
2. Create Virtual Environment & Install Dependencies
bash


python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate
pip install -r requirements.txt
3. Train the Machine Learning Model
This script processes the raw data, trains three models, evaluates them, and serializes the best performing pipeline into the models/ directory.

bash


python src/train_model.py
4. Run the Streamlit Dashboard (Frontend)
Launch the interactive web application to interface with the model.

bash


streamlit run app/streamlit_app.py
5. Run the FastAPI Backend (Optional)
If you wish to serve the model as a REST API for other microservices:

bash


uvicorn api.main:app --reload
API documentation automatically available at http://127.0.0.1:8000/docs.

📈 Model Performance
Because the dataset is imbalanced (significantly more non-churners than churners), Accuracy is a misleading metric. Optimization was focused on Recall (to catch as many true churners as possible) and F1-Score.

Detailed comparison tables and confusion matrices are generated automatically in the outputs/ directory during training.

💡 Future Improvements
Implement hyperparameter tuning (GridSearchCV/Optuna) for enhanced XGBoost performance.
Transition from SQLite to PostgreSQL for production-grade prediction logging.
Containerize the application stack using Docker.
