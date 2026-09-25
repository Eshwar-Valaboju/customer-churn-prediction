# Customer Churn Prediction & Explainable Retention System

**Live Demo**: [Streamlit Community Cloud Link (To be added after deployment)]  
**GitHub Repository**: [GitHub Link (To be added after push)]

## 1. Problem Statement
Customer churn is a critical issue for telecommunications companies. Acquiring a new customer is significantly more expensive than retaining an existing one. Without proactive identification of at-risk customers, companies lose revenue and market share.

## 2. Project Objective
To build a realistic, end-to-end machine learning application that predicts whether a telecom customer is likely to churn. The project emphasizes clear architecture, proper ML practices (handling imbalanced data, avoiding data leakage), and business-friendly explainability.

## 3. Features
- **Data Preprocessing Pipeline**: Automated handling of missing values, scaling, and categorical encoding.
- **Multiple ML Models**: Comparison between Logistic Regression, Random Forest, and XGBoost.
- **Explainability**: SHAP integration to provide transparent reasons for customer churn risk.
- **Interactive Dashboard**: A Streamlit application for stakeholders to input customer data and view predictions.
- **FastAPI Backend**: A robust REST API for serving the trained model to other applications.
- **Prediction Logging**: An SQLite database to track a history of predictions for auditing.

## 4. Architecture
```text
Raw CSV -> Data Cleaning -> EDA -> Feature Engineering -> Train/Test Split
-> Preprocessing Pipeline -> Train Models (LR, RF, XGB) -> Evaluate
-> Save Best Model Pipeline -> FastAPI / Streamlit -> SQLite Logging
```

## 5. Dataset
We use the **IBM Telco Customer Churn dataset**, containing approximately 7,043 customers with 21 features indicating their demographics, services, account information, and churn status.

## 6. Tech Stack
- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Explainability**: SHAP
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend Dashboard**: Streamlit
- **Database**: SQLite
- **Model Serialization**: Joblib

## 7. Project Structure
```text
customer-churn-prediction/
├── data/                      # Raw dataset
├── notebooks/                 # Jupyter notebooks for EDA
├── src/                       # Machine Learning scripts (preprocessing, train, evaluate, predict)
├── models/                    # Saved Joblib pipelines
├── api/                       # FastAPI application
├── app/                       # Streamlit dashboard
├── database/                  # SQLite database and utils
├── outputs/                   # Saved evaluation metrics and charts
├── docs/                      # Interview preparation and resume bullets
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 8. Installation
```bash
# 1. Clone the repository
git clone <your-github-repo-url>
cd customer-churn-prediction

# 2. Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# For Mac/Linux:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 9. How to Train the Model
```bash
python src/train_model.py
```
*This will process the data, train the models, evaluate them, generate confusion matrices in `outputs/`, and save the best pipeline in `models/`.*

## 10. How to Run the API
```bash
uvicorn api.main:app --reload
```
*The API will be available at http://127.0.0.1:8000/docs for interactive testing.*

## 11. How to Run the Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
*The dashboard will automatically open in your web browser.*

## 12. Model Evaluation
Models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Because the dataset is imbalanced (more non-churners than churners), **Recall** and **F1-Score** were prioritized to ensure we catch as many true churners as possible without excessive false positives.

## 13. Explainability
The project uses SHAP and feature importance metrics to explain *why* a customer is at risk. Key factors usually include having a Month-to-month contract, low tenure, and high monthly charges. The dashboard translates these technical insights into actionable business recommendations (e.g., "Offer a 1-year contract discount").

## 14. Screenshots
*(Add screenshots of your Streamlit Dashboard here once deployed)*

## 15. Limitations
- The business recommendations are hard-coded rules based on EDA, not causal AI models.
- The dataset is a static snapshot; real-world churn models require temporal validation (time-series splits).

## 16. Future Improvements
- Implement hyperparameter tuning (GridSearchCV) for better model performance.
- Use a cloud database (PostgreSQL) instead of SQLite for production logs.
- Containerize the application using Docker for easier deployment.

## 17. Disclaimer
This project is for educational and portfolio purposes. The recommendations generated by the app are illustrative and do not guarantee actual customer retention in a real-world scenario.
