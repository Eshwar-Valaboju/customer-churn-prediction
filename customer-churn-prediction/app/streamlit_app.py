import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# Add src and database to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../database'))

from prediction import predict_churn
from db_utils import log_prediction, get_history

"""
WHAT: This is the interactive frontend built with Streamlit.
WHY: Stakeholders and non-technical users need a UI to interact with the model, view performance metrics, and understand churn predictions.
HOW: Streamlit allows us to build web apps purely in Python. We import our prediction and database logic directly.
INTERVIEW: "I built a frontend dashboard using Streamlit to make the ML model accessible. It allows users to input customer data, see the predicted churn risk along with retention recommendations, and review past predictions via an SQLite database."
"""

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Customer Prediction", "Model Performance", "Prediction History"])

# 1. Overview Page
if page == "Overview":
    st.title("Customer Churn Prediction System")
    st.markdown("""
    Welcome to the Customer Churn Prediction dashboard.
    This system uses machine learning (Logistic Regression / Random Forest / XGBoost) to predict if a telecom customer is likely to leave.
    
    ### Business Problem
    Customer churn is a major problem for telecom companies. Acquiring a new customer is far more expensive than retaining an existing one. By predicting churn risk, we can proactively offer retention incentives.
    """)
    
    try:
        df = pd.read_csv("data/Telco-Customer-Churn.csv")
        st.subheader("Dataset Snapshot")
        st.dataframe(df.head())
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{len(df):,}")
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        col3.metric("Avg Monthly Charge", f"${df['MonthlyCharges'].mean():.2f}")
        
    except Exception as e:
        st.warning("Dataset not found. Please ensure it is downloaded in the data/ folder.")

# 2. Customer Prediction
elif page == "Customer Prediction":
    st.title("Predict Customer Churn")
    
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            
        with col2:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            
        with col3:
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly Charges", value=50.0)
            total = st.number_input("Total Charges", value=600.0)
            
        submit = st.form_submit_button("Predict Churn Risk")
        
    if submit:
        customer_data = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi, "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": device, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": billing, "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total
        }
        
        try:
            # We attempt to find the model relative to this file
            model_path = os.path.join(os.path.dirname(__file__), '../models/churn_model.pkl')
            result = predict_churn(customer_data, model_path=model_path)
            
            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.subheader("Prediction Result")
                st.metric("Risk Level", result['risk_level'])
                st.metric("Churn Probability", f"{result['churn_probability']*100:.1f}%")
                
            with col_res2:
                st.subheader("Business Recommendation")
                st.info(result['recommendation'])
                
            # Log to SQLite
            log_prediction(customer_data, result['prediction'], result['churn_probability'], result['risk_level'])
            st.success("Prediction logged to history.")
            
        except FileNotFoundError:
            st.error("Model file not found. Have you trained the model yet? (Run `python src/train_model.py`)")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# 3. Model Performance
elif page == "Model Performance":
    st.title("Model Performance & Evaluation")
    
    st.markdown("""
    Here we compare the performance of Logistic Regression, Random Forest, and XGBoost on our hold-out test set.
    """)
    
    try:
        results_df = pd.read_csv("outputs/model_results/model_comparison.csv")
        st.dataframe(results_df.style.highlight_max(subset=['ROC-AUC', 'F1'], color='lightgreen'))
        
        st.subheader("Confusion Matrices")
        cols = st.columns(3)
        for i, model_name in enumerate(results_df['Model']):
            safe_name = model_name.replace(" ", "_").lower()
            img_path = f"outputs/figures/cm_{safe_name}.png"
            if os.path.exists(img_path):
                cols[i%3].image(img_path, caption=model_name, use_column_width=True)
                
    except FileNotFoundError:
        st.info("Model evaluation results not found. Train the model first.")

# 4. Prediction History
elif page == "Prediction History":
    st.title("Prediction History Logs")
    st.markdown("This table logs every prediction made through the dashboard, backed by an SQLite database.")
    
    try:
        history_df = get_history()
        st.dataframe(history_df)
    except Exception as e:
        st.error(f"Could not load history: {e}")
