import joblib
import pandas as pd
import shap
import numpy as np
import os

"""
WHAT: This script loads the trained model and makes predictions on new data.
WHY: We need a reusable module to handle incoming requests from the API or Streamlit dashboard, process the data, and return predictions + business logic.
HOW: It takes a dictionary of customer data, converts it to a pandas DataFrame, runs it through the saved pipeline, and applies business rules for risk categorization.
INTERVIEW: "I separated the prediction logic from the API. This module loads the pipeline, computes the probability of churn, and categorizes it into Low/Medium/High risk. It also includes basic recommendation logic based on the customer's profile."
"""

def load_model(model_path="../models/churn_model.pkl"):
    """Loads the saved ML pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)

def get_risk_level(probability):
    """Business logic for risk categorization."""
    if probability < 0.30:
        return "LOW RISK"
    elif probability <= 0.60:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

def get_recommendations(customer_dict, risk_level):
    """Simple business rules to recommend retention actions."""
    if risk_level == "LOW RISK":
        return "No immediate action required. Maintain good service."
    
    actions = []
    if customer_dict.get('Contract') == 'Month-to-month':
        actions.append("Offer a discount for upgrading to a 1-year or 2-year contract.")
    if customer_dict.get('TechSupport') == 'No':
        actions.append("Offer a free trial of Tech Support.")
    if customer_dict.get('InternetService') == 'Fiber optic':
        actions.append("Review Fiber optic performance in their area or offer a personalized plan.")
    
    if not actions:
        actions.append("Schedule a check-in call to discuss their satisfaction.")
        
    return " ".join(actions)

def predict_churn(customer_dict, model_path="../models/churn_model.pkl"):
    """
    Takes a single customer dictionary, returns prediction, probability, risk level, and recommendations.
    """
    model = load_model(model_path)
    
    # Convert dict to DataFrame
    df = pd.DataFrame([customer_dict])
    
    # Predict
    prob = model.predict_proba(df)[0][1]
    pred = int(model.predict(df)[0])
    
    risk = get_risk_level(prob)
    rec = get_recommendations(customer_dict, risk)
    
    # Optional: Basic SHAP explainability for tree-based models
    # Note: Extracting SHAP values from a full pipeline with OneHotEncoder is complex.
    # For a beginner project, we can do a simplified feature importance extraction
    # or use shap.KernelExplainer (which can be slow). We will return basic factors.
    
    return {
        "prediction": "Churn" if pred == 1 else "No Churn",
        "churn_probability": float(prob),
        "risk_level": risk,
        "recommendation": rec
    }
