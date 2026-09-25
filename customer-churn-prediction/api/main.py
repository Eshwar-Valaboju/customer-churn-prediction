from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
import os

# Add src to path so we can import our prediction module
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from prediction import predict_churn

"""
WHAT: This is the FastAPI backend.
WHY: We need a way for frontends (like our Streamlit app) or other services to interact with our machine learning model via a standard REST API.
HOW: We define a Pydantic model to strictly validate incoming JSON data, then pass it to our prediction script.
INTERVIEW: "I built a REST API using FastAPI. I used Pydantic models for request validation to ensure the ML model always receives the correct data types, preventing runtime crashes."
"""

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict telecom customer churn risk.",
    version="1.0.0"
)

# Pydantic model for request validation
class CustomerData(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=1)
    PhoneService: str = Field(..., example="No")
    MultipleLines: str = Field(..., example="No phone service")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=29.85)
    TotalCharges: float = Field(..., example=29.85)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Churn Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        # Convert pydantic model to dict
        customer_dict = customer.model_dump()
        
        # We need to make sure the model path is correct relative to the script execution
        model_path = os.path.join(os.path.dirname(__file__), '../models/churn_model.pkl')
        
        result = predict_churn(customer_dict, model_path=model_path)
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail="Model not trained yet. Please run src/train_model.py first.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
