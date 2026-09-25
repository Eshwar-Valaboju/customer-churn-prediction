import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

"""
WHAT: This module handles data cleaning, feature engineering, and constructs the preprocessing pipeline.
WHY: Raw data cannot be fed directly to ML models. We need to handle missing values, encode text, and scale numbers.
HOW: We use scikit-learn's Pipeline and ColumnTransformer to create a robust, leak-free sequence of transformations.
INTERVIEW: "I used Scikit-learn's Pipeline and ColumnTransformer to prevent data leakage and ensure that the exact same transformations applied to the training data are seamlessly applied to new incoming customer data."
"""

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add interpretable features.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # 1. tenure_group: grouping tenure into bins
        # This helps models capture non-linear relationships with tenure
        bins = [0, 12, 24, 48, 60, np.inf]
        labels = ['0-1 Year', '1-2 Years', '2-4 Years', '4-5 Years', '5+ Years']
        X_out['tenure_group'] = pd.cut(X_out['tenure'], bins=bins, labels=labels, right=False).astype(str)
        
        # 2. service_count: count how many extra services a customer has
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        # Count if the value is 'Yes'
        X_out['service_count'] = X_out[services].apply(lambda row: (row == 'Yes').sum(), axis=1)
        
        return X_out

def load_and_clean_data(filepath="data/Telco-Customer-Churn.csv"):
    """Loads dataset and performs basic cleaning before pipeline."""
    df = pd.read_csv(filepath)
    
    # Drop customerID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    # Handle TotalCharges missing values (blanks to NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing TotalCharges as it's very small (11 rows)
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # Convert Target 'Churn' to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def get_preprocessing_pipeline():
    """Builds and returns the scikit-learn preprocessing pipeline."""
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'service_count']
    
    # We include our engineered 'tenure_group' here as well
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                            'PhoneService', 'MultipleLines', 'InternetService', 
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 
                            'Contract', 'PaperlessBilling', 'PaymentMethod',
                            'tenure_group']

    # Preprocessing for numerical data: impute missing and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data: impute missing and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # drop='first' avoids dummy variable trap
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Combine feature engineering and standard preprocessing
    full_pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

if __name__ == "__main__":
    # Simple test
    df = load_and_clean_data("../data/Telco-Customer-Churn.csv")
    print(f"Data shape after cleaning: {df.shape}")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    pipeline = get_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(X)
    print(f"Processed feature matrix shape: {X_processed.shape}")
