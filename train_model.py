import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from data_preprocessing import load_and_clean_data, get_preprocessing_pipeline
from evaluate_model import evaluate_all_models

"""
WHAT: This script trains multiple ML models, evaluates them, and saves the best one.
WHY: We train multiple models (Logistic Regression, Random Forest, XGBoost) to see which algorithm learns the churn patterns best.
HOW: It splits the data into training (80%) and testing (20%), fits each model inside a Pipeline, evaluates them, and uses joblib to save the winner.
INTERVIEW: "I tested Logistic Regression for baseline interpretability, Random Forest to handle non-linearities, and XGBoost for maximum performance. I evaluated them on a hold-out test set and saved the best complete pipeline so there is no mismatch between training and production preprocessing."
"""

def main():
    # 1. Load Data
    print("Loading data...")
    df = load_and_clean_data("../data/Telco-Customer-Churn.csv")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 2. Train/Test Split (Stratified ensures churn ratio is maintained in both sets)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Get Preprocessing Pipeline
    preprocessor = get_preprocessing_pipeline()
    
    # 4. Define Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    trained_pipelines = {}
    
    # 5. Train Models
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Combine preprocessing and the model into one pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        full_pipeline.fit(X_train, y_train)
        trained_pipelines[name] = full_pipeline
        
    # 6. Evaluate Models
    print("\nEvaluating models...")
    results_df = evaluate_all_models(trained_pipelines, X_test, y_test, output_dir="../outputs")
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    
    # 7. Model Selection
    # Let's select the model with the highest ROC-AUC
    # We sort the dataframe by ROC-AUC descending and take the top model's name
    best_model_name = results_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"\nBest model selected based on ROC-AUC: {best_model_name}")
    
    # 8. Save Best Model
    os.makedirs("../models", exist_ok=True)
    best_pipeline = trained_pipelines[best_model_name]
    joblib.dump(best_pipeline, "../models/churn_model.pkl")
    print(f"Model saved to ../models/churn_model.pkl")

if __name__ == "__main__":
    main()
