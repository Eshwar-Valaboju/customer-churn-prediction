import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import os

"""
WHAT: This module evaluates trained machine learning models.
WHY: We need to know how well our models perform on unseen data. Accuracy alone is misleading for imbalanced datasets, so we check Precision, Recall, F1, and ROC-AUC.
HOW: It computes multiple metrics, saves a comparison table to CSV, and generates confusion matrices.
INTERVIEW: "I didn't just look at accuracy because churn datasets are imbalanced. Instead, I focused on Recall (to catch as many churners as possible) and F1-score. I created a reusable evaluation script that saves a comparison table."
"""

def evaluate_all_models(models_dict, X_test, y_test, output_dir="../outputs"):
    """
    Evaluates a dictionary of trained pipelines/models.
    models_dict: {'Logistic Regression': model1, 'Random Forest': model2, ...}
    Returns a DataFrame with the comparison.
    """
    results = []
    
    os.makedirs(f"{output_dir}/model_results", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        
        # Some models might not support predict_proba natively in a pipeline if misconfigured,
        # but LogisticRegression, RandomForest, and XGBoost all do.
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": roc_auc
        })
        
        # Print classification report
        print(f"--- {name} ---")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, name, output_dir)
        
    df_results = pd.DataFrame(results)
    
    # Save the table
    df_results.to_csv(f"{output_dir}/model_results/model_comparison.csv", index=False)
    
    return df_results

def plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    safe_name = model_name.replace(" ", "_").lower()
    plt.savefig(f"{output_dir}/figures/cm_{safe_name}.png", bbox_inches='tight')
    plt.close()
