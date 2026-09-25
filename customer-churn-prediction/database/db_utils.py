import sqlite3
import datetime
import os
import pandas as pd

"""
WHAT: This module handles basic SQLite database operations for logging predictions.
WHY: Storing a history of model predictions allows businesses to monitor model usage and track decisions over time.
HOW: It connects to a local SQLite database, creates a table if it doesn't exist, and inserts prediction records.
INTERVIEW: "I used a lightweight SQLite database to log all predictions made through the dashboard. This simulates a real-world scenario where prediction tracking is necessary for auditing and monitoring model drift."
"""

DB_PATH = os.path.join(os.path.dirname(__file__), 'predictions.db')

def init_db():
    """Initializes the database and creates the history table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            gender TEXT,
            tenure INTEGER,
            monthly_charges REAL,
            prediction TEXT,
            probability REAL,
            risk_level TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(customer_dict, prediction, probability, risk_level):
    """Logs a single prediction into the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO prediction_history 
        (timestamp, gender, tenure, monthly_charges, prediction, probability, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, 
        customer_dict.get('gender', 'N/A'),
        customer_dict.get('tenure', 0),
        customer_dict.get('MonthlyCharges', 0.0),
        prediction,
        probability,
        risk_level
    ))
    
    conn.commit()
    conn.close()

def get_history():
    """Retrieves the prediction history as a Pandas DataFrame."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY timestamp DESC", conn)
    conn.close()
    return df
