import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. SETUP DATA
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '../Eksperimen_SML_FayzulHaq/preprocessing/data_clean.csv')

if not os.path.exists(data_path):
    print(f"Data tidak ditemukan di: {data_path}")
    data_path = 'Eksperimen_SML_FayzulHaq/preprocessing/data_clean.csv'

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. TRAINING DENGAN AUTOLOG

mlflow.autolog()

mlflow.set_experiment("Telco_Churn_Basic")

print("Memulai Training dengan MLflow Autolog...")

with mlflow.start_run() as run:
    # Model Sederhana (Tanpa Tuning)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    print(f"Training Selesai! Run ID: {run.info.run_id}")
    print("Metrik dan Parameter telah otomatis dicatat oleh Autolog.")