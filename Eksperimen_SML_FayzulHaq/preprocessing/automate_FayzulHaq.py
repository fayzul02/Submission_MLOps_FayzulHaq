import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess_data():
    # 1. Tentukan Path 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, '../data_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    output_path = os.path.join(base_dir, 'data_clean.csv')
    
    print(f"Loading data from: {input_path}")
    
    # Cek apakah file ada
    if not os.path.exists(input_path):
        input_path = 'Eksperimen_SML_NamaAnda/data_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
        output_path = 'Eksperimen_SML_NamaAnda/preprocessing/data_clean.csv'

    df = pd.read_csv(input_path)
    
    # 2. Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    # Drop ID
    df = df.drop(['customerID'], axis=1)
    
    # 3. Encoding
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 4. Scaling
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # 5. Save
    df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai! Data disimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_data()