import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# Load Data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '../../Eksperimen_SML_FayzulHaq/preprocessing/data_clean.csv')

if not os.path.exists(data_path):
    print(f"Data tidak ditemukan di: {data_path}")
    data_path = 'Eksperimen_SML_FayzulHaq/preprocessing/data_clean.csv'

print(f"Loading data: {data_path}")
df = pd.read_csv(data_path)

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model 
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# Logging ke MLflow
# mlflow.set_experiment("Telco_Churn_CI_Pipeline")

with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(sk_model=rf, artifact_path="model")
    
    # PENTING: Simpan Run ID ke file txt
    # Ini agar GitHub Action bisa membaca ID ini untuk proses build docker
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    
    with open("run_id.txt", "w") as f:
        f.write(run_id)