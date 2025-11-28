import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# SETUP & LOAD DATA
print("Memulai proses training...")

# Menggunakan relative path agar fleksibel
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '../Eksperimen_SML_FayzulHaq/preprocessing/data_clean.csv')

# Cek file
if not os.path.exists(data_path):
    # Fallback path jika struktur folder berbeda
    data_path = 'Eksperimen_SML_FayzulHaq/preprocessing/data_clean.csv'

print(f"Membaca data dari: {data_path}")
df = pd.read_csv(data_path)

# Pisahkan Fitur (X) dan Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# HYPERPARAMETER TUNING
# Random Forest
rf = RandomForestClassifier(random_state=42)

# Definisikan Grid Parameter untuk dicoba
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

print("Melakukan Hyperparameter Tuning...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Params: {best_params}")

# 3. EVALUASI MODEL
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

# 4. MLFLOW LOGGING (MANUAL - SYARAT SKILLED/ADVANCE)

mlflow.set_experiment("Telco_Churn_Experiment")

with mlflow.start_run():
    print("Mengirim log ke DagsHub...")
    
    # Log Parameters (Manual)
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    
    # Log Metrics (Manual)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # C. Log Model
    mlflow.sklearn.log_model(best_model, "model")
    
    # D. Confusion Matrix Plot
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    # E. Feature Importance Plot
    plt.figure(figsize=(10,6))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Ambil top 10 features
    top_indices = indices[:10]
    plt.title('Top 10 Feature Importances')
    plt.bar(range(len(top_indices)), importances[top_indices], align='center')
    plt.xticks(range(len(top_indices)), [X.columns[i] for i in top_indices], rotation=45)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    print("Logging selesai! Cek DagsHub.")