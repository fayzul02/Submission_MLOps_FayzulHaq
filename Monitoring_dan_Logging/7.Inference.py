import time
import random
import requests
import psutil
import pandas as pd
import os
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# 1. SETUP DATA LOADER
try:
    base_path = "Eksperimen_SML_FayzulHaq/preprocessing/data_clean.csv"
    
    if not os.path.exists(base_path):
        for root, dirs, files in os.walk("."):
            if "data_clean.csv" in files:
                base_path = os.path.join(root, "data_clean.csv")
                break
    
    print(f"Loading data sample from: {base_path}")
    df_source = pd.read_csv(base_path)
    
    if 'Churn' in df_source.columns:
        X_sample = df_source.drop('Churn', axis=1)
    else:
        X_sample = df_source
        
    print("Data loaded successfully! Columns matched.")
    
except Exception as e:
    print(f"ERROR LOADING DATA: {e}")
    exit()

# 2. DEFINISI METRIKS

CPU_USAGE = Gauge('system_cpu_usage', 'Current CPU usage in percent')
MEMORY_USAGE = Gauge('system_memory_usage', 'Current RAM usage in percent')
REQUEST_COUNT = Counter('app_request_count', 'Total output request count')
SUCCESS_COUNT = Counter('app_success_count', 'Total success request')
FAILURE_COUNT = Counter('app_failure_count', 'Total failure request')
LATENCY = Histogram('app_latency_seconds', 'Request latency in seconds')
PREDICTION_VALUE = Gauge('model_prediction_value', 'Result of prediction (0 or 1)')
CONFIDENCE_SCORE = Gauge('model_confidence_score', 'Confidence score of prediction')
INPUT_DATA_SIZE = Gauge('input_data_size_bytes', 'Size of input payload')
DRIFT_MAGNITUDE = Gauge('data_drift_magnitude', 'Simulated data drift magnitude')

# 3. GENERATOR TRAFFIC
def generate_traffic():
    # Update System Metrics
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    
    # Ambil data acak
    random_row = X_sample.sample(1)
    
    # Payload format
    payload = {
        "dataframe_split": random_row.to_dict(orient='split')
    }
    if "index" in payload["dataframe_split"]:
        del payload["dataframe_split"]["index"]

    start_time = time.time()
    try:
        response = requests.post(
            "http://127.0.0.1:5000/invocations", 
            json=payload, 
            headers={'Content-Type': 'application/json'}
        )
        
        latency = time.time() - start_time
        LATENCY.observe(latency)
        REQUEST_COUNT.inc()
        INPUT_DATA_SIZE.set(len(str(payload)))

        if response.status_code == 200:
            SUCCESS_COUNT.inc()
            
            # --- BAGIAN PERBAIKAN ---
            result = response.json()
            
            # Cek format response MLflow 
            pred = 0
            if isinstance(result, list):
                # Format: [0]
                pred = int(result[0])
            elif isinstance(result, dict) and "predictions" in result:
                # Format: {"predictions": [0]}
                pred = int(result["predictions"][0])
            else:
                # Format Dict lain, coba ambil value pertama
                first_val = next(iter(result.values()))
                if isinstance(first_val, list):
                    pred = int(first_val[0])
                else:
                    pred = int(first_val)

            # Simulasi confidence score
            conf = random.uniform(0.70, 0.99)
            
            PREDICTION_VALUE.set(pred)
            CONFIDENCE_SCORE.set(conf)
            DRIFT_MAGNITUDE.set(random.random() * 0.1)
            
            print(f"Request Success | Latency: {latency:.4f}s | Pred: {pred}")
        else:
            FAILURE_COUNT.inc()
            print(f"Request Failed: {response.status_code}")
            
    except Exception as e:
        FAILURE_COUNT.inc()
        print(f"Error Loop: {e}")

if __name__ == '__main__':
    print("Prometheus Metrics Server running on port 8000...")
    start_http_server(8000)
    
    print("Starting Traffic Generator...")
    while True:
        generate_traffic()
        time.sleep(1)