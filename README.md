# 📡 Telco Customer Churn Prediction: End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

## 📖 Deskripsi Proyek
Proyek ini adalah implementasi sistem **Machine Learning Operations (MLOps)** lengkap untuk memprediksi potensi *Churn* (berhenti berlangganan) pada pelanggan layanan telekomunikasi.

Sistem ini dirancang untuk menangani seluruh siklus hidup pengembangan model Machine Learning, mulai dari pemrosesan data otomatis, pelacakan eksperimen model, penerapan otomatis (CI/CD), hingga pemantauan kinerja model dan sistem secara *real-time* di lingkungan produksi.

---

## ✨ Fitur Utama & Implementasi

### 1. 🔄 Automated Data Pipeline
* **Preprocessing Otomatis:** Script Python yang membersihkan data mentah, menangani *missing values*, melakukan *encoding* (Label & One-Hot), dan *scaling* fitur numerik.
* **Integrasi Git:** Pipeline data terintegrasi yang memastikan model selalu dilatih menggunakan versi data terbersih.

### 2. 🧪 Model Experimentation & Tracking
Menggunakan **MLflow** yang terhubung dengan **DagsHub** sebagai remote server untuk manajemen siklus hidup model.
* **Autologging:** Pencatatan otomatis parameter dan metrik dasar.
* **Hyperparameter Tuning:** Eksperimen tingkat lanjut menggunakan GridSearch dengan pencatatan manual untuk metrik presisi.
* **Artifact Management:** Penyimpanan artefak model seperti *Confusion Matrix* dan *Feature Importance Plot*.

### 3. 🚀 CI/CD & Dockerization
Workflow otomatis menggunakan **GitHub Actions**:
* **Automated Training:** Memicu pelatihan ulang (`mlflow run`) secara otomatis setiap kali ada perubahan kode di repository.
* **Docker Build:** Mengemas model yang telah dilatih ke dalam *Docker Container* yang siap didistribusikan ke **Docker Hub**.

### 4. 📊 Real-time Monitoring & Alerting
Sistem pemantauan kesehatan model pasca-deployment menggunakan stack **Prometheus** dan **Grafana**.
* **Metrik Sistem:** Penggunaan CPU dan Memory.
* **Metrik Bisnis & Model:** Total Request, Latency, Success/Failure Rate, dan hasil prediksi.
* **Alerting System:** Notifikasi otomatis jika terjadi anomali (misal: Latency tinggi atau Error rate meningkat).

---

## 📂 Struktur Proyek

```text
.
├── .github/workflows/          # Konfigurasi CI/CD (GitHub Actions)
├── Eksperimen_SML_FayzulHaq/   # Modul Data & Preprocessing
│   ├── data_raw/               # Dataset Sumber (CSV)
│   └── preprocessing/          # Script Otomasi & Notebook Eksperimen
├── Membangun_model/            # Modul Training Model
│   ├── modelling.py            # Script Training (Basic)
│   ├── modelling_tuning.py     # Script Training (Advance/Tuning)
│   └── data_clean.csv          # Data hasil preprocessing
├── Workflow-CI/                # Konfigurasi MLflow Project
│   └── MLProject/              # Definisi Environment & Entry Point
└── Monitoring dan Logging/     # Modul Monitoring
    ├── 2.prometheus.yml        # Config Prometheus
    ├── 3.prometheus_exporter.py# Middleware Exporter Metrik
    ├── Inference.py            # Script Simulasi Traffic/Inference
    └── ...                     # Bukti Screenshot Dashboard

```

# 🚀 Cara Menjalankan Proyek Secara Lokal

Ikuti langkah-langkah berikut untuk menjalankan sistem ini di komputer lokal Anda.

---

## 💻 1. Instalasi & Persiapan

Clone repository dan install dependencies:

```bash
git clone https://github.com/fayzul02/Submission_MLOps_FayzulHaq.git
cd Submission_MLOps_FayzulHaq

# Buat Virtual Environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install Library
pip install -r Membangun_model/requirements.txt
pip install prometheus-client psutil requests
````

---

## 🔧 2. Konfigurasi Tracking URI (DagsHub)

Set kredensial DagsHub di terminal (PowerShell):

```powershell
# Windows (PowerShell)
$env:MLFLOW_TRACKING_URI="https://dagshub.com/USERNAME/REPO.mlflow"
$env:MLFLOW_TRACKING_USERNAME="username"
$env:MLFLOW_TRACKING_PASSWORD="token"
```

> Ganti `USERNAME`, `REPO`, `username`, dan `token` sesuai akun Anda.

---

## 🧠 3. Training Model

Jalankan script untuk melatih model dengan hyperparameter tuning dan mengirim log ke DagsHub:

```bash
python Membangun_model/modelling_tuning.py
```

---

## 📡 4. Menjalankan Sistem Monitoring (Full Stack)

Anda perlu membuka **3 Terminal Terpisah**.

### 🟦 Terminal A — Model Serving (API)

Menyalakan REST API model pada port 5000:

```bash
mlflow models serve -m "Membangun_model/model" -p 5000 --no-conda
```

### 🟩 Terminal B — Prometheus (Database)

Menjalankan Prometheus untuk mengumpulkan metrik:

```bash
cd "Monitoring dan Logging"
.\prometheus.exe --config.file=2.prometheus.yml
```

### 🟧 Terminal C — Traffic Generator (Simulasi User)

Mengirim dummy request untuk menggerakkan grafik monitoring:

```bash
python "Monitoring dan Logging/Inference.py"
```

---

## 📈 5. Akses Dashboard

Setelah semua terminal berjalan, akses monitoring melalui browser:

| Layanan               | URL                                                          |
| --------------------- | ------------------------------------------------------------ |
| **Grafana Dashboard** | [http://localhost:3000](http://localhost:3000)               |
| **Prometheus UI**     | [http://localhost:9090](http://localhost:9090)               |
| **Model API Health**  | [http://localhost:5000/health](http://localhost:5000/health) |


