<div align="center">

<img src="assets/banner-tepco.png" alt="TEPCO Forecasting Banner" width="100%">

# ⚡ TEPCO Electricity Demand Forecasting
### 東京電力電力需要予測 (Tokyo Electric Power Company)

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Prophet](https://img.shields.io/badge/Prophet-Time%20Series-blueviolet?style=for-the-badge)](https://facebook.github.io/prophet/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

<p align="center">
  <b>Navigation / ナビゲーション / Navigasi</b><br>
  <a href="#-english">🇬🇧 English</a> | 
  <a href="#-日本語-japanese">🇯🇵 日本語</a> | 
  <a href="#-bahasa-indonesia">🇮🇩 Bahasa Indonesia</a>
</p>

</div>

---

## 🇬🇧 English

### 📌 Project Overview
A production-ready time-series forecasting pipeline predicting hourly electricity consumption for **TEPCO (Tokyo Electric Power Company)** in the Kanto region. This project addresses the critical challenge of **grid stability** and **energy load balancing** by leveraging deep learning (LSTM) and additive models (Prophet).

**Business Value:**
* **💰 Cost Reduction:** Improves procurement planning accuracy, minimizing expensive spot-market purchases.
* **⚡ Grid Stability:** Predicts demand spikes (>28°C) to prevent outages and optimize battery storage deployment.

### 🚀 Key Features
* **Hybrid Modeling:** Comparison between **Facebook Prophet** (Seasonality) and **LSTM** (Complex temporal dependencies).
* **Robust ETL Pipeline:** Handles multi-year raw data (CSV/Shift-JIS encoding) and merges external Weather API data.
* **Interactive Dashboard:** A Streamlit-based web app for stakeholders to visualize forecasts and simulate peak-shaving.

### 📊 Model Performance (2024 Test Set)
| Model | MAE (Mean Absolute Error) | RMSE | Interpretation |
|-------|--------------------------|------|----------------|
| **Prophet** | ~250 MW | ~320 MW | Strong baseline; captures weekly seasonality well. |
| **LSTM** | **~150 MW** | **~190 MW** | **Best Performer.** Captures non-linear spikes during extreme weather. |

---

## 🇯🇵 日本語 (Japanese)

### 📌 プロジェクト概要
関東エリアにおける**東京電力（TEPCO）**の1時間ごとの電力需要を予測する、本番環境を意識した時系列予測パイプラインです。本プロジェクトは、ディープラーニング（LSTM）と加法モデル（Prophet）を活用し、**電力グリッドの安定性**と**需給バランス**という重要な課題に取り組みます。

**ビジネス価値:**
* **コスト削減:** 調達計画の精度を向上させ、高額なスポット市場での電力購入を最小限に抑えます。
* **グリッドの安定化:** 気温上昇（28°C以上）による需要急増を予測し、停電リスクを回避します。

### 🚀 主な機能
* **ハイブリッド・モデリング:** **Facebook Prophet**（季節性分析）と **LSTM**（複雑な時間依存性）の性能比較。
* **堅牢なETLパイプライン:** 複数年の生データ（CSV/Shift-JIS文字コード）を処理し、外部の気象APIデータと統合。
* **インタラクティブ・ダッシュボード:** Streamlitを使用したWebアプリにより、関係者が予測を可視化し、ピークカットのシミュレーションが可能。

---

## 🇮🇩 Bahasa Indonesia

### 📌 Ringkasan Proyek
Pipeline *forecasting* deret waktu (time-series) untuk memprediksi konsumsi listrik per jam **TEPCO (Tokyo Electric Power Company)** di wilayah Kanto, Jepang. Proyek ini menjawab tantangan krusial dalam **stabilitas jaringan listrik** dan **penyeimbangan beban energi** menggunakan Deep Learning (LSTM).

**Nilai Bisnis:**
* **Efisiensi Biaya:** Meningkatkan akurasi perencanaan pengadaan energi dan mengurangi pemborosan.
* **Mitigasi Risiko:** Memprediksi lonjakan permintaan ekstrem akibat cuaca panas (>28°C).

### 🚀 Fitur Utama
* **ETL Pipeline V2:** Otomasi pembersihan data multi-tahun, penanganan *encoding* Jepang, dan integrasi data Cuaca.
* **Komparasi Model:** Analisis performa antara Prophet (Tren Musiman) vs LSTM (Akurasi Tinggi).
* **Dashboard Streamlit:** Visualisasi data *real-time* untuk pengambilan keputusan operasional.

---

## 📂 Project Structure

```bash
TEPCO-Demand-Forecasting/
├── 📂data/
│   ├── raw/                 # Original TEPCO files (2020-2025)
│   ├── processed/           # Merged & Cleaned datasets
│   ├── external/            # Weather data (Tokyo)
├── 📂notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_FeatureEngineering.ipynb # Lag, Rolling, interaction features
│   ├── 03_Modeling_Prophet.ipynb  # Baseline Model
│   ├── 04_Modeling_LSTM.ipynb     # Deep Learning Model
│   └── 05_Evaluation_Insights.ipynb # Metrics & Business Analysis
├── 📂scripts/
│   ├── etl_v2.py            # Main ETL pipeline
│   ├── train_lstm.py        # PyTorch LSTM training script
│   ├── weather_fetcher.py   # API Data Fetcher
│   └── ...
├── 📂visualizations/          # Generated plots for PPT/Reports
├── dashboard.py             # Streamlit Application
└── requirements.txt         # Project Dependencies

```

---

## 🛠️ Installation & Usage

### 1. Clone & Install
```bash
git clone [https://github.com/yourusername/TEPCO-Forecasting.git](https://github.com/yourusername/TEPCO-Forecasting.git)
cd TEPCO-Forecasting
pip install -r requirements.txt
```

### 2. Data Preparation (ETL)
Run the pipeline to process raw TEPCO data and fetch weather info:
```bash
python scripts/etl_v2.py        # Parse TEPCO data
python scripts/weather_fetcher.py # Download Weather Data
python scripts/merge_weather.py   # Merge Datasets
```

### 3. Training
Train the LSTM Neural Network (ensure PyTorch is installed):
```bash
python scripts/train_lstm.py
```

### 4. Launch Dashboard
Visualize the results locally:
```bash
streamlit run dashboard.py
```

---

## 🔮 Future Improvements
* **Transformer Architecture:** Implement **Temporal Fusion Transformer (TFT)** for better interpretability of variable importance.
* **CI/CD Integration:** Automate model retraining pipelines using GitHub Actions.
* **Containerization:** Dockerize the application for scalable cloud deployment (AWS/GCP).

---
<div align="center">

**Developed by Muhamad Juwandi**
<br>
*Data Science Student & Graphic Designer*
<br>
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/muhamadjuwandi)
[![Portfolio](https://img.shields.io/badge/Portfolio-Keunal.id-ff69b4?style=flat&logo=dribbble)](https://keunal.id)

</div>
