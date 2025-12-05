# 🎬 MAL Recommender Hybrid System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-ff4b4b)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688)
![Status](https://img.shields.io/badge/Status-Completed-success)

[English](#english) | [日本語 (Japanese)](#japanese) | [Bahasa Indonesia](#indonesian)

<br>

<div align="center">
  <img src="images/dashboard_preview.png" alt="Dashboard Preview" width="700"/>
  <br>
  <em>Preview of the Recommendation Dashboard</em>
</div>

---

<a name="english"></a>
## 🇬🇧 English

### Overview
**MAL Recommender Hybrid** is a scalable recommendation engine built on the MyAnimeList dataset. It utilizes a **Hybrid Filtering** approach, combining **Collaborative Filtering (SVD)** to capture user latent preferences and **Content-Based Filtering (TF-IDF)** to recommend similar items based on genres and metadata.

This project demonstrates an end-to-end Data Science workflow: from efficient data processing (Parquet/Sampling) to model deployment via a Streamlit Dashboard and REST API.

### Key Features
* **Hybrid Engine:** Weighted combination of SVD (Matrix Factorization) and TF-IDF (Cosine Similarity).
* **Cold Start Handling:** Automatically suggests popular anime for new/anonymous users.
* **Memory Efficient:** Implements data sampling and Parquet storage to run on standard hardware.
* **Interactive UI:** A user-friendly dashboard to explore recommendations and visualize EDA.

### Project Structure
* `src/`: Core algorithms (Preprocessing, Model logic).
* `dashboard/`: Frontend application using Streamlit.
* `api/`: Backend service using FastAPI.
* `data/`: Storage for processed Parquet files.

---

<a name="japanese"></a>
## 🇯🇵 日本語 (Japanese)

### 概要
**MAL Recommender Hybrid** は、MyAnimeList データセットを用いたスケーラブルな推薦システムです。**SVD（協調フィルタリング）** と **TF-IDF（内容ベースフィルタリング）** を組み合わせたハイブリッド手法を採用し、ユーザーの潜在的嗜好とアニメの特徴情報を統合して最適な推薦を行います。

このプロジェクトは、データ処理（Parquet/サンプリング）からモデルのデプロイ（Streamlit ダッシュボードと REST API）までのエンドツーエンドなデータサイエンスワークフローを示しています。

### 主な特徴
* ハイブリッドエンジン（SVD + TF-IDF）
* コールドスタート対策（新規ユーザーに人気アニメを推薦）
* メモリ効率の高い設計（Parquet形式・サンプリング）
* StreamlitによるインタラクティブなUI

### プロジェクト構成
* `src/`: コアアルゴリズム（前処理、モデルロジック）
* `dashboard/`: Streamlitフロントエンドアプリ
* `api/`: FastAPIバックエンド
* `data/`: Parquetファイル格納

---

<a name="indonesian"></a>
## 🇮🇩 Bahasa Indonesia

### Gambaran Umum
**MAL Recommender Hybrid** adalah sistem rekomendasi berskala besar yang dibangun menggunakan dataset MyAnimeList. Proyek ini menggunakan pendekatan **Hybrid Filtering**, yaitu kombinasi antara **Collaborative Filtering (SVD)** untuk menangkap preferensi pengguna dan **Content-Based Filtering (TF-IDF)** untuk merekomendasikan anime berdasarkan genre dan metadata.

Proyek ini menunjukkan alur kerja Data Science secara menyeluruh, mulai dari pengolahan data (Parquet/Sampling) hingga deployment model menggunakan Streamlit Dashboard dan REST API.

### Fitur Utama
* Mesin Hybrid (SVD + TF-IDF)
* Penanganan Cold Start (rekomendasi anime populer untuk pengguna baru)
* Efisien dalam penggunaan memori
* Antarmuka interaktif dengan Streamlit Dashboard

### Struktur Proyek
* `src/`: Algoritma utama (Preprocessing, Model)
* `dashboard/`: Aplikasi Streamlit frontend
* `api/`: Layanan backend FastAPI
* `data/`: Penyimpanan data hasil olahan Parquet

---

### 🚀 How to Run / 実行方法 / Cara Menjalankan
```bash
# 1. Clone the Repository / リポジトリをクローン / Clone Repository
git clone https://github.com/MuhamadJuwandi/data-science-portfolio.git
cd data-science-portfolio/mal-recommender-hybrid

# 2. Install Dependencies / 依存関係をインストール / Instalasi Dependensi
pip install -r requirements.txt

# 3. Run the Dashboard / ダッシュボードを起動 / Jalankan Dashboard
python -m streamlit run dashboard/app.py
