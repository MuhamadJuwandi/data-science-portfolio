# 🎬 MAL Recommender Hybrid System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-ff4b4b)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688)
![Status](https://img.shields.io/badge/Status-Completed-success)

[🇬🇧 English](#english) | [🇯🇵 日本語 (Japanese)](#japanese) | [🇮🇩 Bahasa Indonesia](#indonesian)

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

### 🚀 How to Run
**1. Clone the Repository**
```bash
git clone [https://github.com/MuhamadJuwandi/data-science-portfolio.git](https://github.com/MuhamadJuwandi/data-science-portfolio.git)
cd data-science-portfolio/mal-recommender-hybrid
2. Install Dependencies

Bash

pip install -r requirements.txt
3. Run Dashboard

Bash

streamlit run dashboard/app.py
<a name="japanese"></a>

🇯🇵 日本語 (Japanese)
概要 (Overview)
MAL Recommender Hybrid は、MyAnimeListのデータセットに基づいて構築されたスケーラブルな推薦エンジンです。ハイブリッド・フィルタリングのアプローチを採用しており、ユーザーの潜在的な好みを捉える協調フィルタリング (SVD) と、ジャンルやメタデータに基づいて類似作品を推奨するコンテンツベース・フィルタリング (TF-IDF) を組み合わせています。

このプロジェクトは、効率的なデータ処理（Parquet/サンプリング）から、StreamlitダッシュボードおよびREST APIによるモデルのデプロイまで、エンドツーエンドのデータサイエンスワークフローを実証しています。

主な機能 (Key Features)
ハイブリッドエンジン: SVD（行列分解）とTF-IDF（コサイン類似度）の加重組み合わせ。

コールドスタート対策: 新規または匿名のユーザーに対して、人気のアニメを自動的に提案します。

メモリ効率: データサンプリングとParquetストレージを実装し、一般的なハードウェアでも動作するように設計されています。

インタラクティブUI: おすすめの探索やEDA（探索的データ分析）を視覚化するための使いやすいダッシュボード。

プロジェクト構成 (Structure)
src/: コアアルゴリズム（前処理、モデルロジック）。

dashboard/: Streamlitを使用したフロントエンドアプリケーション。

api/: FastAPIを使用したバックエンドサービス。

data/: 処理済みParquetファイルの保存場所。

🚀 実行方法 (How to Run)
コマンドラインで以下を実行してください：

1. リポジトリのクローン

Bash

git clone [https://github.com/MuhamadJuwandi/data-science-portfolio.git](https://github.com/MuhamadJuwandi/data-science-portfolio.git)
cd data-science-portfolio/mal-recommender-hybrid
2. 依存関係のインストール

Bash

pip install -r requirements.txt
3. ダッシュボードの起動

Bash

streamlit run dashboard/app.py
<a name="indonesian"></a>

🇮🇩 Bahasa Indonesia
Ringkasan
MAL Recommender Hybrid adalah mesin rekomendasi yang dibangun menggunakan dataset MyAnimeList. Sistem ini menggunakan pendekatan Hybrid Filtering, menggabungkan Collaborative Filtering (SVD) untuk menangkap preferensi pengguna dan Content-Based Filtering (TF-IDF) untuk merekomendasikan item serupa berdasarkan genre.

Proyek ini mendemonstrasikan alur kerja Data Science secara menyeluruh (end-to-end): mulai dari pemrosesan data yang efisien hingga deployment model melalui Dashboard Streamlit dan REST API.

Fitur Utama
Mesin Hybrid: Kombinasi berbobot antara SVD dan TF-IDF.

Penanganan Cold Start: Otomatis menyarankan anime populer untuk pengguna baru.

Efisiensi Memori: Menggunakan format penyimpanan Parquet agar hemat memori.

UI Interaktif: Dashboard yang mudah digunakan untuk melihat hasil rekomendasi.

Struktur Proyek
src/: Algoritma inti (Preprocessing, Logika Model).

dashboard/: Aplikasi tampilan depan menggunakan Streamlit.

api/: Layanan backend menggunakan FastAPI.

data/: Penyimpanan file data yang sudah diproses.
