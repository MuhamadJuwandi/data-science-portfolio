# 🛒 Olist E-Commerce Analytics: Business Insights & Forecasting

![Project Banner](visuals/banner_olist_project.png)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.30%2B-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Prophet](https://img.shields.io/badge/Prophet-Forecasting-red?style=for-the-badge&logo=facebook&logoColor=white)](https://facebook.github.io/prophet/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Navigation / ナビゲーション / Navigasi</b><br>
  <a href="#-english-overview">🇬🇧 English</a> | 
  <a href="#-プロジェクト概要">🇯🇵 日本語</a> | 
  <a href="#-ringkasan-proyek">🇮🇩 Indonesia</a>
</p>

---

## 📸 Dashboard Preview
![Dashboard Demo](visuals/dashboard_demo.gif)
---

<a name="-english-overview"></a>
## 🇬🇧 English Overview

### 🧐 Business Problem
The Brazilian e-commerce sector generates massive transactional data, yet many platforms fail to leverage this for strategic decision-making. Key challenges identified in the **Olist** dataset include:
* **Customer Churn:** 90% of customers are one-time buyers.
* **Inventory Risks:** Lack of demand forecasting leads to stockouts during peak seasons (e.g., Black Friday).
* **Logistics Impact:** No correlation analysis between delivery times and customer satisfaction (CSAT).

### 💡 Solution & Methodology
This project implements an **End-to-End Data Science Workflow** to transform raw data into actionable insights:
1.  **ETL Pipeline:** Cleaned and merged 9 relational tables (100k+ rows) using SQL-style joins in Pandas.
2.  **Customer Segmentation:** Applied **RFM Analysis (Recency, Frequency, Monetary)** and **K-Means Clustering** to identify "Champions" vs "Hibernating" customers.
3.  **Demand Forecasting:** Utilized **Meta's Prophet** (Additive Model) to predict sales trends for the next 90 days, accounting for seasonality.
4.  **Interactive Dashboard:** Built a user-friendly Streamlit app for stakeholders to monitor KPIs dynamically.

### 📊 Key Insights
* **Seller Concentration:** The top 10% of sellers generate **80% of total revenue**.
* **Delivery Sensitivity:** Each day of delivery delay correlates with a **0.5-star drop** in review scores.
* **Seasonality:** Health & Beauty products show a consistent spike in early weeks of the month.

### 🛠 Tech Stack
* **Data Processing:** Pandas, NumPy, SQL
* **Machine Learning:** Scikit-Learn (K-Means), Prophet (Time Series)
* **Visualization:** Plotly, Matplotlib, Seaborn
* **Deployment:** Streamlit Cloud

---

<a name="-プロジェクト概要"></a>
## 🇯🇵 プロジェクト概要

### 🧐 背景と課題
ブラジルのEコマース市場では、膨大な取引データが生成されていますが、多くのプラットフォームはそれを戦略的な意思決定に活用できていません。本プロジェクトでは、**Olist**（ブラジルのECサイト）の公開データセットを使用し、以下の課題に取り組みました。
* **顧客離れ:** 顧客の90%が一度きりの購入で終わっている。
* **在庫リスク:** 需要予測の欠如により、繁忙期（ブラックフライデーなど）に在庫切れが発生。
* **物流の影響:** 配送遅延が顧客満足度（CSAT）に与える影響が可視化されていない。

### 💡 ソリューション・手法
データを実用的なインサイトに変換するために、**エンドツーエンドのデータサイエンスワークフロー**を構築しました。
1.  **ETL処理:** 9つのリレーショナルテーブル（10万行以上）をPandasで結合・クレンジング。
2.  **顧客セグメンテーション:** **RFM分析**と**K-Meansクラスタリング**（機械学習）を用い、顧客を「優良顧客」や「休眠顧客」に分類。
3.  **需要予測:** **Prophet**（時系列モデル）を使用し、季節性を考慮した向こう90日間の売上を予測。
4.  **ダッシュボード:** Streamlitを使用し、非技術者（ステークホルダー）でもKPIを監視できるWebアプリを開発。

### 📊 主な発見
* **パレートの法則:** 上位10%の販売者が、総売上の**80%**を生み出している。
* **配送とレビュー:** 配送が1日遅れるごとに、レビュー評価が平均**0.5**ポイント低下する。
* **季節性:** 「ヘルス＆ビューティー」カテゴリーは、毎月初旬に売上が急増する傾向がある。

---

<a name="-ringkasan-proyek"></a>
## 🇮🇩 Ringkasan Proyek

### 🧐 Latar Belakang Bisnis
Banyak platform e-commerce memiliki data transaksi besar namun gagal mengolahnya menjadi strategi bisnis. Berdasarkan dataset **Olist Brazil**, ditemukan masalah utama: sulitnya mengidentifikasi pelanggan loyal, ketidakpastian stok saat *peak season*, dan kurangnya analisis dampak logistik terhadap kepuasan pelanggan.

### 💡 Solusi Teknis
Proyek ini bukan sekadar analisis, melainkan solusi **Data Science End-to-End**:
* **Segmentasi Pelanggan:** Menggabungkan RFM Analysis dengan Machine Learning (K-Means) untuk personalisasi marketing.
* **Forecasting (Peramalan):** Menggunakan algoritma Prophet untuk memprediksi permintaan produk di masa depan, membantu manajemen stok gudang.
* **Dashboard Interaktif:** Membangun *tools* berbasis Streamlit agar tim bisnis dapat memantau performa penjualan secara *real-time*.

---

## 📂 Project Structure

```bash
olist-dashboard-project/
├── 📁 data/                  # Raw & Processed CSVs (Git ignored)
├── 📁 notebooks/             # Jupyter Notebooks for Experimentation
│   ├── 1_Data_Cleaning_ETL.ipynb
│   ├── 2_RFM_Segmentation_Clustering.ipynb
│   └── 3_Demand_Forecasting_Prophet.ipynb
├── 📁 dashboard/             # Streamlit Production Code
│   ├── main.py               # Main App Entry Point
│   └── utils.py              # Helper Functions
├── 📁 visuals/               # Images for README & Presentation
├── requirements.txt          # Python Dependencies
└── README.md                 # Project Documentation

```

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone [https://github.com/MuhamadJuwandi/olist-ecommerce-analytics.git](https://github.com/MuhamadJuwandi/olist-ecommerce-analytics.git)
cd olist-ecommerce-analytics

```

### 2. Install Dependencies

Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt

```

### 3. Run Streamlit Dashboard

```bash
streamlit run dashboard/main.py

```

---

## 📬 Contact & Portfolio

**Muhamad Juwandi** *Data Science Enthusiast | Graphic Designer*

---

<p align="center">Made with ❤️ and ☕ in Bogor, Indonesia</p>

```

### ⚙️ Action Items untuk Juwandi:

1.  **Folder Structure:** Pastikan struktur folder di laptop Anda sesuai dengan diagram `Project Structure` di atas. Ini menunjukkan kerapian manajemen file (Sangat penting untuk penilaian Git).
2.  **Requirements.txt:** Jalankan perintah `pip freeze > requirements.txt` di environment Anda sebelum upload, tapi pastikan bersihkan library yang tidak terpakai agar file tidak bengkak.
      * *Pastikan library ini ada:* `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `prophet`, `streamlit`, `plotly`.
3.  **Visual Assets:**
      * Buat Banner (`visuals/banner_olist_project.png`) dengan sentuhan desain grafis Anda. Ini "unfair advantage" Anda dibanding Data Scientist lain yang visualnya kaku.
      * Buat GIF (`visuals/dashboard_demo.gif`) yang menunjukkan Anda mengklik filter tanggal atau kategori di Streamlit.
4.  **Pin Repository:** Setelah di-push, gunakan perintah `/pin` (secara manual di settings GitHub) untuk menaruh proyek ini di halaman depan profil Anda.

```
