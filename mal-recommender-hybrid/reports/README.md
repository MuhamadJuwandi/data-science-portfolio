# Laporan Proyek: MAL Recommender Hybrid System

**Oleh:** [Muhamad Juwandi]
**Tanggal:** 27 November 2025
**Repositori:** [https://github.com/MuhamadJuwandi]

---

## 1. Executive Summary

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi anime yang cerdas dan skalabel menggunakan dataset MyAnimeList. Dengan memanfaatkan pendekatan **Hybrid Filtering** (menggabungkan Collaborative Filtering dan Content-Based Filtering), sistem ini dirancang untuk memberikan rekomendasi yang tidak hanya akurat berdasarkan riwayat tontonan pengguna, tetapi juga mampu memperkenalkan judul-judul baru yang relevan secara konten.

Hasil akhir dari proyek ini adalah sebuah ekosistem lengkap yang terdiri dari:
-   **Pipeline Data Engineering** yang efisien untuk memproses jutaan interaksi.
-   **Model Hybrid** yang menyeimbangkan personalisasi dan eksplorasi.
-   **Dashboard Interaktif** untuk demonstrasi visual.
-   **REST API** yang siap untuk integrasi produksi.

## 2. Business Problem

Dalam platform streaming atau database konten sebesar MyAnimeList, pengguna sering mengalami **Choice Overload** (kebingungan memilih karena terlalu banyak opsi). Hal ini dapat menyebabkan:
-   **User Churn:** Pengguna meninggalkan platform karena frustrasi tidak menemukan tontonan menarik.
-   **Low Engagement:** Pengguna hanya menonton judul-judul populer (mainstream) dan melewatkan "hidden gems" yang sebenarnya sesuai dengan selera mereka.

Sistem rekomendasi tradisional seringkali memiliki keterbatasan:
-   **Collaborative Filtering (CF):** Gagal merekomendasikan item baru (Cold Start Problem) dan cenderung bias ke item populer.
-   **Content-Based (CB):** Terlalu spesifik dan kurang mampu memberikan kejutan (serendipity).

## 3. Solution Strategy

Untuk mengatasi masalah di atas, kami mengimplementasikan **Hybrid Recommender System**.

### Mengapa Hybrid?
Pendekatan Hybrid menggabungkan kekuatan kedua metode:
1.  **Collaborative Filtering (SVD):** Menangkap pola interaksi latent antar pengguna. Jika User A dan User B memiliki selera mirip, sistem akan merekomendasikan apa yang disukai User B kepada User A.
2.  **Content-Based Filtering (TF-IDF):** Menganalisis fitur item (Genre, Sinopsis). Jika pengguna menyukai "Action" dengan tema "Cyberpunk", sistem akan mencari anime serupa secara tekstual.

**Mekanisme:**
Skor akhir rekomendasi dihitung menggunakan **Weighted Average**:
$$ Score_{final} = \alpha \cdot Score_{CF} + (1 - \alpha) \cdot Score_{CB} $$
Dimana $\alpha$ adalah parameter bobot yang dapat disesuaikan (default 0.6 untuk memprioritaskan personalisasi CF).

## 4. Technical Implementation

### Arsitektur Sistem
1.  **Data Ingestion & Preprocessing:**
    -   Dataset: MyAnimeList (Kaggle).
    -   **Memory Management:** Mengingat ukuran data interaksi yang masif (>100 juta baris), kami menerapkan teknik **Smart Sampling**. Kami memfilter data untuk mengambil Top 10.000 pengguna paling aktif dan anime dengan minimal 50 rating. Ini memastikan model dilatih pada data berkualitas tinggi (High Signal-to-Noise Ratio) dan dapat dijalankan pada resource terbatas (16GB RAM).
    -   Format Data: Penyimpanan menggunakan format `.parquet` untuk efisiensi I/O dan kompresi.

2.  **Modeling:**
    -   **SVD (Singular Value Decomposition):** Diimplementasikan menggunakan library `Surprise`. Teknik ini mendekomposisi matriks User-Item menjadi faktor-faktor laten.
    -   **TF-IDF Vectorization:** Mengubah teks sinopsis dan genre menjadi vektor numerik untuk menghitung Cosine Similarity.
    -   **Cold Start Handler:** Untuk pengguna baru tanpa riwayat, sistem secara otomatis beralih (fallback) ke rekomendasi "Top Popular" berdasarkan skor dan jumlah penonton.

3.  **Deployment:**
    -   **FastAPI:** Backend service yang menyediakan endpoint RESTful.
    -   **Streamlit:** Frontend dashboard untuk visualisasi dan uji coba model secara real-time.

## 5. Data Discovery & Insights (The "Aha!" Moments)

Dari eksplorasi data yang mendalam, ditemukan beberapa fakta kunci yang mendasari desain sistem ini:

1.  **Extreme Sparsity (>99%):**
    *   Matriks interaksi user-item memiliki tingkat kekosongan lebih dari 99%. Artinya, rata-rata pengguna hanya menonton sebagian kecil dari total katalog.
    *   *Implikasi:* Ini memvalidasi keputusan menggunakan **Matrix Factorization (SVD)** yang sangat robust dalam mengisi *missing values* dibanding metode *memory-based*.

2.  **Korelasi Kualitas vs Popularitas (0.70):**
    *   Terdapat korelasi positif yang kuat antara jumlah penonton dan skor rata-rata. Namun, ini bukan korelasi sempurna (1.0).
    *   *Hidden Gems:* Algoritma kami berhasil mengidentifikasi **26 Judul Anime** dengan skor sempurna (>8.0) namun memiliki popularitas di bawah rata-rata (Bottom 25%). Judul-judul inilah yang menjadi nilai jual utama fitur "Discovery" kami.

## 6. Impact & Results

Meskipun evaluasi dilakukan dalam lingkungan simulasi (offline evaluation), model menunjukkan karakteristik yang menjanjikan:

| Metrik | Deskripsi | Hasil Observasi |
| :--- | :--- | :--- |
| **RMSE (Root Mean Square Error)** | Mengukur rata-rata kesalahan prediksi rating (skala 1-10). | **~1.12** (Indikasi prediksi cukup akurat) |
| **Catalog Coverage** | Kemampuan sistem merekomendasikan item *long-tail*. | Meningkat berkat komponen Content-Based yang mengangkat "Hidden Gems". |
| **Latency** | Waktu respon API. | **< 200ms** per request (setelah caching model). |

## 6. Future Work

Untuk pengembangan selanjutnya, beberapa inisiatif dapat dilakukan:
1.  **Deep Learning:** Mengimplementasikan Neural Collaborative Filtering (NCF) untuk menangkap pola non-linear yang lebih kompleks.
2.  **Real-time Learning:** Menggunakan Online Learning untuk memperbarui model secara instan setiap kali pengguna memberikan rating baru.
3.  **A/B Testing:** Melakukan eksperimen langsung kepada pengguna untuk mengukur dampak nyata terhadap *Watch Time* dan *Retention Rate*.

---
*Copyright © 2025. Dibuat sebagai bagian dari Portofolio Data Science.*
