# SIPADMA
## Sistem Prediksi Anomali Distribusi Air
### Deteksi Indikasi Kebocoran Air Berbasis Machine Learning — PDAM

---

## CARA INSTALL & JALANKAN

```bash
# 1. Install library
pip install -r requirements.txt

# 2. Generate data simulasi (skip kalau sudah punya data asli)
python ml/generate_data.py

# 3. Jalankan sistem
python app.py

# 4. Buka browser
# http://localhost:5000
```

---

## PAKAI DATA ASLI PDAM

Siapkan file Excel dengan kolom:

| Kolom            | Keterangan              | Wajib |
|------------------|-------------------------|-------|
| id_pelanggan     | Kode unik pelanggan     | ✅    |
| bulan            | Angka bulan (1–12)      | ✅    |
| tahun            | Tahun 4 digit           | ✅    |
| volume_m3        | Pemakaian air (m³)      | ✅    |
| nama_pelanggan   | Nama pelanggan          | ⬜    |
| wilayah          | Kelurahan/zona          | ⬜    |
| golongan_tarif   | Golongan tarif (R1 dst) | ⬜    |

Lalu jalankan sistem dan upload lewat menu **Upload Data**.

---

## STRUKTUR FOLDER

```
sipadma/
├── app.py                  ← Jalankan ini
├── requirements.txt
├── ml/
│   ├── generate_data.py    ← Generator data simulasi
│   ├── detector.py         ← Core ML engine
│   ├── model.pkl           ← Model tersimpan
│   └── scaler.pkl          ← Scaler tersimpan
├── data/
│   └── data_pdam.csv       ← Data utama
├── templates/              ← Halaman web
└── uploads/                ← File upload sementara
```

---

## METODE ML

| Komponen         | Detail                                      |
|------------------|---------------------------------------------|
| Algoritma utama  | Isolation Forest (scikit-learn)             |
| Metode pendukung | Z-Score per pelanggan                       |
| Jenis ML         | Unsupervised Learning                       |
| Fitur input      | volume, mean, std, delta, pct_change,       |
|                  | zscore, rolling_mean_3, rolling_mean_6,     |
|                  | rasio_rolling3                              |
| Contamination    | 0.07 (7% data dianggap anomali)             |
| Output           | anomaly label, anomaly score, risiko,alasan |

---

## JAWABAN SIDANG

**"Kenapa tidak perlu data semua 64 ribu pelanggan?"**
Isolation Forest bekerja dengan sampling statistik. Sampel representatif 2.000–5.000 pelanggan dari semua wilayah dan golongan sudah menghasilkan model yang generalizable. Menggunakan seluruh 64 ribu pelanggan justru akan memperlambat proses tanpa peningkatan akurasi yang signifikan untuk skala penelitian ini.

**"Kenapa Isolation Forest bukan algoritma lain?"**
Isolation Forest dipilih karena: (1) unsupervised — tidak butuh label data bocor/tidak; (2) efisien untuk data tabular berdimensi tinggi; (3) tidak sensitif terhadap distribusi data yang tidak normal; (4) terbukti efektif untuk anomaly detection pada data time-series utilitas.

**"Bagaimana kalau ada data baru?"**
Sistem dirancang untuk update berkelanjutan. Petugas cukup upload file Excel bulanan via menu Upload Data. Sistem otomatis menggabungkan data lama dan baru, melatih ulang model, dan memperbarui hasil deteksi tanpa perlu mengulang dari awal.
