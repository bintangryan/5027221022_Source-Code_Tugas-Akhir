Online Recruitment Fraud (ORF) Detection — Environments & Dependencies

> Sistem cerdas untuk mendeteksi lowongan pekerjaan palsu (*Online Recruitment Fraud*) berbahasa Indonesia. Memadukan model **IndoBERT**, teknik **Feature Fusion** metadata (pada notebook eksperimen), serta implementasi **Explainable AI (XAI)** berbasis SHAP untuk transparansi hasil deteksi.

---

## 📁 Struktur Proyek

```text
5027221022_Source-Code_Tugas-Akhir/
├── frontend/               # Antarmuka pengguna (UI/UX) berbasis Web (Next.js)
├── backend/                # REST API, Inferensi Model ML & XAI Engine (Python/FastAPI)
└── notebook/               # Eksperimen, preprocessing, & training model (Jupyter)
```

---

## 🌐 Frontend Environment (Web App)

| Item | Detail |
| :--- | :--- |
| **Project Name** | ORF Detection Frontend |
| **Language** | TypeScript, JavaScript, HTML, CSS |
| **Framework** | Next.js `16.2.6` (App Router) |
| **Runtime Environment** | Node.js |
| **Package Manager** | npm (`package.json`, `package-lock.json`) |

### Core & UI Libraries

- **Next.js (`16.2.6`)** — framework utama berbasis React menggunakan App Router (`app/page.tsx`, `app/layout.tsx`, `next.config.ts`); menangani routing, rendering, dan bundling aplikasi.
- **React & React DOM (`19.2.4`)** — library inti untuk membangun dan merender komponen UI secara deklaratif.
- **axios (`^1.16.1`)** — HTTP client yang dipakai untuk memanggil endpoint backend (`POST /predict`) dan mengirim data lowongan kerja untuk diprediksi.
- **lucide-react (`^1.16.0`)** — kumpulan komponen icon SVG siap pakai untuk mempercantik UI (ikon tombol, indikator status, dsb).
- **clsx (`^2.1.1`)** — utility kecil untuk menggabungkan className secara kondisional (misalnya styling berbeda tergantung status "FRAUD" atau "LEGIT").
- **tailwind-merge (`^3.6.0`)** — menggabungkan beberapa className Tailwind tanpa terjadi konflik/duplikat style yang saling override.
- **Tailwind CSS (`^4`)** — *utility-first CSS framework* untuk styling seluruh komponen (dikonfigurasi lewat `globals.css`).
- **PostCSS (`@tailwindcss/postcss`)** — alat pemroses CSS yang menjembatani Tailwind dengan build pipeline Next.js (`postcss.config.mjs`).

### Linter & Dev Tools

- **TypeScript (`^5`)** — menambahkan *static type checking* ke seluruh kode frontend, dikonfigurasi di `tsconfig.json`.
- **ESLint (`^9`) + eslint-config-next** — menjaga standar kualitas & konsistensi gaya kode sesuai rekomendasi Next.js (`eslint.config.mjs`).
- **@types/node, @types/react, @types/react-dom** — definisi tipe pendukung supaya TypeScript mengenali API Node.js dan React dengan benar.
- **SVG Assets & Icons** — aset visual pendukung UI (`public/`, `favicon.ico`).
- **AI Dev Docs (`AGENTS.md`, `CLAUDE.md`)** — dokumentasi referensi khusus untuk agen AI coding assistant yang bekerja di repo ini.

---

## ⚙️ Backend Environment (REST API & XAI Engine)

| Item | Detail |
| :--- | :--- |
| **Language** | Python (`main.py`, `model_loader.py`, `xai_engine.py`) |
| **Base Image** | `python:3.11-slim` (`Dockerfile`) |
| **Deployment** | Docker → Google Cloud Run (port dinamis lewat env `PORT`, default `8080`) |
| **Dependency Manager** | `pip` via `requirements.txt` |

### API & Server

- **FastAPI** — framework REST API utama (`app = FastAPI(title="Online Recruitment Fraud Diagnostic System API")` di `main.py`); menangani routing dan endpoint `/predict`.
- **Uvicorn** — ASGI server yang menjalankan aplikasi FastAPI (`uvicorn main:app --host 0.0.0.0 --port 8080`), dipanggil baik lewat `CMD` di Dockerfile maupun langsung di `if __name__ == "__main__"`.
- **Pydantic (`BaseModel`)** — mendefinisikan skema request body (`JobInput`: title_id, company_profile_id, description_id, requirements_id, benefits_id) sekaligus validasi otomatis tipe data yang masuk.
- **python-multipart** — dependency FastAPI untuk parsing form-data/file upload (tercantum di `requirements.txt`, disiapkan untuk mendukung request non-JSON bila dibutuhkan).
- **CORSMiddleware (`fastapi.middleware.cors`)** — mengizinkan request lintas origin dari frontend (`allow_origins=["*"]`) supaya Next.js bisa memanggil API tanpa diblokir browser.

### Machine Learning, NLP & Explainable AI (XAI)

- **PyTorch (`torch`, `torch.nn`)** — framework *deep learning* utama; dipakai di `model_loader.py` untuk mendefinisikan arsitektur `MainModelTextOnly` dan memuat bobot hasil training (`model_IndoBERT_Benchmark_S1_20_TEXT_ONLY.pth`), juga untuk inferensi (`torch.inference_mode()`, `torch.autocast`) di `main.py`.
- **Transformers (`AutoModel`, `AutoTokenizer`)** — memuat backbone IndoBERT pretrained (`indobenchmark/indobert-base-p2`) beserta tokenizer-nya; model yang dipakai backend adalah versi **text-only** (tanpa cabang fitur numerik fusion yang ada di notebook training).
- **SHAP (`shap.KernelExplainer`)** — inti dari fitur Explainable AI; menghitung nilai SHapley untuk menjelaskan seberapa besar kontribusi tiap field teks (title, profile, description, dst) terhadap prediksi fraud/legit. Diinisialisasi saat startup server dengan *background data* 10 sampel legit dari `train_20.csv`.
- **Pandas & NumPy** — manipulasi data tabular; membaca `dataset/train_20.csv` dan `dataset/test_20f.csv`, menghitung statistik dinamis per field (`calculate_dynamic_stats` di `xai_engine.py`), serta mengolah array probabilitas hasil prediksi.
- **re (regex)** — dipakai dalam fungsi `clean_text()` di `main.py` untuk membersihkan input user sebelum diproses model: menghapus sisa HTML, menormalisasi spasi antar kata yang menempel (misal `kataA-kataB`), menghapus URL, dan menyeragamkan simbol.
- **BeautifulSoup (`bs4`)** — menghapus tag HTML dari teks input pengguna sebelum masuk ke tokenizer (`BeautifulSoup(text, "html.parser").get_text()`).
- **contextlib (`nullcontext`)** — context manager kondisional; dipakai supaya `torch.autocast` (mixed precision) hanya aktif kalau device-nya CUDA, dan otomatis "dilewati" (no-op) kalau jalan di CPU.

> **Catatan penting:** Model yang di-deploy di backend (`MainModelTextOnly`, di `model_loader.py`) adalah versi **text-only** dari arsitektur `MainModel` yang dilatih di notebook — cabang fitur numerik (fusion) tidak dipakai di web app. Saat load bobot, `strict=False` sengaja dipasang supaya PyTorch tidak error walau ada layer fusion di file `.pth` yang tidak dipakai/di-load ke `MainModelTextOnly`.

---

## 📓 Notebook Environment (Preprocessing & Training)

Kedua notebook dijalankan di **Kaggle Notebook**, Python `3.12.13`, dengan akses internet dinonaktifkan (dataset & model pretrained diambil dari path lokal Kaggle Datasets).

### 1️⃣ Data Preparation (`notebook/data_preparation.ipynb`)

| Item | Detail | Keterangan |
|---|---|---|
| Sumber Data Awal | `fake_job_postings.csv` | Dataset mentah asli (EMSCAD — Employment Scam Aegean Dataset) |
| Sumber Data Antara | `dataset_translated_hierarchy.csv` | Dataset hasil translasi (dimuat ulang setelah proses translasi selesai) |
| Output Akhir | `train_20.csv`, `train_40.csv`, `test_20f.csv` | Hasil akhir data preparation — dipakai di tahap model training, dan juga di-*bundle* ke `backend/dataset/` untuk keperluan SHAP background data & statistik dinamis |

#### 📦 Libraries

**Data Handling**
- **`pandas`** — membaca CSV, manipulasi DataFrame, cek missing value, filtering, grouping.
- **`numpy`** — operasi numerik/array, perhitungan statistik distribusi fitur.

**Text Cleaning & Preprocessing**
- **`re`** (regex) — membersihkan noise di teks (simbol berulang, karakter tidak perlu, dsb) via fungsi `clean_noise()`.
- **`bs4` (`BeautifulSoup`)** — membersihkan tag HTML yang menempel di teks mentah (banyak field job posting mengandung HTML dari web scraping).
- **`warnings`** — mengatur/menyaring warning yang muncul saat proses (misalnya dari BeautifulSoup atau pandas).

**Translasi**
- **`deep_translator` (`GoogleTranslator`)** — menerjemahkan kolom teks (title, company_profile, description, requirements, benefits) dari Bahasa Inggris ke Bahasa Indonesia, menghasilkan kolom `*_id`.
- **`tqdm`** — progress bar saat proses translasi berlangsung (karena translasi per baris bisa memakan waktu lama).
- `!pip install deep_translator` — instalasi library translasi karena tidak tersedia default di environment Kaggle.

**Encoding & Feature Scaling**
- **`sklearn.preprocessing.LabelEncoder`** — encode kolom kategorikal (misalnya `employment_type`, `required_experience`) menjadi representasi numerik.
- **`sklearn.preprocessing.MinMaxScaler`** — menormalisasi fitur numerik/kontekstual (misalnya panjang karakter/token) ke rentang tertentu (0–1) sebelum dipakai model.

**Feature Extraction & Analysis**
- **`sklearn.feature_extraction.text.CountVectorizer`** — mengekstrak kata-kata yang paling sering muncul (top words) per kelas (fraud vs non-fraud) untuk analisis eksploratif.
- **`scipy.stats.chi2_contingency`** — uji Chi-Square untuk menguji hubungan/asosiasi antara fitur kategorikal dengan target `fraudulent`.
- **`sklearn.ensemble` (`RandomForestClassifier`, `GradientBoostingClassifier`)** — model pohon untuk mengukur feature importance non-teks (mana fitur yang paling berpengaruh terhadap label fraud).
- **`sklearn.inspection.permutation_importance`** — mengukur pentingnya suatu fitur dengan mengacak nilainya dan melihat penurunan performa model, sebagai validasi tambahan selain feature importance bawaan Random Forest.

**Data Splitting**
- **`sklearn.model_selection.train_test_split`** — membagi data menjadi set training dan testing, dengan skenario rasio fraud:non-fraud tertentu (mis. 1:4 untuk skenario S1).

**Visualization**
- **`matplotlib.pyplot`** — plotting umum (distribusi fitur, bar chart).
- **`matplotlib.gridspec`** — mengatur layout grid untuk menampilkan banyak subplot sekaligus secara rapi.
- **`seaborn`** — visualisasi statistik (distribusi kategorikal, heatmap, dsb) dengan tampilan lebih informatif.

**Utilities**
- **`IPython.display.display`** — menampilkan output (DataFrame, tabel) dengan format lebih rapi di dalam notebook, terutama untuk komparasi before/after cleaning teks.

#### 🔄 Alur Proses Data Preparation (ringkas)
1. **Load data mentah** (`fake_job_postings.csv`) → cek shape, kolom, missing value.
2. **Eksplorasi kategorikal** → distribusi & persentase per kolom kategorikal.
3. **Text cleaning** → hapus HTML (`BeautifulSoup`), noise, dsb → hasil disimpan sebagai `df_clean`.
4. **Perbandingan before/after** cleaning per kolom teks.
5. **Analisis top words** per kelas (fraud vs non-fraud) via `CountVectorizer`.
6. **Balancing eksploratif** → sampling non-fraud (5x jumlah fraud) untuk analisis lanjutan.
7. **Translasi teks** Inggris → Indonesia (`GoogleTranslator`), dengan preview hasil translasi untuk QA.
8. **Post-processing teks hasil translasi** (`clean_noise`), cek ulang panjang karakter.
9. **Ekstraksi fitur kontekstual** untuk skema *feature fusion* (fitur non-teks: panjang teks, has_company_logo, required_education, dsb) + encoding (`LabelEncoder`).
10. **Analisis fitur non-teks**: distribusi fraud vs non-fraud, feature importance (Random Forest + permutation importance), Chi-Square test.
11. **Scaling fitur numerik/kontekstual** (`MinMaxScaler`).
12. **Split data** menjadi skenario training (S1: 20% fraud rasio 1:4, S2: 40%) dan test set.
13. **Export** hasil akhir ke `train_20.csv`, `train_40.csv`, `test_20f.csv` — siap dipakai di tahap *model training*.

### 2️⃣ Model Training (`notebook/model_training.ipynb`)

| Item | Detail | Keterangan |
|---|---|---|
| Device | `torch.device("cuda" if available else "cpu")` | Menentukan apakah training jalan di GPU (cuda) atau CPU, otomatis menyesuaikan hardware yang tersedia |
| Seed | `torch.manual_seed(42)`, `np.random.seed(42)` | Mengunci random seed supaya hasil training bisa direproduksi (reproducibility) |
| Validasi | `StratifiedKFold`, 5 fold | K-Fold Cross Validation menjaga proporsi kelas tetap seimbang di tiap fold |

#### 📦 Libraries

**Data Handling**
- **`pandas`** — membaca file CSV (`train_20.csv`, dll), manipulasi dan filtering DataFrame (misalnya slicing fold train/val).
- **`numpy`** — operasi numerik/array, dipakai untuk perhitungan statistik (mean, percentile) dan array numerik fitur.

**Deep Learning (PyTorch)**
- **`torch`** — library inti untuk tensor, autograd (perhitungan gradien), dan operasi GPU/CUDA.
- **`torch.nn`** — modul untuk membangun arsitektur neural network: `nn.Module` (base class model), `nn.Linear`, `nn.LayerNorm`, `nn.Dropout`, `nn.MultiheadAttention` (self-attention antar fitur teks), `nn.CrossEntropyLoss` (loss function klasifikasi).
- **`torch.optim` (`AdamW`)** — optimizer untuk mengupdate bobot model (termasuk bobot IndoBERT yang di-fine-tune) selama training, dengan weight decay untuk regularisasi.
- **`torch.utils.data` (`Dataset`, `DataLoader`)** — `Dataset` untuk membungkus data custom (tokenisasi 5 kolom teks + fitur numerik per baris), `DataLoader` untuk membentuk batch dan shuffle data saat training/evaluasi.
- **`torch.amp` (`autocast`, `GradScaler`)** — mixed precision training; mempercepat proses training dan menghemat penggunaan VRAM GPU tanpa banyak mengorbankan akurasi numerik.

**NLP / Transformers**
- **`transformers` (Hugging Face)**
  - **`AutoTokenizer`** — mengubah teks mentah menjadi token ID yang dipahami model (tokenisasi, padding, truncation).
  - **`AutoModel`** — memuat backbone model pretrained (IndoBERT/mBERT) yang nantinya di-fine-tune bersama classifier custom.

**Evaluation & Statistics**
- **`scikit-learn`**
  - **`sklearn.metrics.confusion_matrix`** — menghitung TP/TN/FP/FN sebagai dasar perhitungan accuracy, precision, recall, F1, specificity.
  - **`sklearn.model_selection.StratifiedKFold`** — membagi data menjadi K-Fold dengan menjaga proporsi kelas (fraud vs non-fraud) tetap seimbang di tiap fold.
- **`scipy.stats` (`ttest_rel`, `wilcoxon`)** — uji signifikansi statistik untuk membandingkan performa dua skenario model (text-only vs fusion): paired t-test (parametrik) dan Wilcoxon signed-rank test (non-parametrik).

**Visualization**
- **`matplotlib`** — membuat dan menyimpan plot (figure, save ke PNG).
- **`seaborn`** — membuat heatmap confusion matrix agar lebih mudah dibaca secara visual.

**Utilities**
- **`tqdm`** — progress bar visual saat iterasi training/evaluasi per batch, supaya bisa memantau progres real-time.
- **`gc`** — garbage collection manual, membersihkan memori Python setelah tiap fold selesai.
- **`os`** — operasi sistem file (path handling).

#### 🤖 Pretrained Models
- **`indobenchmark/indobert-base-p2`** — backbone IndoBERT utama yang di-fine-tune untuk tugas klasifikasi; ini juga model yang akhirnya dipakai di backend (versi text-only).
- **`indolem/indobert-base-uncased`** — backbone IndoBERT varian lain, dipakai sebagai perbandingan/benchmark.
- **`bert-base-multilingual-cased` (mBERT)** — model multilingual, dipakai sebagai eksperimen/komparasi tambahan.

---

## 🔗 Keterkaitan Antar Komponen

```text
notebook/  →  hasil training (.pth) & dataset CSV
                    │
                    ▼
backend/model/model_IndoBERT_Benchmark_S1_20_TEXT_ONLY.pth
backend/dataset/{train_20,train_40,test_20f}.csv
                    │
                    ▼
backend (FastAPI + PyTorch + SHAP)  →  REST API /predict
                    │
                    ▼
frontend (Next.js + axios)  →  tampilkan hasil prediksi + XAI highlight ke user
```
