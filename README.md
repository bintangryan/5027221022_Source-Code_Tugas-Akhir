Online Recruitment Fraud (ORF) Detection — Environments & Dependencies

> Sistem cerdas untuk mendeteksi lowongan pekerjaan palsu (*Online Recruitment Fraud*) berbahasa Indonesia. Memadukan model **IndoBERT**, teknik **Feature Fusion** metadata, serta implementasi **Explainable AI (XAI)** untuk transparansi hasil deteksi.

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
| **Framework** | Next.js (App Router) |
| **Runtime Environment** | Node.js |
| **Package Manager** | npm (`package.json`, `package-lock.json`) |

### Core & UI Libraries

| Library | Keterangan |
| :--- | :--- |
| **Next.js** | Framework utama berbasis React menggunakan App Router (`app/page.tsx`, `app/layout.tsx`, `next.config.ts`) |
| **React & React DOM** | Library inti untuk merender antarmuka pengguna |
| **Tailwind CSS** | *Utility-first CSS framework* untuk styling komponen |
| **PostCSS** | Alat pemroses CSS, terintegrasi melalui `postcss.config.mjs` |

### Linter & Tools

| Library / Tool | Keterangan |
| :--- | :--- |
| **TypeScript** | *Static type checking* yang dikonfigurasi pada `tsconfig.json` |
| **ESLint** | Alat penjaga standar kualitas kode (`eslint.config.mjs`) |
| **SVG Assets & Icons** | Aset visual pendukung UI (`public/`, `favicon.ico`) |
| **AI Dev Docs** | Dokumentasi referensi agen AI (`AGENTS.md`, `CLAUDE.md`) |

---

## ⚙️ Backend Environment (REST API & XAI Engine)

| Item | Detail |
| :--- | :--- |
| **Language** | Python (`main.py`, `model_loader.py`, `xai_engine.py`) |
| **Deployment** | Docker Engine (`Dockerfile`, `.dockerignore`) |
| **Dependency Manager**| `pip` via `requirements.txt` |

### API & Server

| Library / Tool | Keterangan |
| :--- | :--- |
| **FastAPI / Flask** | Framework pengembangan REST API untuk memproses *request* dari *frontend* |
| **Uvicorn / Gunicorn**| ASGI/WSGI server untuk menjalankan aplikasi backend |

### Machine Learning, NLP & Explainable AI (XAI)

| Library | Modul / File Terkait | Keterangan |
| :--- | :--- | :--- |
| **PyTorch** | `torch` (`model/model_IndoBERT_Benchmark_S1_20_TEXT_ONLY.pth`) | Framework *deep learning* utama untuk menjalankan file model `.pth` |
| **Transformers** | `transformers` (Hugging Face) | Memuat arsitektur dan *tokenizer* untuk model bahasa **IndoBERT** |
| **SHAP** | `shap` (`xai_engine.py`) | Menghasilkan nilai *Explainable AI* (SHapley Additive exPlanations) |
| **Pandas & NumPy** | `pandas`, `numpy` | Manipulasi data, memproses *feature fusion*, dan membaca dataset (`dataset/*.csv`) |

---

## Notebook Environment (Preprocessing & Training)
# Model Training Environments & Libraries
 
## 🖥️ Environment
 
| Item | Detail | Keterangan |
|---|---|---|
| Platform | Kaggle Notebook | Tempat notebook dijalankan (menyediakan GPU gratis & input dataset via Kaggle Datasets) |
| Python | 3.12.13 | Versi interpreter Python |
| Internet Access | Disabled | Dataset & model pretrained diambil dari path lokal Kaggle (`/kaggle/input/...`), bukan diunduh langsung dari internet saat runtime |
| Device | `torch.device("cuda" if available else "cpu")` | Menentukan apakah training jalan di GPU (cuda) atau CPU, otomatis menyesuaikan hardware yang tersedia |
| Seed | `torch.manual_seed(42)`, `np.random.seed(42)` | Mengunci random seed supaya hasil training bisa direproduksi (reproducibility) |
 
## 📦 Libraries
 
### Data Handling
- **`pandas`** — membaca file CSV (`train_20.csv`, dll), manipulasi dan filtering DataFrame (misalnya slicing fold train/val).
- **`numpy`** — operasi numerik/array, dipakai untuk perhitungan statistik (mean, percentile) dan array numerik fitur.
### Deep Learning (PyTorch)
- **`torch`** — library inti untuk tensor, autograd (perhitungan gradien), dan operasi GPU/CUDA.
- **`torch.nn`** — modul untuk membangun arsitektur neural network: `nn.Module` (base class model), `nn.Linear`, `nn.LayerNorm`, `nn.Dropout`, `nn.MultiheadAttention` (self-attention antar fitur teks), `nn.CrossEntropyLoss` (loss function klasifikasi).
- **`torch.optim` (`AdamW`)** — optimizer untuk mengupdate bobot model (termasuk bobot IndoBERT) selama training, dengan weight decay untuk regularisasi.
- **`torch.utils.data` (`Dataset`, `DataLoader`)** — `Dataset` untuk membungkus data custom (tokenisasi 5 kolom teks + fitur numerik per baris), `DataLoader` untuk membentuk batch dan shuffle data saat training/evaluasi.
- **`torch.amp` (`autocast`, `GradScaler`)** — mixed precision training; mempercepat proses training dan menghemat penggunaan VRAM GPU tanpa banyak mengorbankan akurasi numerik.
### NLP / Transformers
- **`transformers` (Hugging Face)**
  - **`AutoTokenizer`** — mengubah teks mentah menjadi token ID yang dipahami model (tokenisasi, padding, truncation).
  - **`AutoModel`** — memuat backbone model pretrained (IndoBERT/mBERT) yang nantinya di-fine-tune bersama classifier custom.
### Evaluation & Statistics
- **`scikit-learn`**
  - **`sklearn.metrics.confusion_matrix`** — menghitung TP/TN/FP/FN sebagai dasar perhitungan accuracy, precision, recall, F1, specificity.
  - **`sklearn.model_selection.StratifiedKFold`** — membagi data menjadi K-Fold dengan menjaga proporsi kelas (fraud vs non-fraud) tetap seimbang di tiap fold.
- **`scipy.stats` (`ttest_rel`, `wilcoxon`)** — uji signifikansi statistik untuk membandingkan performa dua skenario model (text-only vs fusion): paired t-test (parametrik) dan Wilcoxon signed-rank test (non-parametrik).
### Visualization
- **`matplotlib`** — membuat dan menyimpan plot (figure, save ke PNG).
- **`seaborn`** — membuat heatmap confusion matrix agar lebih mudah dibaca secara visual.
### Utilities
- **`tqdm`** — progress bar visual saat iterasi training/evaluasi per batch, supaya bisa memantau progres real-time.
- **`gc`** — garbage collection manual, membersihkan memori Python setelah tiap fold selesai.
- **`os`** — operasi sistem file (path handling).
## 🤖 Pretrained Models
- **`indobenchmark/indobert-base-p2`** — backbone IndoBERT utama yang di-fine-tune untuk tugas klasifikasi.
- **`indolem/indobert-base-uncased`** — backbone IndoBERT varian lain, dipakai sebagai perbandingan/benchmark.
- **`bert-base-multilingual-cased` (mBERT)** — model multilingual, dipakai sebagai eksperimen/komparasi tambahan.
## 📁 Datasets
- **`train_20.csv`**, **`train_40.csv`** — data training (dua ukuran/versi berbeda).
- **`test_20f.csv`** — data testing/holdout.
- Source: `/kaggle/input/datasets/bintangryan/orf-cleaned-dataset/`
 

# Data Preparation Environments & Libraries
 
## 🖥️ Environment
 
| Item | Detail | Keterangan |
|---|---|---|
| Platform | Kaggle Notebook | Tempat notebook dijalankan, dataset diakses via Kaggle Datasets |
| Language | Python | — |
| Sumber Data Awal | `/kaggle/input/datasets/bintangryan/original-emscad/fake_job_postings.csv` | Dataset mentah asli (EMSCAD — Employment Scam Aegean Dataset) |
| Sumber Data Antara | `/kaggle/input/datasets/bintangryan/translated-hierarchy/dataset_translated_hierarchy.csv` | Dataset hasil translasi (dimuat ulang setelah proses translasi selesai) |
| Output Akhir | `train_20.csv`, `train_40.csv`, `test_20f.csv` | Hasil akhir data preparation, siap dipakai di tahap model training |
 
## 📦 Libraries
 
### Data Handling
- **`pandas`** — membaca CSV, manipulasi DataFrame, cek missing value, filtering, grouping.
- **`numpy`** — operasi numerik/array, perhitungan statistik distribusi fitur.
### Text Cleaning & Preprocessing
- **`re`** (regex) — membersihkan noise di teks (simbol berulang, karakter tidak perlu, dsb) via fungsi `clean_noise()`.
- **`bs4` (`BeautifulSoup`)** — membersihkan tag HTML yang menempel di teks mentah (banyak field job posting mengandung HTML dari web scraping).
- **`warnings`** — mengatur/menyaring warning yang muncul saat proses (misalnya dari BeautifulSoup atau pandas).
### Translasi
- **`deep_translator` (`GoogleTranslator`)** — menerjemahkan kolom teks (title, company_profile, description, requirements, benefits) dari Bahasa Inggris ke Bahasa Indonesia, menghasilkan kolom `*_id`.
- **`tqdm`** — progress bar saat proses translasi berlangsung (karena translasi per baris bisa memakan waktu lama).
- `!pip install deep_translator` — instalasi library translasi karena tidak tersedia default di environment Kaggle.
### Encoding & Feature Scaling
- **`sklearn.preprocessing.LabelEncoder`** — encode kolom kategorikal (misalnya `employment_type`, `required_experience`) menjadi representasi numerik.
- **`sklearn.preprocessing.MinMaxScaler`** — menormalisasi fitur numerik/kontekstual (misalnya panjang karakter/token) ke rentang tertentu (0–1) sebelum dipakai model.
### Feature Extraction & Analysis
- **`sklearn.feature_extraction.text.CountVectorizer`** — mengekstrak kata-kata yang paling sering muncul (top words) per kelas (fraud vs non-fraud) untuk analisis eksploratif.
- **`scipy.stats.chi2_contingency`** — uji Chi-Square untuk menguji hubungan/asosiasi antara fitur kategorikal dengan target `fraudulent`.
- **`sklearn.ensemble` (`RandomForestClassifier`, `GradientBoostingClassifier`)** — model pohon untuk mengukur feature importance non-teks (mana fitur yang paling berpengaruh terhadap label fraud).
- **`sklearn.inspection.permutation_importance`** — mengukur pentingnya suatu fitur dengan mengacak nilainya dan melihat penurunan performa model, sebagai validasi tambahan selain feature importance bawaan Random Forest.
### Data Splitting
- **`sklearn.model_selection.train_test_split`** — membagi data menjadi set training dan testing, dengan skenario rasio fraud:non-fraud tertentu (mis. 1:4 untuk skenario S1).
### Visualization
- **`matplotlib.pyplot`** — plotting umum (distribusi fitur, bar chart).
- **`matplotlib.gridspec`** — mengatur layout grid untuk menampilkan banyak subplot sekaligus secara rapi.
- **`seaborn`** — visualisasi statistik (distribusi kategorikal, heatmap, dsb) dengan tampilan lebih informatif.
### Utilities
- **`IPython.display.display`** — menampilkan output (DataFrame, tabel) dengan format lebih rapi di dalam notebook, terutama untuk komparasi before/after cleaning teks.
## 🔄 Alur Proses Data Preparation (ringkas)
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
