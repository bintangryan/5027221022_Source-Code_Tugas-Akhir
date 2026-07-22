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

| Item | Detail |
| :--- | :--- |
| **Environment** | Jupyter Notebook / Google Colab (`.ipynb`) |
| **File Utama** | `notebook/data_preparation.ipynb` |

### Data Processing & NLP

| Library / Tool | Modul yang Digunakan | Keterangan |
| :--- | :--- | :--- |
| **Pandas & NumPy** | Eksplorasi Data | Pembersihan data dan eksplorasi dataset awal (`train_20.csv`, `train_40.csv`, `test_20f.csv`) |
| **NLTK / Sastrawi** | Stemming, Stopword Removal | *Preprocessing* teks khusus untuk pembersihan bahasa Indonesia |

### Training, Evaluasi & Visualisasi

| Library / Tool | Modul yang Digunakan | Keterangan |
| :--- | :--- | :--- |
| **Transformers & PyTorch** | Fine-tuning model | Proses pelatihan ulang (*training*) model IndoBERT |
| **Scikit-learn** | `sklearn` | Pembagian data latih/uji (*data splitting*) serta evaluasi metrik model (F1-Score, Recall, Precision, Accuracy) |
| **Matplotlib & Seaborn** | Visualisasi grafis | Pembuatan grafik distribusi kelas dan *confusion matrix* |

---

## Global Tools & Version Control

| Tool / Config | Keterangan |
| :--- | :--- |
| **Git** | Sistem *version control* utama (`.gitignore`, `.gitattributes`) |
| **Docker** | Isolasi lingkungan backend (`Dockerfile`) |
| **Docker Ignore** | Mengeliminasi file tidak penting saat proses *build* (`.dockerignore`) |

---

## 🔄 Alur Keseluruhan Sistem

```text
Jupyter / Colab (Training)        →   Backend API & XAI Engine         →   Web App (Frontend)
━━━━━━━━━━━━━━━━━━━━━━━━━━            ━━━━━━━━━━━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━━━━
Python + PyTorch + Sastrawi           FastAPI + Transformers + SHAP    Next.js + Tailwind CSS
Fine-tuning IndoBERT                  Inferensi & Feature Fusion       Antarmuka Pengguna
Ekstraksi Data & Preprocessing        Penjelasan Prediksi (XAI)        Visualisasi & Interaksi
