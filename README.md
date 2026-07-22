Environments & Libraries
Berikut adalah rincian environments, libraries utama, dan versi (rekomendasi/spesifik) yang digunakan dalam pengembangan sistem ini:

1. Frontend Environment
Dibangun menggunakan framework berbasis React dengan arsitektur App Router.

Runtime Environment: Node.js (v18.x atau v20.x)

Framework: Next.js (v14.x)

Language: TypeScript (v5.x)

Styling:

Tailwind CSS (v3.x)

PostCSS (v8.x)

Linting & Formatting: ESLint (v8.x)

2. Backend Environment
Berfokus pada penyediaan REST API untuk inferensi model dan komputasi nilai Explainable AI.

Runtime Environment: Python (v3.9 - v3.11)

Core Libraries:

torch (PyTorch v2.x) - Deep learning framework untuk memuat model.

transformers (Hugging Face v4.x) - Library utama untuk arsitektur IndoBERT.

shap / lime - Library untuk Explainable AI (XAI Engine).

fastapi & uvicorn / flask - Framework untuk membangun REST API.

pandas & numpy - Manipulasi data input/output.

Model: model_IndoBERT_Benchmark_S1_20_TEXT_ONLY.pth

Deployment: Docker Engine (v24.x+)

3. Notebook / Data Science Environment
Digunakan untuk pembersihan data, ekstraksi fitur (termasuk feature fusion metadata), dan pelatihan model.

Environment: Jupyter Notebook / Google Colab

Core Libraries:

pandas & numpy - Eksplorasi dan pembersihan data dataset (train_20.csv).

scikit-learn - Data splitting, evaluasi metrik (F1-Score, Recall, Precision).

transformers & torch - Finetuning model IndoBERT.

matplotlib & seaborn - Visualisasi data dan confusion matrix.

Panduan Instalasi & Menjalankan Aplikasi (Local Development)
Menjalankan Frontend
Buka terminal dan arahkan ke direktori frontend/.

Instal semua dependensi:

Bash
npm install
Jalankan development server:

Bash
npm run dev
Aplikasi dapat diakses melalui http://localhost:3000.

Menjalankan Backend (Tanpa Docker)
Buka terminal dan arahkan ke direktori backend/.

(Opsional namun disarankan) Buat virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # Untuk Linux/Mac
venv\\Scripts\\activate     # Untuk Windows
Instal dependensi dari requirements.txt:

Bash
pip install -r requirements.txt
Pastikan file model model_IndoBERT_Benchmark_S1_20_TEXT_ONLY.pth berada di dalam folder model/.

Jalankan server backend (misal menggunakan Uvicorn untuk FastAPI):

Bash
uvicorn main:app --reload
Menjalankan Backend (Menggunakan Docker)
Pastikan Docker sudah terinstal dan berjalan.

Di dalam direktori backend/, jalankan perintah:

Bash
docker build -t orf-backend .
docker run -p 8000:8000 orf-backend
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(markdown_content)

print("File generated successfully.")


```text?code_stdout&code_event_index=1
