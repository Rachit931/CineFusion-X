
# 🎬 CineFusion-X

A multimodal deep learning framework for movie understanding that combines **movie posters**, **plot summaries**, and **tabular metadata** to solve multiple downstream prediction tasks.

---

# ✨ Features

- 🎞️ Vision Transformer (ViT) for movie posters
- 📝 DistilBERT for plot summaries
- 📊 MLP for tabular metadata
- 🔀 Multimodal feature fusion
- ⭐ Rating prediction
- 💰 Box office tier prediction
- 🎭 Genre prediction
- 🎯 Latent movie clustering
- ⚙️ Reproducible preprocessing pipeline
- 📁 Configurable dataset paths using `.env`

---

# 📂 Project Structure

```text
CineFusion-X/
│
├── config/
│   ├── __init__.py
│   └── paths.py
│
├── notebooks/
├── src/
│   ├── dataset/
│   ├── models/
│   ├── training/
│   └── utils/
│
├── data/
│   ├── raw/
│   │   ├── imdb/
│   │   ├── tmdb/
│   │   └── posters/
│   └── processed/
│       ├── image/
│       ├── text/
│       ├── tabular/
│       ├── checkpoints/
│       └── multimodal_dataset.csv
│
├── requirements.txt
├── pyproject.toml
├── .env.example
├── README.md
└── .gitignore
```

> Dataset files are ignored by Git and stored locally or on mounted cloud storage.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/<username>/CineFusion-X.git
cd CineFusion-X
```

## Install uv

Follow the official installation guide:

https://docs.astral.sh/uv/getting-started/installation/

## Create a virtual environment

```bash
uv venv
```

Activate it

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies

```bash
uv sync
```

---

# ⚙️ Environment Configuration

Copy the example environment file

```bash
cp .env.example .env
```

Windows PowerShell

```powershell
Copy-Item .env.example .env
```

Example `.env`

```env
CINEFUSION_DATA_ROOT=/path/to/CineFusion-X
TMDB_API_KEY=your_tmdb_api_key
```

Example `.env.example`

```env
CINEFUSION_DATA_ROOT=/path/to/CineFusion-X
TMDB_API_KEY=your_tmdb_api_key_here
```

---

# 📦 Dataset Setup

## Option 1 — Download Prepared Dataset

Download the prepared dataset, extract it anywhere, and update:

```env
CINEFUSION_DATA_ROOT=/path/to/CineFusion-X
```

No code changes are required.

---

## Option 2 — Build Dataset Yourself

```text
IMDb TSV Files
      │
      ▼
01_imdb_preprocessing.py
      │
      ▼
imdb_movies_clean.csv
      │
      ▼
02_tmdb_collection.py
      │
      ├────────────► Posters
      ▼
TMDB Metadata
      │
      ▼
03_merge_datasets.py
      │
      ▼
processed/multimodal_dataset.csv
```

---

# 🤝 Collaboration

Collaborators can share one dataset through Google Drive (mounted with `rclone`).

Each contributor only changes their own `.env`:

```env
CINEFUSION_DATA_ROOT=/home/user/GoogleDrive/CineFusion-X
```

The source code never needs to be modified.

---

# 🧠 Model Architecture

```text
Movie Poster ─────► ViT ───────┐
                               │
Plot Summary ───► DistilBERT ──┤
                               ├──► Multimodal Fusion
Tabular Data ───► MLP ─────────┘
                         │
                         ▼
               Shared Representation
                         │
        ┌────────┬──────────┬─────────┐
        ▼        ▼          ▼         ▼
     Rating   Box Office  Genre   Clustering
```

---

# 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- timm
- Pandas
- NumPy
- Scikit-learn
- OpenCV
- IMDb Dataset
- TMDB API
- uv

---

# 🚀 Roadmap

- Cross-attention fusion
- SHAP explainability
- Grad-CAM visualizations
- MLflow experiment tracking
- DVC dataset versioning
- FastAPI inference
- Docker deployment

---

# 📄 License

Released under the MIT License.

---

# 🙏 Acknowledgements

- IMDb
- TMDB
- Hugging Face
- PyTorch
