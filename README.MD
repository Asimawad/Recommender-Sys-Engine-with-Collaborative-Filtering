# Movie Recommendation System using Collaborative Filtering

[![Live Demo](https://img.shields.io/badge/🎬_Live_Demo-Try_it!-blue?style=for-the-badge)](https://asimawad.github.io/projects/aims/recommender-demo.html)
[![API](https://img.shields.io/badge/API-Railway-purple?style=for-the-badge)](https://web-production-bf50f.up.railway.app/)
[![Report](https://img.shields.io/badge/📄_Full_Report-PDF-red?style=for-the-badge)](https://asimawad.github.io/assets/recommender-system/AMLAS_Recommender_System_Report.pdf)

## 🎯 Try It Live!

**[→ Interactive Demo](https://asimawad.github.io/projects/aims/recommender-demo.html)** — Search any movie and get real-time recommendations!

**[→ API Endpoint](https://web-production-bf50f.up.railway.app/)** — FastAPI backend for programmatic access

### API Usage

```bash
# Search movies
curl "https://web-production-bf50f.up.railway.app/search?q=inception"

# Get recommendations by title
curl "https://web-production-bf50f.up.railway.app/recommend?movie=Matrix"

# Get recommendations by movie ID
curl "https://web-production-bf50f.up.railway.app/recommend/1"
```

---

## Introduction

This project implements a movie recommendation system using collaborative filtering techniques, specifically matrix factorization. The system is designed to predict user preferences for movies based on historical ratings and to generate personalized movie recommendations. By leveraging latent factor models, the system captures underlying patterns in user behavior and movie characteristics. The project uses the MovieLens dataset, which provides a rich source of user ratings and movie metadata.

---

## Table of Contents

- [Try It Live!](#-try-it-live)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Methodology](#methodology)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## Project Overview

The goal of this project is to build a scalable and efficient movie recommendation system using collaborative filtering. The key features include:

- **Data Preprocessing:** Efficient handling of large datasets (e.g., 32 million ratings).
- **Model Training:** Implementation of matrix factorization with bias terms and regularization.
- **Performance Optimization:** Use of Numba JIT compilation — **70x speedup** (15 min → 13 sec per epoch).
- **Evaluation:** Achieves **0.77 RMSE** on MovieLens 32M dataset.
- **Live Demo:** Interactive web demo with real-time recommendations.
- **REST API:** FastAPI backend deployed on Railway.

---

## Results

| Dataset | Model | RMSE | Training Time |
|---------|-------|------|---------------|
| ML-100K | Bias Only | 0.81 | ~2 sec |
| ML-32M | Bias + U + V | 0.77 | ~22 min (100 epochs) |
| ML-32M | Bias + U + V + F | 0.77 | ~25 min (100 epochs) |

### Sample Recommendations

**If you liked "Toy Story":**
- Shrek (2001)
- Toy Story 2 (1999)
- Finding Nemo (2003)
- Lion King (1994)
- Monsters, Inc. (2001)

**If you liked "The Matrix":**
- Fight Club (1999)
- Saving Private Ryan (1998)
- Star Wars: Episode V
- LOTR: Fellowship of the Ring

---

## Project Structure

```
├── Data_processing.py      # Data preprocessing & train/test split
├── Main.py                 # Training loop
├── Inference.py            # Generate recommendations
├── src/
│   ├── bias_only_model.py
│   ├── Bias_U_V_Model.py
│   └── Bias_U_V_F_Model.py
├── Experiments_ml-32m/     # Trained model weights
├── Data/                   # MovieLens datasets
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/Asimawad/Recommender-Sys-Engine-with-Collaborative-Filtering.git
cd Recommender-Sys-Engine-with-Collaborative-Filtering

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data Preparation

1. Download [MovieLens Dataset](https://grouplens.org/datasets/movielens/) (25M or 32M)
2. Extract to `Data/` directory
3. Run preprocessing:
   ```bash
   python Data_processing.py
   ```

---

## Methodology

The recommendation system uses **Alternating Least Squares (ALS)** for matrix factorization:

```
R ≈ U × V^T + b_u + b_i
```

Where:
- **U** = User latent factors (embeddings)
- **V** = Movie latent factors (embeddings)
- **b_u** = User bias
- **b_i** = Item bias

### Three Models Implemented

1. **Bias-Only ALS** — Captures global trends
2. **Bias + U + V** — Adds latent factors for personalization
3. **Bias + U + V + F** — Adds genre features for cold-start handling

---

## Model Training

```bash
python Main.py
```

Hyperparameters (32M dataset):
- Latent dimensions: 30
- λ (regularization): 1.0
- τ (factor regularization): 10
- Epochs: 100

---

## Usage

### Generate Recommendations

```bash
python Inference.py
```

Or use the live API:

```python
import requests

response = requests.get(
    "https://web-production-bf50f.up.railway.app/recommend",
    params={"movie": "Inception"}
)
recommendations = response.json()["recommendations"]
```

---

## References

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) (Koren et al., 2009)

---

## Author

**Asim Osman** — AIMS South Africa (AI for Science, DeepMind Partnership)

- [Portfolio](https://asimawad.github.io)
- [Live Demo](https://asimawad.github.io/projects/aims/recommender-demo.html)
