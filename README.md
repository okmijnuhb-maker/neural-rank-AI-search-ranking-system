# NeuralRank — AI-Powered Search Ranking System

A senior-level Learning-to-Rank (LTR) system that uses machine learning to rank search results 17.5% better than traditional keyword search.

## Live Demo

Frontend: https://okmijnuhb-maker.github.io/neural-rank-AI-search-ranking-system/frontend/index.html

Backend API: https://neural-rank-ai-search-ranking-system.onrender.com

Demo Login: username: admin | password: admin123

## What is this project?

NeuralRank replicates how Google, Microsoft and Amazon rank search results — using BM25 as baseline and LambdaMART (XGBoost) as the ML ranking model, trained on the real MS MARCO dataset.

## Key Results

| Metric | BM25 | LambdaMART | Improvement |
|--------|------|------------|-------------|
| NDCG@10 | 0.5069 | 0.5956 | +17.5% |
| MRR | 0.3573 | 0.4703 | +31.6% |
| CTR Lift | - | 20.1% | p=0.041 |

## Tech Stack

**Machine Learning**
- XGBoost LambdaMART (rank:ndcg objective)
- BM25Okapi baseline
- SHAP TreeExplainer
- scikit-learn TF-IDF + LSA

**Backend**
- FastAPI
- SQLAlchemy + SQLite
- Uvicorn

**Frontend**
- HTML5 + CSS3 + JavaScript
- Tailwind CSS
- Plotly.js

**Dataset**
- MS MARCO v1.1 (via Hugging Face)
- 4,851 clean queries
- 39,962 passages

## Features

- Live search demo — BM25 vs LambdaMART side by side
- Model comparison with interactive Plotly charts
- SHAP explainability — why each document was ranked
- A/B testing with statistical significance
- Admin dashboard with real-time query logs
- User authentication (signup/login)
- Dark/light mode

## How to Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Frontend
Open `frontend/index.html` in browser

## Results

- Trained on MS MARCO — real Microsoft Bing search dataset
- 23 engineered features across 4 categories
- NDCG@10 improved from 0.5069 to 0.5956
- Statistically proven with A/B testing (Mann-Whitney U, p=0.041)
- Full explainability with SHAP values

## Interview Summary

> "I built a search ranking system similar to how Google ranks results — using BM25 as baseline and LambdaMART as the ML model. Trained on MS MARCO dataset with 23 engineered features. Achieved 17.5% NDCG improvement, proved with A/B testing (p=0.041), explained with SHAP, and deployed as a full-stack web app with FastAPI backend and SQLite database."
