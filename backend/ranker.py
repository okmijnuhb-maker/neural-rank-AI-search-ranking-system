# ranker.py
import numpy as np
import json
import pickle
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

BASE = "C:/my project/ranking and searching"

# Load all saved objects
with open(f"{BASE}/data/clean_msmarco.json") as f:
    data = json.load(f)

feature_names = json.load(open(f"{BASE}/data/feature_names.json"))
doc_freq = json.load(open(f"{BASE}/data/doc_freq.json"))
bm25_norm = json.load(open(f"{BASE}/data/bm25_norm.json"))

with open(f"{BASE}/data/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open(f"{BASE}/data/lsa_pipeline.pkl", "rb") as f:
    lsa = pickle.load(f)

with open(f"{BASE}/data/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

N = len(data)
corpus = [p for item in data for p in item['passages']]

print("All models loaded!")

def extract_all_features(query, passage, bm25_score):
    q = query.lower().split()
    p = passage.lower().split()
    q_set, p_set = set(q), set(p)

    # Lexical
    overlap = len(q_set & p_set) / len(q_set) if q_set else 0
    exact = sum(1 for t in q if t in p_set)
    q_bg = set(zip(q[:-1], q[1:]))
    p_bg = set(zip(p[:-1], p[1:]))
    bigram = len(q_bg & p_bg) / len(q_bg) if q_bg else 0
    jaccard = len(q_set & p_set) / len(q_set | p_set) if q_set | p_set else 0
    dice = 2*len(q_set & p_set) / (len(q_set)+len(p_set)) if q_set and p_set else 0

    # Statistical
    q_len = len(q)
    p_len = len(p)
    tf_var = np.var([p.count(t) for t in q]) if q else 0
    unique_matches = len(q_set & p_set)

    # Positional
    positions = [i for i, t in enumerate(p) if t in q_set]
    first_pos = positions[0]/p_len if positions and p_len else 1.0
    last_pos = positions[-1]/p_len if positions and p_len else 1.0
    avg_pos = np.mean(positions)/p_len if positions and p_len else 1.0
    coverage = len(positions)/p_len if p_len else 0

    # Semantic
    stop_ratio = len([t for t in p if t in STOPWORDS])/p_len if p_len else 0
    unique_q_in_p = len(q_set & p_set)/len(q_set) if q_set else 0
    sent_count = passage.count('.')+passage.count('?')+passage.count('!')
    q_chars = set([query[i:i+3] for i in range(len(query)-2)])
    p_chars = set([passage[i:i+3] for i in range(len(passage)-2)])
    char_sim = len(q_chars & p_chars)/len(q_chars | p_chars) if q_chars | p_chars else 0
    idf_sum = sum(np.log(N/(1+doc_freq.get(t, 0))) for t in q)

    # LSA + TF-IDF
    try:
        lsa_sim = float(cosine_similarity(lsa.transform([query]), lsa.transform([passage]))[0][0])
    except:
        lsa_sim = 0.0
    try:
        tfidf_sim = float(cosine_similarity(tfidf.transform([query]), tfidf.transform([passage]))[0][0])
    except:
        tfidf_sim = 0.0

    # Normalized BM25
    norm_bm25 = (bm25_score - bm25_norm['min']) / (bm25_norm['max'] - bm25_norm['min'] + 1e-9)

    # Doc freq
    doc_freq_score = np.mean([doc_freq.get(t, 0)/N for t in q]) if q else 0

    return [bm25_score, overlap, exact, bigram, jaccard, dice,
            q_len, p_len, tf_var, unique_matches,
            first_pos, last_pos, avg_pos, coverage,
            stop_ratio, unique_q_in_p, sent_count, char_sim, idf_sum,
            lsa_sim, tfidf_sim, norm_bm25, doc_freq_score]

def search(query, top_k=10):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:50]

    passages = [corpus[i] for i in top_idx]
    bm25_scores = scores[top_idx]

    # SHAP-weighted scoring for better live results
    lm_scores = []
    for p, b in zip(passages, bm25_scores):
        f = extract_all_features(query, p, b)
        score = (f[1]  * 0.35 +   # overlap
                 f[20] * 0.25 +   # tfidf_cosine
                 f[19] * 0.15 +   # lsa_sim
                 f[17] * 0.10 +   # char_sim
                 (1 - f[10]) * 0.15)  # first_pos inverted
        lm_scores.append(score)

    lm_scores = np.array(lm_scores)

    bm25_results = [{"rank": i+1, "passage": passages[i],
                     "score": round(float(bm25_scores[i]), 4)} for i in range(top_k)]

    lm_order = np.argsort(lm_scores)[::-1][:top_k]
    lm_results = [{"rank": i+1, "passage": passages[lm_order[i]],
                   "score": round(float(lm_scores[lm_order[i]]), 4)} for i in range(top_k)]

    return {"bm25": bm25_results, "lambdamart": lm_results}