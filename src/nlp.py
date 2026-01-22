import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import SKILL_TOKENS

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_skills(text: str) -> List[str]:
    t = normalize_ws(text).lower()
    found = []
    for tok in SKILL_TOKENS:
        if tok.lower() in t:
            found.append(tok)
    # unique while preserving order
    seen = set()
    out = []
    for x in found:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def _embed_sentence_transformers(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)


def _embed_tfidf(texts: List[str]) -> np.ndarray:
    # TF-IDF embedding; cheap and no heavy downloads
    vec = TfidfVectorizer(stop_words="english", max_features=30000)
    X = vec.fit_transform(texts)
    # return dense array for cosine
    return X.toarray()


def compute_similarity(query_text: str, documents: List[str]) -> Tuple[np.ndarray, str]:
    """Return (scores in [0..1], method)."""
    texts = [query_text] + documents
    if _HAS_ST:
        try:
            emb = _embed_sentence_transformers(texts)
            q = emb[0:1]
            d = emb[1:]
            scores = cosine_similarity(q, d)[0]
            return scores.astype(float), "sentence-transformers (all-MiniLM-L6-v2)"
        except Exception:
            pass
    emb = _embed_tfidf(texts)
    q = emb[0:1]
    d = emb[1:]
    scores = cosine_similarity(q, d)[0]
    return scores.astype(float), "tf-idf"


def explain_match(ao_text: str, cv_text: str) -> Dict:
    ao_sk = set(extract_skills(ao_text))
    cv_sk = set(extract_skills(cv_text))
    overlap = sorted(list(ao_sk & cv_sk))
    missing = sorted(list(ao_sk - cv_sk))
    return {
        "ao_skills": sorted(list(ao_sk)),
        "cv_skills": sorted(list(cv_sk)),
        "overlap": overlap,
        "missing": missing,
    }
