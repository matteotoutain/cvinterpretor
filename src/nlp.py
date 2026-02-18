# nlp.py

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List

_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 1) EMBEDDINGS
# ============================================================

def embed_text(text: str):
    return _model.encode(text, normalize_embeddings=True)


def compute_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    e1 = embed_text(text1)
    e2 = embed_text(text2)
    return float(np.dot(e1, e2))


# ============================================================
# 2) KEYWORD BLOCKS
# ============================================================

def build_blocks_from_keywords(keywords_json: Dict) -> Dict[str, str]:
    """
    keywords_json attendu:
    {
        "tech_skills": [...],
        "domain_knowledge": [...],
        "experience_summary": "...",
        "certifications": [...],
        "seniority": "Junior/Senior/Manager"
    }
    """

    return {
        "tech_skills": " ".join(keywords_json.get("tech_skills", [])),
        "domain_knowledge": " ".join(keywords_json.get("domain_knowledge", [])),
        "experience": keywords_json.get("experience_summary", ""),
        "certifications": " ".join(keywords_json.get("certifications", [])),
    }


# ============================================================
# 3) SCORING AVEC PONDERATION USER
# ============================================================

def score_blocks(cv_blocks: Dict[str, str],
                 ao_blocks: Dict[str, str],
                 weights: Dict[str, float]):

    scores = {}
    total = 0.0

    for block in weights.keys():
        sim = compute_similarity(
            cv_blocks.get(block, ""),
            ao_blocks.get(block, "")
        )
        scores[block] = sim
        total += sim * weights[block]

    return total, scores


# ============================================================
# 4) CHUNKING POUR VECTOR SEARCH
# ============================================================

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def find_relevant_chunks(query: str, text: str, top_k: int = 3):
    chunks = chunk_text(text)
    if not chunks:
        return []

    query_emb = embed_text(query)
    chunk_embs = _model.encode(chunks, normalize_embeddings=True)

    sims = np.dot(chunk_embs, query_emb)

    ranked = sorted(
        zip(chunks, sims),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [{"chunk": c[0], "score": float(c[1])} for c in ranked]
