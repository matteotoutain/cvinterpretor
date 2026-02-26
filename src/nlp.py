import re
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _embed_sentence_transformers(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)


def _embed_tfidf(texts: List[str]) -> np.ndarray:
    vec = TfidfVectorizer(stop_words="english", max_features=30000)
    X = vec.fit_transform(texts)
    return X.toarray()


def compute_similarity(query_text: str, documents: List[str]) -> Tuple[np.ndarray, str]:
    """
    Return (scores in [0..1], method).
    """
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


# -----------------------------
# Block building from Mistral JSON
# -----------------------------
def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [normalize_ws(str(i)) for i in x if normalize_ws(str(i))]
    if isinstance(x, str):
        # allow comma-separated
        parts = [normalize_ws(p) for p in x.split(",")]
        return [p for p in parts if p]
    return [normalize_ws(str(x))]


def build_ao_blocks(ao_struct: Dict[str, Any], ao_fallback_text: str = "") -> Dict[str, str]:
    """
    Create textual blocks for AO.
    Keys must match build_cv_blocks.
    """
    title = normalize_ws(str(ao_struct.get("titre_poste") or ""))
    context = normalize_ws(str(ao_struct.get("contexte_mission") or ""))

    hard = ", ".join(_as_list(ao_struct.get("competences_techniques")))
    soft = ", ".join(_as_list(ao_struct.get("competences_metier")))
    domain = normalize_ws(str(ao_struct.get("secteur") or ao_struct.get("secteur_principal") or ""))  # optional
    exp = normalize_ws(str(ao_struct.get("experience_requise") or ""))
    langs = ", ".join(_as_list(ao_struct.get("langues_requises")))

    # minimal fallback if some fields missing
    if not (title or context or hard or soft or domain or exp or langs):
        context = ao_fallback_text[:4000]

    return {
        "skills_like": normalize_ws(" ".join([title, hard])),
        "experience_like": normalize_ws(" ".join([context, exp])),
        "domain_like": normalize_ws(" ".join([domain, title])),
        "soft_like": normalize_ws(" ".join([soft, langs])),
        "full": normalize_ws(" ".join([title, context, hard, soft, domain, exp, langs])),
    }


def build_cv_blocks(cv_struct: Dict[str, Any], cv_fallback_text: str = "") -> Dict[str, str]:
    """
    Create textual blocks for CV.
    Expected fields are produced by your Mistral extraction.
    """
    role = normalize_ws(str(cv_struct.get("role_principal") or ""))
    seniority = normalize_ws(str(cv_struct.get("seniorite") or ""))
    sector = normalize_ws(str(cv_struct.get("secteur_principal") or ""))
    tech = ", ".join(_as_list(cv_struct.get("technologies")))
    langs = ", ".join(_as_list(cv_struct.get("langues")))

    # Optional richer fields if you extract them (recommended)
    hard_skills = ", ".join(_as_list(cv_struct.get("hard_skills")))
    soft_skills = ", ".join(_as_list(cv_struct.get("soft_skills")))

    experiences_txt = ""
    exps = cv_struct.get("experiences")
    if isinstance(exps, list):
        chunks = []
        for e in exps[:10]:
            if isinstance(e, dict):
                mission = normalize_ws(str(e.get("mission") or e.get("title") or ""))
                stack = ", ".join(_as_list(e.get("stack")))
                dom = normalize_ws(str(e.get("secteur") or e.get("domain") or ""))
                dur = normalize_ws(str(e.get("duree") or e.get("duration") or ""))
                chunks.append(normalize_ws(" ".join([mission, stack, dom, dur])))
            else:
                chunks.append(normalize_ws(str(e)))
        experiences_txt = normalize_ws(" | ".join([c for c in chunks if c]))

    if not experiences_txt:
        # fallback: use main cv_text slice if no structured experience
        experiences_txt = normalize_ws(cv_fallback_text[:2500])

    # if older extraction used only technologies/langues etc., keep it usable
    skills_like = normalize_ws(" ".join([role, tech, hard_skills]))
    soft_like = normalize_ws(" ".join([soft_skills, langs]))
    domain_like = normalize_ws(" ".join([sector, role]))
    experience_like = normalize_ws(" ".join([experiences_txt, seniority]))

    full = normalize_ws(" ".join([skills_like, experience_like, domain_like, soft_like]))

    return {
        "skills_like": skills_like,
        "experience_like": experience_like,
        "domain_like": domain_like,
        "soft_like": soft_like,
        "full": full,
    }


def score_blocks(
    ao_blocks: Dict[str, str],
    cv_blocks: Dict[str, str],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Compute similarity per block and weighted global score.
    """
    if weights is None:
        weights = {
            "skills_like": 0.40,
            "experience_like": 0.30,
            "domain_like": 0.20,
            "soft_like": 0.10,
        }

    method_used = None
    per_block = {}

    for k in ["skills_like", "experience_like", "domain_like", "soft_like"]:
        q = ao_blocks.get(k, "") or ""
        d = cv_blocks.get(k, "") or ""
        scores, method = compute_similarity(q, [d])
        method_used = method
        per_block[k] = float(scores[0])

    global_score = 0.0
    for k, w in weights.items():
        global_score += per_block.get(k, 0.0) * float(w)

    out = dict(per_block)
    out["global_score"] = float(global_score)

    return out, (method_used or "unknown")


def verdict_from_score(s: float) -> str:
    if s >= 0.75:
        return "Strong Fit"
    if s >= 0.55:
        return "Moderate Fit"
    return "Low Fit"
