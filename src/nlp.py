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


# =========================
# Embedding model (E5)
# =========================
E5_MODEL_NAME = "BAAI/bge-m3"

#E5_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _embed_sentence_transformers(texts: List[str]) -> np.ndarray:
    """
    E5 best practice: prefix queries with 'query:' and documents with 'passage:'.
    Here we don't know which are which, so compute_similarity() will build that properly.
    """
    model = SentenceTransformer(E5_MODEL_NAME)
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
    query_text = normalize_ws(query_text)
    documents = [normalize_ws(d) for d in documents]

    if _HAS_ST:
        try:
            # E5: query/passsage prefixes
            q_in = [f"query: {query_text}"]
            d_in = [f"passage: {d}" for d in documents]

            emb = _embed_sentence_transformers(q_in + d_in)
            q = emb[0:1]
            d = emb[1:]
            scores = cosine_similarity(q, d)[0]
            return scores.astype(float), f"sentence-transformers ({E5_MODEL_NAME})"
        except Exception:
            pass

    # fallback tf-idf
    texts = [query_text] + documents
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
    domain = normalize_ws(str(ao_struct.get("secteur") or ao_struct.get("secteur_principal") or ""))
    exp = normalize_ws(str(ao_struct.get("experience_requise") or ""))
    langs = ", ".join(_as_list(ao_struct.get("langues_requises")))

    # NEW: certifications expected in AO (optional)
    certs_req = ", ".join(_as_list(ao_struct.get("certifications_requises")))

    if not (title or context or hard or soft or domain or exp or langs or certs_req):
        context = ao_fallback_text[:4000]

    return {
        "skills_like": normalize_ws(" ".join([title, hard])),
        "experience_like": normalize_ws(" ".join([context, exp])),
        "domain_like": normalize_ws(" ".join([domain, title])),
        # Replaces soft_like:
        "certification_like": normalize_ws(" ".join([certs_req, langs])),
        "full": normalize_ws(" ".join([title, context, hard, domain, exp, langs, certs_req])),
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

    hard_skills = ", ".join(_as_list(cv_struct.get("hard_skills")))
    soft_skills = ", ".join(_as_list(cv_struct.get("soft_skills")))

    # NEW: certifications field (list or comma-separated string)
    certifications = ", ".join(_as_list(cv_struct.get("certifications")))

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
        experiences_txt = normalize_ws(cv_fallback_text[:2500])

    skills_like = normalize_ws(" ".join([role, tech, hard_skills]))
    domain_like = normalize_ws(" ".join([sector, role]))
    experience_like = normalize_ws(" ".join([experiences_txt, seniority]))

    # Replaces soft_like:
    certification_like = normalize_ws(" ".join([certifications, langs]))

    full = normalize_ws(" ".join([skills_like, experience_like, domain_like, certification_like]))

    return {
        "skills_like": skills_like,
        "experience_like": experience_like,
        "domain_like": domain_like,
        "certification_like": certification_like,
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
            "certification_like": 0.10,
        }

    method_used = None
    per_block = {}

    for k in ["skills_like", "experience_like", "domain_like", "certification_like"]:
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
