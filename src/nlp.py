import re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# ============================================================
# Utils
# ============================================================
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


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


def _lower(s: str) -> str:
    return (s or "").lower()


def normalize_seniority_label(raw: str) -> str:
    """
    Map arbitrary seniority strings to one of: Junior | Senior | Manager | Unknown

    Simple + deterministic: seniority is used ONLY as a late filter (business logic).
    """
    s = _lower(raw)

    if not s.strip():
        return "Unknown"

    # Manager-ish keywords
    if any(
        k in s
        for k in [
            "manager",
            "lead",
            "head",
            "director",
            "principal",
            "partner",
            "responsable",
            "chef de",
            "team lead",
        ]
    ):
        return "Manager"

    # Explicit senior / junior
    if any(k in s for k in ["junior", "débutant", "debutant", "entry", "0-2", "0–2", "1-2", "1–2"]):
        return "Junior"
    if any(k in s for k in ["senior", "confirmé", "confirme", "confirmed", "experienced", "expert"]):
        return "Senior"

    # If there are years, use a rough bucketing
    years = None
    m = re.search(r"(\d{1,2})\s*\+?\s*(ans|years|year)", s)
    if m:
        try:
            years = int(m.group(1))
        except Exception:
            years = None

    if years is not None:
        if years <= 2:
            return "Junior"
        if years <= 7:
            return "Senior"
        return "Manager"

    return "Unknown"


# ============================================================
# Embeddings / Similarity
# ============================================================
# BONUS: better multilingual embedder (FR/EN) + caching (faster + stable)
_ST_MODEL: Optional["SentenceTransformer"] = None
_ST_MODEL_NAME = "intfloat/multilingual-e5-base"


def _get_st_model() -> "SentenceTransformer":
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(_ST_MODEL_NAME)
    return _ST_MODEL


def _embed_sentence_transformers(texts: List[str]) -> np.ndarray:
    model = _get_st_model()
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
            return scores.astype(float), f"sentence-transformers ({_ST_MODEL_NAME})"
        except Exception:
            pass

    emb = _embed_tfidf(texts)
    q = emb[0:1]
    d = emb[1:]
    scores = cosine_similarity(q, d)[0]
    return scores.astype(float), "tf-idf"


# ============================================================
# Block building (keyword-first)
# ============================================================
BLOCK_KEYS = ["tech_skills", "experience", "domain_knowledge", "certifications"]


def build_ao_blocks(ao_struct: Dict[str, Any], ao_fallback_text: str = "") -> Dict[str, str]:
    """
    Create textual blocks for AO (Appel d'Offre).
    Expected to receive a "keyword pack" style AO JSON (but works with partials).
    """
    title = normalize_ws(str(ao_struct.get("titre_poste") or ""))
    summary = normalize_ws(str(ao_struct.get("mission_summary") or ao_struct.get("contexte_mission") or ""))

    tech_req = ", ".join(_as_list(ao_struct.get("tech_skills_required") or ao_struct.get("competences_techniques")))
    domain_req = ", ".join(_as_list(ao_struct.get("domain_knowledge_required") or ao_struct.get("secteur")))
    cert_req = ", ".join(_as_list(ao_struct.get("certifications_required")))
    exp_req = normalize_ws(str(ao_struct.get("experience_requise") or ao_struct.get("experience_required") or ""))

    if not (title or summary or tech_req or domain_req or cert_req or exp_req):
        summary = ao_fallback_text[:4000]

    return {
        "tech_skills": normalize_ws(" ".join([title, tech_req])),
        "experience": normalize_ws(" ".join([summary, exp_req])),
        "domain_knowledge": normalize_ws(" ".join([domain_req, title])),
        "certifications": normalize_ws(cert_req),
        "full": normalize_ws(" ".join([title, summary, tech_req, domain_req, cert_req, exp_req])),
    }


def build_cv_blocks(cv_struct: Dict[str, Any], cv_fallback_text: str = "") -> Dict[str, str]:
    """
    Create textual blocks for CV.

    Works best if cv_struct already contains:
      - tech_skills (list)
      - domain_knowledge (list)
      - certifications (list)
      - experiences (list of dicts)
    """
    role = normalize_ws(str(cv_struct.get("role_principal") or ""))

    tech = ", ".join(_as_list(cv_struct.get("tech_skills") or cv_struct.get("technologies") or cv_struct.get("hard_skills")))
    domain = ", ".join(_as_list(cv_struct.get("domain_knowledge") or cv_struct.get("secteur_principal")))
    certs = ", ".join(_as_list(cv_struct.get("certifications")))

    # Experience block: structured experiences first, else fallback text
    experiences_txt = ""
    exps = cv_struct.get("experiences")
    if isinstance(exps, list):
        chunks = []
        for e in exps[:10]:
            if isinstance(e, dict):
                mission = normalize_ws(str(e.get("mission") or e.get("title") or ""))
                stack = ", ".join(_as_list(e.get("stack") or e.get("tech") or e.get("technologies")))
                dom = normalize_ws(str(e.get("secteur") or e.get("domain") or ""))
                dur = normalize_ws(str(e.get("duree") or e.get("duration") or ""))
                chunks.append(normalize_ws(" ".join([mission, stack, dom, dur])))
            else:
                chunks.append(normalize_ws(str(e)))
        experiences_txt = normalize_ws(" | ".join([c for c in chunks if c]))

    if not experiences_txt:
        experiences_txt = normalize_ws(cv_fallback_text[:2500])

    return {
        "tech_skills": normalize_ws(" ".join([role, tech])),
        "experience": normalize_ws(" ".join([role, experiences_txt])),
        "domain_knowledge": normalize_ws(" ".join([domain, role])),
        "certifications": normalize_ws(certs),
        "full": normalize_ws(" ".join([role, tech, domain, certs, experiences_txt])),
    }


# ============================================================
# Scoring
# ============================================================
def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(weights.get(k, 0.0)) for k in BLOCK_KEYS}
    s = sum(max(v, 0.0) for v in w.values())
    if s <= 0:
        # safe fallback (equal)
        return {k: 1.0 / len(BLOCK_KEYS) for k in BLOCK_KEYS}
    return {k: max(v, 0.0) / s for k, v in w.items()}


def _softmax_pool(scores: List[float], tau: float = 0.20) -> float:
    """
    Smooth max pooling:
      - tau small -> closer to max
      - tau large -> closer to mean
    """
    if not scores:
        return 0.0
    x = np.array(scores, dtype=float)
    x = np.clip(x, 0.0, 1.0)
    m = float(x.max())
    ex = np.exp((x - m) / max(tau, 1e-6))
    return float((ex * x).sum() / ex.sum())


def score_blocks(
    ao_blocks: Dict[str, str],
    cv_blocks: Dict[str, str],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Less-tatillon scoring (NO hardcode of industries/keywords):
    - For each AO block, compare against multiple CV evidences (not only the matching block).
    - Pool with smooth-max so a single strong evidence is enough.
    - Keep the same output schema as before.
    """
    if weights is None:
        weights = {k: 1.0 / len(BLOCK_KEYS) for k in BLOCK_KEYS}
    weights = _normalize_weights(weights)

    method_used = None
    per_block: Dict[str, float] = {}

    # Structural pooling only (not domain-specific rules)
    evidence_map = {
        "tech_skills": ["tech_skills", "experience", "full"],
        "experience": ["experience", "full"],
        "domain_knowledge": ["domain_knowledge", "experience", "full"],
        "certifications": ["certifications", "full"],
    }

    for k in BLOCK_KEYS:
        q = ao_blocks.get(k, "") or ""

        candidates: List[str] = []
        for cv_key in evidence_map.get(k, [k, "full"]):
            txt = cv_blocks.get(cv_key, "") or ""
            if txt.strip():
                candidates.append(txt)

        if not candidates:
            per_block[k] = 0.0
            continue

        scores, method = compute_similarity(q, candidates)
        method_used = method

        pooled = _softmax_pool(scores.tolist(), tau=0.20)
        per_block[k] = float(pooled)

    global_score = 0.0
    for k, w in weights.items():
        global_score += per_block.get(k, 0.0) * float(w)

    out = dict(per_block)
    out["global_score"] = float(global_score)
    out["weights_used"] = dict(weights)

    return out, (method_used or "unknown")


def verdict_from_score(s: float) -> str:
    if s >= 0.75:
        return "Strong Fit"
    if s >= 0.55:
        return "Moderate Fit"
    return "Low Fit"


# ============================================================
# Vector search for citations (chunked CV)
# ============================================================
def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple chunking by characters with overlap.
    Good enough to cite "passages" without relying on PDF coordinates.
    """
    t = normalize_ws(text)
    if not t:
        return []
    chunks: List[str] = []
    i = 0
    n = len(t)
    while i < n:
        j = min(n, i + max_chars)
        chunk = t[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks


def vector_search_passages(
    query: str,
    cv_text: str,
    top_k: int = 3,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Returns top passages with scores:
      [{"rank": 1, "score": 0.83, "text": "..."}]
    """
    chunks = chunk_text(cv_text)
    if not chunks:
        return [], "no_text"

    scores, method = compute_similarity(query, chunks)
    idxs = np.argsort(scores)[::-1][:top_k]

    out: List[Dict[str, Any]] = []
    for r, idx in enumerate(idxs, start=1):
        out.append(
            {
                "rank": r,
                "score": float(scores[idx]),
                "text": chunks[int(idx)],
            }
        )
    return out, method
