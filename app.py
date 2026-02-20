import streamlit as st
import pandas as pd
import json
import math
import plotly.graph_objects as go
from typing import Any, Dict, List, Tuple

from src.config import APP_NAME
from src.db import connect, init_db, upsert_cv, list_cvs, get_cv_texts, delete_cv
from src.extract import extract_text_generic, stable_id_from_bytes
from src.nlp import (
    build_ao_blocks,
    build_cv_blocks,
    score_blocks,
    verdict_from_score,
    normalize_seniority_label,
    vector_search_passages,
    compute_similarity,   # NEW (semantic overlap)
    normalize_ws,         # NEW (small cleanup)
)

from src.mistral_client import (
    call_mistral_cv_keyword_pack,
    call_mistral_ao_keyword_pack,
    call_mistral_json_explanation,
    call_mistral_json_gap_to_ideal,
    DEFAULT_MODEL,
)

# ============================================================
# UI Style (simple + clean)
# ============================================================
st.set_page_config(page_title=APP_NAME, layout="wide")
st.markdown(
    """
    <style>
      .cv-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 14px;
        background: rgba(255,255,255,0.02);
      }
      .muted { opacity: 0.72; }
      .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        border: 1px solid rgba(49, 51, 63, 0.25);
        margin-right: 6px;
        font-size: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_NAME)
st.caption("Focus: **Batch CV Import** (stockage local) + **AO Import & NLP Analysis** (ponctuel, sans stockage AO).")

# ============================================================
# DB init
# ============================================================
conn = connect()
init_db(conn)

tabs = st.tabs(["1) Import CVs (Batch)", "2) Import AO & Analyse (Ponctuel)", "DB: gérer les CVs"])


# ============================================================
# Helpers
# ============================================================
def _csv_join(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return ", ".join([str(i).strip() for i in x if str(i).strip()])
    return str(x).strip()


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""


def _safe_json_loads(s: Any) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def _contains_all_terms(haystack: str, terms: List[str]) -> bool:
    h = (haystack or "").lower()
    for t in terms:
        if (t or "").strip() and (t or "").lower() not in h:
            return False
    return True


def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def radar_plot(scores: Dict[str, Any], title: str = ""):
    labels = ["Tech", "Experience", "Domain", "Certifs"]
    values = [
        clamp01(scores.get("tech_skills", 0.0)),
        clamp01(scores.get("experience", 0.0)),
        clamp01(scores.get("domain_knowledge", 0.0)),
        clamp01(scores.get("certifications", 0.0)),
    ]
    values = values + values[:1]
    theta = labels + labels[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=theta,
            fill="toself",
            name="match",
        )
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=16, r=16, t=38, b=16),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0])),
        title=dict(text=title, x=0.0, xanchor="left"),
        height=260,
    )
    return fig


def set_from_pack(pack: Dict[str, Any], key: str) -> set:
    x = pack.get(key)
    if x is None:
        return set()
    if isinstance(x, list):
        items = x
    else:
        items = [p.strip() for p in str(x).split(",")]
    out = set()
    for it in items:
        s = str(it).strip().lower()
        if s:
            out.add(s)
    return out


# ----------------------------
# NEW: semantic overlap (less tatillon, no hardcoded dictionaries)
# ----------------------------
def _list_from_pack(pack: Dict[str, Any], key: str) -> List[str]:
    x = pack.get(key)
    if x is None:
        return []
    if isinstance(x, list):
        items = x
    else:
        items = [p.strip() for p in str(x).split(",")]
    out: List[str] = []
    for it in items:
        s = normalize_ws(str(it)).strip().lower()
        if s:
            out.append(s)
    return out


def _semantic_overlap_and_gaps(
    ao_terms: List[str],
    cv_terms: List[str],
    threshold: float,
) -> Tuple[List[str], List[str]]:
    ao_terms = [t for t in ao_terms if (t or "").strip()]
    cv_terms = [t for t in cv_terms if (t or "").strip()]

    if not ao_terms:
        return [], []
    if not cv_terms:
        return [], ao_terms

    overlap: List[str] = []
    missing: List[str] = []

    for a in ao_terms:
        scores, _ = compute_similarity(a, cv_terms)
        best = float(scores.max()) if len(scores) else 0.0
        if best >= threshold:
            overlap.append(a)
        else:
            missing.append(a)

    return overlap, missing


def overlap_and_gaps(ao_pack: Dict[str, Any], cv_pack: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Previous version was exact string set overlap (too tatillon).
    Now: semantic overlap based on embeddings similarity (still deterministic, no hardcoded mapping).
    Output schema unchanged.
    """
    ao_tech = _list_from_pack(ao_pack, "tech_skills_required")
    ao_dom = _list_from_pack(ao_pack, "domain_knowledge_required")
    ao_cert = _list_from_pack(ao_pack, "certifications_required")

    cv_tech = _list_from_pack(cv_pack, "tech_skills") + _list_from_pack(cv_pack, "technologies")
    cv_dom = _list_from_pack(cv_pack, "domain_knowledge") + _list_from_pack(cv_pack, "secteur_principal")
    cv_cert = _list_from_pack(cv_pack, "certifications")

    # Thresholds are category-level, not hardcoded per domain
    over_tech, miss_tech = _semantic_overlap_and_gaps(ao_tech, cv_tech, threshold=0.58)
    over_dom, miss_dom = _semantic_overlap_and_gaps(ao_dom, cv_dom, threshold=0.50)
    over_cert, miss_cert = _semantic_overlap_and_gaps(ao_cert, cv_cert, threshold=0.60)

    return {
        "overlap_tech": sorted(over_tech),
        "missing_tech": sorted(miss_tech),
        "overlap_domain": sorted(over_dom),
        "missing_domain": sorted(miss_dom),
        "overlap_cert": sorted(over_cert),
        "missing_cert": sorted(miss_cert),
    }


def pretty_list(xs: List[str], max_items: int = 10) -> str:
    if not xs:
        return "—"
    cut = xs[:max_items]
    suffix = " …" if len(xs) > max_items else ""
    return ", ".join(cut) + suffix


# ============================================================
# TAB 1 — Import CVs
# ============================================================
with tabs[0]:
    st.subheader("Étape 1 — Uploader des CVs (batch) → stockage en base locale (SQLite)")
    st.write("Formats supportés : **.pptx**, **.pdf**, **.docx**, **.txt**. (Pas d’OCR ici.)")

    st.markdown("### Options d'extraction (Mistral)")
    use_mistral = st.checkbox(
        "Utiliser Mistral pour extraire un 'keyword pack' (tech, domaine, certifs, expériences…)",
        value=True,
    )
    mistral_model = st.text_input("Modèle Mistral (CV)", value=DEFAULT_MODEL, disabled=not use_mistral)

    files = st.file_uploader("Dépose tes CVs ici", type=["pptx", "pdf", "docx", "txt"], accept_multiple_files=True)
    do_extract = st.checkbox("Extraire + stocker maintenant", value=True)

    if st.button("Lancer l'import batch"):
        if not files:
            st.warning("Aucun fichier.")
        else:
            with st.spinner("Traitement…"):
                n_ok = 0
                for f in files:
                    raw = f.read()
                    file_id = stable_id_from_bytes(raw)
                    text = extract_text_generic(f.name, raw)

                    cv_keywords = {}
                    if use_mistral:
                        cv_keywords = call_mistral_cv_keyword_pack(text, mistral_model=mistral_model) or {}
                    senior_raw = str(cv_keywords.get("seniorite_raw") or cv_keywords.get("seniorite") or "")
                    senior_label = cv_keywords.get("seniorite_label") or normalize_seniority_label(senior_raw)

                    upsert_cv(
                        conn,
                        cv_id=file_id,
                        filename=f.name,
                        cv_text=text,
                        cv_struct_json=_safe_json_dumps(cv_keywords),
                        cv_keywords_json=_safe_json_dumps(cv_keywords),
                        nom=str(cv_keywords.get("nom") or ""),
                        role_principal=str(cv_keywords.get("role_principal") or ""),
                        seniorite=str(senior_label or "Unknown"),
                    )
                    n_ok += 1

            st.success(f"Import OK: {n_ok} fichier(s).")


# ============================================================
# TAB 2 — Import AO & Analyse
# ============================================================
with tabs[1]:
    st.subheader("Étape 2 — Uploader un AO et scorer les CVs (ponctuel)")

    st.write("Tu importes un **AO** (pdf/docx/txt/pptx) → extraction → (optionnel) Mistral pack AO → matching vs DB.")

    st.markdown("### 2.1 Import AO")
    use_mistral_ao = st.checkbox("Utiliser Mistral pour extraire un 'keyword pack' AO", value=True)
    mistral_model_ao = st.text_input("Modèle Mistral (AO)", value=DEFAULT_MODEL, disabled=not use_mistral_ao)

    ao_file = st.file_uploader("AO (pdf/docx/txt/pptx)", type=["pptx", "pdf", "docx", "txt"], accept_multiple_files=False)

    st.markdown("### 2.2 Filtres & pondérations")
    st.caption("Filtres : keywords (optionnel) + seniority label. Pondérations : Tech / Experience / Domain / Certifs.")
    selected_terms = st.text_input("Filtre mots-clés (optionnel, séparés par virgule)", value="")
    terms = [t.strip() for t in selected_terms.split(",") if t.strip()]

    colA, colB, colC, colD = st.columns(4)
    w_tech = colA.slider("Poids Tech", 0.0, 1.0, 0.35, 0.05)
    w_exp = colB.slider("Poids Experience", 0.0, 1.0, 0.35, 0.05)
    w_dom = colC.slider("Poids Domain", 0.0, 1.0, 0.20, 0.05)
    w_cert = colD.slider("Poids Certifs", 0.0, 1.0, 0.10, 0.05)

    weights = {"tech_skills": w_tech, "experience": w_exp, "domain_knowledge": w_dom, "certifications": w_cert}

    if st.button("Lancer l'analyse"):
        if not ao_file:
            st.warning("Aucun AO.")
        else:
            raw = ao_file.read()
            ao_text = extract_text_generic(ao_file.name, raw)

            ao_pack = {}
            if use_mistral_ao:
                with st.spinner("Mistral: extraction AO pack…"):
                    ao_pack = call_mistral_ao_keyword_pack(ao_text, mistral_model=mistral_model_ao) or {}

            ao_blocks = build_ao_blocks(ao_pack if ao_pack else {}, ao_fallback_text=ao_text)

            # Load CVs from DB
            df = list_cvs(conn)
            if df.empty:
                st.warning("DB vide. Importe des CVs d'abord.")
            else:
                rows = []
                for _, r in df.iterrows():
                    cv_text = r.get("cv_text") or ""
                    cv_pack = _safe_json_loads(r.get("cv_keywords_json")) or _safe_json_loads(r.get("cv_struct_json")) or {}

                    cv_blocks = build_cv_blocks(cv_pack, cv_fallback_text=cv_text)
                    scores, method = score_blocks(ao_blocks, cv_blocks, weights=weights)

                    senior_label = str(r.get("seniorite") or "Unknown")

                    if terms:
                        hay = " ".join(
                            [
                                str(r.get("tech_skills") or ""),
                                str(r.get("domain_knowledge") or ""),
                                str(r.get("certifications") or ""),
                                str(r.get("technologies") or ""),
                                str(r.get("secteur_principal") or ""),
                                str(r.get("role_principal") or ""),
                                str(cv_text[:1200] or ""),
                            ]
                        )
                        if not _contains_all_terms(hay, terms):
                            continue

                    rows.append(
                        {
                            "cv_id": r["cv_id"],
                            "filename": r["filename"],
                            "nom": r.get("nom") or "",
                            "role_principal": r.get("role_principal") or "",
                            "seniorite": senior_label,
                            "score_global": scores["global_score"],
                            "tech_skills": scores.get("tech_skills", 0.0),
                            "experience": scores.get("experience", 0.0),
                            "domain_knowledge": scores.get("domain_knowledge", 0.0),
                            "certifications": scores.get("certifications", 0.0),
                            "method": method,
                        }
                    )

                if not rows:
                    st.warning("Aucun candidat ne passe les filtres (ou DB vide).")
                else:
                    out = pd.DataFrame(rows).sort_values("score_global", ascending=False).reset_index(drop=True)

                    st.info(f"{len(out)} candidat(s) après filtres — affichage Top {min(10, len(out))}.")

                    # Keep the dataframe (useful for debugging / export)
                    with st.expander("Table brute (debug)", expanded=False):
                        st.dataframe(out, use_container_width=True)

                    st.markdown("### Résultats (cards + radar + match/missing)")
                    top_k = min(10, len(out))

                    for i in range(top_k):
                        row = out.iloc[i]
                        cv_id = row["cv_id"]

                        full = df[df["cv_id"] == cv_id].iloc[0]
                        cv_text = full.get("cv_text") or ""
                        cv_pack = _safe_json_loads(full.get("cv_keywords_json")) or _safe_json_loads(full.get("cv_struct_json")) or {}

                        cv_blocks = build_cv_blocks(cv_pack, cv_fallback_text=cv_text)
                        scores, method = score_blocks(ao_blocks, cv_blocks, weights=weights)
                        verdict = verdict_from_score(scores["global_score"])

                        og = overlap_and_gaps(ao_pack if ao_pack else {}, cv_pack if cv_pack else {})

                        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
                        left, mid, right = st.columns([1.55, 1.15, 1.30])

                        with left:
                            st.subheader(f"#{i+1} — {row['filename']}")
                            if row.get("nom"):
                                st.write(f"**{row['nom']}**")
                            if row.get("role_principal"):
                                st.write(f"🧩 {row['role_principal']}")

                            st.markdown(
                                f"""
                                <span class="pill">Seniority: {row.get('seniorite','Unknown')}</span>
                                <span class="pill">Verdict: {verdict}</span>
                                <span class="pill">Method: {method}</span>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.metric("Score global", f"{scores['global_score']:.3f}")

                        with mid:
                            fig = radar_plot(scores, title="Match par bloc")
                            st.plotly_chart(fig, use_container_width=True, key=f"radar_{cv_id}")

                        with right:
                            st.markdown("**Ce qui colle**")
                            st.write(f"- Tech: {pretty_list(og['overlap_tech'])}")
                            st.write(f"- Domain: {pretty_list(og['overlap_domain'])}")
                            st.write(f"- Certifs: {pretty_list(og['overlap_cert'])}")

                            st.markdown("**Ce qui manque**")
                            st.write(f"- Tech: {pretty_list(og['missing_tech'])}")
                            st.write(f"- Domain: {pretty_list(og['missing_domain'])}")
                            st.write(f"- Certifs: {pretty_list(og['missing_cert'])}")

                        st.markdown("</div>", unsafe_allow_html=True)

                        with st.expander("Détails (justification + citations + gaps vs idéal)", expanded=False):
                            colX, colY = st.columns(2)
                            do_explain = colX.button("Générer justification (match)", key=f"explain_{cv_id}")
                            do_gaps = colY.button("Générer manques vs idéal", key=f"gaps_{cv_id}")

                            if do_explain:
                                with st.spinner("Mistral: justification…"):
                                    expl = call_mistral_json_explanation(
                                        ao_pack=ao_pack if ao_pack else {},
                                        cv_pack=cv_pack if cv_pack else {},
                                        scores=scores,
                                        mistral_model=mistral_model_ao,
                                    )
                                if expl.get("error"):
                                    st.error(expl["error"])
                                else:
                                    st.json(expl)

                                    st.markdown("#### Citations (preuves dans le CV)")
                                    for section_key in ["strengths", "gaps"]:
                                        items = expl.get(section_key) or []
                                        if not items:
                                            continue
                                        st.markdown(f"**{section_key.upper()}**")
                                        for j, it in enumerate(items, start=1):
                                            title = it.get("title") or it.get("item") or f"Item {j}"
                                            q_terms = it.get("query_terms") or []
                                            st.markdown(f"- **{title}**")
                                            if q_terms:
                                                for qt in q_terms[:5]:
                                                    passages, _m = vector_search_passages(str(qt), cv_text, top_k=2)
                                                    for p in passages:
                                                        st.markdown(
                                                            f"  - *(score {p['score']:.2f})* {p['text']}",
                                                        )

                            if do_gaps:
                                with st.spinner("Mistral: gaps vs idéal…"):
                                    gaps = call_mistral_json_gap_to_ideal(
                                        ao_pack=ao_pack if ao_pack else {},
                                        cv_pack=cv_pack if cv_pack else {},
                                        mistral_model=mistral_model_ao,
                                    )
                                if gaps.get("error"):
                                    st.error(gaps["error"])
                                else:
                                    st.json(gaps)


# ============================================================
# TAB 3 — DB management
# ============================================================
with tabs[2]:
    st.subheader("Base locale — CVs stockés")
    df_list = list_cvs(conn)

    # ✅ compat: ta DB expose "seniorite_label", pas "seniorite"
    cols = ["cv_id", "filename", "nom", "role_principal", "seniorite_label"]
    cols = [c for c in cols if c in df_list.columns]

    if df_list.empty:
        st.info("DB vide.")
    else:
        st.dataframe(df_list[cols], use_container_width=True)

        st.markdown("### Supprimer un CV")
        choice = st.selectbox(
            "Choisir un CV à supprimer",
            df_list["cv_id"].tolist(),
            format_func=lambda cid: df_list[df_list["cv_id"] == cid]["filename"].iloc[0],
        )
        if st.button("Supprimer", type="secondary"):
            delete_cv(conn, choice)
            st.success("Supprimé. Recharge la page si besoin.")
