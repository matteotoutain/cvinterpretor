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


def overlap_and_gaps(ao_pack: Dict[str, Any], cv_pack: Dict[str, Any]) -> Dict[str, List[str]]:
    ao_tech = set_from_pack(ao_pack, "tech_skills_required")
    ao_dom = set_from_pack(ao_pack, "domain_knowledge_required")
    ao_cert = set_from_pack(ao_pack, "certifications_required")

    cv_tech = set_from_pack(cv_pack, "tech_skills") | set_from_pack(cv_pack, "technologies")
    cv_dom = set_from_pack(cv_pack, "domain_knowledge") | set_from_pack(cv_pack, "secteur_principal")
    cv_cert = set_from_pack(cv_pack, "certifications")

    return {
        "overlap_tech": sorted(list(ao_tech & cv_tech)),
        "missing_tech": sorted(list(ao_tech - cv_tech)),
        "overlap_domain": sorted(list(ao_dom & cv_dom)),
        "missing_domain": sorted(list(ao_dom - cv_dom)),
        "overlap_cert": sorted(list(ao_cert & cv_cert)),
        "missing_cert": sorted(list(ao_cert - cv_cert)),
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

    if files and do_extract:
        st.write(f"📦 {len(files)} fichier(s) reçu(s).")

        for f in files:
            raw_bytes = f.getvalue()
            cv_id = stable_id_from_bytes(raw_bytes)
            filename = f.name

            with st.expander(f"CV: {filename}", expanded=False):
                with st.spinner("Extraction texte…"):
                    doc_type, text = extract_text_generic(filename=filename, file_bytes=raw_bytes)
                    text = (text or "").strip()

                if not text:
                    st.error("Extraction texte vide. (Pas d'OCR ici.)")
                    continue

                extracted = {}
                if use_mistral:
                    with st.spinner("Mistral: keyword pack CV…"):
                        extracted = call_mistral_cv_keyword_pack(cv_text=text, mistral_model=mistral_model)
                    if extracted.get("error"):
                        st.error(extracted["error"])
                        extracted = {}

                nom = extracted.get("nom") if extracted else ""
                role = extracted.get("role_principal") if extracted else ""

                senior_raw = extracted.get("seniorite_raw") or extracted.get("seniorite") or ""
                senior_label = extracted.get("seniorite_label") or normalize_seniority_label(str(senior_raw))

                row = {
                    "cv_id": cv_id,
                    "filename": filename,
                    "nom": nom or "",
                    "role_principal": role or "",
                    "seniorite_raw": str(senior_raw or ""),
                    "seniorite_label": str(senior_label or "Unknown"),
                    "secteur_principal": extracted.get("secteur_principal") or "",
                    "tech_skills": _csv_join(extracted.get("tech_skills") or []),
                    "domain_knowledge": _csv_join(extracted.get("domain_knowledge") or []),
                    "certifications": _csv_join(extracted.get("certifications") or []),
                    # legacy
                    "technologies": _csv_join(extracted.get("technologies") or extracted.get("hard_skills") or []),
                    "langues": _csv_join(extracted.get("langues") or []),
                    "cv_text": text,
                    "cv_struct_json": _safe_json_dumps(extracted) if extracted else "",
                    "cv_keywords_json": _safe_json_dumps(extracted) if extracted else "",
                }

                upsert_cv(conn, row)

                st.success("✅ Stocké en DB")
                st.write(
                    {
                        "nom": row["nom"],
                        "role_principal": row["role_principal"],
                        "seniorite_label": row["seniorite_label"],
                        "tech_skills": (row["tech_skills"][:200] + ("…" if len(row["tech_skills"]) > 200 else "")),
                        "domain_knowledge": (row["domain_knowledge"][:200] + ("…" if len(row["domain_knowledge"]) > 200 else "")),
                        "certifications": (row["certifications"][:200] + ("…" if len(row["certifications"]) > 200 else "")),
                    }
                )


# ============================================================
# TAB 2 — AO & Analyse
# ============================================================
with tabs[1]:
    st.subheader("Étape 2 — Import AO → scoring + explications + radar + citations (sans stocker l'AO)")

    st.markdown("### AO input")
    use_mistral_ao = st.checkbox("Utiliser Mistral pour nettoyer l'AO en 'keyword pack'", value=True)
    mistral_model_ao = st.text_input("Modèle Mistral (AO)", value=DEFAULT_MODEL, disabled=not use_mistral_ao)

    ao_file = st.file_uploader("Dépose ton AO ici", type=["pdf", "docx", "txt", "pptx"], accept_multiple_files=False)

    ao_text = ""
    if ao_file:
        ao_type, ao_text = extract_text_generic(filename=ao_file.name, file_bytes=ao_file.getvalue())
        ao_text = (ao_text or "").strip()

    ao_pack: Dict[str, Any] = {}
    if ao_text and use_mistral_ao:
        with st.spinner("Mistral: keyword pack AO…"):
            ao_pack = call_mistral_ao_keyword_pack(ao_text=ao_text, mistral_model=mistral_model_ao)
        if ao_pack.get("error"):
            st.error(ao_pack["error"])
            ao_pack = {}

    st.markdown("### Pondération des blocs (user-driven)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        w_tech = st.slider("Tech skills", 0.0, 1.0, 0.35, 0.05)
    with c2:
        w_exp = st.slider("Experience", 0.0, 1.0, 0.30, 0.05)
    with c3:
        w_dom = st.slider("Domain knowledge", 0.0, 1.0, 0.25, 0.05)
    with c4:
        w_cert = st.slider("Certifications", 0.0, 1.0, 0.10, 0.05)

    weights = {"tech_skills": w_tech, "experience": w_exp, "domain_knowledge": w_dom, "certifications": w_cert}

    st.markdown("### Filtres (après scoring)")
    senior_filter = st.multiselect(
        "Séniorité",
        ["Junior", "Senior", "Manager", "Unknown"],
        default=["Junior", "Senior", "Manager", "Unknown"],
    )

    ao_skill_suggestions: List[str] = []
    if ao_pack:
        ao_skill_suggestions = list(dict.fromkeys((ao_pack.get("tech_skills_required") or []) + (ao_pack.get("domain_knowledge_required") or [])))

    selected_terms = st.multiselect(
        "Filtrer sur compétences/domaines (le candidat doit contenir TOUS les termes)",
        ao_skill_suggestions,
        default=[],
    )

    run = st.button("Lancer l'analyse", type="primary", disabled=not bool(ao_text))

    if run:
        df = get_cv_texts(conn)
        if df.empty:
            st.warning("Aucun CV en base. Va d'abord dans l'onglet 1.")
        else:
            ao_blocks = build_ao_blocks(ao_pack if ao_pack else {}, ao_fallback_text=ao_text)

            rows: List[Dict[str, Any]] = []
            for _, r in df.iterrows():
                cv_text = r.get("cv_text") or ""
                cv_pack = _safe_json_loads(r.get("cv_keywords_json")) or _safe_json_loads(r.get("cv_struct_json")) or {}
                if not cv_pack:
                    cv_pack = {
                        "role_principal": r.get("role_principal") or "",
                        "technologies": r.get("technologies") or "",
                        "secteur_principal": r.get("secteur_principal") or "",
                        "certifications": r.get("certifications") or "",
                    }

                cv_blocks = build_cv_blocks(cv_pack, cv_fallback_text=cv_text)
                scores, method = score_blocks(ao_blocks, cv_blocks, weights=weights)

                senior_label = r.get("seniorite_label") or normalize_seniority_label(str(r.get("seniorite_raw") or "")) or "Unknown"
                if senior_label not in senior_filter:
                    continue

                if selected_terms:
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
                    if not _contains_all_terms(hay, selected_terms):
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
                        st.plotly_chart(fig, use_container_width=True)

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
                                        title = it.get("title") or f"{section_key} {j}"
                                        q_terms = it.get("query_terms") or []
                                        query = " ".join(q_terms) if q_terms else title
                                        hits, _ = vector_search_passages(query=query, cv_text=cv_text, top_k=2)
                                        with st.expander(f"{j}. {title} — query: {query}", expanded=False):
                                            if not hits:
                                                st.write("No supporting passage found.")
                                            else:
                                                for h in hits:
                                                    st.write(f"Score: {h['score']:.3f}")
                                                    st.write(h["text"])

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

                                st.markdown("#### Citations (vérif dans le CV)")
                                for section_key in ["must_have_missing", "nice_to_have_missing", "unclear"]:
                                    items = gaps.get(section_key) or []
                                    if not items:
                                        continue
                                    st.markdown(f"**{section_key.upper()}**")
                                    for j, it in enumerate(items, start=1):
                                        item = it.get("item") or f"{section_key} {j}"
                                        q_terms = it.get("query_terms") or []
                                        query = " ".join(q_terms) if q_terms else item
                                        hits, _ = vector_search_passages(query=query, cv_text=cv_text, top_k=2)
                                        with st.expander(f"{j}. {item} — query: {query}", expanded=False):
                                            if not hits:
                                                st.write("No supporting passage found.")
                                            else:
                                                for h in hits:
                                                    st.write(f"Score: {h['score']:.3f}")
                                                    st.write(h["text"])


# ============================================================
# TAB 3 — DB management
# ============================================================
with tabs[2]:
    st.subheader("Base locale — CVs stockés")
    df_list = list_cvs(conn)
    st.dataframe(df_list, use_container_width=True)

    st.markdown("### Supprimer un CV")
    if not df_list.empty:
        choice = st.selectbox(
            "Choisir un CV à supprimer",
            df_list["cv_id"].tolist(),
            format_func=lambda cid: df_list[df_list["cv_id"] == cid]["filename"].iloc[0],
        )
        if st.button("Supprimer", type="secondary"):
            delete_cv(conn, choice)
            st.success("Supprimé. Recharge la page si besoin.")
    else:
        st.info("DB vide.")
