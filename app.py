import streamlit as st
import pandas as pd
import json

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
    BLOCK_KEYS,
)

from src.mistral_client import (
    call_mistral_cv_keyword_pack,
    call_mistral_ao_keyword_pack,
    call_mistral_json_explanation,
    call_mistral_json_gap_to_ideal,
    DEFAULT_MODEL,
)


# ============================================================
# Small helpers
# ============================================================
def _csv_join(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return ", ".join([str(i).strip() for i in x if str(i).strip()])
    return str(x).strip()


def _safe_json_dumps(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""


def _safe_json_loads(s):
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def _contains_all_terms(haystack: str, terms: list[str]) -> bool:
    h = (haystack or "").lower()
    return all((t or "").lower() in h for t in terms if (t or "").strip())


# ============================================================
# App
# ============================================================
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Focus: **Batch CV Import** (stockage local) + **AO Import & NLP Analysis** (ponctuel, sans stockage AO).")

conn = connect()
init_db(conn)

tabs = st.tabs(["1) Import CVs (Batch)", "2) Import AO & Analyse (Ponctuel)", "DB: gérer les CVs"])


# ---------------------------
# 1) Import CVs
# ---------------------------
with tabs[0]:
    st.subheader("Étape 1 — Uploader des CVs (batch) → stockage en base locale (SQLite)")
    st.write("Formats supportés : **.pptx** (priorité), **.pdf**, **.docx**, **.txt**.")

    st.markdown("### Options d'extraction (Mistral)")
    use_mistral = st.checkbox(
        "Utiliser Mistral pour extraire un 'keyword pack' (tech, domaine, certifs, expériences…)",
        value=True,
    )
    mistral_model = st.text_input("Modèle Mistral (CV)", value=DEFAULT_MODEL, disabled=not use_mistral)

    files = st.file_uploader("Dépose tes CVs ici", type=["pptx", "pdf", "docx", "txt"], accept_multiple_files=True)

    colA, colB = st.columns([1, 1])
    with colA:
        do_extract = st.checkbox("Extraire + stocker maintenant", value=True)
    with colB:
        st.info("Conseil : charge 5–10 CVs pour valider le pipeline, puis scale.")

    if files and do_extract:
        st.write(f"📦 {len(files)} fichier(s) reçu(s).")
        for f in files:
            raw_bytes = f.getvalue()
            cv_id = stable_id_from_bytes(raw_bytes)
            filename = f.name

            with st.expander(f"CV: {filename}", expanded=False):
                with st.spinner("Extraction texte…"):
                    text = extract_text_generic(filename=filename, file_bytes=raw_bytes)

                if not text.strip():
                    st.error("Extraction texte vide. (Pas d'OCR ici, demandé explicitement.)")
                    continue

                extracted = {}
                if use_mistral:
                    with st.spinner("Mistral: keyword pack…"):
                        extracted = call_mistral_cv_keyword_pack(cv_text=text, mistral_model=mistral_model)

                    if extracted.get("error"):
                        st.error(extracted["error"])
                        extracted = {}

                # Fallback minimal if no Mistral
                nom = extracted.get("nom") if extracted else None
                role = extracted.get("role_principal") if extracted else None

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
                    # legacy fields (keep for compatibility)
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
                        "tech_skills": row["tech_skills"][:200] + ("…" if len(row["tech_skills"]) > 200 else ""),
                        "domain_knowledge": row["domain_knowledge"][:200] + ("…" if len(row["domain_knowledge"]) > 200 else ""),
                        "certifications": row["certifications"][:200] + ("…" if len(row["certifications"]) > 200 else ""),
                    }
                )


# ---------------------------
# 2) Import AO & Analyse
# ---------------------------
with tabs[1]:
    st.subheader("Étape 2 — Import AO → scoring + explications (sans stocker l'AO)")

    st.markdown("### AO input")
    use_mistral_ao = st.checkbox("Utiliser Mistral pour nettoyer l'AO en 'keyword pack'", value=True)
    mistral_model_ao = st.text_input("Modèle Mistral (AO)", value=DEFAULT_MODEL, disabled=not use_mistral_ao)

    ao_file = st.file_uploader("Dépose ton AO ici", type=["pdf", "docx", "txt", "pptx"], accept_multiple_files=False)
    ao_text = ""
    if ao_file:
        ao_text = extract_text_generic(filename=ao_file.name, file_bytes=ao_file.getvalue())

    ao_pack = {}
    if ao_text.strip() and use_mistral_ao:
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

    st.markdown("### Filtres (à la toute fin)")
    senior_filter = st.multiselect("Séniorité", ["Junior", "Senior", "Manager", "Unknown"], default=["Junior", "Senior", "Manager", "Unknown"])

    # Optional: filter on specific skills (based on AO pack if available)
    ao_skill_suggestions = []
    if ao_pack:
        ao_skill_suggestions = list(dict.fromkeys((ao_pack.get("tech_skills_required") or []) + (ao_pack.get("domain_knowledge_required") or [])))
    selected_terms = st.multiselect(
        "Filtrer sur ces compétences / domaines (candidats doivent contenir TOUS les termes sélectionnés)",
        ao_skill_suggestions,
        default=[],
    )

    run = st.button("Lancer l'analyse", type="primary", disabled=not ao_text.strip())

    if run:
        df = get_cv_texts(conn)
        if df.empty:
            st.warning("Aucun CV en base. Va d'abord dans l'onglet 1.")
        else:
            ao_blocks = build_ao_blocks(ao_pack if ao_pack else {}, ao_fallback_text=ao_text)

            rows = []
            for _, r in df.iterrows():
                cv_text = r.get("cv_text") or ""
                cv_pack = _safe_json_loads(r.get("cv_keywords_json")) or _safe_json_loads(r.get("cv_struct_json")) or {}
                if not cv_pack:
                    cv_pack = {"role_principal": r.get("role_principal") or "", "technologies": r.get("technologies") or "", "secteur_principal": r.get("secteur_principal") or ""}

                cv_blocks = build_cv_blocks(cv_pack, cv_fallback_text=cv_text)
                scores, method = score_blocks(ao_blocks, cv_blocks, weights=weights)

                senior_label = r.get("seniorite_label") or normalize_seniority_label(str(r.get("seniorite_raw") or "")) or "Unknown"
                # Filter late (per requirement)
                if senior_label not in senior_filter:
                    continue

                # Filter by selected terms (late)
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

                st.markdown("### Résultats (triés par score)")
                st.dataframe(out, use_container_width=True)

                st.markdown("### Explication + citations (Vector search)")
                top_n = min(10, len(out))
                candidate_labels = [
                    f"{i+1}. {out.loc[i,'filename']} — {out.loc[i,'nom']} — {out.loc[i,'score_global']:.3f}"
                    for i in range(top_n)
                ]
                pick = st.selectbox("Choisir un candidat à expliquer", candidate_labels, index=0)
                pick_idx = int(pick.split(".")[0]) - 1
                pick_cv_id = out.loc[pick_idx, "cv_id"]

                # Fetch full CV row
                full = df[df["cv_id"] == pick_cv_id].iloc[0]
                cv_text = full.get("cv_text") or ""
                cv_pack = _safe_json_loads(full.get("cv_keywords_json")) or _safe_json_loads(full.get("cv_struct_json")) or {}

                # Recompute for that CV (keeps consistent with filters/weights)
                cv_blocks = build_cv_blocks(cv_pack, cv_fallback_text=cv_text)
                scores, method = score_blocks(ao_blocks, cv_blocks, weights=weights)
                verdict = verdict_from_score(scores["global_score"])

                st.write({"verdict_from_score": verdict, "global_score": scores["global_score"], "vector_method": method})

                colX, colY = st.columns(2)
                with colX:
                    do_explain = st.button("Générer justification (match)", key="btn_explain")
                with colY:
                    do_gaps = st.button("Générer analyse des manques (vs candidat parfait)", key="btn_gaps")

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
                        st.success("✅ Justification générée")
                        st.json(expl)

                        # Citations from CV for each strength / gap
                        st.markdown("#### Citations (passages retrouvés dans le CV)")
                        for section_key in ["strengths", "gaps"]:
                            items = expl.get(section_key) or []
                            if not items:
                                continue
                            st.markdown(f"**{section_key.upper()}**")
                            for i, it in enumerate(items, start=1):
                                title = it.get("title") or f"{section_key} {i}"
                                q_terms = it.get("query_terms") or []
                                query = " ".join(q_terms) if q_terms else title
                                hits, _ = vector_search_passages(query=query, cv_text=cv_text, top_k=2)
                                with st.expander(f"{i}. {title} — query: {query}", expanded=False):
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
                        st.success("✅ Analyse des manques générée")
                        st.json(gaps)

                        st.markdown("#### Citations (vérif dans le CV)")
                        for section_key in ["must_have_missing", "nice_to_have_missing", "unclear"]:
                            items = gaps.get(section_key) or []
                            if not items:
                                continue
                            st.markdown(f"**{section_key.upper()}**")
                            for i, it in enumerate(items, start=1):
                                item = it.get("item") or f"{section_key} {i}"
                                q_terms = it.get("query_terms") or []
                                query = " ".join(q_terms) if q_terms else item
                                hits, _ = vector_search_passages(query=query, cv_text=cv_text, top_k=2)
                                with st.expander(f"{i}. {item} — query: {query}", expanded=False):
                                    if not hits:
                                        st.write("No supporting passage found.")
                                    else:
                                        for h in hits:
                                            st.write(f"Score: {h['score']:.3f}")
                                            st.write(h["text"])


# ---------------------------
# 3) DB management
# ---------------------------
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
