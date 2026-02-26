import streamlit as st
import pandas as pd
import io
import zipfile
import datetime
import json
import plotly.graph_objects as go

from src.config import APP_NAME
from src.db import connect, init_db, upsert_cv, list_cvs, get_cv_texts, delete_cv
from src.extract import extract_text_generic, stable_id_from_bytes
from src.nlp import build_ao_blocks, build_cv_blocks, score_blocks, verdict_from_score

from src.mistral_client import (
    call_mistral_json_extraction,
    call_mistral_json_explanation,
    DEFAULT_MODEL,
)


def create_radar_chart(skills_like, experience_like, domain_like, certification_like, global_score):
    categories = ['Skills-like', 'Experience-like', 'Domain-like', 'Certification-like', 'Global']
    values = [skills_like, experience_like, domain_like, certification_like, global_score]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Match Scores',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        title=dict(text='Profil de correspondance (par blocs)', x=0.5, xanchor='center')
    )
    return fig


st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("PoC focalisé sur : **Batch CV Import** (stockage local) + **AO Import & NLP Analysis** (ponctuel, sans stockage AO).")

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
        "Utiliser Mistral pour structurer le CV (blocs: skills, expériences, domaine, certifications)",
        value=True,
    )
    mistral_model = st.text_input("Modèle Mistral (CV)", value=DEFAULT_MODEL, disabled=not use_mistral)

    files = st.file_uploader("Dépose tes CVs ici", type=["pptx", "pdf", "docx", "txt"], accept_multiple_files=True)

    colA, colB = st.columns([1, 1])
    with colA:
        do_extract = st.checkbox("Extraire + stocker maintenant", value=True)
    with colB:
        st.info("Conseil UX : commence par charger 5–10 CVs pour valider le pipeline, puis scale.")

    if files and do_extract:
        ok, ko = 0, 0
        with st.status("Import en cours…", expanded=True) as status:
            for f in files:
                try:
                    b = f.getvalue()
                    cv_id = stable_id_from_bytes(b)

                    doc_type, text = extract_text_generic(f.name, b)
                    text = (text or "").strip()
                    if not text:
                        raise ValueError("Document vide après extraction")

                    row = {
                        "cv_id": cv_id,
                        "filename": f.name,
                        "nom": None,
                        "role_principal": None,
                        "seniorite": None,
                        "secteur_principal": None,
                        "technologies": None,
                        "langues": None,
                        "cv_text": text,
                        "cv_struct_json": None,
                    }

                    if use_mistral:
                        cv_extraction_prompt = (
                            "Extract the following detailed information from the provided CV text, translated in english. "
                            "Output the result as a single JSON object. If a field is not found, use `null`.\n\n"
                            "{\n"
                            '  "nom": "Full name of the person",\n'
                            '  "role_principal": "Main role or title",\n'
                            '  "seniorite": "Seniority level or years of experience if present",\n'
                            '  "secteur_principal": "Main industry sector(s), comma-separated",\n'
                            '  "technologies": "Key tools/tech mentioned, comma-separated",\n'
                            '  "langues": "Languages, comma-separated",\n'
                            '  "certifications": ["List of certifications (e.g., Salesforce Certified Associate, PMP, AWS, Azure...)"],\n'
                            '  "hard_skills": ["List of hard skills / technologies as items"],\n'
                            '  "soft_skills": ["List of soft skills as items"],\n'
                            '  "experiences": [\n'
                            '     {\n'
                            '       "mission": "Short mission title/summary",\n'
                            '       "secteur": "Domain/industry if stated",\n'
                            '       "stack": ["Tech stack used on that mission"],\n'
                            '       "duree": "Duration if stated"\n'
                            '     }\n'
                            "  ],\n"
                            '  "cv_text": "The CV main text, focusing on experience and key skills."\n'
                            "}\n"
                        )

                        extracted = call_mistral_json_extraction(
                            text_input=text,
                            user_prompt=cv_extraction_prompt,
                            mistral_model=mistral_model,
                        )

                        if extracted and not extracted.get("error"):
                            for k in ["nom", "role_principal", "seniorite", "secteur_principal", "technologies", "langues", "cv_text"]:
                                if k in extracted and extracted[k] is not None:
                                    row[k] = extracted[k]
                            row["cv_struct_json"] = json.dumps(extracted, ensure_ascii=False)
                        else:
                            st.write(f"⚠️ Mistral extraction failed for {f.name}: {extracted.get('error') if extracted else 'Unknown error'}")

                    upsert_cv(conn, row)
                    st.write(f"✅ {f.name} ({doc_type}) — stocké (cv_id={cv_id})")
                    ok += 1

                except Exception as e:
                    st.write(f"❌ {f.name} — {e}")
                    ko += 1

            status.update(label=f"Terminé — {ok} OK / {ko} KO", state="complete")

    st.divider()
    st.subheader("CVs actuellement en base")
    df = list_cvs(conn)
    st.dataframe(df, width=True, hide_index=True)


# ---------------------------
# 2) AO Import & NLP Analysis
# ---------------------------
with tabs[1]:
    st.subheader("Étape 2 — Uploader un Appel d'Offre → Analyse par blocs → Matching + explication")
    st.write("Ici l'AO n'est **pas** stocké : on calcule à la volée.")

    st.markdown("### Options d'analyse AO (Mistral)")
    use_mistral_ao = st.checkbox("Utiliser Mistral pour structurer l'AO (blocs: skills, contexte, domaine, certifications)", value=True)
    mistral_model_ao = st.text_input("Modèle Mistral (AO)", value=DEFAULT_MODEL, disabled=not use_mistral_ao)

    st.markdown("### Options d'explication (Mistral)")
    use_mistral_explain = st.checkbox("Générer une explication structurée (JSON) après scoring", value=True)
    mistral_model_explain = st.text_input("Modèle Mistral (Explain)", value=DEFAULT_MODEL, disabled=not use_mistral_explain)

    ao_file = st.file_uploader("Dépose ton AO", type=["pdf", "docx", "txt", "pptx"], accept_multiple_files=False)
    top_k = st.slider("Nombre de profils à afficher", 3, 30, 10)

    if "ao_analysis_results" not in st.session_state:
        st.session_state.ao_analysis_results = None

    if ao_file:
        try:
            ao_bytes = ao_file.getvalue()
            ao_type, ao_text = extract_text_generic(ao_file.name, ao_bytes)
            ao_text = (ao_text or "").strip()
            if not ao_text:
                raise ValueError("AO vide après extraction")

            st.success(f"AO chargé ({ao_type}). Longueur texte: {len(ao_text):,} caractères.")
            with st.expander("Voir l'extrait de l'AO"):
                st.write(ao_text[:2500] + ("…" if len(ao_text) > 2500 else ""))

            col1, col2 = st.columns([2, 1])
            with col1:
                launch_analysis = st.button("🚀 Lance l'analyse", type="primary", key="launch_analysis")
            with col2:
                st.info(f"CVs en base : {len(get_cv_texts(conn))}")

            if launch_analysis or st.session_state.ao_analysis_results is not None:
                with st.status("Analyse en cours…", expanded=True) as status:
                    ao_struct = None
                    if use_mistral_ao:
                        ao_extraction_prompt = (
                            "Extract the following detailed information from the provided AO text, translated in english. "
                            "Output the result as a single JSON object. If a field is not found, use `null`.\n\n"
                            "{\n"
                            '  "titre_poste": "Job title or main role for the mission",\n'
                            '  "contexte_mission": "Brief summary of the mission context and objectives",\n'
                            '  "competences_techniques": ["List of required technical skills"],\n'
                            '  "competences_metier": ["List of business/soft skills expected"],\n'
                            '  "secteur": "Industry domain if stated (banking, insurance, public sector, etc.)",\n'
                            '  "experience_requise": "Required experience level or years",\n'
                            '  "langues_requises": ["Languages required"],\n'
                            '  "certifications_requises": ["Certifications explicitly required or strongly desired"]\n'
                            "}\n"
                        )

                        ao_struct = call_mistral_json_extraction(
                            text_input=ao_text,
                            user_prompt=ao_extraction_prompt,
                            mistral_model=mistral_model_ao,
                        )

                    cvs = get_cv_texts(conn)
                    if cvs.empty:
                        st.warning("Aucun CV en base. Reviens à l'étape 1.")
                        status.update(label="Analyse impossible (0 CV)", state="error")
                    else:
                        if isinstance(ao_struct, dict) and not ao_struct.get("error"):
                            ao_blocks = build_ao_blocks(ao_struct, ao_fallback_text=ao_text)
                        else:
                            ao_struct = {"error": ao_struct.get("error")} if isinstance(ao_struct, dict) else {"error": "AO struct failed"}
                            ao_blocks = build_ao_blocks({}, ao_fallback_text=ao_text)

                        rows = []
                        method_used = None

                        for _, r in cvs.iterrows():
                            cv_text = r.get("cv_text") or ""
                            cv_struct_json = r.get("cv_struct_json")

                            if cv_struct_json:
                                try:
                                    cv_struct = json.loads(cv_struct_json)
                                except Exception:
                                    cv_struct = {}
                            else:
                                cv_struct = {}

                            cv_blocks = build_cv_blocks(cv_struct, cv_fallback_text=cv_text)
                            scores, method = score_blocks(ao_blocks, cv_blocks)
                            method_used = method_used or method

                            rows.append({
                                "cv_id": r["cv_id"],
                                "filename": r["filename"],
                                "nom": r.get("nom"),
                                "role_principal": r.get("role_principal"),
                                "seniorite": r.get("seniorite"),
                                "secteur_principal": r.get("secteur_principal"),
                                "technologies": r.get("technologies"),
                                "langues": r.get("langues"),
                                "cv_text": cv_text,
                                "cv_struct_json": cv_struct_json,
                                "skills_like": scores["skills_like"],
                                "experience_like": scores["experience_like"],
                                "domain_like": scores["domain_like"],
                                "certification_like": scores["certification_like"],
                                "global_score": scores["global_score"],
                                "verdict": verdict_from_score(scores["global_score"]),
                            })

                        res = pd.DataFrame(rows).sort_values("global_score", ascending=False).head(int(top_k))

                        st.session_state.ao_analysis_results = {
                            "ao_struct": ao_struct,
                            "ao_blocks": ao_blocks,
                            "cvs": res,
                            "method": method_used or "unknown",
                            "ao_text": ao_text,
                            "ao_file_name": ao_file.name,
                        }

                        status.update(label="Analyse terminée ✅", state="complete")

                if st.session_state.ao_analysis_results:
                    results = st.session_state.ao_analysis_results
                    cvs = results["cvs"]
                    ao_struct = results["ao_struct"]
                    method = results["method"]

                    if ao_struct and isinstance(ao_struct, dict) and not ao_struct.get("error"):
                        st.subheader("AO — résumé structuré (Mistral)")
                        st.json(ao_struct)
                    elif ao_struct and isinstance(ao_struct, dict) and ao_struct.get("error"):
                        st.warning(f"⚠️ Mistral AO extraction failed: {ao_struct.get('error')}")

                    st.caption(f"Méthode embeddings : **{method}** (cosine similarity).")
                    st.info("Matching basé sur **blocs** (skills-like / experience-like / domain-like / certification-like).")

                    st.divider()
                    st.subheader("📥 Télécharger les meilleurs CVs")

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        num_to_download = st.slider(
                            "Nombre de meilleurs CVs à télécharger",
                            1,
                            min(10, len(cvs)),
                            min(3, len(cvs)),
                            key="num_download"
                        )
                    with col2:
                        st.info(f"Score moyen : {cvs['global_score'].mean():.3f}")

                    top_cvs_to_download = cvs.head(num_to_download)

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for idx, (_, row) in enumerate(top_cvs_to_download.iterrows(), 1):
                            cv_content = (row["cv_text"] or "").encode("utf-8")
                            safe_filename = f"{idx}_{row['filename']}.txt"
                            zip_file.writestr(safe_filename, cv_content)

                    zip_buffer.seek(0)
                    st.download_button(
                        label=f"⬇️ Télécharger {num_to_download} meilleurs CVs (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"best_cvs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                    )

                    st.divider()
                    st.subheader("Résultats détaillés")

                    for idx, (_, row) in enumerate(cvs.iterrows(), 1):
                        title = f"{idx}. {row['filename']} — global {row['global_score']:.3f} — {row['verdict']}"
                        if idx <= num_to_download:
                            title = f"⭐ {title}"

                        with st.expander(title, expanded=(idx <= 3)):
                            c1, c2, c3, c4, c5 = st.columns(5)
                            with c1:
                                st.metric("Skills-like", f"{row['skills_like']:.3f}")
                            with c2:
                                st.metric("Experience-like", f"{row['experience_like']:.3f}")
                            with c3:
                                st.metric("Domain-like", f"{row['domain_like']:.3f}")
                            with c4:
                                st.metric("Certification-like", f"{row['certification_like']:.3f}")
                            with c5:
                                st.metric("Global", f"{row['global_score']:.3f}")

                            st.plotly_chart(
                                create_radar_chart(
                                    row["skills_like"],
                                    row["experience_like"],
                                    row["domain_like"],
                                    row["certification_like"],
                                    row["global_score"],
                                ),
                                use_container_width=True
                            )

                            st.markdown("**Info CV**")
                            info_data = {
                                "Nom": row["nom"] or "—",
                                "Rôle": row["role_principal"] or "—",
                                "Seniorité": row["seniorite"] or "—",
                                "Secteur": row["secteur_principal"] or "—",
                                "Technologies": row["technologies"] or "—",
                                "Langues": row["langues"] or "—",
                            }
                            st.table(pd.DataFrame(info_data.items(), columns=["Champ", "Valeur"]))

                            st.markdown("**Extrait CV**")
                            st.text_area(
                                "Contenu",
                                (row["cv_text"] or "")[:1500] + ("…" if len(row["cv_text"] or "") > 1500 else ""),
                                height=180,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"cv_text_{row['cv_id']}"
                            )

                            if use_mistral_explain:
                                st.markdown("**Explication IA (structurée)**")
                                cv_struct = {}
                                if row.get("cv_struct_json"):
                                    try:
                                        cv_struct = json.loads(row["cv_struct_json"])
                                    except Exception:
                                        cv_struct = {}

                                ao_struct_for_explain = ao_struct if isinstance(ao_struct, dict) and not ao_struct.get("error") else {}
                                score_payload = {
                                    "skills_like": float(row["skills_like"]),
                                    "experience_like": float(row["experience_like"]),
                                    "domain_like": float(row["domain_like"]),
                                    "certification_like": float(row["certification_like"]),
                                    "global_score": float(row["global_score"]),
                                    "verdict": row["verdict"],
                                }

                                explain = call_mistral_json_explanation(
                                    ao_struct=ao_struct_for_explain,
                                    cv_struct=cv_struct,
                                    scores=score_payload,
                                    mistral_model=mistral_model_explain,
                                )

                                if explain and not explain.get("error"):
                                    st.json(explain)
                                else:
                                    st.warning(f"⚠️ Explain failed: {explain.get('error') if isinstance(explain, dict) else 'Unknown error'}")

        except Exception as e:
            st.error(str(e))


# ---------------------------
# 3) Manage DB
# ---------------------------
with tabs[2]:
    st.subheader("🗂️ Gestion de la Base de Données CV")

    st.markdown("### 📊 Vue d'ensemble")
    df_all = list_cvs(conn)
    df_all_with_text = get_cv_texts(conn)

    if df_all.empty:
        st.info("Aucun CV en base. Commencez par l'étape 1.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total CVs", len(df_all))
        with col2:
            filled_nom = df_all["nom"].notna().sum()
            st.metric("CVs avec nom", filled_nom)
        with col3:
            filled_struct = df_all_with_text["cv_struct_json"].notna().sum()
            st.metric("CVs structurés (JSON)", filled_struct)
        with col4:
            st.metric("Fichiers catalogués", len(df_all))

        st.markdown("### 🔍 Filtrer & Analyser")
        col1, col2 = st.columns([2, 1])
        with col1:
            search_term = st.text_input("Rechercher par nom de fichier ou nom de personne", "")
        with col2:
            show_full = st.checkbox("Afficher texte complet", value=False)

        df_filtered = df_all.copy()
        if search_term.strip():
            s = search_term.lower()
            df_filtered = df_filtered[
                (df_filtered["filename"].str.lower().str.contains(s, na=False)) |
                (df_filtered["nom"].fillna("").str.lower().str.contains(s, na=False))
            ]

        st.markdown(f"**{len(df_filtered)} CV(s) correspondant(s)**")
        display_cols = ["filename", "nom", "role_principal", "seniorite", "secteur_principal", "technologies", "langues"]
        st.dataframe(df_filtered[display_cols], width=True, hide_index=True, use_container_width=True)

        st.markdown("### 🔎 Détails & Actions")
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_filename = st.selectbox(
                "Sélectionner un CV pour voir les détails",
                df_filtered["filename"].tolist() if len(df_filtered) > 0 else ["— Aucun CV —"],
                key="cv_selector"
            )
        with col2:
            if st.button("🔄 Rafraîchir", key="refresh_db"):
                st.rerun()

        if selected_filename != "— Aucun CV —" and selected_filename in df_filtered["filename"].values:
            selected_row = df_all_with_text[df_all_with_text["filename"] == selected_filename].iloc[0]

            with st.expander(f"📄 Détails de {selected_filename}", expanded=True):
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.write("**Métadonnées extraites**")
                    meta_data = {
                        "cv_id": selected_row["cv_id"],
                        "Nom": selected_row["nom"] or "—",
                        "Rôle": selected_row["role_principal"] or "—",
                        "Seniorité": selected_row["seniorite"] or "—",
                        "Secteur": selected_row["secteur_principal"] or "—",
                        "Technologies": selected_row["technologies"] or "—",
                        "Langues": selected_row["langues"] or "—",
                        "Struct JSON": "✅" if selected_row.get("cv_struct_json") else "—",
                    }
                    for k, v in meta_data.items():
                        st.write(f"**{k}:** {v}")

                with col2:
                    st.write("**Actions**")
                    if st.button(f"📥 Télécharger CV", key=f"dl_{selected_row['cv_id']}"):
                        cv_text = selected_row["cv_text"] or "Contenu non disponible"
                        st.download_button(
                            label="⬇️ Télécharger en .txt",
                            data=cv_text.encode("utf-8"),
                            file_name=f"{selected_filename}.txt",
                            mime="text/plain",
                            key=f"download_{selected_row['cv_id']}"
                        )

                    if st.button(f"🗑️ Supprimer ce CV", key=f"del_{selected_row['cv_id']}", type="secondary"):
                        try:
                            delete_cv(conn, selected_row["cv_id"])
                            st.success(f"✅ {selected_filename} supprimé")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Erreur : {e}")

                if show_full and selected_row["cv_text"]:
                    st.markdown("**Contenu complet du CV**")
                    st.text_area("Texte", selected_row["cv_text"], height=300, disabled=True, label_visibility="collapsed")
                else:
                    st.markdown("**Aperçu du contenu** (2000 premiers caractères)")
                    preview = (selected_row["cv_text"] or "")[:2000] + ("…" if len(selected_row["cv_text"] or "") > 2000 else "")
                    st.text_area("Aperçu", preview, height=150, disabled=True, label_visibility="collapsed")

        st.divider()
        st.markdown("### ⚠️ Actions en masse")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.warning("Attention : les actions en masse sont définitives")
        with col2:
            if st.button("🗑️ Vider la base complète", key="clear_all", type="secondary"):
                if st.session_state.get("confirm_clear", False):
                    for _, r in df_all.iterrows():
                        delete_cv(conn, r["cv_id"])
                    st.success("✅ Base vidée")
                    st.rerun()
                else:
                    st.error("Êtes-vous sûr ? Cliquez à nouveau pour confirmer.")
                    st.session_state.confirm_clear = True
