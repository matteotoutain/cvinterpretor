import streamlit as st
import pandas as pd
import io
import zipfile
import datetime
import plotly.graph_objects as go

from src.config import APP_NAME
from src.db import connect, init_db, upsert_cv, list_cvs, get_cv_texts, delete_cv
from src.extract import extract_text_generic, stable_id_from_bytes
from src.nlp import compute_similarity, explain_match

# NEW: Mistral (as in your notebook style)
from src.mistral_client import call_mistral_json_extraction, DEFAULT_MODEL

# Function to create radar chart
def create_radar_chart(nlp_score, skill_score, seniority_score, global_score):
    """Create a radar chart showing all 4 match scores."""
    categories = ['NLP Score', 'Skills Score', 'Seniority Score', 'Global Score']
    values = [nlp_score, skill_score, seniority_score, global_score]
    
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
        height=400,
        margin=dict(l=60, r=60, t=60, b=60),
        title=dict(text='Profil de correspondance', x=0.5, xanchor='center')
    )
    
    return fig

st.set_page_config(page_title=APP_NAME, layout="wide")

st.title(APP_NAME)
st.caption(
    "PoC focalisé sur : **Batch CV Import** (stockage local) + **AO Import & NLP Analysis** (ponctuel, sans stockage AO)."
)

conn = connect()
init_db(conn)

tabs = st.tabs(["1) Import CVs (Batch)", "2) Import AO & Analyse (Ponctuel)", "DB: gérer les CVs"])

# ---------------------------
# 1) Import CVs
# ---------------------------
with tabs[0]:
    st.subheader("Étape 1 — Uploader des CVs (batch) → stockage en base locale (SQLite)")
    st.write("Formats supportés : **.pptx** (priorité), **.pdf**, **.docx**, **.txt**.")

    # NEW: Mistral options (CV)
    st.markdown("### Options d'extraction (Mistral)")
    use_mistral = st.checkbox(
        "Utiliser Mistral pour extraire des champs (nom, rôle, seniorité, technologies, langues, etc.)",
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

                    # Base row (structure inchangée)
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
                    }

                    # NEW: Mistral CV extraction (fills existing columns only)
                    if use_mistral:
                        initial_extraction_prompt = (
                            "Extract the following detailed information from the provided CV text, translated in english. "
                            "Output the result as a single JSON object. If a field is not found, use `null`.\n\n"
                            "{\n"
                            '  "nom": "Full name of the person",\n'
                            '  "role_principal": "Main role or title (e.g., \\"Senior Data Engineer\\", \\"Data Scientist NLP\\")",\n'
                            '  "seniorite": "Seniority level (e.g., \\"Senior\\", \\"Confirmé\\", \\"Junior\\"); if you see Senior, Confirmé or Junior in the whole text then it should be that word.",\n'
                            '  "secteur_principal": "Main industry sector(s) (e.g., \\"Banking\\", \\"Insurance\\", \\"Retail\\"), translated in english. List multiple if present, separated by comma.",\n'
                            '  "technologies": "Key technologies, tools, or programming languages mentioned (e.g., \\"Python, SQL, Spark\\"). List multiple if present, separated by comma.",\n'
                            '  "langues": "Languages spoken, translated in english. List multiple if present, separated by comma.",\n'
                            '  "cv_text": "The CV main text, focusing on experience and key skills."\n'
                            "}\n"
                        )

                        extracted = call_mistral_json_extraction(
                            text_input=text,
                            user_prompt=initial_extraction_prompt,
                            mistral_model=mistral_model,
                        )

                        if extracted and not extracted.get("error"):
                            for k in [
                                "nom",
                                "role_principal",
                                "seniorite",
                                "secteur_principal",
                                "technologies",
                                "langues",
                                "cv_text",
                            ]:
                                if k in extracted and extracted[k] is not None:
                                    row[k] = extracted[k]
                        else:
                            st.write(
                                f"⚠️ Mistral extraction failed for {f.name}: "
                                f"{extracted.get('error') if extracted else 'Unknown error'}"
                            )

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
    st.subheader("Étape 2 — Uploader un Appel d'Offre → Analyse NLP → Matching + explication")
    st.write("Ici l'AO n'est **pas** stocké : on calcule à la volée.")

    # NEW: Mistral options (AO)
    st.markdown("### Options d'analyse AO (Mistral)")
    use_mistral_ao = st.checkbox("Utiliser Mistral pour structurer l'AO (résumé + exigences)", value=True)
    mistral_model_ao = st.text_input("Modèle Mistral (AO)", value=DEFAULT_MODEL, disabled=not use_mistral_ao)

    ao_file = st.file_uploader("Dépose ton AO", type=["pdf", "docx", "txt", "pptx"], accept_multiple_files=False)
    top_k = st.slider("Nombre de profils à afficher", 3, 30, 10)

    # Store analysis results in session state
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

            # Launch analysis button
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
                            '  "titre_poste": "Job title or main role for the mission (e.g., \\"Senior Data Engineer\\", \\"Data Scientist NLP\\")",\n'
                            '  "contexte_mission": "Brief summary of the mission context and objectives",\n'
                            '  "competences_techniques": "Key technical skills required (e.g., \\"Python, SQL, Spark\\"). List multiple if present, separated by comma.",\n'
                            '  "competences_metier": "Key business or soft skills required (e.g., \\"Project Management, Client Relationship\\"). List multiple if present, separated by comma.",\n'
                            '  "experience_requise": "Required experience level or years (e.g., \\"7+ years\\", \\"Junior\\", \\"Senior\\")",\n'
                            '  "langues_requises": "Languages required for the mission, translated in english. List multiple if present, separated by comma."\n'
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
                    else:
                        scores, method = compute_similarity(ao_text, cvs["cv_text"].tolist())
                        cvs = cvs.copy()
                        cvs["score"] = scores
                        cvs = cvs.sort_values("score", ascending=False).head(int(top_k))

                        # Store results in session state
                        st.session_state.ao_analysis_results = {
                            "ao_struct": ao_struct,
                            "cvs": cvs,
                            "method": method,
                            "ao_text": ao_text,
                            "ao_file_name": ao_file.name,
                        }

                        status.update(label="Analyse terminée ✅", state="complete")

                # Display results if available
                if st.session_state.ao_analysis_results:
                    results = st.session_state.ao_analysis_results
                    cvs = results["cvs"]
                    ao_struct = results["ao_struct"]
                    method = results["method"]
                    ao_text = results["ao_text"]

                    if ao_struct and not ao_struct.get("error"):
                        st.subheader("AO — résumé structuré (Mistral)")
                        st.json(ao_struct)
                    elif ao_struct and ao_struct.get("error"):
                        st.warning(f"⚠️ Mistral AO extraction failed: {ao_struct.get('error')}")

                    st.caption(f"Méthode de matching : **{method}** (cosine similarity).")

                    st.info(
                        "Matching basé sur le texte AO complet. Le résumé structuré sert à mieux expliquer les résultats."
                    )

                    # Download section
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
                        st.info(f"Score moyen : {cvs['score'].mean():.3f}")

                    top_cvs_to_download = cvs.head(num_to_download)

                    # Create zip file with best CVs
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
                        title = f"{idx}. {row['filename']} — score {row['score']:.3f}"
                        if idx <= num_to_download:
                            title = f"⭐ {title}"

                        with st.expander(title, expanded=(idx <= 3)):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.markdown("**Score NLP**")
                                st.metric("Similarité", f"{row['score']:.3f}", delta=None)

                            # Get seniority values from ao_struct and CV
                            ao_seniority = ao_struct.get("experience_requise", "") if isinstance(ao_struct, dict) else ""
                            cv_seniority = row.get("seniorite", "") or ""
                            
                            expl = explain_match(ao_text, row["cv_text"], ao_seniority, cv_seniority)
                            
                            # Skill overlap ratio score
                            skill_ratio = expl.get("skill_overlap_ratio", 0.0)
                            with col2:
                                st.markdown("**Score Compétences**")
                                st.metric("Overlap Ratio", f"{skill_ratio:.3f}", delta=None)
                            
                            # Seniority match score
                            seniority_score = expl.get("seniority_match_score", 0.5)
                            with col3:
                                st.markdown("**Score Seniorité**")
                                st.metric("Match Seniorité", f"{seniority_score:.3f}", delta=None)
                            
                            # Calculate and display global score (weighted)
                            # Weights: NLP 50%, Skills 30%, Seniority 20%
                            global_score = (row['score'] * 0.5) + (skill_ratio * 0.3) + (seniority_score * 0.2)
                            with col4:
                                st.markdown("**Score Global**")
                                st.metric("Score Pondéré", f"{global_score:.3f}", delta=None)
                            
                            # Radar chart visualization
                            st.markdown("**Visualisation des scores**")
                            radar_fig = create_radar_chart(row['score'], skill_ratio, seniority_score, global_score)
                            st.plotly_chart(radar_fig, use_container_width=True)
                            
                            # Info CV
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
                                height=200,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"cv_text_{row['cv_id']}"
                            )

                            st.markdown("**Analyse de correspondance**")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"✅ **Compétences correspondent** ({len(expl['overlap'])})")
                                st.write(", ".join(expl["overlap"][:15]) if expl["overlap"] else "—")
                            with col_b:
                                st.write(f"❌ **Compétences manquantes** ({len(expl['missing'])})")
                                st.write(", ".join(expl["missing"][:15]) if expl["missing"] else "—")

        except Exception as e:
            st.error(str(e))

        except Exception as e:
            st.error(str(e))

# ---------------------------
# 3) Manage DB - IMPROVED
# ---------------------------
with tabs[2]:
    st.subheader("🗂️ Gestion de la Base de Données CV")
    
    # Overview section
    st.markdown("### 📊 Vue d'ensemble")
    df_all = list_cvs(conn)
    df_all_with_text = get_cv_texts(conn)  # Full data including cv_text
    
    if df_all.empty:
        st.info("Aucun CV en base. Commencez par l'étape 1.")
    else:
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total CVs", len(df_all))
        with col2:
            filled_nom = df_all["nom"].notna().sum()
            st.metric("CVs avec nom", filled_nom)
        with col3:
            filled_tech = df_all["technologies"].notna().sum()
            st.metric("CVs enrichis", filled_tech)
        with col4:
            avg_text_len = df_all["filename"].apply(lambda x: len(x) if x else 0).mean()
            st.metric("Fichiers catalogués", len(df_all))
        
        # Filters
        st.markdown("### 🔍 Filtrer & Analyser")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_term = st.text_input("Rechercher par nom de fichier ou nom de personne", "")
        with col2:
            show_full = st.checkbox("Afficher texte complet", value=False)
        
        # Apply filters
        df_filtered = df_all.copy()
        if search_term.strip():
            search_lower = search_term.lower()
            df_filtered = df_filtered[
                (df_filtered["filename"].str.lower().str.contains(search_lower, na=False)) |
                (df_filtered["nom"].fillna("").str.lower().str.contains(search_lower, na=False))
            ]
        
        st.markdown(f"**{len(df_filtered)} CV(s) correspondant(s)**")
        
        # Display table with actions
        st.markdown("### 📋 Liste des CVs")
        
        display_cols = ["filename", "nom", "role_principal", "seniorite", "secteur_principal", "technologies", "langues"]
        df_display = df_filtered[display_cols].copy()
        st.dataframe(df_display, width=True, hide_index=True, use_container_width=True)
        
        
        # Detailed view
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
            # Get full row with cv_text from df_all_with_text
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
                    st.text_area(
                        "Texte",
                        selected_row["cv_text"],
                        height=300,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                else:
                    st.markdown("**Aperçu du contenu** (2000 premiers caractères)")
                    preview = (selected_row["cv_text"] or "")[:2000] + ("…" if len(selected_row["cv_text"] or "") > 2000 else "")
                    st.text_area(
                        "Aperçu",
                        preview,
                        height=150,
                        disabled=True,
                        label_visibility="collapsed"
                    )
        
        # Bulk actions
        st.divider()
        st.markdown("### ⚠️ Actions en masse")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.warning("Attention : les actions en masse sont définitives")
        with col2:
            if st.button("🗑️ Vider la base complète", key="clear_all", type="secondary"):
                if st.session_state.get("confirm_clear", False):
                    for _, row in df_all.iterrows():
                        delete_cv(conn, row["cv_id"])
                    st.success("✅ Base vidée")
                    st.rerun()
                else:
                    st.error("Êtes-vous sûr ? Cliquez à nouveau pour confirmer.")
                    st.session_state.confirm_clear = True
