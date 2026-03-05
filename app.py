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
from src.nlp import build_ao_blocks, build_cv_blocks, score_blocks_enhanced, verdict_from_score

from src.mistral_client import (
    call_mistral_json_extraction,
    call_mistral_json_explanation,
    DEFAULT_MODEL,
)


def create_radar_chart(nlp_score, skills_score, seniority_score, domain_score, language_score, global_score):
    categories = ['NLP', 'Skills', 'Seniority', 'Domain', 'Language', 'Global']
    values = [nlp_score, skills_score, seniority_score, domain_score, language_score, global_score]

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
        title=dict(text='Score Breakdown (NLP + Non-NLP Features)', x=0.5, xanchor='center')
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
                        cv_extraction_prompt = f"""
Extract the following detailed information from the provided CV text and translate extracted values into English.

You MUST output ONLY valid JSON (no markdown, no extra text).

CRITICAL RULE: Categories must be STRICTLY SEPARATED ("airtight buckets").
- Do NOT copy the same content across multiple fields.
- Do NOT derive or infer items from other categories.

STANDARDIZED SKILLS REFERENCE (use these exact terms when extracting technologies):
Programming: python, java, c++, c#, javascript, typescript, go, rust, php, ruby, scala, kotlin, swift, objective-c, perl, r, matlab, lua, bash, shell, powershell, groovy, clojure, haskell, elixir
Web: django, flask, fastapi, react, angular, vue.js, node.js, express.js, spring, spring boot, asp.net, laravel, ruby on rails, next.js, nuxt.js, svelte, ember.js, backbone.js, meteor
Databases: sql, postgresql, mysql, mongodb, elasticsearch, cassandra, redis, dynamodb, oracle, sql server, mariadb, firebase, neo4j, couchdb, influxdb, cockroachdb
Cloud/DevOps: aws, azure, gcp, docker, kubernetes, terraform, ansible, jenkins, gitlab ci, github actions, bitbucket, circleci, travis ci, heroku, netlify, cloudformation, sam, serverless, lambda, ec2, s3, ecs, eks
Data/ML: machine learning, deep learning, neural networks, tensorflow, pytorch, scikit-learn, pandas, numpy, apache spark, hadoop, kafka, airflow, bigquery, snowflake, tableau, power bi, looker, dbt, data analytics
DevOps/Infrastructure: linux, windows, macos, nginx, apache, ssl, tls, vpn, firewall, monitoring, prometheus, grafana, elk stack, datadog, new relic
Tools: git, svn, jira, confluence, trello, slack, asana
APIs: rest, graphql, grpc, soap, mqtt, amqp, rabbitmq, activemq
Testing: junit, pytest, mocha, jasmine, selenium, cypress, testng, cucumber, postman, loadrunner, jmeter
Mobile: ios, android, flutter, react native, xamarin, cordova

Allowed content per category:
- experiences: ONLY concrete professional/academic experiences (missions/projects/roles). Each experience must be an actual described experience.
  - It may include tools/stack ONLY if explicitly written as used in that specific experience (not global skills).
- hard_skills / technologies: ONLY explicit skills/technologies from the STANDARDIZED SKILLS list above. Use exact standardized names.
  - DO NOT paste experience descriptions here.
- soft_skills: ONLY explicit behavioral/soft skills (e.g., "communication", "leadership").
  - DO NOT infer soft skills from role descriptions.
- certifications: ONLY explicit certification names (e.g., "PMP", "Salesforce Certified Associate", "AWS SAA").
  - DO NOT include trainings, courses, or degrees unless clearly stated as a certification.
- secteur_principal / secteur: ONLY the industry/domain (e.g., banking, insurance, retail). Not tools.
- langues: ONLY languages.

NO INFERENCE:
- If not explicitly stated in the text, output null (for scalars) or [] (for lists).

DEDUPLICATION:
- Do not repeat the same token across lists unless it is truly a different item (e.g., "AWS" vs "AWS SAA").

Output the result as a single JSON object with the following schema:
{{
  "nom": "Full name of the person",
  "role_principal": "Main role or title",
  "seniorite": "Seniority level (one of: Junior, Senior, Manager) or years of experience if present",
  "secteur_principal": "Main industry sector(s), comma-separated",
  "technologies": "Key tools/tech from STANDARDIZED SKILLS list, comma-separated",
  "langues": "Languages, comma-separated",
  "certifications": ["List of certifications (e.g., Salesforce Certified Associate, PMP, AWS, Azure...)"],
  "hard_skills": ["List of hard skills / technologies as items from STANDARDIZED SKILLS"],
  "soft_skills": ["List of soft skills as items"],
  "experiences": [
    {{
      "mission": "Short mission title/summary",
      "secteur": "Domain/industry if stated",
      "stack": ["Tech stack used on that mission"],
      "duree": "Duration if stated"
    }}
  ],
  "cv_text": "The CV main text, focusing on experience and key skills."
}}
"""

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
                            "STANDARDIZED SKILLS REFERENCE (use these exact terms for technical skills):\n"
                            "Programming: python, java, c++, c#, javascript, typescript, go, rust, php, ruby, scala, kotlin, swift, objective-c, perl, r, matlab, lua, bash, shell, powershell, groovy, clojure, haskell, elixir\n"
                            "Web: django, flask, fastapi, react, angular, vue.js, node.js, express.js, spring, spring boot, asp.net, laravel, ruby on rails, next.js, nuxt.js, svelte, ember.js, backbone.js, meteor\n"
                            "Databases: sql, postgresql, mysql, mongodb, elasticsearch, cassandra, redis, dynamodb, oracle, sql server, mariadb, firebase, neo4j, couchdb, influxdb, cockroachdb\n"
                            "Cloud/DevOps: aws, azure, gcp, docker, kubernetes, terraform, ansible, jenkins, gitlab ci, github actions, bitbucket, circleci, travis ci, heroku, netlify, cloudformation, sam, serverless, lambda, ec2, s3, ecs, eks\n"
                            "Data/ML: machine learning, deep learning, neural networks, tensorflow, pytorch, scikit-learn, pandas, numpy, apache spark, hadoop, kafka, airflow, bigquery, snowflake, tableau, power bi, looker, dbt, data analytics\n"
                            "DevOps/Infrastructure: linux, windows, macos, nginx, apache, ssl, tls, vpn, firewall, monitoring, prometheus, grafana, elk stack, datadog, new relic\n"
                            "Tools: git, svn, jira, confluence, trello, slack, asana\n"
                            "APIs: rest, graphql, grpc, soap, mqtt, amqp, rabbitmq, activemq\n"
                            "Testing: junit, pytest, mocha, jasmine, selenium, cypress, testng, cucumber, postman, loadrunner, jmeter\n"
                            "Mobile: ios, android, flutter, react native, xamarin, cordova\n\n"
                            "{\n"
                            '  "titre_poste": "Job title or main role for the mission",\n'
                            '  "contexte_mission": "Brief summary of the mission context and objectives",\n'
                            '  "competences_techniques": ["List of required technical skills from STANDARDIZED SKILLS list"],\n'
                            '  "competences_metier": ["List of business/soft skills expected"],\n'
                            '  "secteur": "Industry domain if stated (banking, insurance, public sector, etc.)",\n'
                            '  "experience_requise": "Required seniority level (one of: Junior, Senior, Manager) or years of experience",\n'
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
                            
                            # Use enhanced scoring with non-NLP features
                            scores, method = score_blocks_enhanced(
                                ao_blocks, cv_blocks, ao_struct or {}, cv_struct
                            )
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
                                # New enhanced scoring fields
                                "nlp_score": scores["nlp_score"],
                                "skills_score": scores["skills_score"],
                                "seniority_score": scores["seniority_score"],
                                "domain_score": scores["domain_score"],
                                "language_score": scores["language_score"],
                                "global_score": scores["global_score"],
                                "skill_details": scores["skill_details"],
                                "ao_seniority": scores.get("ao_seniority", ""),
                                "cv_seniority": scores.get("cv_seniority", ""),
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

                    st.caption(f"Méthode embeddings : **{method}** (cosine similarity avec features structurelles).")
                    st.info("Matching basé sur : **NLP (50%)** + **Skills (20%)** + **Seniority (15%)** + **Domain (10%)** + **Language (5%)**.")

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
                            # Score breakdown (6 columns)
                            c1, c2, c3, c4, c5, c6 = st.columns(6)
                            with c1:
                                st.metric("NLP", f"{row['nlp_score']:.3f}")
                            with c2:
                                st.metric("Skills", f"{row['skills_score']:.3f}")
                            with c3:
                                st.metric("Seniority", f"{row['seniority_score']:.3f}")
                            with c4:
                                st.metric("Domain", f"{row['domain_score']:.3f}")
                            with c5:
                                st.metric("Language", f"{row['language_score']:.3f}")
                            with c6:
                                st.metric("Global", f"{row['global_score']:.3f}")

                            st.plotly_chart(
                                create_radar_chart(
                                    row["nlp_score"],
                                    row["skills_score"],
                                    row["seniority_score"],
                                    row["domain_score"],
                                    row["language_score"],
                                    row["global_score"],
                                ),
                                use_container_width=True
                            )

                            # Skill Analysis
                            skill_details = row.get("skill_details", {})
                            if skill_details:
                                st.markdown("**🎯 Skill Analysis**")
                                skill_cols = st.columns([2, 1, 1])
                                
                                with skill_cols[0]:
                                    st.write(f"**Overlap Ratio**: {skill_details.get('overlap_ratio', 0):.1%}")
                                    if skill_details.get("overlap"):
                                        st.write(f"✓ Found: {', '.join(skill_details['overlap'][:5])}" + 
                                                ("..." if len(skill_details['overlap']) > 5 else ""))
                                
                                with skill_cols[1]:
                                    st.write(f"**Required**: {len(skill_details.get('ao_skills', []))}")
                                
                                with skill_cols[2]:
                                    st.write(f"**Has**: {len(skill_details.get('cv_skills', []))}")
                                
                                if skill_details.get("missing"):
                                    st.warning(f"⚠️ Missing skills: {', '.join(skill_details['missing'][:5])}" +
                                             ("..." if len(skill_details['missing']) > 5 else ""))

                            # Seniority & Domain
                            st.markdown("**📋 Seniority & Domain**")
                            sren_cols = st.columns([1, 1, 1])
                            with sren_cols[0]:
                                st.write(f"**AO Seniority**: {row['ao_seniority'] or '—'}")
                            with sren_cols[1]:
                                st.write(f"**CV Seniority**: {row['cv_seniority'] or '—'}")
                            with sren_cols[2]:
                                st.write(f"**Seniority Score**: {row['seniority_score']:.3f}")

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
                                
                                # Use new scoring fields for explanation payload
                                score_payload = {
                                    "nlp_score": float(row["nlp_score"]),
                                    "skills_score": float(row["skills_score"]),
                                    "seniority_score": float(row["seniority_score"]),
                                    "domain_score": float(row["domain_score"]),
                                    "language_score": float(row["language_score"]),
                                    "global_score": float(row["global_score"]),
                                    "skill_details": row.get("skill_details", {}),
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

import json
import streamlit as st
from src.db import connect, init_db, TABLE_NAME

with st.expander("🔍 Debug – View Structured CV JSON in DB"):

    conn = connect()
    init_db(conn)  # important: garantit que la table existe

    c = conn.cursor()
    c.execute(f"SELECT cv_id, filename, cv_struct_json FROM {TABLE_NAME}")
    rows = c.fetchall()

    if not rows:
        st.info("No CVs in database.")
    else:
        for cv_id, filename, cv_struct_json in rows:
            st.markdown(f"### 📄 {filename} (ID: {cv_id})")

            try:
                data = json.loads(cv_struct_json) if cv_struct_json else None
                st.json(data)
            except Exception as e:
                st.error(f"JSON parsing error: {e}")
                st.text(cv_struct_json)

    conn.close()

