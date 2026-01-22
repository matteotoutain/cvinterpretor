import streamlit as st
import pandas as pd

from src.config import APP_NAME
from src.db import connect, init_db, upsert_cv, list_cvs, get_cv_texts, delete_cv
from src.extract import extract_text_generic, stable_id_from_bytes
from src.nlp import compute_similarity, explain_match

# NEW: Mistral (as in your notebook style)
from src.mistral_client import call_mistral_json_extraction, DEFAULT_MODEL

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
    st.markdown("### Options d’extraction (Mistral)")
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
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------
# 2) AO Import & NLP Analysis
# ---------------------------
with tabs[1]:
    st.subheader("Étape 2 — Uploader un Appel d’Offre → Analyse NLP → Matching + explication")
    st.write("Ici l’AO n’est **pas** stocké : on calcule à la volée.")

    # NEW: Mistral options (AO)
    st.markdown("### Options d’analyse AO (Mistral)")
    use_mistral_ao = st.checkbox("Utiliser Mistral pour structurer l’AO (résumé + exigences)", value=True)
    mistral_model_ao = st.text_input("Modèle Mistral (AO)", value=DEFAULT_MODEL, disabled=not use_mistral_ao)

    ao_file = st.file_uploader("Dépose ton AO", type=["pdf", "docx", "txt", "pptx"], accept_multiple_files=False)
    top_k = st.slider("Nombre de profils à afficher", 3, 30, 10)

    if ao_file:
        try:
            ao_bytes = ao_file.getvalue()
            ao_type, ao_text = extract_text_generic(ao_file.name, ao_bytes)
            ao_text = (ao_text or "").strip()
            if not ao_text:
                raise ValueError("AO vide après extraction")

            st.success(f"AO chargé ({ao_type}). Longueur texte: {len(ao_text):,} caractères.")
            with st.expander("Voir l’extrait de l’AO"):
                st.write(ao_text[:2500] + ("…" if len(ao_text) > 2500 else ""))

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

                if ao_struct and not ao_struct.get("error"):
                    st.subheader("AO — résumé structuré (Mistral)")
                    st.json(ao_struct)
                else:
                    st.write(
                        f"⚠️ Mistral AO extraction failed: {ao_struct.get('error') if ao_struct else 'Unknown error'}"
                    )
                    ao_struct = None

            cvs = get_cv_texts(conn)
            if cvs.empty:
                st.warning("Aucun CV en base. Reviens à l’étape 1.")
            else:
                scores, method = compute_similarity(ao_text, cvs["cv_text"].tolist())
                cvs = cvs.copy()
                cvs["score"] = scores
                cvs = cvs.sort_values("score", ascending=False).head(int(top_k))

                st.caption(f"Méthode de matching : **{method}** (cosine similarity).")

                # Optional: show AO fields as a short "what we're matching against"
                if ao_struct and not ao_struct.get("error"):
                    st.info(
                        "Matching basé sur le texte AO complet. Le résumé structuré sert à mieux expliquer les résultats."
                    )

                st.subheader("Résultats")
                for _, row in cvs.iterrows():
                    title = f"{row['filename']} — score {row['score']:.3f}"
                    with st.expander(title, expanded=False):
                        expl = explain_match(ao_text, row["cv_text"])
                        st.markdown("**Extrait CV (début)**")
                        st.write((row["cv_text"] or "")[:1500] + ("…" if len(row["cv_text"] or "") > 1500 else ""))

        except Exception as e:
            st.error(str(e))

# ---------------------------
# 3) Manage DB
# ---------------------------
with tabs[2]:
    st.subheader("Base locale — supprimer un CV si besoin")
    df = list_cvs(conn)
    st.dataframe(df, use_container_width=True, hide_index=True)

    cv_id = st.text_input("cv_id à supprimer (copie/colle depuis le tableau)")
    if st.button("Supprimer", type="secondary", disabled=not cv_id):
        try:
            delete_cv(conn, cv_id.strip())
            st.success("Supprimé.")
            st.experimental_rerun()
        except Exception as e:
            st.error(str(e))
