import streamlit as st
import pandas as pd

from src.config import APP_NAME
from src.db import connect, init_db, upsert_cv, list_cvs, get_cv_texts, delete_cv
from src.extract import extract_text_generic, stable_id_from_bytes
from src.nlp import compute_similarity, explain_match

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
    files = st.file_uploader("Dépose tes CVs ici", type=["pptx","pdf","docx","txt"], accept_multiple_files=True)

    colA, colB = st.columns([1,1])
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
                    }
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

    ao_file = st.file_uploader("Dépose ton AO", type=["pdf","docx","txt","pptx"], accept_multiple_files=False)
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

            cvs = get_cv_texts(conn)
            if cvs.empty:
                st.warning("Aucun CV en base. Reviens à l’étape 1.")
            else:
                scores, method = compute_similarity(ao_text, cvs["cv_text"].tolist())
                cvs = cvs.copy()
                cvs["score"] = scores
                cvs = cvs.sort_values("score", ascending=False).head(int(top_k))

                st.caption(f"Méthode de matching : **{method}** (cosine similarity).")
                st.subheader("Résultats")
                for _, row in cvs.iterrows():
                    title = f"{row['filename']} — score {row['score']:.3f}"
                    with st.expander(title, expanded=False):
                        expl = explain_match(ao_text, row["cv_text"])
                        c1, c2 = st.columns([1,1])
                        with c1:
                            st.markdown("**Ce qui colle (skills en commun)**")
                            if expl["overlap"]:
                                st.write(", ".join(expl["overlap"]))
                            else:
                                st.write("Aucun overlap détecté (dico skills minimal).")
                        with c2:
                            st.markdown("**Ce qui manque côté CV (skills AO non vus)**")
                            if expl["missing"]:
                                st.write(", ".join(expl["missing"][:40]) + ("…" if len(expl["missing"])>40 else ""))
                            else:
                                st.write("Rien de critique détecté.")

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
