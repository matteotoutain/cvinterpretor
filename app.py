import io
import json
import zipfile
import datetime
from typing import Any, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import APP_NAME
from src.db import connect, init_db, upsert_cv, list_cvs, get_cv_texts, delete_cv
from src.extract import extract_text_generic, stable_id_from_bytes
from src.nlp import build_ao_blocks, build_cv_blocks, score_blocks_enhanced, verdict_from_score
from src.mistral_client import (
    call_mistral_json_extraction,
    call_mistral_json_explanation,
    DEFAULT_MODEL,
)

LOGO_URL = "https://www.centrale-mediterranee.fr/themes/custom/central_mars_theme/logo.svg"
PRIMARY_BLUE = "#0B3D91"
SECONDARY_BLUE = "#1D6FD8"
LIGHT_BLUE = "#EAF2FF"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top right, rgba(29,111,216,0.10), transparent 26%),
                    linear-gradient(180deg, #f7faff 0%, #ffffff 35%, #f8fbff 100%);
            }}
            .main .block-container {{
                max-width: 1380px;
                padding-top: 1.4rem;
                padding-bottom: 3rem;
            }}
            .hero-card {{
                background: linear-gradient(135deg, rgba(11,61,145,0.98), rgba(29,111,216,0.90));
                border-radius: 22px;
                padding: 1.5rem 1.6rem;
                color: white;
                box-shadow: 0 14px 36px rgba(11, 61, 145, 0.18);
                border: 1px solid rgba(255,255,255,0.12);
                margin-bottom: 1rem;
            }}
            .hero-title {{
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
                margin: 0 0 .35rem 0;
            }}
            .hero-subtitle {{
                font-size: 1rem;
                opacity: .96;
                margin: 0;
            }}
            .section-card {{
                background: rgba(255,255,255,0.92);
                border: 1px solid rgba(11,61,145,0.08);
                border-radius: 18px;
                padding: 1rem 1rem 0.75rem 1rem;
                box-shadow: 0 10px 24px rgba(11,61,145,0.06);
                margin-bottom: 1rem;
            }}
            .mini-card {{
                background: {LIGHT_BLUE};
                border: 1px solid rgba(11,61,145,0.10);
                border-radius: 16px;
                padding: 0.9rem 1rem;
                min-height: 110px;
            }}
            .mini-card-title {{
                font-size: 0.85rem;
                font-weight: 700;
                color: {PRIMARY_BLUE};
                margin-bottom: .2rem;
                text-transform: uppercase;
                letter-spacing: .02em;
            }}
            .mini-card-text {{
                font-size: 0.95rem;
                color: #18324f;
                margin: 0;
            }}
            .badge-row {{
                display: flex;
                gap: .5rem;
                flex-wrap: wrap;
                margin-top: .75rem;
            }}
            .badge {{
                background: rgba(255,255,255,0.18);
                border: 1px solid rgba(255,255,255,0.18);
                color: white;
                border-radius: 999px;
                padding: .35rem .7rem;
                font-size: .84rem;
                font-weight: 600;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: .35rem;
            }}
            .stTabs [data-baseweb="tab"] {{
                background: white;
                border-radius: 12px 12px 0 0;
                border: 1px solid rgba(11,61,145,0.10);
                padding: 0.65rem 1rem;
            }}
            .stTabs [aria-selected="true"] {{
                color: {PRIMARY_BLUE};
                font-weight: 700;
                box-shadow: inset 0 -3px 0 {SECONDARY_BLUE};
            }}
            div[data-testid="stMetric"] {{
                background: white;
                border: 1px solid rgba(11,61,145,0.10);
                border-radius: 16px;
                padding: 0.8rem;
                box-shadow: 0 8px 18px rgba(11,61,145,0.06);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    left, right = st.columns([4.5, 1.2])
    with left:
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-title">CV Interpretor</div>
                <p class="hero-subtitle">
                    Chargez vos CV et un appel d'offres. Nous structurons les données, scorons les correspondances et vous fournissons des recommandations.
                </p>
                <div class="badge-row">
                    <span class="badge">Extraction</span>
                    <span class="badge">Structuration IA</span>
                    <span class="badge">Scoring multi-critères</span>
                    <span class="badge">Explicabilité</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.image(LOGO_URL, use_container_width=True)


def render_tutorial() -> None:
    with st.expander("📘 Tutoriel d’utilisation et logique métier", expanded=False):
        st.markdown(
            """
            **1. Alimenter la base candidats**  
            Déposez un ou plusieurs CV. L’application extrait le texte, normalise le contenu, puis le transforme en données structurées : rôle principal, niveau de séniorité, secteur, technologies, certifications, langues et expériences clés.

            **2. Charger un appel d’offres**  
            Déposez ensuite un AO. Le moteur reconstruit le besoin métier : contexte de mission, compétences attendues, niveau d’expérience, secteur cible, langues et éventuelles certifications.

            **3. Lancer le matching**  
            Le moteur compare les blocs de l’AO et des CV sur plusieurs dimensions : proximité sémantique, adéquation technique, cohérence sectorielle et compatibilité de séniorité. Chaque profil reçoit un score global et une qualification de pertinence.

            **4. Lire la recommandation**  
            Les résultats affichent les meilleurs profils, les écarts éventuels, les points forts, ainsi qu’une justification exploitable pour un usage staffing / préqualification.
            """
        )


def info_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-card-title">{title}</div>
            <p class="mini-card-text">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_radar_chart(nlp_score: float, skills_score: float, domain_score: float, global_score: float):
    categories = ["NLP", "Skills", "Domain", "Global"]
    values = [nlp_score, skills_score, domain_score, global_score]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            line_color=PRIMARY_BLUE,
            fillcolor="rgba(29, 111, 216, 0.22)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=360,
        margin=dict(l=30, r=30, t=50, b=30),
        title=dict(text="Décomposition du score", x=0.5, xanchor="center"),
    )
    return fig


def build_cv_extraction_prompt() -> str:
    return """Extract the following detailed information from the provided CV text and translate extracted values into English.
You MUST output ONLY valid JSON (no markdown, no extra text).
CRITICAL RULE: Categories must be STRICTLY SEPARATED (\"airtight buckets\").
- Do NOT copy the same content across multiple fields.
- Do NOT derive or infer items from other categories.
Output the result as a single JSON object with the following schema:
{
  \"nom\": \"Full name of the person\",
  \"role_principal\": \"Main role or title\",
  \"seniorite\": \"Seniority level (one of: Junior, Senior, Manager) or years of experience if present\",
  \"secteur_principal\": \"Main industry sector(s), comma-separated\",
  \"technologies\": \"Key tools/tech, comma-separated\",
  \"langues\": \"Languages, comma-separated\",
  \"certifications\": [\"List of certifications\"],
  \"hard_skills\": [\"List of hard skills\"],
  \"soft_skills\": [\"List of soft skills\"],
  \"experiences\": [{\"mission\": \"summary\", \"secteur\": \"domain\", \"stack\": [\"stack\"], \"duree\": \"duration\"}],
  \"cv_text\": \"Main CV text\"
}"""


def build_ao_extraction_prompt() -> str:
    return (
        "Extract the following detailed information from the provided AO text, translated in english. "
        "Output the result as a single JSON object. If a field is not found, use `null`.\n\n"
        "{\n"
        '  "titre_poste": "Job title or main role for the mission",\n'
        '  "contexte_mission": "Brief summary of the mission context and objectives",\n'
        '  "competences_techniques": ["Required technical skills"],\n'
        '  "competences_metier": ["Business or soft skills expected"],\n'
        '  "secteur": "Industry domain",\n'
        '  "experience_requise": "Required seniority level or years of experience",\n'
        '  "langues_requises": ["Languages required"],\n'
        '  "certifications_requises": ["Certifications required or strongly desired"]\n'
        "}\n"
    )


def normalize_weights(nlp_weight: float, skills_weight: float, domain_weight: float) -> Tuple[float, float, float]:
    total = nlp_weight + skills_weight + domain_weight
    if total <= 0:
        return 0.55, 0.30, 0.15
    return nlp_weight / total, skills_weight / total, domain_weight / total


def safe_json_loads(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def make_zip_from_top_cvs(df_top: pd.DataFrame) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, (_, row) in enumerate(df_top.iterrows(), start=1):
            zip_file.writestr(f"{idx:02d}_{row['filename']}.txt", (row.get("cv_text") or "").encode("utf-8"))
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


st.set_page_config(page_title=APP_NAME, layout="wide")
inject_css()
conn = connect()
init_db(conn)
if "ao_analysis_results" not in st.session_state:
    st.session_state.ao_analysis_results = None

render_hero()
render_tutorial()
row1, row2, row3 = st.columns(3)
with row1:
    info_card("Base candidats", "Import batch, stockage local et catalogue de profils consultables.")
with row2:
    info_card("Analyse AO", "Lecture d’un besoin, structuration automatique et scoring multicritère.")
with row3:
    info_card("Explication métier", "Justification lisible pour appuyer une décision de staffing.")

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
cv_count = len(list_cvs(conn))
metric1, metric2, metric3 = st.columns(3)
metric1.metric("CV en base", cv_count)
metric2.metric("Formats acceptés", "pptx / pdf / docx / txt")
metric3.metric("Mode", "IA / NLP")
st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs(["📥 Import des CV", "🎯 Analyse d’un AO", "🗂️ Bibliothèque de CV"])

with tabs[0]:
    st.subheader("Importer et structurer des CV")
    use_mistral = st.checkbox("Structurer automatiquement les CV avec l’IA", value=True)
    mistral_model = st.text_input("Modèle d’extraction CV", value=DEFAULT_MODEL, disabled=not use_mistral)
    uploaded_files = st.file_uploader("Déposez les CV à intégrer", type=["pptx", "pdf", "docx", "txt"], accept_multiple_files=True)
    start_import = st.button("Ajouter les CV à la base", type="primary", use_container_width=True)

    if uploaded_files and start_import:
        ok_count, ko_count = 0, 0
        with st.status("Traitement des documents", expanded=True) as status:
            for f in uploaded_files:
                try:
                    raw_bytes = f.getvalue()
                    cv_id = stable_id_from_bytes(raw_bytes)
                    doc_type, text = extract_text_generic(f.name, raw_bytes)
                    text = (text or "").strip()
                    if not text:
                        raise ValueError("Le document ne contient pas de texte exploitable après extraction.")
                    row = {
                        "cv_id": cv_id, "filename": f.name, "nom": None, "role_principal": None,
                        "seniorite": None, "secteur_principal": None, "technologies": None,
                        "langues": None, "cv_text": text, "cv_struct_json": None,
                    }
                    if use_mistral:
                        extracted = call_mistral_json_extraction(text_input=text, user_prompt=build_cv_extraction_prompt(), mistral_model=mistral_model)
                        if extracted and not extracted.get("error"):
                            for k in ["nom", "role_principal", "seniorite", "secteur_principal", "technologies", "langues", "cv_text"]:
                                if k in extracted and extracted[k] is not None:
                                    row[k] = extracted[k]
                            row["cv_struct_json"] = json.dumps(extracted, ensure_ascii=False)
                    upsert_cv(conn, row)
                    st.write(f"✅ {f.name} intégré avec succès ({doc_type}).")
                    ok_count += 1
                except Exception as e:
                    st.write(f"❌ {f.name} non intégré : {e}")
                    ko_count += 1
            status.update(label=f"Import terminé — {ok_count} succès / {ko_count} échec(s)", state="complete")

    df_preview = list_cvs(conn)
    if df_preview.empty:
        st.info("Aucun CV en base pour le moment.")
    else:
        display_cols = [c for c in ["filename", "nom", "role_principal", "seniorite", "secteur_principal", "technologies", "langues"] if c in df_preview.columns]
        st.dataframe(df_preview[display_cols], use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Analyser un appel d’offres et recommander les meilleurs profils")
    use_mistral_ao = st.checkbox("Structurer automatiquement l’AO", value=True)
    mistral_model_ao = st.text_input("Modèle d’analyse AO", value=DEFAULT_MODEL, disabled=not use_mistral_ao)
    use_mistral_explain = st.checkbox("Afficher une explication IA par profil", value=True)
    mistral_model_explain = st.text_input("Modèle d’explication", value=DEFAULT_MODEL, disabled=not use_mistral_explain)
    ao_file = st.file_uploader("Déposez l’appel d’offres", type=["pdf", "docx", "txt", "pptx"], accept_multiple_files=False)
    top_k = st.slider("Nombre de profils à remonter", 3, 30, 10)
    coef1, coef2, coef3 = st.columns(3)
    with coef1:
        nlp_weight = st.slider("Poids sémantique (NLP)", 0.0, 1.0, 0.55, 0.05)
    with coef2:
        skills_weight = st.slider("Poids compétences", 0.0, 1.0, 0.30, 0.05)
    with coef3:
        domain_weight = st.slider("Poids secteur / domaine", 0.0, 1.0, 0.15, 0.05)

    nlp_w_display, skills_w_display, domain_w_display = normalize_weights(nlp_weight, skills_weight, domain_weight)

    if ao_file is not None:
        ao_bytes = ao_file.getvalue()
        ao_type, ao_text = extract_text_generic(ao_file.name, ao_bytes)
        ao_text = (ao_text or "").strip()
        if ao_text:
            st.success(f"AO chargé avec succès ({ao_type}).")
            with st.expander("Voir un extrait du document"):
                st.write(ao_text[:2500] + ("…" if len(ao_text) > 2500 else ""))
            launch_analysis = st.button("Lancer l’analyse", type="primary", use_container_width=True)
            if launch_analysis:
                with st.status("Analyse en cours", expanded=True) as status:
                    ao_struct = None
                    if use_mistral_ao:
                        ao_struct = call_mistral_json_extraction(text_input=ao_text, user_prompt=build_ao_extraction_prompt(), mistral_model=mistral_model_ao)
                    cvs = get_cv_texts(conn)
                    if cvs.empty:
                        st.warning("Aucun CV n’est disponible en base. Importez d’abord des profils.")
                        status.update(label="Analyse interrompue", state="error")
                    else:
                        ao_blocks = build_ao_blocks(ao_struct if isinstance(ao_struct, dict) and not ao_struct.get("error") else {}, ao_fallback_text=ao_text)
                        results_rows = []
                        method_used = None
                        for _, r in cvs.iterrows():
                            cv_text = r.get("cv_text") or ""
                            cv_struct = safe_json_loads(r.get("cv_struct_json"))
                            cv_blocks = build_cv_blocks(cv_struct, cv_fallback_text=cv_text)
                            scores, method = score_blocks_enhanced(ao_blocks, cv_blocks, ao_struct or {}, cv_struct, nlp_weight=nlp_weight, skills_weight=skills_weight, domain_weight=domain_weight)
                            method_used = method_used or method
                            results_rows.append({
                                "cv_id": r["cv_id"], "filename": r["filename"], "nom": r.get("nom"), "role_principal": r.get("role_principal"),
                                "seniorite": r.get("seniorite"), "secteur_principal": r.get("secteur_principal"), "technologies": r.get("technologies"),
                                "langues": r.get("langues"), "cv_text": cv_text, "cv_struct_json": r.get("cv_struct_json"),
                                "nlp_score": scores["nlp_score"], "skills_score": scores["skills_score"], "seniority_score": scores["seniority_score"],
                                "domain_score": scores["domain_score"], "global_score": scores["global_score"], "skill_details": scores.get("skill_details", {}),
                                "ao_seniority": scores.get("ao_seniority", ""), "cv_seniority": scores.get("cv_seniority", ""),
                                "verdict": verdict_from_score(scores["global_score"]),
                            })
                        st.session_state.ao_analysis_results = {
                            "ao_struct": ao_struct,
                            "cvs": pd.DataFrame(results_rows).sort_values("global_score", ascending=False).head(int(top_k)).reset_index(drop=True),
                            "method": method_used or "unknown",
                            "weights": {"nlp": nlp_w_display, "skills": skills_w_display, "domain": domain_w_display},
                        }
                        status.update(label="Analyse terminée", state="complete")

    if st.session_state.ao_analysis_results:
        results = st.session_state.ao_analysis_results
        cvs = results["cvs"].copy()
        ao_struct = results["ao_struct"]
        weights = results["weights"]
        sum1, sum2, sum3, sum4 = st.columns(4)
        sum1.metric("Profils classés", len(cvs))
        sum2.metric("Score moyen", f"{cvs['global_score'].mean():.3f}")
        sum3.metric("Meilleur score", f"{cvs['global_score'].max():.3f}")
        sum4.metric("Moteur", results["method"])
        if ao_struct and isinstance(ao_struct, dict) and not ao_struct.get("error"):
            with st.expander("Résumé structuré du besoin"):
                st.json(ao_struct)
        st.info(f"Scoring combiné : NLP {weights['nlp']:.0%} · Skills {weights['skills']:.0%} · Domaine {weights['domain']:.0%}.")
        seniority_filter = st.selectbox("Filtrer les résultats par séniorité détectée", ["Tous", "Junior", "Senior", "Manager"])
        filtered_cvs = cvs.copy()
        if seniority_filter != "Tous":
            filtered_cvs = filtered_cvs[filtered_cvs["cv_seniority"].astype(str).str.contains(seniority_filter, case=False, na=False)]
            if filtered_cvs.empty:
                st.warning("Aucun profil sur ce filtre. Affichage complet rétabli.")
                filtered_cvs = cvs.copy()
        num_to_download = st.slider("Nombre de CV à inclure dans l’archive", 1, min(10, len(filtered_cvs)), min(3, len(filtered_cvs)))
        st.download_button(label=f"Télécharger les {num_to_download} meilleurs CV (ZIP)", data=make_zip_from_top_cvs(filtered_cvs.head(num_to_download)), file_name=f"best_cvs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")
        st.dataframe(filtered_cvs[["filename", "nom", "role_principal", "secteur_principal", "global_score", "verdict"]].sort_values("global_score", ascending=False), use_container_width=True, hide_index=True)
        for idx, (_, row) in enumerate(filtered_cvs.iterrows(), start=1):
            with st.expander(f"{idx}. {row['filename']} · score {row['global_score']:.3f} · {row['verdict']}", expanded=(idx <= 3)):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("NLP", f"{row['nlp_score']:.3f}")
                c2.metric("Skills", f"{row['skills_score']:.3f}")
                c3.metric("Domaine", f"{row['domain_score']:.3f}")
                c4.metric("Global", f"{row['global_score']:.3f}")
                st.plotly_chart(create_radar_chart(row["nlp_score"], row["skills_score"], row["domain_score"], row["global_score"]), use_container_width=True)
                st.write(f"**Séniorité AO** : {row['ao_seniority'] or '—'}")
                st.write(f"**Séniorité CV** : {row['cv_seniority'] or '—'}")
                st.write(f"**Rôle détecté** : {row.get('role_principal') or '—'}")
                st.write(f"**Secteur** : {row.get('secteur_principal') or '—'}")
                preview_text = (row.get("cv_text") or "")[:1600] + ("…" if len(row.get("cv_text") or "") > 1600 else "")
                st.text_area("Aperçu du CV", preview_text, height=180, disabled=True, key=f"preview_{row['cv_id']}")
                if use_mistral_explain:
                    explain = call_mistral_json_explanation(
                        ao_struct=ao_struct if isinstance(ao_struct, dict) and not ao_struct.get("error") else {},
                        cv_struct=safe_json_loads(row.get("cv_struct_json")),
                        scores={
                            "nlp_score": float(row["nlp_score"]), "skills_score": float(row["skills_score"]), "seniority_score": float(row["seniority_score"]),
                            "domain_score": float(row["domain_score"]), "global_score": float(row["global_score"]), "skill_details": row.get("skill_details", {}), "verdict": row["verdict"],
                        },
                        mistral_model=mistral_model_explain,
                    )
                    if explain and not explain.get("error"):
                        st.info(f"**NLP** : {explain.get('nlp_why', '—')}")
                        st.info(f"**Skills** : {explain.get('skills_why', '—')}")
                        st.info(f"**Domaine** : {explain.get('domain_why', '—')}")
                        st.success(f"**Overall fit** : {explain.get('overall_fit', 'Unknown')}")
                        st.write(f"**Point fort clé** : {explain.get('key_strength', '—')}")
                        gap = explain.get("key_gap", "—")
                        st.write(f"**Principal écart** : {gap if gap != 'None' else 'Aucun écart majeur détecté'}")

with tabs[2]:
    st.subheader("Bibliothèque de CV")
    df_all = list_cvs(conn)
    df_all_with_text = get_cv_texts(conn)
    if df_all.empty:
        st.info("La base est vide. Importez des CV depuis l’onglet précédent.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CV au total", len(df_all))
        m2.metric("Avec nom détecté", int(df_all["nom"].notna().sum()) if "nom" in df_all else 0)
        m3.metric("Avec structuration IA", int(df_all_with_text["cv_struct_json"].notna().sum()) if "cv_struct_json" in df_all_with_text else 0)
        m4.metric("Catalogue actif", len(df_all))
        search_term = st.text_input("Rechercher par nom de fichier ou nom de personne")
        show_full = st.checkbox("Afficher le texte complet")
        df_filtered = df_all.copy()
        if search_term.strip():
            s = search_term.lower()
            df_filtered = df_filtered[(df_filtered["filename"].astype(str).str.lower().str.contains(s, na=False)) | (df_filtered["nom"].fillna("").astype(str).str.lower().str.contains(s, na=False))]
        st.dataframe(df_filtered[[c for c in ["filename", "nom", "role_principal", "seniorite", "secteur_principal", "technologies", "langues"] if c in df_filtered.columns]], use_container_width=True, hide_index=True)
        selected_filename = st.selectbox("Sélectionner un CV pour voir le détail", df_filtered["filename"].tolist() if len(df_filtered) > 0 else ["— Aucun CV —"])
        if selected_filename != "— Aucun CV —" and selected_filename in df_filtered["filename"].values:
            selected_row = df_all_with_text[df_all_with_text["filename"] == selected_filename].iloc[0]
            st.write(f"**Nom** : {selected_row.get('nom') or '—'}")
            st.write(f"**Rôle** : {selected_row.get('role_principal') or '—'}")
            st.write(f"**Séniorité** : {selected_row.get('seniorite') or '—'}")
            st.write(f"**Secteur** : {selected_row.get('secteur_principal') or '—'}")
            st.write(f"**Technologies** : {selected_row.get('technologies') or '—'}")
            st.write(f"**Langues** : {selected_row.get('langues') or '—'}")
            cv_text = selected_row.get("cv_text") or "Contenu non disponible"
            st.download_button(label="Télécharger le contenu texte", data=cv_text.encode("utf-8"), file_name=f"{selected_filename}.txt", mime="text/plain", use_container_width=True)
            if st.button("Supprimer ce CV", type="secondary", use_container_width=True):
                delete_cv(conn, selected_row["cv_id"])
                st.success(f"{selected_filename} a été supprimé de la base.")
                st.rerun()
            preview = cv_text if show_full else cv_text[:2000] + ("…" if len(cv_text) > 2000 else "")
            st.text_area("Contenu", preview, height=300 if show_full else 180, disabled=True)
            structured = safe_json_loads(selected_row.get("cv_struct_json"))
            if structured:
                with st.expander("Voir la structuration IA du CV"):
                    st.json(structured)
        if st.button("Vider entièrement la base", type="secondary", use_container_width=True):
            if st.session_state.get("confirm_clear", False):
                for _, r in df_all.iterrows():
                    delete_cv(conn, r["cv_id"])
                st.session_state.confirm_clear = False
                st.success("La base candidats a été vidée.")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.error("Cliquez une seconde fois pour confirmer la suppression complète.")
