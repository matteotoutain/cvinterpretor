# CV ↔ AO Matcher (Streamlit PoC)

PoC minimaliste centré sur **2 blocs** :
1. **Batch CV Import** : upload de CVs → extraction texte → stockage en **SQLite local**
2. **AO Import & NLP Analysis** : upload d’un AO → matching à la volée (AO non stocké) → résultats + explications

## Lancer en local

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

## Formats supportés
- CV : `.pptx` (priorité), `.pdf`, `.docx`, `.txt`
- AO : `.pdf`, `.docx`, `.txt`, `.pptx`

## Comment ça marche (v1)
- **Stockage CV** : SQLite `data/cv_database.db`, table `cv_profiles`
- **Matching** : cosine similarity
  - tente `sentence-transformers/all-MiniLM-L6-v2` si dispo
  - fallback TF-IDF si le modèle n’est pas téléchargeable
- **Explication** : overlap de compétences via un petit dictionnaire (`src/config.py → SKILL_TOKENS`)

## Extensions faciles (si tu veux level-up)
- Ajouter une étape d’extraction structurée (nom / rôle / seniorité / tech) via LLM **ou** règles + NER
- Enrichir le dico skills (ou un référentiel type ESCO)
- Ajouter un “why” plus solide : passages CV les plus proches de l’AO, citations, etc.
