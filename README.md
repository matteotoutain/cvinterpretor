# CV ↔ AO Matcher (Streamlit PoC)

PoC minimaliste centré sur **2 blocs** :
1. **Batch CV Import** : upload de CVs → extraction texte → stockage en **SQLite local**
2. **AO Import & NLP Analysis** : upload d’un AO → matching à la volée (AO non stocké) → résultats + explications

## Formats supportés
- CV : `.pptx` (priorité), `.pdf`, `.docx`, `.txt`
- AO : `.pdf`, `.docx`, `.txt`, `.pptx`

## Comment ça marche (v1)
- **Stockage CV** : SQLite `data/cv_database.db`, table `cv_profiles`
- **Matching** : cosine similarity
  - tente `sentence-transformers/all-MiniLM-L6-v2` si dispo
  - fallback TF-IDF si le modèle n’est pas téléchargeable
