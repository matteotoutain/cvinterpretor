# CV ↔ AO Matcher (Streamlit PoC)

PoC minimaliste centré sur **2 blocs** :
1. **Batch CV Import** : upload de CVs → extraction texte → stockage en **SQLite local**
2. **AO Import & NLP Analysis** : upload d’un AO → matching (AO non stocké) → résultats + explications

## Formats supportés
- CV : `.pptx` (priorité), `.pdf`, `.docx`, `.txt`
- AO : `.pdf`, `.docx`, `.txt`, `.pptx`

## Comment ça marche (v1)
- **Stockage CV** : SQLite `data/cv_database.db`, table `cv_profiles`
- **Matching** : cosine similarity
