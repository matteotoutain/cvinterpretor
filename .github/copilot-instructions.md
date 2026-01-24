# CV Interpretor - AI Coding Agent Instructions

## Project Overview

**CV Interpretor** is a Streamlit-based PoC (Proof of Concept) for matching CVs against Job Offers (AOs - Appels d'Offre). The architecture follows a two-phase workflow: persistent CV storage with optional LLM enrichment, and ephemeral AO analysis with on-the-fly matching.

### Core Architecture

```
app.py (Streamlit UI - 3 tabs)
├── Tab 1: Batch CV Import → extract text + Mistral enrichment → SQLite upsert
├── Tab 2: AO Analysis & Matching
│   ├── Upload AO → extract + optional Mistral structuring
│   ├── 🚀 "Lance l'analyse" button (on-demand, session-state persistent)
│   ├── Compute cosine similarity (semantic + TF-IDF fallback)
│   └── 📥 Download top N candidates as ZIP
└── Tab 3: DB Management (Professional-grade)
    ├── 📊 KPI Dashboard (total, enriched, searchable)
    ├── 🔍 Search & filter by name/filename
    ├── 📋 List with all metadata columns
    ├── 🔎 Detail view with download/delete actions
    └── ⚠️ Bulk actions (clear DB with confirmation)

Data Flow:
1. CV Upload → extract_text_generic() → optional Mistral JSON extraction → upsert_cv()
2. AO Upload → extract_text_generic() → optional Mistral structuring → compute_similarity()
3. Similarity Scores + Skill Overlap → explain_match() (currently stubbed)
4. Session State Persistence: ao_analysis_results stored across UI interactions
```

## Key Components & Responsibilities

### `src/extract.py` - Multi-format Document Parser
Handles text extraction from `.pptx`, `.pdf`, `.docx`, `.txt` with deterministic content hashing:
- **Extraction priority**: PPTX > PDF > DOCX > TXT (PPTX prioritized for CVs)
- **stable_id_from_bytes()**: SHA256 hash (first 16 chars) prevents duplicate uploads
- Each format parser normalizes whitespace and joins multi-element content (slides/pages/paragraphs)

**Pattern**: Use `extract_text_generic(filename, file_bytes)` for any document upload; it returns `(doc_type, text)`.

### `src/db.py` - SQLite Persistence Layer
Single table `cv_profiles` with columns: `cv_id` (PK), `filename`, and enrichment fields (`nom`, `role_principal`, `seniorite`, `secteur_principal`, `technologies`, `langues`, `cv_text`).

**Key operations**:
- `list_cvs()`: Returns DataFrame WITHOUT `cv_text` (lightweight, for tables/filtering)
- `get_cv_texts()`: Returns DataFrame WITH full `cv_text` (for detail views & downloads)
- `upsert_cv()`: Idempotent insert/update by `cv_id` (prevents duplicates)
- `delete_cv()`: Remove single CV
- `init_db()`: Lazy table creation on first connection

**Important Pattern**: Always use `list_cvs()` for UI lists, `get_cv_texts()` only when needing full text (performance-critical distinction)

### `src/mistral_client.py` - LLM Integration (Optional)
Uses Mistral API (via `mistralai` SDK) with JSON extraction mode:
- **Environment**: Requires `MISTRAL_API_KEY` in `.env`
- **Default model**: `mistral-medium-latest` (configurable per tab)
- **Pattern**: Call `call_mistral_json_extraction(text_input, user_prompt, mistral_model)` with system prompt "expert at extracting structured information from text" and `response_format={"type": "json_object"}`
- **Fallback**: Returns `{"error": "..."}` on API failure; UI gracefully skips enrichment

**Enrichment Fields**:
- **CV extraction**: nom, role_principal, seniorite, secteur_principal, technologies, langues
- **AO extraction**: titre_poste, contexte_mission, competences_techniques, competences_metier, experience_requise, langues_requises

### `src/nlp.py` - Similarity & Explanation Engine
Two matching strategies with automatic fallback:

**compute_similarity(query, documents)**: 
- **Primary**: Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`) - semantic embeddings, requires download (~100MB), may fail offline
- **Fallback**: TF-IDF vectorization - stateless, no external models, ~95% accuracy vs semantic for this domain
- Returns `(cosine_scores, method_name)` where scores ∈ [0,1]

**explain_match(ao_text, cv_text)**:
- Currently relies on `extract_skills()` which is **stubbed out** (returns empty list)
- Compares skill sets: `{"overlap": [...], "missing": [...], "ao_skills": [...], "cv_skills": [...]}`
- **TODO**: Implement skill extraction via regex pattern matching or Mistral (see README)

### `src/config.py` - Static Configuration
Minimal config: paths (`BASE_DIR`, `DATA_DIR`, `DB_PATH`), app name. No runtime settings.

## Developer Workflows

### Local Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
echo MISTRAL_API_KEY=<your_api_key> > .env
streamlit run app.py
```

### Running the App
```bash
streamlit run app.py
# Opens http://localhost:8501
```

### Testing Patterns
- **Manual testing**: Upload 3-5 sample CVs to verify extraction quality
- **Regression risk**: Mistral API changes or downtime; always test fallback (TF-IDF matching still works)
- **Database state**: `.env` controls API key; `data/cv_database.db` is local and persistent

## Project-Specific Conventions

### Naming & Structure
- **French UI labels**: Tab titles, field prompts, and messages are French (cf. app.py). Keep UI French for consistency.
- **Column naming**: Snake case with French roots (e.g., `role_principal`, `seniorite`, `competences_techniques`)
- **Error handling**: Streamlit UI catches exceptions; return error dicts from Mistral (not raising), allow TF-IDF to silently fallback

### Streamlit UI Patterns (app.py)

**Tab 2 - AO Analysis (Lines ~140-310):**
- Launch button triggers analysis → stores results in `st.session_state.ao_analysis_results` 
- Results persist across sidebar changes (no recalc when filtering)
- ZIP download: `st.download_button()` with `zipfile.ZipFile()` for batch export
- Result display: expanders per candidate with metadata table + score metric + skill analysis

**Tab 3 - Database Management (Lines ~318-463):**
- `df_all = list_cvs(conn)` for stats/table display
- `df_all_with_text = get_cv_texts(conn)` loaded upfront for details view (KEY FIX: avoids KeyError)
- Search filter: case-insensitive substring match on filename + nom fields
- Detail selector: dropdown populates from filtered results
- Selected row fetched from `df_all_with_text` (not `df_filtered`) to access `cv_text`
- Bulk delete: confirm pattern with `st.session_state.confirm_clear` (double-click safety)

**Key Import Additions** (for new features):
```python
import io, zipfile, datetime  # Tab 2 download feature
```

### Design Patterns
1. **Graceful degradation**: No Mistral API → skip enrichment (UI shows warning, CV still imported)
2. **Idempotent upserts**: Re-upload same CV → updates if changed, no duplicates
3. **Lazy initialization**: DB table created on first connection, not at import time
4. **On-the-fly AO processing**: AOs never persisted; ephemeral analysis only
5. **Session state for analysis persistence**: `st.session_state.ao_analysis_results` stores matching results across UI interactions (avoids recalc on filter changes)
6. **Dual DataFrame loading**: `list_cvs()` for UI, `get_cv_texts()` for details (performance optimization)

### Mistral JSON Extraction Protocol
Always include: system prompt (expert framing) + user prompt (detailed field spec) + `response_format={"type": "json_object"}` + `temperature=0.2` (deterministic). Return plain dict on success or `{"error": "..."}` on parse failure.

## Integration Points & Dependencies

- **Streamlit**: UI framework; leverage `st.expander`, `st.status`, `st.file_uploader` for UX
- **sqlite3**: Standard library, no ORM; use parameterized queries (SQL injection prevention)
- **Mistral API**: External dependency; handle API key missing/invalid gracefully
- **Sentence Transformers**: Optional (~100MB model); project supports offline fallback
- **python-pptx, pdfplumber, python-docx**: Format-specific parsers; each has unique edge cases (see `extract.py`)

## Common Pitfalls & Bug Prevention

### 1. DataFrame Column Access (Tab 3 Bug - FIXED)
**Problem:** `list_cvs()` has NO `cv_text` → accessing it raises `KeyError`  
**Solution:** Load `df_all_with_text = get_cv_texts(conn)` upfront, use it for details view  
**Location:** app.py lines 325, 387 (shows correct pattern)

### 2. Session State Initialization
**Pattern:** Always check before use:
```python
if "ao_analysis_results" not in st.session_state:
    st.session_state.ao_analysis_results = None
```
**Why:** Streamlit reruns script on every interaction; state persists but needs initialization

### 3. File Upload Encoding Issues
**Risk:** PPTX/PDF encoding errors silently fail  
**Mitigation:** `extract_text_generic()` returns `(doc_type, text)`; always validate text not empty  
**Example:** Tab 1 line ~58 checks `if not text: raise ValueError("Document vide après extraction")`

## Common Extension Points

1. **Skill enrichment** (`extract_skills()`): Implement regex-based skill matching or ESCO vocabulary mapping
2. **Structured extraction**: Enhance Mistral prompts for experience years, location, salary bands
3. **Database indexing**: Add `CREATE INDEX` on `technologies`, `secteur_principal` for large-scale queries
4. **Persistent AO cache**: Store AO + extractions to enable historical comparison
5. **Multi-AO batch analysis**: Extend Tab 2 to accept multiple AOs with aggregated matching

## Recent Changes & Bug Fixes

### KeyError: 'cv_text' Fix (24 Jan 2026)
**Issue:** Tab 3 detail view crashed when accessing `cv_text` from filtered results  
**Root:** `list_cvs()` used for filtering; didn't include `cv_text` column  
**Fix:** Load `df_all_with_text = get_cv_texts(conn)` at Tab 3 start; fetch selected row from this full dataset  
**Files:** app.py lines 325 (load) + 387 (fetch row)

### Tab 2 & Tab 3 UI Enhancements (24 Jan 2026)
- ✅ "Lance l'analyse" button with session state persistence
- ✅ ZIP download for top candidates
- ✅ Professional DB management (stats, search, bulk actions)
- ✅ Dual DataFrame pattern for performance

## Gotchas & Tips

- **Large documents**: Text extraction can be slow (5-10s for 100+ page PDFs); Streamlit UI remains responsive
- **Unicode in PPTX**: Some PPTX files have encoding issues; fallback to TXT if extraction fails
- **Mistral rate limits**: No retry logic; consider adding exponential backoff if scaling
- **TF-IDF dimensions**: Max 30,000 features; very large CV bases may hit memory limits
- **Duplicate CV detection**: Based on content hash, not filename; same CV with different names detected correctly
- **Session state scope**: Per-browser session; not shared across tabs or users (single-user PoC limitation)
