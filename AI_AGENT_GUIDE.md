# 🤖 AI Agent Quick Reference

**If you're an AI agent working on CV Interpretor, read this first:**

## Critical Patterns You MUST Know

### 1. **Database DataFrame Pattern** ⭐⭐⭐
```python
# FAST - for tables & filtering
df_all = list_cvs(conn)  
# Contains: cv_id, filename, nom, role, seniorite, secteur, tech, langues
# Missing: cv_text

# FULL - for detail views & downloads  
df_all_with_text = get_cv_texts(conn)
# Same fields + cv_text (the full CV document)
```

**Rule:** Use `list_cvs()` for UI tables. Use `get_cv_texts()` ONLY when viewing/downloading details.

### 2. **Session State for Persistent Analysis** ⭐⭐⭐
```python
# Initialize once per session
if "ao_analysis_results" not in st.session_state:
    st.session_state.ao_analysis_results = None

# Store results after "Lance l'analyse" button click
st.session_state.ao_analysis_results = {
    "ao_struct": ...,
    "cvs": ...,
    "method": ...,
    "ao_text": ...,
}

# Display results without recalculating
if st.session_state.ao_analysis_results:
    results = st.session_state.ao_analysis_results
    # Show results...
```

**Why:** Streamlit reruns the entire script on every interaction. Session state persists data across reruns, preventing expensive recalculations.

### 3. **Tab 3 KeyError Prevention** ⭐⭐⭐
```python
# ✅ CORRECT - Load full data upfront
df_all_with_text = get_cv_texts(conn)

# ✅ CORRECT - Fetch selected row from full dataset
selected_row = df_all_with_text[df_all_with_text["filename"] == selected].iloc[0]

# ❌ WRONG - This would fail
selected_row = df_filtered[df_filtered["filename"] == selected].iloc[0]  # No cv_text!
cv_text = selected_row["cv_text"]  # KeyError!
```

**Lesson:** Always fetch detailed data from the FULL dataset, not the filtered one.

---

## The 3 Tabs Explained

### Tab 1: Import CVs
- Upload CVs (PPTX/PDF/DOCX/TXT)
- Extract text with `extract_text_generic()`
- Optional Mistral enrichment (nom, role, tech, etc.)
- Upsert to SQLite with `upsert_cv()`
- **Pattern:** Batch processing with error handling

### Tab 2: Analyze AO & Match
- Upload Job Offer (AO)
- Extract text + optional Mistral structuring
- **🚀 "Lance l'analyse" button** → stores in session state
- Compute similarity (Semantic or TF-IDF fallback)
- **📥 ZIP download** for top candidates
- **Pattern:** On-demand analysis with persistent results

### Tab 3: Manage Database
- 📊 KPI Dashboard (stats)
- 🔍 Search & filter CVs
- 📋 List all CVs with metadata
- 🔎 Detail view (full CV + download/delete)
- ⚠️ Bulk actions
- **Pattern:** Professional database UI

---

## What This Project Does NOT Have (Yet)

These are TODOs - don't assume they exist:

- ❌ **Skill extraction:** `extract_skills()` is stubbed (returns empty)
- ❌ **AO persistence:** AOs are ephemeral, never stored
- ❌ **Multi-AO analysis:** Only one AO at a time
- ❌ **Database indexes:** Performance will degrade with 1000+ CVs
- ❌ **CSV/PDF export:** Results only display in UI

---

## Mistral Integration Rules

When using `call_mistral_json_extraction()`:

```python
extracted = call_mistral_json_extraction(
    text_input=document_text,
    user_prompt="Extract [fields] as JSON object...",
    mistral_model="mistral-medium-latest"  # configurable
)

# Returns:
# ✅ {"nom": "...", "role": "..."}  on success
# ❌ {"error": "API failed"}         on failure

if extracted and not extracted.get("error"):
    # Use extracted data
else:
    # Graceful fallback - continue without enrichment
```

**Pattern:** Always check for `{"error": ...}` dict. Never raise exceptions.

---

## French UI Convention

**IMPORTANT:** All user-facing text is in FRENCH. Keep it that way!

```python
# ✅ DO THIS
st.write("🚀 Lance l'analyse")
st.markdown("### 📊 Vue d'ensemble")
st.error("❌ Erreur : document vide")

# ❌ DON'T MIX LANGUAGES
st.write("🚀 Launch analysis")  # Wrong!
```

---

## Performance Tips

- **Text extraction:** Can take 5-10 seconds for 100+ page PDFs (normal)
- **Semantic similarity:** ~5-10s on first run (downloads 100MB model), then fast
- **TF-IDF fallback:** ~1s, always available (no external model)
- **ZIP creation:** <1s even for 100 CVs

---

## Common Gotchas

1. **PPTX encoding:** Some files have encoding issues → fallback to TXT
2. **Session state persistence:** Exists only for current browser session (not persistent across restarts)
3. **Mistral rate limits:** No retry logic → add if scaling
4. **Large CV bases:** TF-IDF max 30,000 features → may hit memory limit with 10,000+ CVs
5. **Duplicate detection:** Based on content HASH, not filename → same CV with different name detected

---

## Files You'll Touch Most

- **`app.py`** (463 lines) - All UI logic
- **`src/db.py`** - Database queries
- **`src/nlp.py`** - Similarity matching
- **`src/extract.py`** - File text extraction
- **`src/mistral_client.py`** - LLM API calls

---

**Last Updated:** 24 Jan 2026  
**Difficulty Level for AI:** Medium (Streamlit state management + SQL patterns)
