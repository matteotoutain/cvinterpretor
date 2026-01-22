from pathlib import Path

APP_NAME = "CV ↔ AO Matcher (PoC)"
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

DB_PATH = DATA_DIR / "cv_database.db"

# Small pragmatic skill dictionary (extend freely)
SKILL_TOKENS = [
    # Data / dev
    "python","pandas","numpy","scikit-learn","sklearn","pytorch","tensorflow","spark","databricks",
    "sql","postgres","mysql","bigquery","snowflake",
    "ml","machine learning","nlp","llm","rag","embeddings",
    "docker","kubernetes","git","ci/cd","linux",
    # Cloud / enterprise
    "salesforce","data cloud","marketing cloud","service cloud","crm","apex","lwc","soql","flow",
    "aws","azure","gcp",
    # PM / business
    "agile","scrum","product","stakeholder","workshop","requirements","specifications",
    # Languages
    "français","anglais","italien","japonais",
]
