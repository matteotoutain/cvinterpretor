from pathlib import Path

APP_NAME = "CV Interpretor (PoC)"
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

DB_PATH = DATA_DIR / "cv_database.db"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

