import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import DB_PATH


TABLE_NAME = "cv_profiles"


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def init_db(conn: sqlite3.Connection) -> None:
    sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        cv_id TEXT PRIMARY KEY,
        filename TEXT,
        nom TEXT,
        role_principal TEXT,
        seniorite TEXT,
        secteur_principal TEXT,
        technologies TEXT,
        langues TEXT,
        cv_text TEXT
    );
    """
    conn.execute(sql)
    conn.commit()


def upsert_cv(conn: sqlite3.Connection, row: Dict) -> None:
    cols = ["cv_id","filename","nom","role_principal","seniorite","secteur_principal","technologies","langues","cv_text"]
    values = [row.get(c) for c in cols]
    placeholders = ",".join(["?"]*len(cols))
    sql = f"""
    INSERT INTO {TABLE_NAME} ({",".join(cols)})
    VALUES ({placeholders})
    ON CONFLICT(cv_id) DO UPDATE SET
        filename=excluded.filename,
        nom=excluded.nom,
        role_principal=excluded.role_principal,
        seniorite=excluded.seniorite,
        secteur_principal=excluded.secteur_principal,
        technologies=excluded.technologies,
        langues=excluded.langues,
        cv_text=excluded.cv_text;
    """
    conn.execute(sql, values)
    conn.commit()


def list_cvs(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT cv_id, filename, nom, role_principal, seniorite, secteur_principal, technologies, langues FROM {TABLE_NAME} ORDER BY filename", conn)


def get_cv_texts(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT cv_id, filename, nom, role_principal, seniorite, secteur_principal, technologies, langues, cv_text FROM {TABLE_NAME}", conn)


def delete_cv(conn: sqlite3.Connection, cv_id: str) -> None:
    conn.execute(f"DELETE FROM {TABLE_NAME} WHERE cv_id = ?", (cv_id,))
    conn.commit()
