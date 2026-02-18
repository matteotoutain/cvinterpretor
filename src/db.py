# db.py

import sqlite3

DB_PATH = "cv_database.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS cv_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nom TEXT,
        seniorite TEXT,
        tech_skills TEXT,
        domain_knowledge TEXT,
        certifications TEXT,
        cv_text TEXT,
        keyword_json TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_cv(nom, seniorite, tech, domain, certifs, cv_text, keyword_json):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    INSERT INTO cv_profiles
    (nom, seniorite, tech_skills, domain_knowledge, certifications, cv_text, keyword_json)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (nom, seniorite, tech, domain, certifs, cv_text, keyword_json))

    conn.commit()
    conn.close()


def get_all_cvs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM cv_profiles")
    rows = c.fetchall()
    conn.close()
    return rows
