import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# Comprehensive skill dictionary (normalized skills)
SKILL_DICT = {
    # ---------------------------------------------------------
    # 1. Programming & Scripting
    # ---------------------------------------------------------
    "python": ["python", "py", "pandas", "numpy"], # Souvent implicite via les libs
    "r": [r"\br\b", "r programming", "rstudio"],
    "sql": ["sql", "t-sql", "pl/sql", "plsql", "ansi sql"],
    "scala": ["scala"],
    "java": ["java", "jvm"],
    "csharp": ["c#", "csharp", ".net", "dotnet"],
    "cpp": ["c++", "cpp"],
    "go": [r"\bgo\b", "golang"],
    "rust": ["rust"],
    "julia": ["julia"], # Niche mais présent en optimisation/maths
    "sas": [r"\bsas\b", "sas base", "sas eg", "sas enterprise guide"], # Encore très fort en banque/assurance
    "matlab": ["matlab"],
    "bash": ["bash", "shell", "scripting shell", "sh"],
    "vba": ["vba", "visual basic"], # Souvent demandé pour migration legacy

    # ---------------------------------------------------------
    # 2. Web & App Development (Fullstack context)
    # ---------------------------------------------------------
    "javascript": ["javascript", "js", "node", "nodejs"],
    "typescript": ["typescript", "ts"],
    "html_css": ["html", "css", "html5", "css3"],
    "react": ["react", "reactjs", "react.js"],
    "angular": ["angular", "angularjs"],
    "vue": ["vue", "vuejs"],
    "flask": ["flask"],
    "django": ["django"],
    "fastapi": ["fastapi"],
    "spring": ["spring", "spring boot"],

    # ---------------------------------------------------------
    # 3. Cloud Providers (CSP)
    # ---------------------------------------------------------
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "sage maker", "sagemaker"],
    "azure": ["azure", "microsoft azure", "azure data factory", "adf", "synapse"],
    "gcp": ["gcp", "google cloud", "google cloud platform", "bigquery", "vertex ai"],
    "ovh": ["ovh", "ovhcloud"], # Spécifique marché FR/EU
    "scaleway": ["scaleway"],   # Spécifique marché FR

    # ---------------------------------------------------------
    # 4. Big Data & Distributed Computing
    # ---------------------------------------------------------
    "spark": ["spark", "pyspark", "apache spark", "sparksql"],
    "databricks": ["databricks", "delta lake"],
    "hadoop": ["hadoop", "hdfs", "mapreduce"],
    "hive": ["hive", "hiveql"],
    "kafka": ["kafka", "confluent"],
    "flink": ["flink", "apache flink"],
    "snowflake": ["snowflake", "snowsql"],
    "presto_trino": ["presto", "trino"],

    # ---------------------------------------------------------
    # 5. Data Engineering & ETL/ELT
    # ---------------------------------------------------------
    "dbt": [r"\bdbt\b", "data build tool"],
    "airflow": ["airflow", "apache airflow", "cloud composer", "mwaa"],
    "talend": ["talend", "talend open studio"],
    "informatica": ["informatica", "powercenter"],
    "ssis": ["ssis", "sql server integration services"],
    "fivetran": ["fivetran"],
    "matillion": ["matillion"],
    "glue": ["aws glue", "glue"],
    "nifi": ["nifi", "apache nifi"],

    # ---------------------------------------------------------
    # 6. Databases (SQL, NoSQL, NewSQL)
    # ---------------------------------------------------------
    "postgresql": ["postgresql", "postgres", "pgadmin"],
    "mysql": ["mysql", "mariadb"],
    "oracle": ["oracle db", "oracle database"],
    "sql_server": ["sql server", "mssql", "ms sql"],
    "mongodb": ["mongodb", "mongo"],
    "cassandra": ["cassandra"],
    "elasticsearch": ["elasticsearch", "elk", "elastic stack"],
    "redis": ["redis"],
    "dynamodb": ["dynamodb"],
    "cosmosdb": ["cosmosdb"],
    "teradata": ["teradata"],
    "redshift": ["redshift"],
    "bigquery": ["bigquery"],

    # ---------------------------------------------------------
    # 7. Data Science & Machine Learning (Classic)
    # ---------------------------------------------------------
    "scikit-learn": ["scikit-learn", "sklearn"],
    "tensorflow": ["tensorflow", "tf", "tfx"],
    "pytorch": ["pytorch", "torch"],
    "keras": ["keras"],
    "xgboost": ["xgboost", "xgb"],
    "lightgbm": ["lightgbm", "lgbm"],
    "catboost": ["catboost"],
    "statsmodels": ["statsmodels"],
    "optimization": ["cplex", "gurobi", "linear programming", "operations research", "roadef"],
    "spacy": ["spacy", "nltk"],

    # ---------------------------------------------------------
    # 8. GenAI & LLM (Tendance actuelle forte)
    # ---------------------------------------------------------
    "llm": ["llm", "large language model", "gpt", "llama", "mistral"],
    "openai": ["openai", "chatgpt", "gpt-4"],
    "huggingface": ["hugging face", "huggingface", "transformers"],
    "langchain": ["langchain"],
    "rag": [r"\brag\b", "retrieval augmented generation"],
    "vector_db": ["pinecone", "milvus", "weaviate", "qdrant", "chroma", "vector database"],
    "prompt_engineering": ["prompt engineering", "prompting"],

    # ---------------------------------------------------------
    # 9. MLOps & Industrialization
    # ---------------------------------------------------------
    "mlflow": ["mlflow"],
    "kubeflow": ["kubeflow"],
    "docker": ["docker", "container"],
    "kubernetes": ["kubernetes", "k8s", "aks", "eks", "gke"],
    "ci_cd": ["ci/cd", "cicd", "jenkins", "gitlab ci", "github actions", "circleci"],
    "terraform": ["terraform", "iac", "infrastructure as code"],
    "ansible": ["ansible"],
    "git": ["git", "github", "gitlab", "bitbucket"],

    # ---------------------------------------------------------
    # 10. BI & Data Visualization
    # ---------------------------------------------------------
    "power_bi": ["power bi", "powerbi", "dax", "power query"],
    "tableau": ["tableau", "tableau desktop", "tableau server"],
    "qlik": ["qlik", "qliksense", "qlikview"],
    "looker": ["looker", "lookml"],
    "streamlit": ["streamlit"],
    "dash": ["plotly dash", "dash"],
    "excel": ["excel", "spreadsheet"],
    "microstrategy": ["microstrategy"],
    "grafana": ["grafana"], # Souvent pour le monitoring technique

    # ---------------------------------------------------------
    # 11. Project Management & Methodologies
    # ---------------------------------------------------------
    "agile": ["agile", "scrum", "kanban", "sprint"],
    "safe": [r"\bsafe\b", "scaled agile framework"], # Très fréquent grands comptes
    "jira": ["jira", "atlassian"],
    "confluence": ["confluence"],
    "devops": ["devops"],
    "finops": ["finops"], # Gestion des coûts cloud
    "dataops": ["dataops"],

    # ---------------------------------------------------------
    # 12. Soft Skills & Roles (Contextes AO Français)
    # ---------------------------------------------------------
    "english": ["anglais", "english", "bilingue"],
    "french": ["français", "french"],
    "communication": ["communication", "vulgarisation", "storytelling"],
    "leadership": ["leadership", "management", "encadrement"],
    "problem_solving": ["problem solving", "résolution de problèmes"],
    "amoa": ["amoa", "maîtrise d'ouvrage", "product owner", "po", "business analyst", "ba"],
    "moe": ["moe", "maîtrise d'oeuvre"],
    
    # ---------------------------------------------------------
    # 13. Governance, Compliance & Architecture
    # ---------------------------------------------------------
    "gdpr": ["gdpr", "rgpd", "cnil", "données personnelles"],
    "data_governance": ["data governance", "gouvernance des données", "data catalog", "data quality", "collibra", "atlan"],
    "data_mesh": ["data mesh"],
    "data_fabric": ["data fabric"],
    "togaf": ["togaf", "architecture d'entreprise"], # Architecture
    "iso27001": ["iso 27001", "secnumcloud"], # Sécurité
}


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_skills(text: str) -> List[str]:
    """Extract skills from text using skill dictionary."""
    t = normalize_ws(text).lower()
    found = set()
    
    # Search for each skill and its aliases
    for skill_name, aliases in SKILL_DICT.items():
        for alias in aliases:
            # Use word boundaries for more accurate matching
            if alias.startswith("^"):
                # Regex pattern
                if re.search(alias, t):
                    found.add(skill_name)
            else:
                # Word boundary matching
                if re.search(r"\b" + re.escape(alias) + r"\b", t):
                    found.add(skill_name)
    
    return sorted(list(found))


def _embed_sentence_transformers(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)


def _embed_tfidf(texts: List[str]) -> np.ndarray:
    # TF-IDF embedding; cheap and no heavy downloads
    vec = TfidfVectorizer(stop_words="english", max_features=30000)
    X = vec.fit_transform(texts)
    # return dense array for cosine
    return X.toarray()


def compute_similarity(query_text: str, documents: List[str]) -> Tuple[np.ndarray, str]:
    """Return (scores in [0..1], method)."""
    texts = [query_text] + documents
    if _HAS_ST:
        try:
            emb = _embed_sentence_transformers(texts)
            q = emb[0:1]
            d = emb[1:]
            scores = cosine_similarity(q, d)[0]
            return scores.astype(float), "sentence-transformers (all-MiniLM-L6-v2)"
        except Exception:
            pass
    emb = _embed_tfidf(texts)
    q = emb[0:1]
    d = emb[1:]
    scores = cosine_similarity(q, d)[0]
    return scores.astype(float), "tf-idf"


def explain_match(
    ao_text: str,
    cv_text: str,
    ao_seniority: Optional[str] = None,
    cv_seniority: Optional[str] = None,
) -> Dict:
    """
    Compare skills and seniority in AO vs CV.
    
    Returns:
        Dict with keys: ao_skills, cv_skills, overlap, missing,
                       skill_overlap_ratio (0-1), seniority_match_score (0-1)
    """
    ao_sk = set(extract_skills(ao_text))
    cv_sk = set(extract_skills(cv_text))
    overlap = sorted(list(ao_sk & cv_sk))
    missing = sorted(list(ao_sk - cv_sk))
    
    # Calculate skill overlap ratio
    total_ao_skills = max(len(ao_sk), 1)
    skill_overlap_ratio = len(overlap) / total_ao_skills
    
    # Calculate seniority match score
    seniority_match_score = _calculate_seniority_match(ao_seniority, cv_seniority)
    
    return {
        "ao_skills": sorted(list(ao_sk)),
        "cv_skills": sorted(list(cv_sk)),
        "overlap": overlap,
        "missing": missing,
        "skill_overlap_ratio": skill_overlap_ratio,
        "ao_seniority": ao_seniority,
        "cv_seniority": cv_seniority,
        "seniority_match_score": seniority_match_score,
    }


def _calculate_seniority_match(
    ao_seniority: Optional[str],
    cv_seniority: Optional[str]
) -> float:
    """
    Calculate seniority match score (0-1).
    
    Seniority levels: Junior (0-2 yrs), Confirmé (2-5 yrs), Senior (5-10 yrs), Expert (10+ yrs)
    """
    seniority_values = {
        "junior": 1,
        "confirmé": 2,
        "senior": 3,
        "expert": 4,
        "0": 0.5, "1": 1, "2": 1, "3": 2, "4": 2, "5": 3, "10": 4,
    }
    
    def parse_seniority(s: Optional[str]) -> Optional[int]:
        if not s:
            return None
        s_lower = s.lower().strip()
        for key, val in seniority_values.items():
            if key in s_lower:
                return val
        return None
    
    ao_val = parse_seniority(ao_seniority)
    cv_val = parse_seniority(cv_seniority)
    
    if ao_val is None or cv_val is None:
        # If either is missing, neutral score
        return 0.5
    
    # Perfect match = 1.0, one level off = 0.75, two levels = 0.5, three+ = 0.25
    diff = abs(ao_val - cv_val)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.75
    elif diff == 2:
        return 0.5
    else:
        return 0.25
