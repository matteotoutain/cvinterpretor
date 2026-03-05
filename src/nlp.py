import re
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# =========================
# Embedding model (E5)
# =========================
E5_MODEL_NAME = "BAAI/bge-m3"

#E5_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# =========================
# Standardized Skill Dictionary (Common Base for NLP + LLM)
# =========================
STANDARDIZED_SKILLS = {
    # Programming Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust", "php",
    "ruby", "scala", "kotlin", "swift", "objective-c", "perl", "r", "matlab", "lua",
    "bash", "shell", "powershell", "groovy", "clojure", "haskell", "elixir",

    # Web Frameworks
    "django", "flask", "fastapi", "react", "angular", "vue.js", "node.js", "express.js",
    "spring", "spring boot", "asp.net", "laravel", "ruby on rails", "next.js", "nuxt.js",
    "svelte", "ember.js", "backbone.js", "meteor",

    # Databases
    "sql", "postgresql", "mysql", "mongodb", "elasticsearch", "cassandra", "redis",
    "dynamodb", "oracle", "sql server", "mariadb", "firebase", "neo4j", "couchdb",
    "influxdb", "cockroachdb",

    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "gitlab ci", "github actions", "bitbucket", "circleci", "travis ci", "heroku", "netlify",
    "cloudformation", "sam", "serverless", "lambda", "ec2", "s3", "ecs", "eks",

    # Data & ML/AI
    "machine learning", "deep learning", "neural networks", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "apache spark", "hadoop", "kafka", "airflow",
    "bigquery", "snowflake", "tableau", "power bi", "looker", "dbt", "data analytics",

    # DevOps & Infrastructure
    "linux", "windows", "macos", "nginx", "apache", "ssl", "tls", "vpn", "firewall",
    "monitoring", "prometheus", "grafana", "elk stack", "datadog", "new relic",

    # Version Control & Tools
    "git", "svn", "jira", "confluence", "trello", "slack", "asana",

    # API & Messaging
    "rest", "graphql", "grpc", "soap", "mqtt", "amqp", "rabbitmq", "activemq",

    # Testing & QA
    "junit", "pytest", "mocha", "jasmine", "selenium", "cypress", "testng",
    "cucumber", "postman", "loadrunner", "jmeter",

    # Mobile Development
    "ios", "android", "flutter", "react native", "xamarin", "cordova",

    # Soft Skills (for completeness, though less relevant for technical matching)
    "leadership", "communication", "teamwork", "agile", "scrum", "kanban",
    "project management", "problem solving", "critical thinking", "creativity",
    "adaptability", "time management", "mentoring", "coaching",
}

# Legacy SKILL_DICT for backward compatibility (maps to standardized)
SKILL_DICT = STANDARDIZED_SKILLS

# Skill normalization mapping (common variations → standardized)
SKILL_NORMALIZATION = {
    # Programming languages
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "cpp": "c++",
    "csharp": "c#",
    "objectivec": "objective-c",
    "sh": "shell",

    # Frameworks
    "vue": "vue.js",
    "nodejs": "node.js",
    "express": "express.js",
    "springboot": "spring boot",
    "rails": "ruby on rails",
    "nextjs": "next.js",
    "nuxt": "nuxt.js",
    "ember": "ember.js",
    "backbone": "backbone.js",

    # Databases
    "mssql": "sql server",
    "postgres": "postgresql",

    # Cloud/DevOps
    "k8s": "kubernetes",
    "gitlab": "gitlab ci",
    "github": "github actions",
    "circle": "circleci",
    "travis": "travis ci",
    "cf": "cloudformation",
    "sam": "serverless application model",

    # Data/ML
    "ml": "machine learning",
    "dl": "deep learning",
    "spark": "apache spark",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
    "torch": "pytorch",
    "bi": "business intelligence",
    "elt": "data analytics",

    # DevOps
    "elk": "elk stack",
    "newrelic": "new relic",

    # Testing
    "test": "testing",
}


def extract_skills(text: str) -> List[str]:
    """
    Extract and normalize recognized skills from text using word-boundary matching.
    Returns standardized skill names in order of appearance.
    Case-insensitive, handles common variations/abbreviations.
    """
    if not text:
        return []

    text_lower = text.lower()
    found = []
    seen = set()

    # First pass: exact matches with standardized skills
    for skill in STANDARDIZED_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            if skill not in seen:
                found.append(skill)
                seen.add(skill)

    # Second pass: normalization mapping for common variations
    for variation, standard in SKILL_NORMALIZATION.items():
        if standard not in seen:  # Only add if we haven't already found the standard form
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, text_lower):
                if standard not in seen:
                    found.append(standard)
                    seen.add(standard)

    return found


def normalize_skill(skill: str) -> str:
    """
    Normalize a single skill to its standardized form.
    """
    if not skill:
        return ""

    skill_lower = skill.lower().strip()

    # Check if it's already standardized
    if skill_lower in STANDARDIZED_SKILLS:
        return skill_lower

    # Check normalization mapping
    return SKILL_NORMALIZATION.get(skill_lower, skill_lower)


def normalize_skill_list(skills: List[str]) -> List[str]:
    """
    Normalize a list of skills to standardized forms, removing duplicates.
    """
    if not skills:
        return []

    normalized = []
    seen = set()

    for skill in skills:
        std_skill = normalize_skill(skill)
        if std_skill and std_skill not in seen:
            normalized.append(std_skill)
            seen.add(std_skill)

    return normalized


def _parse_seniority(text: str) -> Optional[int]:
    """
    Parse seniority level from text (3 levels only).
    Returns: 0=Junior, 1=Senior, 2=Manager, None=Unknown
    Also recognizes 'X years' format.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Explicit level match
    if re.search(r'\b(manager|lead|head|director|principal)\b', text_lower):
        return 2
    if re.search(r'\b(senior|expert|confirmé?|confirmée?)\b', text_lower):
        return 1
    if re.search(r'\bjunior\b', text_lower):
        return 0
    
    # Years of experience match
    years_match = re.search(r'(\d+)\+?\s*(?:ans?|yea?rs?)', text_lower)
    if years_match:
        years = int(years_match.group(1))
        if years < 2:
            return 0  # Junior: < 2 years
        elif years < 5:
            return 1  # Senior: 2-5 years
        else:
            return 2  # Manager: 5+ years
    
    return None


def _calculate_seniority_score(ao_seniority: Optional[int], cv_seniority: Optional[int]) -> float:
    """
    Calculate seniority alignment score [0-1] for 3-level system.
    Exact match: 1.0 | One level off: 0.8 | Two levels off: 0.4
    If either is None: return 0.5 (neutral).
    """
    if ao_seniority is None or cv_seniority is None:
        return 0.5
    
    diff = abs(ao_seniority - cv_seniority)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.8
    else:  # diff >= 2 (max 2 in 3-level system)
        return 0.4


def extract_languages(text: str) -> List[str]:
    """
    Extract language names from text (English, French, Spanish, German, etc.).
    """
    if not text:
        return []
    
    languages = {
        "english", "french", "spanish", "german", "italian", "portuguese",
        "dutch", "russian", "chinese", "japanese", "korean", "arabic",
        "hebrew", "greek", "turkish", "polish", "czech", "hungarian",
        "romanian", "bulgarian", "swedish", "norwegian", "danish", "finnish",
        "thai", "vietnamese", "indonesian", "malay", "tagalog", "swahili",
    }
    
    text_lower = text.lower()
    found = []
    seen = set()
    
    for lang in languages:
        pattern = r'\b' + re.escape(lang) + r'\b'
        if re.search(pattern, text_lower):
            if lang not in seen:
                found.append(lang)
                seen.add(lang)
    
    return found


def _score_keyword_presence(required_text: str, candidate_text: str) -> float:
    """
    Score based on presence of keywords (not semantic).
    Extracts unique words from required_text and checks coverage in candidate_text.
    Returns score in [0-1].
    """
    if not required_text or not candidate_text:
        return 0.0
    
    # Extract keywords (words longer than 3 chars)
    required_words = set(
        w.lower() for w in re.findall(r'\b\w+\b', required_text)
        if len(w) > 3 and w.lower() not in {'that', 'this', 'with', 'from', 'able', 'have', 'will'}
    )
    
    if not required_words:
        return 0.5
    
    candidate_lower = candidate_text.lower()
    found_count = sum(1 for w in required_words if re.search(r'\b' + re.escape(w) + r'\b', candidate_lower))
    
    return min(1.0, found_count / len(required_words))


def _calculate_domain_match(ao_domain: str, cv_domain: str) -> float:
    """
    Simple domain/sector matching.
    Returns 1.0 for exact match, 0.3 for no match (non-zero for flexibility).
    """
    if not ao_domain or not cv_domain:
        return 0.5  # Neutral if missing
    
    ao_d = normalize_ws(ao_domain).lower()
    cv_d = normalize_ws(cv_domain).lower()
    
    if ao_d == cv_d:
        return 1.0
    
    # Check for overlap (e.g., "IT" in "IT Services")
    if ao_d in cv_d or cv_d in ao_d:
        return 0.75
    
    # Semantic similarity via compute_similarity
    scores, _ = compute_similarity(ao_d, [cv_d])
    domain_sim = float(scores[0])
    
    # Threshold: only count as match if > 0.5, else penalize
    return domain_sim if domain_sim > 0.5 else 0.3


def _embed_sentence_transformers(texts: List[str]) -> np.ndarray:
    """
    E5 best practice: prefix queries with 'query:' and documents with 'passage:'.
    Here we don't know which are which, so compute_similarity() will build that properly.
    """
    model = SentenceTransformer(E5_MODEL_NAME)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)


def _embed_tfidf(texts: List[str]) -> np.ndarray:
    vec = TfidfVectorizer(stop_words="english", max_features=30000)
    X = vec.fit_transform(texts)
    return X.toarray()


def compute_similarity(query_text: str, documents: List[str]) -> Tuple[np.ndarray, str]:
    """
    Return (scores in [0..1], method).
    """
    query_text = normalize_ws(query_text)
    documents = [normalize_ws(d) for d in documents]

    if _HAS_ST:
        try:
            # E5: query/passsage prefixes
            q_in = [f"query: {query_text}"]
            d_in = [f"passage: {d}" for d in documents]

            emb = _embed_sentence_transformers(q_in + d_in)
            q = emb[0:1]
            d = emb[1:]
            scores = cosine_similarity(q, d)[0]
            return scores.astype(float), f"sentence-transformers ({E5_MODEL_NAME})"
        except Exception:
            pass

    # fallback tf-idf
    texts = [query_text] + documents
    emb = _embed_tfidf(texts)
    q = emb[0:1]
    d = emb[1:]
    scores = cosine_similarity(q, d)[0]
    return scores.astype(float), "tf-idf"


# -----------------------------
# Block building from Mistral JSON
# -----------------------------
def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [normalize_ws(str(i)) for i in x if normalize_ws(str(i))]
    if isinstance(x, str):
        parts = [normalize_ws(p) for p in x.split(",")]
        return [p for p in parts if p]
    return [normalize_ws(str(x))]


def build_ao_blocks(ao_struct: Dict[str, Any], ao_fallback_text: str = "") -> Dict[str, str]:
    """
    Create textual blocks for AO.
    Keys must match build_cv_blocks.
    """
    title = normalize_ws(str(ao_struct.get("titre_poste") or ""))
    context = normalize_ws(str(ao_struct.get("contexte_mission") or ""))

    hard = ", ".join(_as_list(ao_struct.get("competences_techniques")))
    soft = ", ".join(_as_list(ao_struct.get("competences_metier")))
    domain = normalize_ws(str(ao_struct.get("secteur") or ao_struct.get("secteur_principal") or ""))
    exp = normalize_ws(str(ao_struct.get("experience_requise") or ""))
    langs = ", ".join(_as_list(ao_struct.get("langues_requises")))

    # NEW: certifications expected in AO (optional)
    certs_req = ", ".join(_as_list(ao_struct.get("certifications_requises")))

    if not (title or context or hard or soft or domain or exp or langs or certs_req):
        context = ao_fallback_text[:4000]

    return {
        "skills_like": normalize_ws(" ".join([title, hard])),
        "experience_like": normalize_ws(" ".join([context, exp])),
        "domain_like": normalize_ws(" ".join([domain, title])),
        # Replaces soft_like:
        "certification_like": normalize_ws(" ".join([certs_req, langs])),
        "full": normalize_ws(" ".join([title, context, hard, domain, exp, langs, certs_req])),
    }


def build_cv_blocks(cv_struct: Dict[str, Any], cv_fallback_text: str = "") -> Dict[str, str]:
    """
    Create textual blocks for CV.
    Expected fields are produced by your Mistral extraction.
    """
    role = normalize_ws(str(cv_struct.get("role_principal") or ""))
    seniority = normalize_ws(str(cv_struct.get("seniorite") or ""))
    sector = normalize_ws(str(cv_struct.get("secteur_principal") or ""))
    tech = ", ".join(_as_list(cv_struct.get("technologies")))
    langs = ", ".join(_as_list(cv_struct.get("langues")))

    hard_skills = ", ".join(_as_list(cv_struct.get("hard_skills")))
    soft_skills = ", ".join(_as_list(cv_struct.get("soft_skills")))

    # NEW: certifications field (list or comma-separated string)
    certifications = ", ".join(_as_list(cv_struct.get("certifications")))

    experiences_txt = ""
    exps = cv_struct.get("experiences")
    if isinstance(exps, list):
        chunks = []
        for e in exps[:10]:
            if isinstance(e, dict):
                mission = normalize_ws(str(e.get("mission") or e.get("title") or ""))
                stack = ", ".join(_as_list(e.get("stack")))
                dom = normalize_ws(str(e.get("secteur") or e.get("domain") or ""))
                dur = normalize_ws(str(e.get("duree") or e.get("duration") or ""))
                chunks.append(normalize_ws(" ".join([mission, stack, dom, dur])))
            else:
                chunks.append(normalize_ws(str(e)))
        experiences_txt = normalize_ws(" | ".join([c for c in chunks if c]))

    if not experiences_txt:
        experiences_txt = normalize_ws(cv_fallback_text[:2500])

    skills_like = normalize_ws(" ".join([role, tech, hard_skills]))
    domain_like = normalize_ws(" ".join([sector, role]))
    experience_like = normalize_ws(" ".join([experiences_txt, seniority]))

    # Replaces soft_like:
    certification_like = normalize_ws(" ".join([certifications, langs]))

    full = normalize_ws(" ".join([skills_like, experience_like, domain_like, certification_like]))

    return {
        "skills_like": skills_like,
        "experience_like": experience_like,
        "domain_like": domain_like,
        "certification_like": certification_like,
        "full": full,
    }


def score_blocks(
    ao_blocks: Dict[str, str],
    cv_blocks: Dict[str, str],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Compute similarity per block and weighted global score.
    """
    if weights is None:
        weights = {
            "skills_like": 0.40,
            "experience_like": 0.30,
            "domain_like": 0.20,
            "certification_like": 0.10,
        }

    method_used = None
    per_block = {}

    for k in ["skills_like", "experience_like", "domain_like", "certification_like"]:
        q = ao_blocks.get(k, "") or ""
        d = cv_blocks.get(k, "") or ""
        scores, method = compute_similarity(q, [d])
        method_used = method
        per_block[k] = float(scores[0])

    global_score = 0.0
    for k, w in weights.items():
        global_score += per_block.get(k, 0.0) * float(w)

    out = dict(per_block)
    out["global_score"] = float(global_score)

    return out, (method_used or "unknown")


def verdict_from_score(s: float) -> str:
    if s >= 0.75:
        return "Strong Fit"
    if s >= 0.55:
        return "Moderate Fit"
    return "Low Fit"


# =========================
# Enhanced Scoring (NLP + Non-NLP Features)
# =========================
def score_blocks_enhanced(
    ao_blocks: Dict[str, str],
    cv_blocks: Dict[str, str],
    ao_struct: Dict[str, Any] = None,
    cv_struct: Dict[str, Any] = None,
    nlp_weight: float = 0.50,
) -> Tuple[Dict[str, Any], str]:
    """
    Enhanced scoring combining NLP similarity + non-NLP features:
    - Skill overlap ratio (exact keyword matching)
    - Seniority alignment (experience leveling)
    - Domain/sector match
    - Language requirement coverage
    
    Returns:
        ({
            'nlp_score': float (0-1),
            'skills_score': float (0-1),
            'seniority_score': float (0-1),
            'domain_score': float (0-1),
            'language_score': float (0-1),
            'global_score': float (0-1),
            'skill_details': {
                'ao_skills': List[str],
                'cv_skills': List[str],
                'overlap': List[str],
                'missing': List[str],
                'overlap_ratio': float,
            }
        }, method_str)
    """
    ao_struct = ao_struct or {}
    cv_struct = cv_struct or {}
    
    # ===== NLP Score (50% default weight) =====
    nlp_scores, method_used = score_blocks(ao_blocks, cv_blocks)
    nlp_score = nlp_scores.get("global_score", 0.5)
    
    # ===== Skill Extraction & Matching =====
    # Prioritize Mistral-extracted skills if available, then normalize them
    ao_skills_raw = ao_struct.get("competences_techniques", [])
    cv_skills_raw = cv_struct.get("technologies", [])

    # Fallback to regex extraction if Mistral fields are empty
    if not ao_skills_raw:
        ao_full_text = ao_blocks.get("full", "")
        ao_skills_raw = extract_skills(ao_full_text)

    if not cv_skills_raw:
        cv_full_text = cv_blocks.get("full", "")
        cv_skills_raw = extract_skills(cv_full_text)

    # Ensure they're lists and normalize all skills to standardized forms
    if isinstance(ao_skills_raw, str):
        ao_skills_raw = [s.strip() for s in ao_skills_raw.split(",") if s.strip()]
    if isinstance(cv_skills_raw, str):
        cv_skills_raw = [s.strip() for s in cv_skills_raw.split(",") if s.strip()]

    # Normalize all skills to standardized forms
    ao_skills_list = normalize_skill_list(ao_skills_raw)
    cv_skills_list = normalize_skill_list(cv_skills_raw)

    # Calculate overlap using normalized skills
    overlap = set(ao_skills_list) & set(cv_skills_list)
    missing = set(ao_skills_list) - overlap

    # Skill overlap ratio (0-1): how many required skills does candidate have?
    if ao_skills_list:
        skill_overlap_ratio = len(overlap) / len(ao_skills_list)
    else:
        skill_overlap_ratio = 0.5  # Neutral if no skills extracted
    
    # Apply keyword presence bonus
    keyword_bonus = _score_keyword_presence(ao_blocks.get("skills_like", ""), cv_blocks.get("skills_like", ""))
    skills_score = 0.6 * skill_overlap_ratio + 0.4 * keyword_bonus
    
    # ===== Seniority Alignment =====
    # Prioritize Mistral-extracted seniority if available
    ao_seniority_str = normalize_ws(str(ao_struct.get("seniorite", "") or ao_struct.get("experience_requise", "")))
    cv_seniority_str = normalize_ws(str(cv_struct.get("seniorite", "")))
    
    ao_seniority_level = _parse_seniority(ao_seniority_str)
    cv_seniority_level = _parse_seniority(cv_seniority_str)
    seniority_score = _calculate_seniority_score(ao_seniority_level, cv_seniority_level)
    
    # ===== Domain/Sector Match =====
    ao_domain = normalize_ws(str(ao_struct.get("secteur_principal", "") or ao_struct.get("secteur", "")))
    cv_domain = normalize_ws(str(cv_struct.get("secteur_principal", "")))
    domain_score = _calculate_domain_match(ao_domain, cv_domain)
    
    # ===== Language Requirements =====
    ao_langs_str = normalize_ws(str(ao_blocks.get("certification_like", "")))
    cv_langs_str = normalize_ws(str(cv_blocks.get("certification_like", "")))
    ao_languages = set(extract_languages(ao_langs_str))
    cv_languages = set(extract_languages(cv_langs_str))
    
    if ao_languages:
        lang_overlap = len(ao_languages & cv_languages) / len(ao_languages)
        language_score = lang_overlap
    else:
        language_score = 0.5  # Neutral if no languages extracted
    
    # ===== Global Score (Weighted Combination) =====
    # nlp_weight (50%) + skills (20%) + seniority (15%) + domain (10%) + language (5%)
    non_nlp_weight = 1.0 - nlp_weight
    global_score = (
        nlp_weight * nlp_score +
        non_nlp_weight * (
            0.40 * skills_score +
            0.30 * seniority_score +
            0.20 * domain_score +
            0.10 * language_score
        )
    )
    
    skill_details = {
        "ao_skills": sorted(ao_skills_list),
        "cv_skills": sorted(cv_skills_list),
        "overlap": sorted(list(overlap)),
        "missing": sorted(list(missing)),
        "overlap_ratio": float(skill_overlap_ratio),
    }
    
    return {
        "nlp_score": float(nlp_score),
        "skills_score": float(skills_score),
        "seniority_score": float(seniority_score),
        "domain_score": float(domain_score),
        "language_score": float(language_score),
        "global_score": float(global_score),
        "skill_details": skill_details,
        "ao_seniority": ao_seniority_str,
        "cv_seniority": cv_seniority_str,
    }, method_used
