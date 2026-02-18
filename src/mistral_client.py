# mistral_client.py

import os
import requests

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-large-latest"

API_URL = "https://api.mistral.ai/v1/chat/completions"


def call_mistral(prompt: str):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    r = requests.post(API_URL, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ============================================================
# 1) KEYWORD PACK EXTRACTION
# ============================================================

def extract_keyword_pack(text: str):

    prompt = f"""
Extract structured keywords from the following text.

Return JSON only:

{{
  "tech_skills": ["..."],
  "domain_knowledge": ["..."],
  "experience_summary": "... concise professional summary ...",
  "certifications": ["..."],
  "seniority": "Junior | Senior | Manager"
}}

Rules:
- Do not invent.
- If unclear, use empty list.
- Keep concise.
- Focus only on relevant professional content.

TEXT:
{text}
"""

    return call_mistral(prompt)


# ============================================================
# 2) GAP TO IDEAL
# ============================================================

def gap_to_ideal(cv_keywords, ao_keywords):

    prompt = f"""
You are a senior staffing analyst.

Compare the candidate to the ideal profile derived from the job offer.

Return structured JSON:

{{
  "must_have_missing": [...],
  "nice_to_have_missing": [...],
  "unclear_elements": [...],
  "risk_analysis": "...",
  "questions_to_validate": [...]
}}

Rules:
- No hallucination.
- If not in CV → mark missing or unclear.
- Be analytical, not generic.

CV:
{cv_keywords}

JOB OFFER:
{ao_keywords}
"""

    return call_mistral(prompt)


# ============================================================
# 3) EXPLANATION POST-SCORING
# ============================================================

def explain_scoring(cv_keywords, ao_keywords, block_scores):

    prompt = f"""
Explain why the candidate matches or not the job offer.

Block scores:
{block_scores}

CV keywords:
{cv_keywords}

AO keywords:
{ao_keywords}

Provide structured analysis:
- Strengths
- Weaknesses
- Overall fit assessment
- Hiring recommendation

Be precise and business-oriented.
"""

    return call_mistral(prompt)
