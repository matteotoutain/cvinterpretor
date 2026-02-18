import os
import json
from typing import Any, Dict

from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "mistral-medium-latest"


def _get_client() -> Mistral:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env var MISTRAL_API_KEY")
    return Mistral(api_key=api_key)


def call_mistral_json_extraction(
    text_input: str,
    user_prompt: str,
    mistral_model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Generic helper: ask for a JSON object and parse it.
    """
    client = _get_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at extracting structured information from text "
                "and outputting it as a JSON object. Only output valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"{user_prompt}\n\nHere is the text to analyze:\n```\n{text_input}\n```\n\n",
        },
    ]

    try:
        chat_response = client.chat.complete(
            model=mistral_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        response_content = chat_response.choices[0].message.content
        return json.loads(response_content)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Keyword packs (Noise removal BEFORE NLP)
# ============================================================
def build_cv_keyword_pack_prompt() -> str:
    return (
        "Extract a CLEAN 'keyword pack' from the provided CV text. Translate everything to English.\n"
        "Output a single JSON object. If a field is not found, use null (or [] for lists).\n\n"
        "{\n"
        '  "nom": "Full name",\n'
        '  "role_principal": "Main role/title",\n'
        '  "seniorite_raw": "Raw seniority phrasing if present (e.g., Senior Data Engineer, 5 years, etc.)",\n'
        '  "seniorite_label": "One of: Junior | Senior | Manager | null (ONLY if you are confident)",\n'
        '  "secteur_principal": "Main industry sector(s), comma-separated",\n'
        '  "tech_skills": ["TECH skills/tools/languages/frameworks/platforms"],\n'
        '  "domain_knowledge": ["Domain knowledge / industry keywords (banking, insurance, public, retail...)"],\n'
        '  "certifications": ["Certifications (Salesforce, AWS, Azure, PMP, etc.)"],\n'
        '  "langues": ["Languages"],\n'
        '  "experiences": [\n'
        "     {\n"
        '       "mission": "Short mission title/summary",\n'
        '       "secteur": "Domain/industry if stated",\n'
        '       "stack": ["Tech stack used"],\n'
        '       "duree": "Duration if stated"\n'
        "     }\n"
        "  ],\n"
        '  "cv_text": "A compact text summary focusing on responsibilities + achievements (no fluff)"\n'
        "}\n\n"
        "Rules:\n"
        "- Do NOT invent facts.\n"
        "- Avoid soft skills unless they are extremely specific and demonstrable.\n"
        "- Prefer lists of concrete nouns/proper names/technologies.\n"
    )


def build_ao_keyword_pack_prompt() -> str:
    return (
        "Extract a CLEAN 'keyword pack' from the provided job description / call for tender (AO). "
        "Translate everything to English.\n"
        "Output a single JSON object. If a field is not found, use null (or [] for lists).\n\n"
        "{\n"
        '  "titre_poste": "Mission title / role",\n'
        '  "mission_summary": "1-3 sentences of mission objective + context (no fluff)",\n'
        '  "tech_skills_required": ["Required technical skills/tools/platforms"],\n'
        '  "domain_knowledge_required": ["Required domain/industry knowledge"],\n'
        '  "certifications_required": ["Required or strongly preferred certifications"],\n'
        '  "experience_required": "Years / level required if stated",\n'
        '  "seniority_target": "One of: Junior | Senior | Manager | null (ONLY if explicit)",\n'
        '  "nice_to_have": ["Nice-to-have skills (technical or domain)"],\n'
        '  "constraints": ["Hard constraints: language, on-site, clearance, etc."]\n'
        "}\n\n"
        "Rules:\n"
        "- Do NOT invent requirements.\n"
        "- Keep lists concrete and deduplicated.\n"
    )


def call_mistral_cv_keyword_pack(
    cv_text: str,
    mistral_model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    return call_mistral_json_extraction(
        text_input=cv_text,
        user_prompt=build_cv_keyword_pack_prompt(),
        mistral_model=mistral_model,
    )


def call_mistral_ao_keyword_pack(
    ao_text: str,
    mistral_model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    return call_mistral_json_extraction(
        text_input=ao_text,
        user_prompt=build_ao_keyword_pack_prompt(),
        mistral_model=mistral_model,
    )


# ============================================================
# Structured explanation + "Perfect candidate gaps"
# ============================================================
def call_mistral_json_explanation(
    ao_pack: Dict[str, Any],
    cv_pack: Dict[str, Any],
    scores: Dict[str, Any],
    mistral_model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Produce a structured explanation (JSON) of why a CV matches an AO.
    Adds query_terms so we can do vector-search citations.
    """
    client = _get_client()

    prompt = f"""
You are a staffing assistant for consulting proposals.
Explain WHY a candidate matches (or not) a job description, based ONLY on the provided inputs.

Return ONLY valid JSON with this schema:
{{
  "verdict": "Strong Fit | Moderate Fit | Low Fit",
  "one_liner": "1 sentence summary",
  "strengths": [
    {{
      "title": "short label",
      "evidence": "what in CV matches AO (be specific)",
      "impact": "why it matters",
      "query_terms": ["terms to retrieve supporting CV passages (2-6 items)"]
    }}
  ],
  "gaps": [
    {{
      "title": "short label",
      "evidence": "what is missing/unclear compared to AO",
      "risk": "why it matters",
      "mitigation": "how to reduce risk",
      "query_terms": ["terms to verify in CV (2-6 items)"]
    }}
  ],
  "recommended_questions": [
    "question to ask candidate"
  ],
  "score_breakdown": {{
    "tech_skills": "what drove the score",
    "experience": "what drove the score",
    "domain_knowledge": "what drove the score",
    "certifications": "what drove the score"
  }}
}}

AO_PACK:
{json.dumps(ao_pack, ensure_ascii=False)}

CV_PACK:
{json.dumps(cv_pack, ensure_ascii=False)}

SCORES:
{json.dumps(scores, ensure_ascii=False)}

Rules:
- Do NOT invent facts not present in AO_PACK or CV_PACK.
- If unclear, explicitly say "unclear" and add a recommended question.
- query_terms must be concrete (tech names, domain keywords, certification names, client/industry terms).
"""

    messages = [
        {"role": "system", "content": "You output ONLY valid JSON. No markdown. No commentary."},
        {"role": "user", "content": prompt},
    ]

    try:
        chat_response = client.chat.complete(
            model=mistral_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        response_content = chat_response.choices[0].message.content
        return json.loads(response_content)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}


def call_mistral_json_gap_to_ideal(
    ao_pack: Dict[str, Any],
    cv_pack: Dict[str, Any],
    mistral_model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Compare CV to an 'ideal' candidate implied by the AO.
    Output is action-oriented and NOT a one-liner.
    """
    client = _get_client()

    prompt = f"""
You are a staffing assistant. Build an "ideal candidate profile" from the AO, then compare the CV against it.
Return ONLY valid JSON with this schema:

{{
  "ideal_profile": {{
    "tech_skills": ["must-have technical skills implied by AO"],
    "domain_knowledge": ["must-have domain knowledge implied by AO"],
    "certifications": ["must-have or strongly preferred certifications"],
    "experience": "what type/level of experience would be ideal",
    "seniority_target": "Junior | Senior | Manager | Unknown"
  }},
  "must_have_missing": [
    {{
      "item": "missing requirement",
      "why_important": "impact on mission delivery",
      "how_to_close": "realistic mitigation (training, pairing, interview check...)",
      "query_terms": ["terms to verify in CV (2-6 items)"]
    }}
  ],
  "nice_to_have_missing": [
    {{
      "item": "nice-to-have gap",
      "value": "why it would help",
      "query_terms": ["terms to verify in CV (2-6 items)"]
    }}
  ],
  "unclear": [
    {{
      "item": "could be present but not explicit in CV",
      "what_to_ask": "question to validate",
      "query_terms": ["terms to search in CV (2-6 items)"]
    }}
  ],
  "top_questions": [
    "best questions to validate the gaps quickly"
  ]
}}

AO_PACK:
{json.dumps(ao_pack, ensure_ascii=False)}

CV_PACK:
{json.dumps(cv_pack, ensure_ascii=False)}

Rules:
- Do NOT invent facts.
- If CV does not explicitly mention something, treat it as missing or unclear.
- Keep items concrete and mission-relevant (avoid generic soft skills).
"""

    messages = [
        {"role": "system", "content": "You output ONLY valid JSON. No markdown. No commentary."},
        {"role": "user", "content": prompt},
    ]

    try:
        chat_response = client.chat.complete(
            model=mistral_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        response_content = chat_response.choices[0].message.content
        return json.loads(response_content)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}
