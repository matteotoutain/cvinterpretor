import os
import json
from typing import Any, Dict

from mistralai import Mistral
from dotenv import load_dotenv

from .nlp import STANDARDIZED_SKILLS, CONSULTING_DOMAINS

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
    Ask for a JSON object and parse it.
    Guides extraction using standardized consulting domains.
    """
    client = _get_client()

    system_message = """You are an expert at extracting structured information from text and outputting it as a JSON object. Only output valid JSON.

CRITICAL: For domain/sector fields, use ONLY these STANDARDIZED CONSULTING DOMAINS:
""" + ", ".join(sorted(CONSULTING_DOMAINS)) + """

If the domain is not in the list, normalize it to the closest matching standardized domain from the list above. Do NOT invent or use undefined domains.

CRITICAL: For skills/technologies/competences_techniques/competences_metier fields, use ONLY these STANDARDIZED SKILLS: """ + ", ".join(sorted(STANDARDIZED_SKILLS)) + """

If the skill is not in the list, normalize it to the closest matching standardized skill from the list above. Do NOT invent or use undefined skills."""

    messages = [
        {
            "role": "system",
            "content": system_message,
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


def call_mistral_json_explanation(
    ao_struct: Dict[str, Any],
    cv_struct: Dict[str, Any],
    scores: Dict[str, Any],
    mistral_model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Produce a personalized, integrated explanation for the 3 scores.
    Returns concise, tailored insights for this specific CV-AO match.
    """
    client = _get_client()

    # Extract key personalized info
    def _as_list_simple(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            return [p.strip() for p in x.split(",") if p.strip()]
        return []

    candidate_name = cv_struct.get("nom", "the candidate")
    candidate_role = cv_struct.get("role_principal", "candidate")
    candidate_skills = ", ".join(_as_list_simple(cv_struct.get("technologies", []))[:3])  # Top 3 skills
    candidate_domain = cv_struct.get("secteur_principal", "various domains")

    ao_title = ao_struct.get("titre_poste", "the position")
    ao_skills = ", ".join(_as_list_simple(ao_struct.get("competences_techniques", []))[:3])  # Top 3 required skills
    ao_domain = ao_struct.get("secteur", "the industry")

    prompt = f"""
You are a staffing assistant evaluating a specific candidate for a job opening. Provide a personalized, actionable explanation of why {candidate_name} ({candidate_role}) matches the {ao_title} position.

Candidate Profile:
- Name: {candidate_name}
- Role: {candidate_role}
- Key Skills: {candidate_skills}
- Domain: {candidate_domain}

Job Requirements:
- Title: {ao_title}
- Required Skills: {ao_skills}
- Domain: {ao_domain}

Use these STANDARDIZED CONSULTING DOMAINS for domain analysis: """ + ", ".join(sorted(CONSULTING_DOMAINS)) + """

Analyze this specific match based on the provided structures and scores. Focus on unique strengths, gaps, and fit for this candidate-job pair.

Return ONLY valid JSON with this simple schema:
{{
  "nlp_why": "2-3 words explaining the semantic match quality for this candidate",
  "skills_why": "Brief explanation of skill alignment + 1 key missing skill if any",
  "domain_why": "Brief explanation of sector fit using standardized domains",
  "overall_fit": "Strong | Moderate | Low",
  "key_strength": "One main strength specific to this candidate",
  "key_gap": "One main gap specific to this candidate or 'None'"
}}

AO_STRUCT:
{json.dumps(ao_struct, ensure_ascii=False)}

CV_STRUCT:
{json.dumps(cv_struct, ensure_ascii=False)}

SCORES:
{json.dumps(scores, ensure_ascii=False)}

Make the explanation personalized to {candidate_name}'s profile and the {ao_title} requirements. Keep explanations brief but specific.
"""

    messages = [
        {
            "role": "system",
            "content": "You output ONLY valid JSON. No markdown, no code blocks, no commentary.",
        },
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
