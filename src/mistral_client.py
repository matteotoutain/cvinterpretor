import os
import json
from typing import Any, Dict, Optional

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
    Ask for a JSON object and parse it.
    """
    client = _get_client()

    messages = [
        {
            "role": "system",
            "content": "You are an expert at extracting structured information from text and outputting it as a JSON object. Only output valid JSON.",
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
    Produce a structured explanation (JSON) of why a CV matches an AO.
    """
    client = _get_client()

    prompt = f"""
You are a staffing assistant for consulting proposals.
You must explain why a candidate CV matches (or not) an AO, based ONLY on the provided structured inputs.

Return ONLY valid JSON with this schema:
{{
  "verdict": "Strong Fit | Moderate Fit | Low Fit",
  "one_liner": "1 sentence summary",
  "strengths": [
    {{"title": "short label", "evidence": "what in CV matches AO", "impact": "why it matters"}}
  ],
  "gaps": [
    {{"title": "short label", "evidence": "what is missing/unclear", "risk": "why it matters", "mitigation": "how to reduce risk"}}
  ],
  "recommended_questions": [
    "question to ask candidate"
  ],
  "score_breakdown": {{
    "skills_like": "explain which blocks drove the score",
    "experience_like": "explain which blocks drove the score",
    "domain_like": "explain which blocks drove the score",
    "certification_like": "explain which blocks drove the score"
  }}
}}

AO_STRUCT:
{json.dumps(ao_struct, ensure_ascii=False)}

CV_STRUCT:
{json.dumps(cv_struct, ensure_ascii=False)}

SCORES:
{json.dumps(scores, ensure_ascii=False)}

Rules:
- Do NOT invent facts not present in AO_STRUCT or CV_STRUCT.
- If unclear, say "unclear" and add a recommended question.
"""

    messages = [
        {
            "role": "system",
            "content": "You output ONLY valid JSON. No markdown. No commentary.",
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
