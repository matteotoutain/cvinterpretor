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
    Same behavior as your notebook: ask for a JSON object and parse it.
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
