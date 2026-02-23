"""
OCR Query Chatbot - Answer questions from extracted document data using SLM.

Uses schema templates + user prompts to map document content into structured answers
with full traceability to source fields. Does NOT modify the extraction pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.config import Config


def _load_schema_context(form_type: str) -> str:
    """Load schema field descriptions for SLM context (helps mapping)."""
    schema_paths = {
        "cms-1500": Path(__file__).parent.parent.parent / "data" / "schemas" / "cms-1500.json",
        "cms1500": Path(__file__).parent.parent.parent / "data" / "schemas" / "cms-1500.json",
        "ub-04": Path(__file__).parent.parent.parent / "data" / "schemas" / "ub-04.json",
        "ub04": Path(__file__).parent.parent.parent / "data" / "schemas" / "ub-04.json",
    }
    key = form_type.lower().replace("-", "").replace("_", "")
    path = schema_paths.get(key)
    if not path or not path.exists():
        return ""
    try:
        with open(path) as f:
            schema = json.load(f)
        fields = schema.get("fields", [])
        lines = [f"- {f.get('id', '')}: {f.get('description', f.get('label', ''))}" for f in fields[:60]]
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""


def query_ocr_slm(
    prompt: str,
    extracted_fields: Dict[str, Any],
    field_details: Optional[List[Dict[str, Any]]] = None,
    form_type: str = "cms-1500",
    model: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    Answer user question from OCR JSON using Ollama SLM.

    Args:
        prompt: User question (e.g., "What is the patient's address?")
        extracted_fields: OCR JSON dict (field_id -> value)
        field_details: Optional list of field metadata for traceability
        form_type: Form type for schema context (cms-1500, ub-04)
        model: Ollama model (default: Config.OLLAMA_MODEL_SLM)

    Returns:
        (answer, source_field_ids) - answer text and list of field IDs used
    """
    model = model or Config.OLLAMA_MODEL_SLM
    schema_context = _load_schema_context(form_type)

    # Build context from extracted fields (filter empty)
    data_str = json.dumps(
        {k: v for k, v in extracted_fields.items() if v and str(v).strip()},
        indent=2
    )

    system_prompt = """You are a document data assistant. Answer questions using ONLY the provided JSON document data.
Rules:
- Use ONLY values from the JSON. Do not invent or guess.
- If the asked for is not in the JSON, say "Not found in document."
- For addresses, combine street, city, state, zip when available.
- Cite source field IDs when answering (e.g., "source: 2_patient_name").
- Be concise and accurate."""
    if schema_context:
        system_prompt += f"\n\nSchema field reference:\n{schema_context}"

    user_content = f"""Document data (OCR JSON):
```json
{data_str}
```

User question: {prompt}

Answer (include source field IDs for traceability):"""

    try:
        import requests
        resp = requests.post(
            f"http://{Config.OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": user_content,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 512},
            },
            timeout=60,
        )
        if not resp.ok:
            return f"Error: Ollama returned {resp.status_code}", []

        data = resp.json()
        answer = (data.get("response") or "").strip()

        # Extract source field IDs from response (simple heuristic)
        source_ids = []
        if field_details:
            for fd in field_details:
                fid = fd.get("id", "")
                if fid and fid in answer:
                    source_ids.append(fid)
        else:
            for fid in extracted_fields:
                if fid in answer:
                    source_ids.append(fid)

        return answer, list(dict.fromkeys(source_ids))  # dedupe
    except Exception as e:
        return f"Error: {str(e)}", []
