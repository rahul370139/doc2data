"""
Validation Agent - field validation and optional LLM QA.

PURPOSE: Validates extracted fields using validators.py (NPI, date, phone,
ICD, etc.). Returns errors and warnings. Optional LLM QA check on full
extraction for sanity review.

USE CASE: Pipeline calls this after assembly. Catches format errors before
output. No manual use; part of MultiAgentPipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.pipelines.core import BaseAgent, DetectedBlock, PipelineConfig
from src.pipelines.validators import validate_field as typed_validate_field

from utils.config import Config


class ValidationAgent(BaseAgent):
    """Field validation and QA checks."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("ValidationAgent")
        self.config = config
    
    async def initialize(self):
        self._initialized = True
    
    def validate_field(self, value: str, field_type: str) -> Tuple[bool, str]:
        """Validate a field value using validators.py."""
        if not value or not field_type:
            return True, ""
        passed, info = typed_validate_field(field_type, value)
        if passed:
            return True, ""
        # Unknown validator = no format check for this type, pass
        if info.get("reason") == "unknown_validator":
            return True, ""
        reason = info.get("reason", "format")
        return False, f"Invalid {field_type} ({reason})"
    
    async def llm_qa_check(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Run LLM QA check on extracted data."""
        if not self.config.enable_llm_qa:
            return []
        
        notes = []
        try:
            import requests
            
            fields_str = "\n".join([f"- {k}: {v}" for k, v in extracted_data.items() if v])
            prompt = f"""Review this medical form extraction for errors:
{fields_str}

List only obvious errors (max 3). If all looks good, say "OK"."""
            
            response = requests.post(
                f"http://{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": self.config.slm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200}
                },
                timeout=30
            )
            
            if response.ok:
                result = response.json().get("response", "").strip()
                if result and "ok" not in result.lower():
                    notes.append(result)
        except Exception:
            pass
        
        return notes
    
    async def process(self, blocks: List[DetectedBlock], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all data."""
        await self.initialize()
        
        validation_results = {
            "errors": [],
            "warnings": [],
            "qa_notes": []
        }
        
        # Field-level validation
        for block in blocks:
            field_type = block.metadata.get("field_type")
            if field_type:
                valid, msg = self.validate_field(block.text, field_type)
                if not valid:
                    validation_results["errors"].append({
                        "field_id": block.id,
                        "message": msg
                    })
        
        # LLM QA
        qa_notes = await self.llm_qa_check(extracted_data)
        validation_results["qa_notes"] = qa_notes
        
        return validation_results
