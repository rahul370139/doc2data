"""
Lightweight Graph Orchestrator for Doc2Data
===========================================

Implements a directed graph architecture for document processing, mirroring
LangGraph principles but without external dependencies.

Goal: Secure, local, modular document extraction for healthcare.

Architecture:
    [Ingest] -> [FormID] -> (Router)
                              |
           +------------------+------------------+
           v                                     v
    [CMS-1500 Branch]                     [General Branch]
    [Align] -> [YOLO Layout]              [LayoutLM/Donut]
           |                                     |
           +------------------+------------------+
                              v
                        [Merge Layout]
                              |
          +----------+--------+--------+----------+
          v          v        v        v          v
       [Text]     [Table]  [Figure]  [Form]   [Check]
          |          |        |        |          |
       [Tiered    [Table    [VLM]    [Tiered    [Density]
        OCR]      Agent]              OCR]
          |          |                 |          |
       [SLM]         |               [SLM]        |
          |          |                 |          |
          +----------+--------+--------+----------+
                              v
                        [Merge Results]
                              v
                        [Validation] -> [Assembly]

"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

# Import existing agents
from src.pipelines.agentic_cms1500 import (
    RegistrationAgent,
    ZoneAgent,
    OCRAgent,
    LLMExtractionAgent
)
from src.pipelines.business_schema import map_to_business_schema
from src.pipelines.ingest import ingest_document
from utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DocGraph")


# ============================================================================
# State Definition
# ============================================================================

class FormType(Enum):
    CMS1500 = "cms-1500"
    UB04 = "ub-04"
    GENERIC = "generic"
    UNKNOWN = "unknown"

@dataclass
class DocState:
    """Shared state passed between graph nodes."""
    # Input
    file_path: str
    images: List[np.ndarray] = field(default_factory=list)
    page_idx: int = 0
    
    # Classification
    form_type: FormType = FormType.UNKNOWN
    form_confidence: float = 0.0
    
    # Layout & Alignment
    aligned_image: Optional[np.ndarray] = None
    alignment_matrix: Optional[np.ndarray] = None
    layout_blocks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Extraction
    ocr_blocks: List[Dict[str, Any]] = field(default_factory=list)
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    validation_errors: List[str] = field(default_factory=list)
    qa_notes: List[str] = field(default_factory=list)
    
    # Final Output
    business_fields: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)

    def log(self, step: str):
        self.history.append(f"{time.strftime('%H:%M:%S')} - {step}")
        logger.info(step)


# ============================================================================
# Graph Node Base
# ============================================================================

class GraphNode:
    """Base unit of work in the pipeline."""
    def __init__(self, name: str):
        self.name = name

    def __call__(self, state: DocState) -> DocState:
        try:
            state.log(f"Starting {self.name}...")
            start = time.time()
            new_state = self.process(state)
            duration = time.time() - start
            state.metadata[f"{self.name}_time"] = duration
            return new_state
        except Exception as e:
            state.log(f"Error in {self.name}: {e}")
            state.metadata[f"{self.name}_error"] = str(e)
            state.metadata[f"{self.name}_trace"] = traceback.format_exc()
            return state

    def process(self, state: DocState) -> DocState:
        raise NotImplementedError


# ============================================================================
# Specialized Nodes
# ============================================================================

class IngestNode(GraphNode):
    def process(self, state: DocState) -> DocState:
        pages = ingest_document(state.file_path, dpi=Config.DPI)
        if pages:
            state.images = [p.image for p in pages]
            state.metadata["page_count"] = len(pages)
        else:
            raise ValueError(f"Failed to ingest {state.file_path}")
        return state


class FormIDNode(GraphNode):
    def process(self, state: DocState) -> DocState:
        # Simple heuristic for now, can be upgraded to ML classifier
        # 1. Check filename hint
        fname = state.file_path.lower()
        if "cms1500" in fname or "cms-1500" in fname:
            state.form_type = FormType.CMS1500
            state.form_confidence = 0.9
            return state
            
        # 2. OCR Header check (if not hinted)
        img = state.images[0]
        # Quick crop of top 15%
        h, w = img.shape[:2]
        header = img[0:int(h*0.15), :]
        
        # Use simple OCR wrapper (assuming imported from ocr.py or similar)
        # For now, we'll assume generic if not explicit
        state.form_type = FormType.GENERIC
        state.form_confidence = 0.5
        return state


class TemplateAlignNode(GraphNode):
    def __init__(self):
        super().__init__("TemplateAlignment")
        self.agent = RegistrationAgent("cms-1500")

    def process(self, state: DocState) -> DocState:
        if state.form_type != FormType.CMS1500:
            state.log("Skipping alignment (not CMS-1500)")
            state.aligned_image = state.images[0]
            return state

        H, shape = self.agent.align(state.images[0])
        if H is not None:
            h, w = state.images[0].shape[:2]
            state.aligned_image = cv2.warpPerspective(state.images[0], H, (w, h))
            state.alignment_matrix = H
            state.metadata["alignment_success"] = True
        else:
            state.log("Alignment failed, using original")
            state.aligned_image = state.images[0]
            state.metadata["alignment_success"] = False
        return state


class CMS1500ExtractionNode(GraphNode):
    """Combines Layout + OCR + SLM for CMS-1500 using existing Agentic logic."""
    def __init__(self):
        super().__init__("CMS1500_Extractor")
        # Reuse the robust logic we built in agentic_cms1500.py
        # This avoids code duplication
        from src.pipelines.agentic_cms1500 import run_cms1500_agentic
        self.runner = run_cms1500_agentic

    def process(self, state: DocState) -> DocState:
        if state.form_type != FormType.CMS1500:
            return state

        # We call the existing pipeline function but we might refactor it 
        # to accept image directly in future. For now, pass file path 
        # as the runner re-ingests. 
        # OPTIMIZATION: Refactor run_cms1500_agentic to take image input.
        # For this prototype, we'll let it re-ingest (caching handles overhead).
        
        result = self.runner(
            state.file_path,
            use_icr=True,
            use_llm=True,
            align_template=state.metadata.get("alignment_success", True)
        )
        
        state.extracted_fields = result.get("extracted_fields", {})
        state.business_fields = result.get("business_fields", {})
        state.ocr_blocks = result.get("ocr_blocks", [])
        state.metadata["field_details"] = result.get("field_details", [])
        
        return state


class GeneralFormNode(GraphNode):
    """Fallback for non-CMS1500 forms."""
    def process(self, state: DocState) -> DocState:
        if state.form_type == FormType.CMS1500:
            return state
            
        from src.pipelines.form_extractor import extract_with_full_pipeline
        result = extract_with_full_pipeline(state.file_path)
        
        state.extracted_fields = result.get("extracted_fields", {})
        state.business_fields = result.get("business_fields", {})
        return state


class ValidationNode(GraphNode):
    def process(self, state: DocState) -> DocState:
        # Cross-field validation logic here
        bus = state.business_fields
        
        # Example: Check Total vs Sum
        # (This logic exists in cms1500_production.py, we can migrate it here)
        return state


# ============================================================================
# Graph Orchestrator
# ============================================================================

class DocumentGraph:
    """
    Executes the secure document processing graph.
    """
    def __init__(self):
        # Define nodes
        self.ingest = IngestNode("Ingest")
        self.identify = FormIDNode("FormID")
        self.align = TemplateAlignNode()
        self.extract_cms = CMS1500ExtractionNode()
        self.extract_gen = GeneralFormNode("GeneralExtraction")
        self.validate = ValidationNode("Validation")

    def run(self, file_path: str) -> Dict[str, Any]:
        """Execute the pipeline flow."""
        # 1. Initialize State
        state = DocState(file_path=file_path)
        
        # 2. Linear Flow (Ingest -> ID)
        state = self.ingest(state)
        state = self.identify(state)
        
        # 3. Conditional Routing
        if state.form_type == FormType.CMS1500:
            state.log("Routing to CMS-1500 Path")
            state = self.align(state)
            state = self.extract_cms(state)
        else:
            state.log("Routing to General Path")
            state = self.extract_gen(state)
            
        # 4. Merge & Validate (Common)
        state = self.validate(state)
        
        # 5. Export
        return self._export_result(state)

    def _export_result(self, state: DocState) -> Dict[str, Any]:
        """Convert state to final JSON response."""
        return {
            "success": not any("_error" in k for k in state.metadata.keys()),
            "form_type": state.form_type.value,
            "business_fields": state.business_fields,
            "extracted_fields": state.extracted_fields,
            "ocr_blocks": state.ocr_blocks,
            "field_details": state.metadata.get("field_details", []),
            "metadata": state.metadata,
            "logs": state.history
        }

# ============================================================================
# CLI Entry
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="File to process")
    args = parser.parse_args()
    
    graph = DocumentGraph()
    result = graph.run(args.input)
    
    print(json.dumps(result["business_fields"], indent=2, default=str))

if __name__ == "__main__":
    main()

