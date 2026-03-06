"""
Pipeline agents for document processing.
"""
from src.pipelines.agents.form_identification import FormIdentificationAgent
from src.pipelines.agents.template_alignment import TemplateAlignmentAgent
from src.pipelines.agents.layout_detection import LayoutDetectionAgent
from src.pipelines.agents.ocr import OCRAgent, PaddleOCRWrapper
from src.pipelines.agents.labeling import LabelingAgent
from src.pipelines.validators import ValidationAgent

__all__ = [
    "FormIdentificationAgent",
    "TemplateAlignmentAgent",
    "LayoutDetectionAgent",
    "OCRAgent",
    "PaddleOCRWrapper",
    "LabelingAgent",
    "ValidationAgent",
]
