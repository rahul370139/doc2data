"""
Document processing pipeline package.

PURPOSE: Re-exports MultiAgentPipeline, PipelineConfig, FormType, and related
types for convenient imports. Use: from src.pipelines import MultiAgentPipeline.

USE CASE: Main entry point for using the pipeline from app, scripts, or tests.
"""
from src.pipelines.multi_agent_pipeline import (
    MultiAgentPipeline,
    PipelineConfig,
    FormType,
    BlockType,
    DetectedBlock,
    FormIdentification,
    AlignmentResult,
)

__all__ = [
    "MultiAgentPipeline",
    "PipelineConfig",
    "FormType",
    "BlockType",
    "DetectedBlock",
    "FormIdentification",
    "AlignmentResult",
]
