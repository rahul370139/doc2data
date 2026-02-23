"""
Base agent class for the multi-agent document processing pipeline.

PURPOSE: Provides a common interface (initialize, process, log) for all pipeline
agents. Every agent (form identification, alignment, layout, OCR, labeling,
validation) inherits from BaseAgent so the orchestrator can treat them uniformly.

USE CASE: When adding a new agent, subclass BaseAgent and implement
initialize() and process(). The pipeline will call them in sequence.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base class for all pipeline agents."""

    def __init__(self, name: str):
        self.name = name
        self._initialized = False

    @abstractmethod
    async def initialize(self):
        """Initialize the agent (lazy loading)."""
        pass

    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """Process input and return result."""
        pass

    def log(self, message: str):
        """Log agent activity."""
        print(f"[{self.name}] {message}")
