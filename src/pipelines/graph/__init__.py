"""
LangGraph-based agentic orchestrator for Doc2Data.

The graph supersedes the linear control flow in ``multi_agent_pipeline.process``
by introducing explicit nodes for planning, execution, reflection and
rescue.  The underlying agents (form ID, alignment, OCR, validation,
business schema mapping) are reused unchanged — the graph owns routing,
loops and escalation policy.

Public API:
    - build_graph(config)  → compiled LangGraph
    - run_graph(config, path) → final state dict
    - GraphState             → TypedDict used by nodes
"""
from src.pipelines.graph.state import GraphState, create_initial_state
from src.pipelines.graph.orchestrator import build_graph, run_graph, run_graph_sync

__all__ = [
    "GraphState",
    "create_initial_state",
    "build_graph",
    "run_graph",
    "run_graph_sync",
]
