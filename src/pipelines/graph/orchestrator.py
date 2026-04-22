"""
LangGraph orchestrator for Doc2Data.

Nodes are defined in ``graph.nodes``.  This module wires them into a
StateGraph with conditional edges for lane selection and the reflect →
rescue loop.

The graph is built once per process and reused across requests.  Each
request creates a fresh state dict and runs ``graph.ainvoke(state)``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from src.pipelines.core.models import PipelineConfig
from src.pipelines.graph.state import GraphState, create_initial_state
from src.pipelines.graph.nodes import (
    align_node, extract_digital_node, extract_scan_node,
    extract_sections_node, extract_tables_node, extract_widgets_node,
    finalize_node, identify_node, load_node,
    plan_node, reflect_node, rescue_node, validate_node,
    route_from_plan, route_after_digital, route_after_reflect,
)


logger = logging.getLogger("graph.orchestrator")

_graph = None
_graph_lock = asyncio.Lock()


def build_graph():
    """Build the Doc2Data LangGraph.

    Flow:

        START → load → identify → plan →
            • Lane A: extract_widgets → extract_tables → validate
            • Lane B: extract_digital → (extract_tables → validate | align → sections → scan → tables → validate)
            • Lane C: align → extract_sections → extract_scan → extract_tables → validate
        validate → reflect →
            • rescue → validate   (one loop max)
            • finalize → END

    ``extract_sections`` is a layout-first Tier-1 step:
        - Spatial clustering of the schema into ~8–22 semantic sections.
        - One VLM call per section with only that section's fields.
        - Its output SEEDS the blocks used by ``extract_scan`` so
          Florence-2 only runs on the residual fields the VLM left empty.
        - Disabling via ``config.enable_sections = False`` makes this a
          no-op pass-through and the graph reverts to pre-section
          behaviour (Florence-2 on every field).

    ``extract_tables`` is a dedicated table-extraction lane.  TABLE
    blocks (e.g. CMS-1500 Box 24) always route through here so we get
    VLM → VLM fallback → Florence-2 row OCR fallback, a separate trace
    entry, and clean writeback into ``extracted_fields``.  Historically
    this logic lived inline in ``extract_scan_node`` and silently
    failed because ``labeling_agent`` was never initialised on the
    graph path.
    """
    g = StateGraph(GraphState)

    g.add_node("load", load_node)
    g.add_node("identify", identify_node)
    g.add_node("plan", plan_node)
    g.add_node("extract_widgets", extract_widgets_node)
    g.add_node("extract_digital", extract_digital_node)
    g.add_node("align", align_node)
    g.add_node("extract_sections", extract_sections_node)
    g.add_node("extract_scan", extract_scan_node)
    g.add_node("extract_tables", extract_tables_node)
    g.add_node("validate", validate_node)
    g.add_node("reflect", reflect_node)
    g.add_node("rescue", rescue_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "load")
    g.add_edge("load", "identify")
    g.add_edge("identify", "plan")

    # Plan → lane
    g.add_conditional_edges(
        "plan", route_from_plan,
        {
            "extract_widgets": "extract_widgets",
            "extract_digital": "extract_digital",
            "align": "align",
        },
    )

    # Lane A → tables → validate.  Widget PDFs rarely have usable
    # TABLE fields but routing through keeps the graph uniform and
    # lets tables work if widgets were only populated on part of the
    # form.
    g.add_edge("extract_widgets", "extract_tables")

    # Lane B → tables → validate (or downgrade to Lane C)
    g.add_conditional_edges(
        "extract_digital", route_after_digital,
        {"align": "align", "validate": "extract_tables"},
    )

    # Lane C: align → extract_sections → extract_scan → extract_tables → validate
    g.add_edge("align", "extract_sections")
    g.add_edge("extract_sections", "extract_scan")
    g.add_edge("extract_scan", "extract_tables")
    g.add_edge("extract_tables", "validate")

    # Validate → reflect
    g.add_edge("validate", "reflect")

    # Reflect → rescue OR finalize
    g.add_conditional_edges(
        "reflect", route_after_reflect,
        {"rescue": "rescue", "finalize": "finalize"},
    )

    # Rescue loops back to validate (one iteration max enforced in route)
    g.add_edge("rescue", "validate")

    # Finalize ends the graph
    g.add_edge("finalize", END)

    return g.compile()


async def _get_graph():
    global _graph
    async with _graph_lock:
        if _graph is None:
            _graph = build_graph()
    return _graph


async def run_graph(
    file_path: str,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """Run the graph end-to-end on a single file.

    Returns the ``final_response`` dict from the ``finalize`` node.
    """
    graph = await _get_graph()
    state = create_initial_state(file_path, config)
    result = await graph.ainvoke(state)
    return result.get("final_response") or {}


def run_graph_sync(
    file_path: str,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper — safe outside an async context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Nested loop scenario (e.g. inside Streamlit)
            return asyncio.run_coroutine_threadsafe(
                run_graph(file_path, config), loop,
            ).result()
    except RuntimeError:
        pass
    return asyncio.run(run_graph(file_path, config))
