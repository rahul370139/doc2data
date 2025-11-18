"""Unit tests for form geometry helpers and validators."""
from utils.models import Block, BlockType
from src.pipelines.segment import (
    assign_columns_to_blocks,
    compute_template_column_centers,
    DEFAULT_TEMPLATE_LAYOUTS,
)
from src.pipelines.validators import validate_field, guess_field_type
from src.pipelines.ocr import compute_label_field_cost


def _make_block(block_id: str, x0: float, y0: float, x1: float, y1: float, page_id: int = 0) -> Block:
    return Block(
        id=block_id,
        type=BlockType.TEXT,
        bbox=(x0, y0, x1, y1),
        page_id=page_id,
        metadata={},
    )


def test_assign_columns_auto_clusters_by_position():
    blocks = [
        _make_block("b1", 50, 100, 150, 200),
        _make_block("b2", 400, 110, 520, 210),
        _make_block("b3", 760, 120, 860, 220),
    ]
    assign_columns_to_blocks(blocks, page_width=1000)
    assert [blk.metadata.get("column_id") for blk in blocks] == [0, 1, 2]


def test_assign_columns_uses_template_centers():
    blocks = [
        _make_block("b1", 100, 100, 200, 200),
        _make_block("b2", 500, 100, 620, 200),
    ]
    centers = compute_template_column_centers(DEFAULT_TEMPLATE_LAYOUTS, "cms-1500", page_width=1000)
    assign_columns_to_blocks(blocks, page_width=1000, template_centers=centers)
    assert blocks[0].metadata.get("column_id") == 0
    assert blocks[1].metadata.get("column_id") in {1, 2, 3}


def test_validator_and_guess_field_type():
    ok, info = validate_field("npi", "1000000008")
    assert ok and info.get("normalized") == "1000000008"
    assert guess_field_type("Provider NPI") == "npi"


def test_compute_label_field_cost_prefers_near_label():
    field = _make_block("field", 400, 200, 700, 260)
    close_label = _make_block("lbl1", 300, 205, 380, 250)
    far_label = _make_block("lbl2", 50, 50, 120, 80)
    close_cost = compute_label_field_cost(field, close_label)
    far_cost = compute_label_field_cost(field, far_label)
    assert close_cost < far_cost
