"""
Visualization utilities for bounding box overlays.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils.models import Block, BlockType, BlockRole


# Color mapping for block types
BLOCK_TYPE_COLORS = {
    BlockType.TEXT: (0, 255, 0),      # Green
    BlockType.TITLE: (255, 0, 0),     # Red
    BlockType.LIST: (0, 0, 255),      # Blue
    BlockType.TABLE: (255, 165, 0),   # Orange
    BlockType.FIGURE: (255, 0, 255),  # Magenta
    BlockType.FORM: (0, 165, 255),    # Orange (forms)
    BlockType.UNKNOWN: (128, 128, 128) # Gray
}

# Color mapping for roles
ROLE_COLORS = {
    BlockRole.TITLE: (255, 0, 0),      # Red
    BlockRole.H1: (200, 0, 0),         # Dark Red
    BlockRole.H2: (150, 0, 0),         # Darker Red
    BlockRole.HEADER: (0, 255, 255),   # Cyan
    BlockRole.FOOTER: (255, 255, 0),   # Yellow
    BlockRole.LIST_ITEM: (0, 0, 255),  # Blue
    BlockRole.CAPTION: (128, 0, 128),  # Purple
}


def get_block_color(block: Block, highlight: bool = False) -> Tuple[int, int, int]:
    """
    Get color for a block.
    
    Args:
        block: Block to get color for
        highlight: Whether to highlight (red)
        
    Returns:
        RGB color tuple
    """
    if highlight:
        return (255, 0, 0)  # Red for highlight
    
    # Try role-based color first
    if block.role and block.role in ROLE_COLORS:
        return ROLE_COLORS[block.role]
    
    # Fall back to type-based color
    return BLOCK_TYPE_COLORS.get(block.type, (128, 128, 128))


def draw_bounding_boxes(
    image: np.ndarray,
    blocks: List[Block],
    highlight_id: Optional[str] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Image to draw on
        blocks: List of blocks to draw
        highlight_id: ID of block to highlight (optional)
        thickness: Line thickness
        
    Returns:
        Image with bounding boxes drawn
    """
    result = image.copy()
    
    for block in blocks:
        x0, y0, x1, y1 = [int(coord) for coord in block.bbox]
        
        # Get color
        highlight = (block.id == highlight_id) if highlight_id else False
        color = get_block_color(block, highlight=highlight)
        
        # Draw rectangle
        cv2.rectangle(result, (x0, y0), (x1, y1), color, thickness)
        
        # Draw label
        label = f"{block.type.value}"
        if block.role:
            label += f" ({block.role.value})"
        
        # Draw text background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            result,
            (x0, y0 - text_height - 4),
            (x0 + text_width, y0),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            result,
            label,
            (x0, y0 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return result


def create_svg_overlay(
    blocks: List[Block],
    width: int,
    height: int,
    highlight_id: Optional[str] = None
) -> str:
    """
    Create SVG overlay for bounding boxes.
    
    Args:
        blocks: List of blocks
        width: Image width
        height: Image height
        highlight_id: ID of block to highlight (optional)
        
    Returns:
        SVG string
    """
    svg_lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    ]
    
    for block in blocks:
        x0, y0, x1, y1 = [float(coord) for coord in block.bbox]
        
        # Get color
        highlight = (block.id == highlight_id) if highlight_id else False
        color = get_block_color(block, highlight=highlight)
        color_str = f"rgb({color[0]}, {color[1]}, {color[2]})"
        
        # Draw rectangle
        rect_width = x1 - x0
        rect_height = y1 - y0
        
        svg_lines.append(
            f'<rect x="{x0}" y="{y0}" width="{rect_width}" height="{rect_height}" '
            f'stroke="{color_str}" stroke-width="2" fill="none" '
            f'data-block-id="{block.id}" '
            f'class="block-overlay {"highlight" if highlight else ""}"/>'
        )
        
        # Draw label
        label = f"{block.type.value}"
        if block.role:
            label += f" ({block.role.value})"
        
        svg_lines.append(
            f'<text x="{x0 + 5}" y="{y0 + 15}" fill="{color_str}" '
            f'font-size="12" font-family="Arial">{label}</text>'
        )
    
    svg_lines.append('</svg>')
    return "\n".join(svg_lines)


def get_block_info(block: Block) -> str:
    """
    Get formatted block information string.
    
    Args:
        block: Block to get info for
        
    Returns:
        Formatted info string
    """
    info_lines = [
        f"ID: {block.id}",
        f"Type: {block.type.value}",
        f"Page: {block.page_id}",
        f"BBox: {block.bbox}",
    ]
    
    if block.role:
        info_lines.append(f"Role: {block.role.value}")
    
    if block.level is not None:
        info_lines.append(f"Level: {block.level}")
    
    if block.text:
        text_preview = block.text[:100] + "..." if len(block.text) > 100 else block.text
        info_lines.append(f"Text: {text_preview}")
    
    if block.confidence < 1.0:
        info_lines.append(f"Confidence: {block.confidence:.2f}")
    
    return "\n".join(info_lines)
