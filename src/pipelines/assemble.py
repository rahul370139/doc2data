"""
Assembly pipeline: JSON hierarchy builder and Markdown generator.
"""
from typing import List, Dict, Any
from utils.models import Document, Block, BlockRole, BlockType, TableBlock, FigureBlock


class DocumentAssembler:
    """Assemble document into JSON and Markdown."""
    
    def __init__(self):
        """Initialize document assembler."""
        pass
    
    def build_hierarchy(self, blocks: List[Block]) -> List[Block]:
        """
        Build parent-child hierarchy based on headers and spatial relationships.
        
        Args:
            blocks: List of blocks
            
        Returns:
            Blocks with children populated
        """
        # Sort blocks by page and reading order
        sorted_blocks = sorted(blocks, key=lambda b: (b.page_id, b.bbox[1], b.bbox[0]))
        
        # Build header stack
        header_stack = []
        
        for block in sorted_blocks:
            # If this is a header, update stack
            if block.role in [BlockRole.TITLE, BlockRole.H1, BlockRole.H2]:
                level = block.level or 0
                
                # Remove headers at same or deeper level
                header_stack = [h for h in header_stack if (h.get("level", 0) or 0) < level]
                
                # Add current header to stack
                header_stack.append({
                    "id": block.id,
                    "level": level
                })
            
            # Assign children based on header stack
            if header_stack:
                parent_id = header_stack[-1]["id"]
                if parent_id != block.id:
                    # Find parent block and add this as child
                    for parent_block in sorted_blocks:
                        if parent_block.id == parent_id:
                            if block.id not in parent_block.children:
                                parent_block.children.append(block.id)
                            break
        
        return sorted_blocks
    
    def assemble_json(self, document: Document) -> Dict[str, Any]:
        """
        Assemble document into JSON structure.
        
        Args:
            document: Document object
            
        Returns:
            JSON dictionary
        """
        # Build hierarchy
        blocks = self.build_hierarchy(document.blocks)
        document.blocks = blocks
        
        # Convert to dictionary
        return document.to_dict()
    
    def assemble_markdown(
        self,
        document: Document,
        include_page_breaks: bool = False
    ) -> str:
        """
        Assemble document into Markdown.
        
        Args:
            document: Document object
            include_page_breaks: Whether to include page breaks
            
        Returns:
            Markdown string
        """
        # Build hierarchy
        blocks = self.build_hierarchy(document.blocks)
        
        # Sort blocks by page and reading order
        sorted_blocks = sorted(blocks, key=lambda b: (b.page_id, b.bbox[1], b.bbox[0]))
        
        markdown_lines = []
        current_page = -1
        
        for block in sorted_blocks:
            # Add page break if needed
            if include_page_breaks and block.page_id != current_page:
                if current_page >= 0:
                    markdown_lines.append("\n---\n")
                current_page = block.page_id
            
            # Convert block to Markdown
            block_md = self._block_to_markdown(block)
            if block_md:
                markdown_lines.append(block_md)
        
        return "\n".join(markdown_lines)
    
    def _block_to_markdown(self, block: Block) -> str:
        """
        Convert a block to Markdown.
        
        Args:
            block: Block to convert
            
        Returns:
            Markdown string
        """
        if isinstance(block, TableBlock):
            return self._table_to_markdown(block)
        elif isinstance(block, FigureBlock):
            return self._figure_to_markdown(block)
        else:
            return self._text_block_to_markdown(block)
    
    def _text_block_to_markdown(self, block: Block) -> str:
        """Convert text block to Markdown."""
        text = block.text or ""
        
        if not text.strip():
            return ""
        
        # Apply role-based formatting
        if block.role == BlockRole.TITLE:
            return f"# {text}\n"
        elif block.role == BlockRole.H1:
            return f"## {text}\n"
        elif block.role == BlockRole.H2:
            return f"### {text}\n"
        elif block.role == BlockRole.LIST_ITEM:
            return f"- {text}\n"
        elif block.role == BlockRole.HEADER:
            return f"**{text}**\n"
        elif block.role == BlockRole.FOOTER:
            return f"*{text}*\n"
        elif block.role == BlockRole.CAPTION:
            return f"*{text}*\n"
        elif block.type == BlockType.FORM:
            return f"[Form {block.id} region]\n"
        else:
            return f"{text}\n"
    
    def _table_to_markdown(self, table_block: TableBlock) -> str:
        """Convert table block to Markdown."""
        if not table_block.headers and not table_block.body:
            if table_block.shape:
                rows, cols = table_block.shape
                return f"[Table {table_block.id} region ~ {rows}x{cols}]\n"
            return f"[Table {table_block.id} region]\n"
        
        lines = []
        
        # Headers
        if table_block.headers:
            for header_row in table_block.headers:
                lines.append("| " + " | ".join(header_row) + " |")
                lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")
        
        # Body
        if table_block.body:
            for row in table_block.body:
                lines.append("| " + " | ".join(row) + " |")
        
        if lines:
            return "\n".join(lines) + "\n"
        else:
            return f"[Table {table_block.id} region]\n"
    
    def _figure_to_markdown(self, figure_block: FigureBlock) -> str:
        """Convert figure block to Markdown."""
        caption = figure_block.caption or f"Figure {figure_block.id}"
        
        if figure_block.data_or_link:
            return f"![{caption}]({figure_block.data_or_link})\n"
        else:
            return f"[{caption}]\n"
