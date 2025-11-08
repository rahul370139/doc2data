"""
Semantic labeling pipeline - STUBBED (Step 4 skipped for local testing).
LLM/VLM integration will be added when GPU stack is available.
"""
from typing import List, Dict, Any, Optional
from utils.models import Block, BlockRole, BlockType


class SLMLabeler:
    """
    Semantic labeling using SLM - STUBBED.
    
    This is a no-op implementation that skips LLM/VLM calls.
    Blocks will retain their basic types but won't have semantic roles assigned.
    To enable: Set ENABLE_SLM=true in .env and ensure Ollama is running.
    """
    
    def __init__(self, enabled: bool = False):
        """
        Initialize SLM labeler.
        
        Args:
            enabled: Whether to enable LLM labeling (default: False for local testing)
        """
        self.enabled = enabled
        self.client = None
        self.model = None
        
        if self.enabled:
            try:
                from src.vlm.ollama_client import get_ollama_client
                from utils.config import Config
                self.client = get_ollama_client()
                self.model = Config.SLM_MODEL
                print("✓ SLM labeling enabled (Ollama required)")
            except Exception as e:
                print(f"⚠ SLM enabled but Ollama unavailable: {e}")
                print("  Continuing with stub (no semantic roles assigned)")
                self.enabled = False
    
    def create_prompt(self, block: Block, context: Dict[str, Any]) -> str:
        """
        Create prompt for semantic labeling - STUBBED.
        
        Returns empty string when disabled.
        When enabled, creates prompt for LLM.
        
        Args:
            block: Block to label
            context: Context information (page, bbox, flags)
            
        Returns:
            Prompt string (empty when disabled)
        """
        if not self.enabled:
            return ""
        text = block.text or ""
        page = context.get("page", block.page_id)
        bbox = context.get("bbox", block.bbox)
        repeated_header = context.get("repeated_header", False)
        near_figure = context.get("near_figure", False)
        
        prompt = f"""You are a document parser. Classify the following text block into one of these labels:

Labels: ["title", "h1", "h2", "header", "footer", "page_num", "list_item", "kv_label", "kv_value", "caption", "paragraph", "unknown"]

Input text: {text[:500]}
Position: page={page}, bbox={bbox}
Repeated header: {repeated_header}
Near figure: {near_figure}

Examples:
- "Introduction" → {{"role": "h1"}}
- "1. Overview" → {{"role": "h2"}}
- "Page 1" → {{"role": "page_num"}}
- "Name: John" → {{"role": "kv_label"}} (for "Name:") and {{"role": "kv_value"}} (for "John")
- "• Item 1" → {{"role": "list_item"}}

Respond with ONLY a JSON object: {{"role": "<LABEL>"}}"""
        
        return prompt
    
    def label_block(self, block: Block, context: Dict[str, Any]) -> Block:
        """
        Label a single block with semantic role - STUBBED.
        
        When disabled, returns block unchanged (no semantic roles assigned).
        
        Args:
            block: Block to label
            context: Context information
            
        Returns:
            Block (unchanged when disabled)
        """
        if not self.enabled:
            # Stub: return block unchanged
            return block
        
        # Skip if not a text block
        if block.type not in [BlockType.TEXT, BlockType.TITLE, BlockType.LIST]:
            return block
        
        # Skip if no text
        if not block.text or len(block.text.strip()) == 0:
            return block
        
        try:
            import json
            # Create prompt
            prompt = self.create_prompt(block, context)
            
            # Call Ollama
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json"
            )
            
            # Parse response
            response_text = response.get("response", "")
            
            # Extract JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                role_str = result.get("role", "unknown")
                
                # Map to BlockRole enum
                try:
                    block.role = BlockRole(role_str)
                except ValueError:
                    block.role = BlockRole.UNKNOWN
                
                # Set header level
                if block.role == BlockRole.TITLE:
                    block.level = 0
                elif block.role == BlockRole.H1:
                    block.level = 1
                elif block.role == BlockRole.H2:
                    block.level = 2
        
        except Exception as e:
            print(f"Error labeling block {block.id}: {e}")
            block.role = BlockRole.UNKNOWN
        
        return block
    
    def label_blocks(
        self,
        blocks: List[Block],
        pages: List[Any]
    ) -> List[Block]:
        """
        Label multiple blocks with semantic roles.
        
        Args:
            blocks: List of blocks to label
            pages: List of page images (for context)
            
        Returns:
            List of labeled blocks
        """
        labeled_blocks = []
        
        for block in blocks:
            # Build context
            context = {
                "page": block.page_id,
                "bbox": block.bbox,
                "repeated_header": block.metadata.get("role_hint") == "header",
                "near_figure": block.metadata.get("caption_candidate", False)
            }
            
            # Label block
            labeled_block = self.label_block(block, context)
            labeled_blocks.append(labeled_block)
        
        return labeled_blocks

