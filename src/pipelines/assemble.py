"""
Assembly pipeline: JSON hierarchy builder and Markdown generator.
"""
from typing import List, Dict, Any, Optional, Tuple
from utils.models import Document, Block, BlockRole, BlockType, TableBlock, FigureBlock, PageImage
from utils.config import Config
from src.pipelines.table_processor import TableProcessor
from src.pipelines.figure_processor import FigureProcessor


class DocumentAssembler:
    """Assemble document into JSON and Markdown."""
    
    def __init__(
        self,
        process_tables: bool = True,
        process_figures: bool = True,
        use_vlm: Optional[bool] = None,
        **_ignored
    ):
        """
        Initialize document assembler.
        
        Args:
            process_tables: Whether to run table extraction
            process_figures: Whether to run figure extraction
            use_vlm: Override for VLM usage (defaults to Config.ENABLE_VLM)
        """
        self.process_tables = process_tables
        self.process_figures = process_figures
        self.use_vlm = Config.ENABLE_VLM if use_vlm is None else use_vlm
        
        self.table_processor = None
        self.figure_processor = None
        
        if process_tables:
            try:
                self.table_processor = TableProcessor(enable_vlm=self.use_vlm)
            except Exception as exc:
                print(f"⚠️ Table processor unavailable: {exc}")
                self.table_processor = None
                self.process_tables = False
        
        if process_figures:
            try:
                self.figure_processor = FigureProcessor(enable_vlm=self.use_vlm)
            except Exception as exc:
                print(f"⚠️ Figure processor unavailable: {exc}")
                self.figure_processor = None
                self.process_figures = False
    
    def _ensure_enriched(self, document: Document):
        """Process tables/figures once per document."""
        if document.metadata.get("_blocks_enriched"):
            return
        document.blocks = self._process_structured_blocks(document.blocks, document.pages)
        document.metadata["_blocks_enriched"] = True
    
    def _process_structured_blocks(
        self,
        blocks: List[Block],
        pages: List[PageImage]
    ) -> List[Block]:
        """Run table + figure processing to enrich block metadata."""
        if not pages or not (self.process_tables or self.process_figures):
            return blocks
        
        page_lookup = {page.page_id: page for page in pages}
        text_by_page: Dict[int, List[Block]] = {}
        for block in blocks:
            if block.type in {BlockType.TEXT, BlockType.TITLE, BlockType.LIST}:
                text_by_page.setdefault(block.page_id, []).append(block)
        
        enriched: List[Block] = []
        # Batch process per page for fewer VLM calls
        for page_id, page in page_lookup.items():
            page_blocks = [b for b in blocks if b.page_id == page_id]
            tables = [
                b for b in page_blocks
                if b.type == BlockType.TABLE and self.table_processor and not b.metadata.get("table_processed")
            ]
            figures = [
                b for b in page_blocks
                if b.type == BlockType.FIGURE and self.figure_processor and not b.metadata.get("figure_processed")
            ]

            table_results: Dict[str, Block] = {}
            if tables:
                processed_tables = self.table_processor.process_table_blocks_batch(
                    tables,
                    page.image,
                    use_vlm=self.use_vlm
                )
                for t in processed_tables:
                    t.metadata = dict(t.metadata)
                    t.metadata["table_processed"] = True
                    table_results[t.id] = t

            figure_results: Dict[str, Block] = {}
            if figures:
                caption_map: Dict[str, str] = {}
                for f in figures:
                    caption = f.metadata.get("caption_candidate_text")
                    if not caption:
                        caption = self.figure_processor.find_caption_for_figure(
                            f,
                            text_by_page.get(f.page_id, [])
                        )
                    if caption:
                        caption_map[f.id] = caption
                processed_figures = self.figure_processor.process_figure_blocks_batch(
                    figures,
                    page.image,
                    captions=caption_map
                )
                for fig in processed_figures:
                    fig.metadata = dict(fig.metadata)
                    fig.metadata["figure_processed"] = True
                    figure_results[fig.id] = fig

            for b in page_blocks:
                if b.id in table_results:
                    enriched.append(table_results[b.id])
                elif b.id in figure_results:
                    enriched.append(figure_results[b.id])
                else:
                    enriched.append(b)
        return enriched
    
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
        Assemble document into production-ready JSON structure with enhanced metadata.
        
        Args:
            document: Document object
            
        Returns:
            Enhanced JSON dictionary with document-level metadata, statistics, and structured content
        """
        self._ensure_enriched(document)
        # Build hierarchy
        blocks = self.build_hierarchy(document.blocks)
        document.blocks = blocks
        
        # Extract document-level information
        doc_metadata = self._extract_document_metadata(document)
        doc_statistics = self._compute_document_statistics(document)
        key_value_pairs = self._extract_key_value_pairs(document.blocks)
        
        # Get base dictionary
        base_dict = document.to_dict()
        
        # Enhance with production-ready structure
        enhanced_json = {
            "document": {
                "id": base_dict.get("doc_id", "unknown"),
                "title": doc_metadata.get("title"),
                "author": doc_metadata.get("author"),
                "date": doc_metadata.get("date"),
                "summary": doc_metadata.get("summary"),
                "type": doc_metadata.get("document_type", "unknown")
            },
            "statistics": doc_statistics,
            "key_value_pairs": key_value_pairs,
            "content": {
                "pages": self._enhance_page_blocks(base_dict.get("pages", []), document.blocks),
                "reading_order": self._compute_reading_order(document.blocks)
            },
            "metadata": {
                **base_dict.get("metadata", {}),
                "processing_timestamp": doc_metadata.get("processing_timestamp"),
                "total_pages": len(document.pages),
                "total_blocks": len(document.blocks),
                "blocks_with_text": doc_statistics.get("blocks_with_text", 0),
                "text_coverage": doc_statistics.get("text_coverage", 0.0)
            }
        }
        form_fields, checkboxes = self._collect_form_outputs(document.blocks)
        enhanced_json["form_fields"] = form_fields
        enhanced_json["checkboxes"] = checkboxes
        enhanced_json["filled_schema"] = self.export_filled_schema(document.blocks)
        return enhanced_json

    def export_filled_schema(self, blocks: List[Block]) -> Dict[str, Any]:
        """
        Export filled schema where keys are schema IDs and values are extracted text/state.
        
        Args:
            blocks: List of blocks
            
        Returns:
            Dictionary mapping schema IDs to values
        """
        filled = {}
        
        for block in blocks:
            meta = getattr(block, "metadata", {}) or {}
            
            # Check checkboxes
            if meta.get("checkbox_state"):
                # Check if this checkbox is linked to a schema field
                # Checkboxes might be linked to a label that has a schema ID?
                # Or the checkbox itself might have been snapped to a schema field
                form_meta = meta.get("form_field")
                schema_id = form_meta.get("schema_id") if form_meta else None
                
                # If check box doesn't have schema_id directly, maybe it's in the label?
                if not schema_id:
                    # This logic depends on how schema is applied to checkboxes
                    # segment.py _apply_template_schema handles FORM blocks
                    # If a checkbox is detected as FORM, it might get schema_id
                    pass
                
                if schema_id:
                    state = meta.get("checkbox_state")
                    # Map to boolean or string
                    value = True if state == "checked" else False
                    filled[schema_id] = value
                elif meta.get("label_text"):
                    # Fallback to label text as key if no schema ID
                    key = meta.get("label_text")
                    state = meta.get("checkbox_state")
                    filled[key] = state
                continue
            
            # Check form fields
            form_meta = meta.get("form_field")
            if form_meta:
                schema_id = form_meta.get("schema_id")
                value = form_meta.get("value_text")
                
                if schema_id:
                    filled[schema_id] = value
                elif form_meta.get("label_text"):
                    # Fallback to label text
                    key = form_meta.get("label_text")
                    filled[key] = value
                    
        return filled

    def _collect_form_outputs(self, blocks: List[Block]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        form_fields: List[Dict[str, Any]] = []
        checkboxes: List[Dict[str, Any]] = []
        for block in blocks:
            meta = getattr(block, "metadata", {}) or {}
            if meta.get("checkbox_state"):
                checkboxes.append({
                    "id": block.id,
                    "page": block.page_id,
                    "bbox": list(block.bbox),
                    "state": meta.get("checkbox_state"),
                    "confidence": meta.get("checkbox_confidence", 0.0),
                    "label_id": meta.get("label_id"),
                    "label_text": meta.get("label_text"),
                    "ocr_confidence": meta.get("ocr_confidence", 0.0),
                    "engine": meta.get("ocr_engine")
                })
                continue
            form_meta = meta.get("form_field")
            if form_meta:
                form_fields.append({
                    "id": block.id,
                    "page": block.page_id,
                    "bbox": list(block.bbox),
                    "label_id": form_meta.get("label_id"),
                    "label_text": form_meta.get("label_text"),
                    "field_type": form_meta.get("field_type"),
                    "value": form_meta.get("value_text"),
                    "validator_passed": form_meta.get("validator_passed", False),
                    "validator_info": form_meta.get("validator_info", {}),
                    "ocr_confidence": meta.get("ocr_confidence", 0.0),
                    "engine": meta.get("ocr_engine")
                })
        return form_fields, checkboxes
    
    def _extract_document_metadata(self, document: Document) -> Dict[str, Any]:
        """
        Extract document-level metadata: title, author, date, summary.
        
        Args:
            document: Document object
            
        Returns:
            Dictionary with extracted metadata
        """
        import re
        from datetime import datetime
        
        metadata = {
            "title": None,
            "author": None,
            "date": None,
            "summary": None,
            "document_type": "unknown",
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Get blocks from first page (usually contains title/header)
        first_page_blocks = [b for b in document.blocks if b.page_id == 0]
        if not first_page_blocks:
            return metadata
        
        # Sort by position (top to bottom, left to right)
        sorted_blocks = sorted(first_page_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
        
        # Extract title (usually first TITLE block or top TEXT block)
        for block in sorted_blocks[:5]:  # Check first 5 blocks
            if block.type == BlockType.TITLE and block.text and block.text.strip():
                metadata["title"] = block.text.strip()
                break
            elif block.type == BlockType.TEXT and block.text and block.text.strip():
                text = block.text.strip()
                # Check if it looks like a title (short, uppercase, or at top of page)
                if block.bbox[1] < document.pages[0].height * 0.2 and len(text) < 200:
                    metadata["title"] = text
                    break
        
        # Extract author (look for patterns like "By:", "Author:", names at top)
        author_patterns = [r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", r"author[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", 
                          r"([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)"]  # "John D. Smith" pattern
        for block in sorted_blocks[:10]:
            if block.text:
                text = block.text.strip()
                for pattern in author_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        metadata["author"] = match.group(1).strip()
                        break
                if metadata["author"]:
                    break
        
        # Extract date (look for date patterns)
        date_patterns = [
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",  # MM/DD/YYYY
            r"\b([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b",  # January 1, 2024
            r"\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b"  # 1 January 2024
        ]
        for block in sorted_blocks:
            if block.text:
                text = block.text.strip()
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        metadata["date"] = match.group(1).strip()
                        break
                if metadata["date"]:
                    break
        
        # Generate summary (first paragraph or first 200 chars of main text)
        for block in sorted_blocks:
            if block.type == BlockType.TEXT and block.text and block.text.strip():
                text = block.text.strip()
                if len(text) > 50:  # Substantial text block
                    metadata["summary"] = text[:300] + "..." if len(text) > 300 else text
                    break
        
        # Determine document type based on content
        all_text = " ".join([b.text for b in document.blocks if b.text])
        if any(word in all_text.lower() for word in ["grant", "proposal", "application"]):
            metadata["document_type"] = "grant_proposal"
        elif any(word in all_text.lower() for word in ["invoice", "bill", "payment"]):
            metadata["document_type"] = "invoice"
        elif any(word in all_text.lower() for word in ["contract", "agreement"]):
            metadata["document_type"] = "contract"
        elif len([b for b in document.blocks if b.type == BlockType.FORM]) > 3:
            metadata["document_type"] = "form"
        elif len([b for b in document.blocks if b.type == BlockType.TABLE]) > 2:
            metadata["document_type"] = "report"
        else:
            metadata["document_type"] = "document"
        
        return metadata
    
    def _compute_document_statistics(self, document: Document) -> Dict[str, Any]:
        """
        Compute document statistics: block counts, text coverage, confidence scores.
        
        Args:
            document: Document object
            
        Returns:
            Dictionary with statistics
        """
        blocks = document.blocks
        
        # Count by type
        type_counts = {}
        for block_type in BlockType:
            type_counts[block_type.value] = len([b for b in blocks if b.type == block_type])
        
        # Text statistics
        blocks_with_text = [b for b in blocks if b.text and b.text.strip()]
        total_text_length = sum(len(b.text) for b in blocks_with_text)
        avg_confidence = sum(b.confidence for b in blocks) / len(blocks) if blocks else 0
        
        # Text coverage (percentage of blocks with text)
        text_coverage = (len(blocks_with_text) / len(blocks) * 100) if blocks else 0
        
        # Word box statistics
        total_words = sum(len(b.word_boxes) for b in blocks if b.word_boxes)
        avg_word_confidence = 0
        if blocks_with_text:
            all_word_confidences = []
            for b in blocks_with_text:
                if b.word_boxes:
                    all_word_confidences.extend([wb.confidence for wb in b.word_boxes])
            avg_word_confidence = sum(all_word_confidences) / len(all_word_confidences) if all_word_confidences else 0
        
        return {
            "total_blocks": len(blocks),
            "blocks_by_type": type_counts,
            "blocks_with_text": len(blocks_with_text),
            "text_coverage_percent": round(text_coverage, 2),
            "total_characters": total_text_length,
            "total_words": total_words,
            "average_block_confidence": round(avg_confidence, 3),
            "average_word_confidence": round(avg_word_confidence, 3),
            "pages_processed": len(document.pages)
        }
    
    def _extract_key_value_pairs(self, blocks: List[Block]) -> List[Dict[str, Any]]:
        """
        Extract key-value pairs from form blocks and structured text.
        
        Args:
            blocks: List of blocks
            
        Returns:
            List of key-value pairs
        """
        import re
        
        kv_pairs = []
        
        # Look for form blocks
        form_blocks = [b for b in blocks if b.type == BlockType.FORM]
        
        # Pattern for key-value pairs: "Label: Value" or "Label Value"
        kv_patterns = [
            r"^([^:]+):\s*(.+)$",  # "Name: John"
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+([A-Z0-9@.]+)$"  # "Email user@example.com"
        ]
        
        for block in blocks:
            if not block.text or not block.text.strip():
                continue
            
            text = block.text.strip()
            
            # Check for key-value patterns
            for pattern in kv_patterns:
                match = re.match(pattern, text)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip() if len(match.groups()) > 1 else ""
                    if len(key) < 50 and len(value) < 200:  # Reasonable sizes
                        kv_pairs.append({
                            "key": key,
                            "value": value,
                            "block_id": block.id,
                            "page": block.page_id,
                            "bbox": block.bbox,
                            "confidence": block.confidence
                        })
                        break
        
        return kv_pairs
    
    def _compute_reading_order(self, blocks: List[Block]) -> List[str]:
        """
        Compute reading order (list of block IDs in reading order).
        
        Args:
            blocks: List of blocks
            
        Returns:
            List of block IDs in reading order
        """
        # Sort by page, then by y-position (top to bottom), then by x-position (left to right)
        sorted_blocks = sorted(blocks, key=lambda b: (b.page_id, b.bbox[1], b.bbox[0]))
        return [block.id for block in sorted_blocks]
    
    def _enhance_page_blocks(
        self,
        pages_list: List[Dict[str, Any]],
        all_blocks: List[Block]
    ) -> List[Dict[str, Any]]:
        """
        Enhance page blocks with detailed metadata for each block.
        
        Args:
            pages_list: List of page dictionaries from base_dict
            all_blocks: All blocks in the document
            
        Returns:
            Enhanced pages list with detailed block information
        """
        block_dict = {block.id: block for block in all_blocks}
        
        enhanced_pages = []
        for page_dict in pages_list:
            enhanced_page = dict(page_dict)
            enhanced_blocks = []
            
            for block_dict_item in page_dict.get("blocks", []):
                block_id = block_dict_item.get("id")
                if block_id in block_dict:
                    block = block_dict[block_id]
                    enhanced_block = self._create_detailed_block_dict(block)
                    enhanced_blocks.append(enhanced_block)
                else:
                    enhanced_blocks.append(block_dict_item)
            
            enhanced_page["blocks"] = enhanced_blocks
            enhanced_pages.append(enhanced_page)
        
        return enhanced_pages
    
    def _create_detailed_block_dict(self, block: Block) -> Dict[str, Any]:
        """
        Create a detailed dictionary for a block with comprehensive metadata.
        
        Args:
            block: Block object
            
        Returns:
            Detailed block dictionary with all metadata and reasoning
        """
        base_dict = block.to_dict()
        
        # Extract detection information
        detection_method = block.metadata.get("detection_method", "unknown")
        detected_by = block.metadata.get("detected_by", "unknown")
        reclassified_by = block.metadata.get("reclassified_by")
        original_type = block.metadata.get("original_type")
        
        # Build detection explanation
        detection_explanation = {
            "primary_method": detection_method,
            "detector": detected_by,
            "confidence": block.confidence,
            "was_reclassified": reclassified_by is not None
        }
        
        if reclassified_by:
            detection_explanation["reclassification"] = {
                "method": reclassified_by,
                "original_type": original_type,
                "reasoning": block.metadata.get("reclassification_reasoning", "texture_analysis")
            }
        
        # Add texture features if available
        texture_features = block.metadata.get("texture_features")
        if texture_features:
            detection_explanation["texture_analysis"] = {
                "text_density": round(texture_features.get("text_density", 0), 4),
                "line_density": round(texture_features.get("line_density", 0), 4),
                "horizontal_density": round(texture_features.get("horizontal_density", 0), 4),
                "vertical_density": round(texture_features.get("vertical_density", 0), 4),
                "stroke_density": round(texture_features.get("stroke_density", 0), 4)
            }
        
        # Add OCR information if available
        ocr_info = {}
        if block.word_boxes:
            ocr_info["word_count"] = len(block.word_boxes)
            if block.word_boxes:
                avg_ocr_confidence = sum(wb.confidence for wb in block.word_boxes) / len(block.word_boxes)
                ocr_info["average_word_confidence"] = round(avg_ocr_confidence, 3)
                ocr_info["text_extracted"] = bool(block.text and block.text.strip())
        
        # Add block characteristics
        block_area = (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1])
        aspect_ratio = (block.bbox[2] - block.bbox[0]) / (block.bbox[3] - block.bbox[1]) if (block.bbox[3] - block.bbox[1]) > 0 else 0
        
        characteristics = {
            "area": round(block_area, 2),
            "aspect_ratio": round(aspect_ratio, 3),
            "width": round(block.bbox[2] - block.bbox[0], 2),
            "height": round(block.bbox[3] - block.bbox[1], 2)
        }
        
        # Add reasoning/explanation
        reasoning = block.metadata.get("reasoning")
        if not reasoning and reclassified_by:
            reasoning = block.metadata.get("reclassification_reasoning")
        
        # Enhanced metadata
        enhanced_metadata = {
            **base_dict.get("metadata", {}),
            "detection": detection_explanation,
            "characteristics": characteristics,
            "ocr": ocr_info if ocr_info else None,
            "reasoning": reasoning
        }
        
        # Add all original metadata fields
        for key, value in block.metadata.items():
            if key not in enhanced_metadata:
                enhanced_metadata[key] = value
        
        base_dict["metadata"] = enhanced_metadata
        
        return base_dict
    
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
        self._ensure_enriched(document)
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
