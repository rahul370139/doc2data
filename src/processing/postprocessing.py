"""
Text post-processing utilities: cleaning, formatting, hyphenation fixes.
"""
import re
from typing import List, Tuple


def fix_hyphenation(text: str) -> str:
    """
    Fix hyphenated line breaks in text.
    
    Args:
        text: Text with potential hyphenation breaks
        
    Returns:
        Text with hyphenation fixed
    """
    # Pattern: word ending with hyphen followed by newline and lowercase letter
    # Example: "exam-\nple" -> "example"
    pattern = r'([a-zA-Z])-\s*\n\s*([a-z])'
    text = re.sub(pattern, r'\1\2', text)
    
    # Also handle cases where hyphen is at end of line
    pattern2 = r'([a-zA-Z])-\s+([a-z])'
    text = re.sub(pattern2, r'\1\2', text)
    
    return text


def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and fixing common issues.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with single newline
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Fix common OCR errors
    text = text.replace('|', 'l')  # Common OCR error
    text = text.replace('0', 'O')  # Context-dependent, might need improvement
    
    return text


def join_hyphenated_breaks(text_lines: List[str]) -> str:
    """
    Join lines that were broken by hyphenation.
    
    Args:
        text_lines: List of text lines
        
    Returns:
        Joined text with hyphenation fixed
    """
    if not text_lines:
        return ""
    
    # Join lines
    text = "\n".join(text_lines)
    
    # Fix hyphenation
    text = fix_hyphenation(text)
    
    # Clean text
    text = clean_text(text)
    
    return text


def strip_whitespace(text: str) -> str:
    """
    Strip whitespace from text while preserving structure.
    
    Args:
        text: Text with potential whitespace issues
        
    Returns:
        Text with whitespace cleaned
    """
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    
    # Remove empty lines at start and end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    
    # Join lines
    return '\n'.join(lines)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Text with inconsistent whitespace
        
    Returns:
        Text with normalized whitespace
    """
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    return text


def postprocess_text(text: str, fix_hyphens: bool = True, clean: bool = True) -> str:
    """
    Apply all post-processing steps to text.
    
    Args:
        text: Raw text
        fix_hyphens: Whether to fix hyphenation
        clean: Whether to clean text
        
    Returns:
        Post-processed text
    """
    if fix_hyphens:
        text = fix_hyphenation(text)
    
    if clean:
        text = clean_text(text)
        text = normalize_whitespace(text)
        text = strip_whitespace(text)
    
    return text

