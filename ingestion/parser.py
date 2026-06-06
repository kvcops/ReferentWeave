import re
import string
import logging
from pathlib import Path
from typing import List, Dict, Any
from core.config import logger
from core.models import Chunk
from google import genai

def is_dirty_pdf(text: str, threshold: float = 0.3) -> bool:
    """
    Checks if a PDF has a high non-alphanumeric ratio, which indicates
    scanning errors, watermarks, or embedded junk.
    """
    if not text:
        return False
    
    # Count alphanumeric characters vs total characters
    total_chars = len(text)
    alnum_chars = sum(1 for c in text if c.isalnum() or c.isspace())
    
    non_alnum_ratio = 1.0 - (alnum_chars / total_chars)
    logger.debug(f"Pre-flight check: non-alphanumeric ratio is {non_alnum_ratio:.2f}")
    return non_alnum_ratio > threshold

def summarize_bloated_table(table_text: str) -> str:
    """
    Uses Gemini 3.1 Flash-Lite to summarize a wide markdown table
    into a 2-sentence structural summary.
    """
    logger.info("Mitigating Table Header Bloat: generating structural summary for markdown table.")
    try:
        from core.config import GEMINI_API_KEY, GEMINI_MODEL
        if GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            client = genai.Client()
        
        prompt = (
            "Summarize the following markdown table into exactly a 2-sentence structural summary. "
            "Highlight the main columns, the entities described, and the key metrics or relationships. "
            "Do not output any introductory or concluding text, only output the 2-sentence summary."
        )
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, table_text]
        )
        summary = response.text.strip()
        logger.info(f"Generated table summary: '{summary}'")
        return f"\n\n[Table Summary: {summary}]\n\n"
    except Exception as e:
        logger.error(f"Failed to generate table summary using Gemini: {e}. Keeping raw table text.")
        return table_text

def parse_with_docling(file_path: Path) -> str:
    """
    Parses a PDF/Doc using IBM Docling, exporting to Markdown.
    """
    logger.info(f"Attempting to parse document {file_path.name} with IBM Docling...")
    from docling.document_converter import DocumentConverter
    
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    markdown_text = result.document.export_to_markdown()
    logger.info("IBM Docling parsing completed successfully.")
    return markdown_text

def parse_with_pypdf(file_path: Path) -> str:
    """
    Fallback parser using pypdf.
    """
    logger.info(f"Using fallback parser (pypdf) for document {file_path.name}...")
    from pypdf import PdfReader
    
    reader = PdfReader(str(file_path))
    pages_text = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)
    
    logger.info(f"pypdf parsing completed. Extracted {len(reader.pages)} pages.")
    return "\n\n".join(pages_text)

def parse_document(file_path: Path) -> str:
    """
    Loads and parses a PDF document using Docling or pypdf fallback.
    Runs pre-flight heuristic checks.
    """
    try:
        # Attempt Docling
        text = parse_with_docling(file_path)
    except Exception as e:
        logger.warning(f"Docling parser failed or not available ({e}). Falling back to pypdf.")
        try:
            text = parse_with_pypdf(file_path)
        except Exception as fallback_err:
            logger.error(f"Fallback parser also failed: {fallback_err}")
            raise fallback_err
            
    # Pre-flight check
    if is_dirty_pdf(text, threshold=0.3):
        logger.warning(
            f"Pre-flight heuristic warning: Document '{file_path.name}' has a high ratio (>30%) of "
            f"non-alphanumeric characters. It may be a scanned or dirty PDF. Route to review or vision parser."
        )
        
    return text

def chunk_document_text(text: str, max_chunk_len: int = 1000) -> List[Dict[str, Any]]:
    """
    Splits the parsed text/markdown into logical chunks (paragraphs and sections).
    Monitors table header bloat and summarizes tables if they exceed threshold.
    """
    logger.info("Splitting document into logical chunks...")
    
    # Extract markdown tables if present
    # A simple regex for markdown tables: rows starting/ending with |
    table_pattern = re.compile(r"((?:^\|.+\|$\n?)+)", re.MULTILINE)
    
    # We replace bloated tables with their summary
    processed_text = text
    matches = list(table_pattern.finditer(text))
    
    # Iterate in reverse to avoid index shifts during replacement
    for match in reversed(matches):
        table_str = match.group(1)
        # Check table header size (first row of table)
        lines = table_str.strip().split("\n")
        if lines:
            header_len = len(lines[0])
            # If the header takes up more than 15% of the max_chunk_len (e.g. 150 chars)
            if header_len > (max_chunk_len * 0.15):
                table_summary = summarize_bloated_table(table_str)
                processed_text = processed_text[:match.start()] + table_summary + processed_text[match.end():]

    # Split by paragraphs or markdown headers
    raw_splits = re.split(r"(?:\n\s*\n|(?=^#{1,4}\s))", processed_text, flags=re.MULTILINE)
    
    chunks = []
    current_chunk = ""
    current_headers = []
    
    for split in raw_splits:
        split = split.strip()
        if not split:
            continue
            
        # Track active headers for metadata
        header_match = re.match(r"^(#{1,4})\s+(.+)$", split)
        if header_match:
            level = len(header_match.group(1))
            header_text = header_match.group(2)
            # Resize headers list
            current_headers = current_headers[:level - 1]
            current_headers.append(header_text)
            
        # Combine splits to reach optimal chunk size
        if len(current_chunk) + len(split) < max_chunk_len:
            current_chunk += "\n\n" + split if current_chunk else split
        else:
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {
                        "headers": list(current_headers)
                    }
                })
            # If the single split is itself too large, break it down
            if len(split) >= max_chunk_len:
                words = split.split()
                sub_chunk = ""
                for word in words:
                    if len(sub_chunk) + len(word) + 1 < max_chunk_len:
                        sub_chunk += " " + word if sub_chunk else word
                    else:
                        chunks.append({
                            "text": sub_chunk.strip(),
                            "metadata": {
                                "headers": list(current_headers)
                            }
                        })
                        sub_chunk = word
                current_chunk = sub_chunk
            else:
                current_chunk = split
                
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "metadata": {
                "headers": list(current_headers)
            }
        })
        
    logger.info(f"Document chunking complete. Created {len(chunks)} chunks.")
    return chunks

def build_context_chunks(raw_chunks: List[Dict[str, Any]], doc_id: str) -> List[Chunk]:
    """
    Pairs each chunk with its predecessor context and builds typed Chunk objects.
    """
    logger.info("Pairing chunks with predecessor context...")
    context_chunks = []
    
    for i, chunk in enumerate(raw_chunks):
        background = raw_chunks[i-1]["text"] if i > 0 else ""
        
        # Incremental stable ID (uint64 format)
        chunk_id = 1000 + i
        
        context_chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            target_text=chunk["text"],
            background_context=background,
            enriched_text="",  # Will be filled by enricher
            metadata=chunk["metadata"]
        ))
        
    logger.info(f"Successfully constructed {len(context_chunks)} sliding paired chunks.")
    return context_chunks
