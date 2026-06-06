import re
import string
import logging
from pathlib import Path
from typing import List, Dict, Any
from core.config import logger, MAX_DOCLING_PAGES
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
    Locally summarizes a markdown table in Python without using any external API calls.
    Extracts headers, row counts, and a brief compact preview to preserve retrieval semantics.
    """
    logger.info("Mitigating Table Header Bloat: generating local structural summary for markdown table.")
    try:
        lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
        if not lines:
            return ""
            
        # Parse headers from the first line
        header_line = lines[0]
        headers = [h.strip() for h in header_line.split("|") if h.strip()]
        
        # Filter out line separators (e.g., containing only dashes, colons, or spaces)
        row_lines = []
        for line in lines[1:]:
            # Check if this line is just a separator like |---|---|
            clean_line = line.replace("|", "").replace("-", "").replace(":", "").strip()
            if clean_line:
                row_lines.append(line)
                
        row_count = len(row_lines)
        headers_str = ", ".join(headers) if headers else "None"
        
        # Build a preview of the first 2 data rows
        preview_rows = []
        for r_line in row_lines[:2]:
            r_cells = [c.strip() for c in r_line.split("|") if c.strip()]
            if r_cells:
                preview_rows.append(" | ".join(r_cells))
                
        preview_str = "; ".join(preview_rows) if preview_rows else "No data"
        
        summary = (
            f"\n\n[Table Summary: Contains {row_count} rows across columns ({headers_str}). "
            f"Preview data: {preview_str}]\n\n"
        )
        logger.info(f"Generated local table summary for table with {row_count} rows.")
        return summary
    except Exception as e:
        logger.error(f"Failed to locally summarize table: {e}. Keeping raw table text.")
        return table_text

def is_digitally_born_pdf(file_path: Path) -> bool:
    """
    Checks if a PDF document has an embedded text layer (is digitally born).
    We check the first few pages to see if any text can be extracted.
    """
    if file_path.suffix.lower() != ".pdf":
        return True
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        num_pages = len(reader.pages)
        if num_pages == 0:
            return False
            
        sample_pages = min(5, num_pages)
        total_text_len = 0
        for i in range(sample_pages):
            page_text = reader.pages[i].extract_text() or ""
            total_text_len += len(page_text.strip())
            
        has_text = total_text_len > (sample_pages * 10)
        logger.info(f"Digitally born check for {file_path.name}: has_text_layer={has_text} (extracted {total_text_len} chars from {sample_pages} pages)")
        return has_text
    except Exception as e:
        logger.warning(f"Error checking if PDF is digitally born: {e}. Defaulting to False.")
        return False

def parse_with_docling(file_path: Path) -> str:
    """
    Parses a PDF/Doc using IBM Docling, exporting to Markdown.
    Skips OCR if the document is digitally born to prevent high memory consumption (OOM / std::bad_alloc).
    """
    digitally_born = is_digitally_born_pdf(file_path)
    
    if digitally_born:
        logger.info(f"Attempting to parse document {file_path.name} with IBM Docling (OCR disabled: digitally born)...")
    else:
        logger.info(f"Attempting to parse document {file_path.name} with IBM Docling (OCR enabled: scanned PDF)...")
        
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat, ConversionStatus
    
    # Configure Docling to be lightweight (disable OCR only if digitally born)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = not digitally_born
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(file_path))
    
    # If the conversion was not a complete success (e.g. PARTIAL_SUCCESS or FAILURE),
    # raise a ValueError so we immediately fall back to pypdf instead of returning a truncated file.
    if result.status != ConversionStatus.SUCCESS:
        raise ValueError(
            f"Docling conversion did not succeed completely (status: {result.status})."
        )
        
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
    text = ""
    # Precheck page count for PDFs to avoid C++ std::bad_alloc (OOM) on large documents
    if file_path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            num_pages = len(reader.pages)
            if num_pages > MAX_DOCLING_PAGES:
                logger.warning(
                    f"Document '{file_path.name}' has {num_pages} pages, which exceeds the limit of "
                    f"{MAX_DOCLING_PAGES} pages for Docling layout processing on this system. "
                    f"Bypassing Docling entirely to avoid out-of-memory errors (std::bad_alloc). Falling back to pypdf."
                )
                return parse_with_pypdf(file_path)
        except Exception as pe:
            logger.warning(f"Failed to check PDF page count: {pe}. Proceeding to Docling.")

    try:
        # Attempt Docling
        text = parse_with_docling(file_path)
        if not text or len(text.strip()) < 100:
            logger.warning("Docling returned empty or extremely short text. Falling back to pypdf.")
            text = parse_with_pypdf(file_path)
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
