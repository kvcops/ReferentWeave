from typing import List
from core.models import Chunk, Resolution
from core.config import logger

def enrich_chunk(chunk: Chunk, resolutions: List[Resolution]) -> Chunk:
    """
    Appends resolved context block containing high-confidence resolutions to the raw text of the chunk.
    Keeps the original raw text fully intact, appending a cheat sheet block at the end.
    """
    raw_text = chunk.target_text
    
    # Filter for safe, high confidence and resolved entities
    safe_resolutions = [
        r for r in resolutions
        if r.confidence == "high" and r.resolved_entity.upper() != "UNCERTAIN"
    ]
    
    if not safe_resolutions:
        logger.info(f"Chunk {chunk.chunk_id}: No high-confidence coreferences to append.")
        chunk.enriched_text = raw_text
        return chunk
        
    # Build the context block
    context_block = "\n\n[Resolved Context Block:\n"
    for r in safe_resolutions:
        context_block += f"- '{r.original_phrase}' = {r.resolved_entity}\n"
    context_block += "]"
    
    chunk.enriched_text = raw_text + context_block
    logger.info(f"Chunk {chunk.chunk_id}: Appended resolved context block with {len(safe_resolutions)} items.")
    return chunk


def normalize_text(t: str) -> str:
    import string
    t = t.lower()
    t = "".join(c for c in t if c not in string.punctuation)
    return " ".join(t.split())


def enrich_document(document_text: str, resolutions: List[Resolution]) -> str:
    """
    Appends resolved context blocks directly to the paragraphs in the document_text
    where the resolved sentences are found.
    """
    if not resolutions:
        return document_text

    # Filter for safe, high confidence and resolved entities
    safe_resolutions = [
        r for r in resolutions
        if r.confidence == "high" and r.resolved_entity.upper() != "UNCERTAIN"
    ]
    
    if not safe_resolutions:
        logger.info("No high-confidence coreferences to enrich in document.")
        return document_text

    # Split document by paragraphs (splitting by double-newlines)
    paragraphs = document_text.split("\n\n")
    
    # Track which resolutions belong to which paragraph index
    paragraph_resolutions = {i: [] for i in range(len(paragraphs))}
    resolved_matched_count = 0
    
    for res in safe_resolutions:
        orig_sent = res.original_sentence.strip() if res.original_sentence else ""
        if not orig_sent:
            continue
            
        norm_sent = normalize_text(orig_sent)
        
        # Try exact matching in paragraphs first
        matched = False
        for i, para in enumerate(paragraphs):
            if orig_sent in para:
                paragraph_resolutions[i].append(res)
                matched = True
                break
        
        # Try normalized matching if exact fails
        if not matched:
            for i, para in enumerate(paragraphs):
                norm_para = normalize_text(para)
                if norm_sent in norm_para:
                    paragraph_resolutions[i].append(res)
                    matched = True
                    break
                    
        if matched:
            resolved_matched_count += 1
        else:
            logger.warning(f"Could not map resolution to any paragraph: '{orig_sent}'")

    # Reconstruct document with appended context blocks
    enriched_paragraphs = []
    for i, para in enumerate(paragraphs):
        para_res = paragraph_resolutions[i]
        if para_res:
            # Build the context block
            context_block = "\n\n[Resolved Context Block:\n"
            for r in para_res:
                context_block += f"- '{r.original_phrase}' = {r.resolved_entity}\n"
            context_block += "]"
            enriched_paragraphs.append(para + context_block)
        else:
            enriched_paragraphs.append(para)
            
    logger.info(f"Enriched document: mapped {resolved_matched_count}/{len(safe_resolutions)} resolutions to paragraphs.")
    return "\n\n".join(enriched_paragraphs)
