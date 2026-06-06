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
