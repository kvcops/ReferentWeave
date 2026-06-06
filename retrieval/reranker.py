import json
from pydantic import BaseModel, Field
from typing import List
from core.config import logger, GEMINI_API_KEY, GEMINI_MODEL, TOP_K_GENERATION
from core.models import QueryResult
from google import genai

class ChunkScore(BaseModel):
    chunk_id: int = Field(description="The unique ID of the chunk being evaluated.")
    relevance_score: float = Field(description="Relevance score from 0.0 (completely irrelevant) to 10.0 (highly relevant to answering the query).")

class RerankingResponse(BaseModel):
    scores: List[ChunkScore] = Field(description="List of relevance scores for each chunk.")

class GeminiReranker:
    def __init__(self):
        if GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.client = genai.Client()
        
    def rerank(self, query: str, candidates: List[QueryResult]) -> List[QueryResult]:
        """
        Reranks top candidates using Gemini 3.1 Flash-Lite as a Cross-Encoder.
        Truncates chunks to 512 tokens (roughly 2000 characters) before sending to preserve latency.
        """
        if not candidates:
            return []
            
        logger.info(f"Reranking {len(candidates)} candidates using Gemini Cross-Encoder...")
        
        # 1. Truncate chunks & build candidate payload
        candidate_items = []
        candidate_map = {}
        for c in candidates:
            # Latency mitigation: truncate to first 2000 characters
            truncated_text = c.text[:2000]
            candidate_items.append({
                "chunk_id": c.chunk_id,
                "text": truncated_text
            })
            candidate_map[c.chunk_id] = c
            
        # 2. Prepare Reranking Prompt
        prompt = (
            "You are a high-performance cross-encoder reranking model. "
            f"User Query: '{query}'\n\n"
            "Evaluate the relevance of each text chunk below to the User Query. "
            "For each chunk, assign a relevance score between 0.0 (not relevant at all) and 10.0 (contains direct answer/crucial context). "
            "Output your scores strictly in the requested JSON schema.\n\n"
            f"Candidate Chunks:\n{json.dumps(candidate_items, indent=2)}"
        )
        
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": RerankingResponse.model_json_schema()
                }
            )
            
            # Parse response
            data = json.loads(response.text.strip())
            scores_obj = RerankingResponse(**data)
            
            # Map scores back to QueryResult objects
            scored_candidates = []
            scored_ids = set()
            for s in scores_obj.scores:
                if s.chunk_id in candidate_map:
                    candidate = candidate_map[s.chunk_id]
                    # Update score with the cross-encoder score
                    candidate.score = s.relevance_score
                    scored_candidates.append(candidate)
                    scored_ids.add(s.chunk_id)
                    
            # Add any candidates that the model missed with a default score of 0
            for cid, candidate in candidate_map.items():
                if cid not in scored_ids:
                    candidate.score = 0.0
                    scored_candidates.append(candidate)
                    
            # Sort candidates by relevance score descending
            sorted_candidates = sorted(scored_candidates, key=lambda c: c.score, reverse=True)
            
            logger.info("Reranking completed successfully.")
            for c in sorted_candidates[:TOP_K_GENERATION]:
                logger.info(f" - Chunk ID: {c.chunk_id} (Reranked Score: {c.score:.2f})")
                
            return sorted_candidates[:TOP_K_GENERATION]
            
        except Exception as e:
            logger.error(f"Failed to rerank candidates: {e}. Falling back to default RRF order.")
            # If cross-encoder fails, fall back to default candidates order
            return candidates[:TOP_K_GENERATION]
