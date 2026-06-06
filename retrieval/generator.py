from typing import List, Dict, Any
from core.config import logger, GEMINI_API_KEY, GEMINI_MODEL
from core.models import QueryResult
from google import genai

class GroundedGenerator:
    def __init__(self):
        if GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.client = genai.Client()
        
    def generate_answer(self, query: str, context_chunks: List[QueryResult]) -> str:
        """
        Synthesizes the final answer using Gemini 3.1 Flash-Lite.
        Strictly uses the raw, untouched text of the top chunks, enforcing citations.
        """
        if not context_chunks:
            return "No relevant context was found to answer your question."
            
        logger.info(f"Generating grounded answer for query: '{query}'...")
        
        # 1. Format the context blocks with chunk IDs
        context_payloads = []
        for c in context_chunks:
            # Note: We use c.text (which is the raw_text, untouched source chunk)
            # as strictly mandated by ReferentWeave design.
            context_payloads.append(
                f"--- CHUNK ID: {c.chunk_id} (Document: {c.doc_id}) ---\n"
                f"{c.text}"
            )
            
        context_str = "\n\n".join(context_payloads)
        
        # 2. Formulate generator prompt
        prompt = (
            "You are an expert enterprise research assistant. "
            "Answer the User Query using ONLY the provided text chunks. "
            "Strictly follow these rules:\n"
            "1. Ground your answer entirely on the provided chunks. Do not make up facts, draw outside knowledge, or assume details.\n"
            "2. Cite your sources inside your answer by appending the Chunk ID in brackets (e.g. [1001], [1002]) immediately after the sentences that use information from that chunk.\n"
            "3. If the provided chunks do not contain enough information to answer the query, say: 'Based on the provided documents, I could not find enough information to answer this question.' Do not attempt to guess or synthesize.\n\n"
            f"Provided Text Chunks:\n{context_str}\n\n"
            f"User Query: {query}"
        )
        
        try:
            # We can also configure thinking if desired, but default Flash-Lite generates quickly
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt]
            )
            answer = response.text.strip()
            logger.info("Grounded answer generation completed successfully.")
            return answer
        except Exception as e:
            logger.error(f"Failed to generate grounded answer: {e}")
            return f"An error occurred while synthesizing the answer: {str(e)}"
