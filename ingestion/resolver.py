import json
from core.config import logger, GEMINI_API_KEY, GEMINI_MODEL
from core.models import ResolutionResponse
from google import genai

class CoreferenceResolver:
    def __init__(self):
        # Initialize client using configuration credentials
        if GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.client = genai.Client()
        logger.info(f"Initialized CoreferenceResolver with model: {GEMINI_MODEL}")
        
    def resolve_document(self, document_text: str) -> ResolutionResponse:
        """
        Resolves coreferences across the entire document text in a single pass.
        Returns a ResolutionResponse containing a list of vague references, their resolutions,
        and the exact sentence in the document where they occur.
        """
        if not document_text:
            return ResolutionResponse(resolutions=[])
            
        logger.info("Running document-wide coreference resolver...")
        
        prompt = (
            "You are a precision coreference resolver.\n"
            "You will be given the full text of a document. Your task is to analyze the document, "
            "locate all sentences or lines containing vague pronouns (e.g. 'it', 'they', 'this', 'these', 'those', 'that', 'the above table', 'the company', 'the project') "
            "and ambiguous nouns whose meaning depends on context from earlier in the document.\n\n"
            "For each vague reference or pronoun, resolve it using context from the document.\n\n"
            "Rules:\n"
            "1. Find the exact sentence/line in the document where the pronoun/reference is used, and set it as `original_sentence`.\n"
            "2. The `original_sentence` MUST match word-for-word a sentence in the provided document.\n"
            "3. Identify the specific pronoun/vague noun (e.g. 'it', 'they') as `original_phrase`.\n"
            "4. Resolve the reference to its fully qualified, concrete, and unambiguous entity as `resolved_entity`. "
            "Ensure the resolved entity is fully self-contained and includes the primary subject or project name (e.g. instead of resolving 'They' to 'depolarization events', resolve it to 'Helios-4 Key Exchange Protocol depolarization events'; instead of 'this hardware upgrade', resolve it to 'Helios-4 active tip-tilt mirrors and adaptive optics upgrade'). This ensures the resolution contains the full context of the main entity name.\n"
            "5. If you are not 100% certain of the resolution, set `resolved_entity` to 'UNCERTAIN' and `confidence` to 'low'.\n"
            "6. Do not invent or guess any facts. Do not make up entities.\n\n"
            f"Document Text:\n{document_text}"
        )
        
        import time
        max_retries = 3
        backoff_sec = 2
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[prompt],
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": ResolutionResponse.model_json_schema(),
                    }
                )
                
                data = json.loads(response.text.strip())
                resolutions_obj = ResolutionResponse(**data)
                
                logger.info(f"Resolved {len(resolutions_obj.resolutions)} references across the document.")
                for res in resolutions_obj.resolutions:
                    logger.info(f" - Line: '{res.original_sentence}' | Found: '{res.original_phrase}' -> '{res.resolved_entity}' (confidence: {res.confidence})")
                    
                return resolutions_obj
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Resolver attempt {attempt + 1} failed: {e}. Retrying in {backoff_sec}s...")
                    time.sleep(backoff_sec)
                    backoff_sec *= 2
                else:
                    logger.error(f"Failed to resolve coreferences after {max_retries + 1} attempts: {e}")
                    return ResolutionResponse(resolutions=[])

    def resolve(self, background_context: str, target_text: str) -> ResolutionResponse:
        """
        Resolves coreferences inside the target_text using background_context.
        Returns a ResolutionResponse object containing high and low confidence resolutions.
        """
        if not target_text:
            return ResolutionResponse(resolutions=[])
            
        logger.info("Running coreference resolver...")
        
        # Prepare system instruction / prompt
        prompt = (
            "You are a precision coreference resolver.\n"
            f"Background Context (preceding chunk for reference lookup):\n{background_context or 'None'}\n\n"
            f"Target Text (chunk to resolve references inside):\n{target_text}\n\n"
            "Identify vague references, pronouns (e.g. 'it', 'they', 'this', 'these', 'those', 'that', 'the above table', 'the company', 'the project'), "
            "and ambiguous nouns in the Target Text.\n"
            "Resolve them using ONLY the Background Context or Target Text.\n"
            "Rules:\n"
            "1. If a reference refers to something defined in the Background Context, resolve it.\n"
            "2. If you are not 100% certain of the resolution, set resolved_entity to 'UNCERTAIN' and confidence to 'low'.\n"
            "3. Do not invent or guess any facts. Do not make up entities."
        )
        
        import time
        max_retries = 3
        backoff_sec = 2
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[prompt],
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": ResolutionResponse.model_json_schema(),
                    }
                )
                
                # Parse structured JSON output
                data = json.loads(response.text.strip())
                resolutions_obj = ResolutionResponse(**data)
                
                # Log resolutions
                logger.info(f"Resolved {len(resolutions_obj.resolutions)} references.")
                for res in resolutions_obj.resolutions:
                    logger.info(f" - Found: '{res.original_phrase}' -> '{res.resolved_entity}' (confidence: {res.confidence})")
                    
                return resolutions_obj
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Legacy resolver attempt {attempt + 1} failed: {e}. Retrying in {backoff_sec}s...")
                    time.sleep(backoff_sec)
                    backoff_sec *= 2
                else:
                    logger.error(f"Failed to resolve coreferences in legacy resolver after {max_retries + 1} attempts: {e}")
                    return ResolutionResponse(resolutions=[])
