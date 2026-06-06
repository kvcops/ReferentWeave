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
            logger.error(f"Failed to resolve coreferences: {e}")
            return ResolutionResponse(resolutions=[])
