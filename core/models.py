from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

class Resolution(BaseModel):
    original_sentence: Optional[str] = Field(default="", description="The exact full sentence or line in the text containing the vague reference/pronoun")
    original_phrase: str = Field(description="The pronoun, ambiguous noun, or vague reference found in the text (e.g. 'it', 'they', 'the project')")
    resolved_entity: str = Field(description="The resolved actual entity name from context, or 'UNCERTAIN' if it cannot be resolved with absolute certainty")
    confidence: Literal["high", "low"] = Field(description="Confidence level of resolution. Mark 'low' if you are guessing or unsure")

class ResolutionResponse(BaseModel):
    resolutions: List[Resolution] = Field(description="List of detected vague references and their resolved entities")

class Chunk(BaseModel):
    chunk_id: int = Field(description="Stable uint64 external index id")
    doc_id: str = Field(description="Identifier of the source document")
    target_text: str = Field(description="Original untouched raw text of the chunk")
    background_context: str = Field(description="Raw text of the predecessor chunk (empty if first chunk)")
    enriched_text: str = Field(description="Raw text with contextual appended block")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata like headers, section level, page number")

class QueryResult(BaseModel):
    chunk_id: int
    score: float
    text: str
    metadata: Dict[str, Any]
    enriched_text: str
    doc_id: str
