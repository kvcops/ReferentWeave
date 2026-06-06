import os
import shutil
import traceback
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.config import logger, UPLOAD_DIR
from core.models import Chunk, QueryResult
from ingestion.parser import parse_document, chunk_document_text, build_context_chunks, build_context_chunks_from_enriched
from ingestion.resolver import CoreferenceResolver
from ingestion.enricher import enrich_chunk, enrich_document
from ingestion.embedder import IndexingEngine
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import GeminiReranker
from retrieval.generator import GroundedGenerator

app = FastAPI(title="ReferentWeave RAG API", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
logger.info("Initializing ReferentWeave indexing and search engines...")
indexer = IndexingEngine()
searcher = HybridSearcher(indexer)
reranker = GeminiReranker()
generator = GroundedGenerator()
resolver = CoreferenceResolver()

# Mount static files folder
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Request/Response Schemas
class QueryRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None

class QueryResponse(BaseModel):
    answer: str
    candidates: List[QueryResult]
    trace: List[str]

class IngestResponse(BaseModel):
    filename: str
    doc_id: str
    chunks_count: int
    resolutions_count: int
    trace: List[str]

@app.get("/")
def read_root():
    """Serves the dashboard home page."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail="Frontend dashboard index.html not found.")

@app.get("/api/documents")
def list_documents():
    """Lists all ingested documents in the system."""
    try:
        docs = indexer.get_all_doc_ids()
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Deletes a document from all indices."""
    try:
        indexer.delete_document(doc_id)
        return {"status": "success", "message": f"Document '{doc_id}' deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingests a document (PDF), chunks it, runs coreference resolution, 
    appends resolved context, embeds, and indexes it.
    """
    trace = []
    doc_id = file.filename
    temp_file_path = UPLOAD_DIR / file.filename
    
    trace.append(f"Received file: {file.filename}")
    logger.info(f"API Ingest: Received file '{file.filename}'")
    
    try:
        # 1. Save uploaded file to disk
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        trace.append(f"Saved file to local storage: {temp_file_path}")
        
        # 2. Parse file
        trace.append("Starting document parsing (Docling parser with pypdf fallback)...")
        text = parse_document(temp_file_path)
        trace.append(f"Successfully extracted {len(text)} characters of text.")
        
        # 3. Coreference resolution on the entire document (Single LLM pass)
        trace.append("Resolving coreferences on the entire document using Gemini 3.1 Flash-Lite...")
        res_response = resolver.resolve_document(text)
        
        high_conf = [r for r in res_response.resolutions if r.confidence == "high" and r.resolved_entity.upper() != "UNCERTAIN"]
        resolutions_count = len(high_conf)
        
        # Log resolution trace for dashboard
        for r in high_conf:
            trace.append(f"Resolved coreference: '{r.original_phrase}' -> '{r.resolved_entity}' in sentence: '{r.original_sentence}'")
            
        # 4. Enrich full document text with context blocks
        trace.append("Enriching document text with resolved context blocks...")
        enriched_text = enrich_document(text, res_response.resolutions)
        
        # 5. Chunk the enriched text
        trace.append("Segmenting enriched text into layout-aware logical chunks...")
        raw_chunks = chunk_document_text(enriched_text)
        trace.append(f"Document chunked into {len(raw_chunks)} sections.")
        
        # 6. Build typed context chunks from enriched chunks
        trace.append("Building sliding window contexts from enriched chunks...")
        resolved_chunks = build_context_chunks_from_enriched(raw_chunks, doc_id)
        
        trace.append(f"Coreference resolution finished. Appended {resolutions_count} high-confidence references.")
        
        # 6. Ingest into vector & keyword search engines
        trace.append("Generating Jina embeddings and committing to TurboVec & Tantivy indices...")
        indexer.ingest_chunks(resolved_chunks)
        trace.append("Document committed and indexes synchronized successfully.")
        
        # Clean up temp file
        if temp_file_path.exists():
            os.remove(temp_file_path)
            
        return IngestResponse(
            filename=file.filename,
            doc_id=doc_id,
            chunks_count=len(resolved_chunks),
            resolutions_count=resolutions_count,
            trace=trace
        )
        
    except Exception as e:
        logger.error(f"Error during ingestion pipeline: {e}")
        logger.error(traceback.format_exc())
        
        # Clean up temp file
        if temp_file_path.exists():
            os.remove(temp_file_path)
            
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Executes the RAG query: Hybrid retrieval, RRF fusion, Cross-encoder rerank,
    and grounded generator answer.
    """
    trace = []
    query = request.query
    doc_ids = request.doc_ids
    
    trace.append(f"Received query: '{query}'")
    if doc_ids:
        trace.append(f"Filters active: document(s) {doc_ids}")
    else:
        trace.append("Filters inactive: global search enabled.")
        
    try:
        # 1. Hybrid Search
        trace.append("Retrieving candidates using dual indexing (Tantivy BM25 + TurboVec dense vector)...")
        candidates = searcher.search(query, doc_ids)
        trace.append(f"Dual indexing returned {len(candidates)} fused candidate chunks (RRF).")
        
        # 2. Rerank
        trace.append("Reranking candidates using Gemini 3.1 Flash-Lite Cross-Encoder...")
        reranked_candidates = reranker.rerank(query, candidates)
        trace.append(f"Reranked and filtered down to top {len(reranked_candidates)} chunks.")
        
        # 3. Grounded Answer
        trace.append("Generating grounded answer with citations using Gemini 3.1 Flash-Lite...")
        answer = generator.generate_answer(query, reranked_candidates)
        trace.append("Answer synthesized successfully.")
        
        return QueryResponse(
            answer=answer,
            candidates=reranked_candidates,
            trace=trace
        )
        
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
