import json
import requests
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from core.config import (
    logger,
    INDEX_DIR,
    JINA_API_KEY,
    JINA_EMBEDDING_MODEL,
    EMBEDDING_DIM
)
from core.models import Chunk
from turbovec import IdMapIndex
import tantivy

# Define Tantivy schema
schema_builder = tantivy.SchemaBuilder()
schema_builder.add_integer_field("chunk_id", stored=True)
schema_builder.add_text_field("doc_id", stored=True)
schema_builder.add_text_field("text", stored=True)  # enriched text
schema_builder.add_text_field("raw_text", stored=True)  # raw target text
schema = schema_builder.build()

class IndexingEngine:
    def __init__(self):
        # Create directories
        INDEX_DIR.mkdir(exist_ok=True, parents=True)
        self.tv_path = INDEX_DIR / "turbovec.tvim"
        self.tantivy_dir = INDEX_DIR / "tantivy"
        self.tantivy_dir.mkdir(exist_ok=True, parents=True)
        self.metadata_path = INDEX_DIR / "chunks_store.json"
        
        # Initialize TurboVec
        if self.tv_path.exists():
            try:
                self.turbovec_index = IdMapIndex.load(str(self.tv_path))
                logger.info(f"Loaded existing TurboVec index from {self.tv_path}")
            except Exception as e:
                logger.warning(f"Could not load TurboVec index: {e}. Recreating...")
                self.turbovec_index = IdMapIndex(dim=EMBEDDING_DIM, bit_width=4)
        else:
            self.turbovec_index = IdMapIndex(dim=EMBEDDING_DIM, bit_width=4)
            
        # Initialize Tantivy
        self.tantivy_index = tantivy.Index(schema, path=str(self.tantivy_dir))
        logger.info(f"Tantivy index initialized at {self.tantivy_dir}")
        
        # Load metadata store
        self.chunk_store = {}
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.chunk_store = json.load(f)
                logger.info(f"Loaded {len(self.chunk_store)} chunk metadata records.")
            except Exception as e:
                logger.error(f"Error loading chunk store: {e}")
                self.chunk_store = {}
                
    def get_jina_embeddings(self, texts: List[str], task: str = "retrieval.passage") -> List[List[float]]:
        """
        Sends texts to the Jina Embeddings API.
        """
        logger.info(f"Generating Jina embeddings for {len(texts)} chunks (task={task})...")
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}"
        }
        data = {
            "model": JINA_EMBEDDING_MODEL,
            "task": task,
            "normalized": True,
            "input": texts
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            res_json = response.json()
            
            embeddings = [item["embedding"] for item in res_json["data"]]
            logger.info("Jina embeddings generated successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to fetch Jina embeddings: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response error content: {e.response.text}")
            raise e
            
    def ingest_chunks(self, chunks: List[Chunk]):
        """
        Ingests parsed, resolved, and enriched chunks into TurboVec, Tantivy, and Metadata store.
        """
        if not chunks:
            logger.info("No chunks provided for ingestion.")
            return
            
        logger.info(f"Starting ingestion pipeline for {len(chunks)} chunks...")
        
        # 1. Get embeddings for enriched text
        texts_to_embed = [c.enriched_text for c in chunks]
        embeddings = self.get_jina_embeddings(texts_to_embed, task="retrieval.passage")
        
        # 2. Add to TurboVec
        vectors_arr = np.array(embeddings, dtype=np.float32)
        ids_arr = np.array([c.chunk_id for c in chunks], dtype=np.uint64)
        
        self.turbovec_index.add_with_ids(vectors_arr, ids_arr)
        self.turbovec_index.write(str(self.tv_path))
        logger.info("Saved TurboVec index changes.")
        
        # 3. Add to Tantivy
        writer = self.tantivy_index.writer()
        for chunk in chunks:
            writer.add_document(tantivy.Document(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.enriched_text,
                raw_text=chunk.target_text
            ))
        writer.commit()
        logger.info("Saved Tantivy index changes.")
        
        # 4. Save metadata
        for chunk in chunks:
            self.chunk_store[str(chunk.chunk_id)] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "target_text": chunk.target_text,
                "background_context": chunk.background_context,
                "enriched_text": chunk.enriched_text,
                "metadata": chunk.metadata
            }
            
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_store, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Ingestion pipeline complete. Metastore size: {len(self.chunk_store)}")
        
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Gets embedding for a single query (task=retrieval.query).
        """
        embeddings = self.get_jina_embeddings([query], task="retrieval.query")
        return np.array(embeddings[0], dtype=np.float32)
        
    def get_all_doc_ids(self) -> List[str]:
        """
        Returns a list of all ingested document IDs.
        """
        doc_ids = set()
        for info in self.chunk_store.values():
            doc_ids.add(info["doc_id"])
        return list(doc_ids)

    def delete_document(self, doc_id: str):
        """
        Deletes all chunks of a document from indices.
        """
        logger.info(f"Deleting document {doc_id} from indices...")
        
        # Find chunks to remove
        chunk_ids_to_remove = []
        for cid_str, info in list(self.chunk_store.items()):
            if info["doc_id"] == doc_id:
                chunk_ids_to_remove.append(int(cid_str))
                self.chunk_store.pop(cid_str)
                
        # Remove from TurboVec
        for cid in chunk_ids_to_remove:
            self.turbovec_index.remove(cid)
        self.turbovec_index.write(str(self.tv_path))
        
        # Remove from Tantivy (Delete by term and commit)
        # Note: tantivy-py doesn't expose easy delete_term, so we recreate the index to delete
        # since it's an in-memory or small local index.
        # But we can also write the remaining documents to a fresh Tantivy index.
        # Let's rebuild the Tantivy index from the remaining chunk store to be safe.
        import shutil
        try:
            shutil.rmtree(self.tantivy_dir)
            self.tantivy_dir.mkdir(exist_ok=True, parents=True)
            self.tantivy_index = tantivy.Index(schema, path=str(self.tantivy_dir))
            
            writer = self.tantivy_index.writer()
            for cid_str, info in self.chunk_store.items():
                writer.add_document(tantivy.Document(
                    chunk_id=info["chunk_id"],
                    doc_id=info["doc_id"],
                    text=info["enriched_text"],
                    raw_text=info["target_text"]
                ))
            writer.commit()
        except Exception as e:
            logger.error(f"Error rebuilding Tantivy index during deletion: {e}")
            
        # Save metadata
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_store, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Deleted {len(chunk_ids_to_remove)} chunks belonging to document: {doc_id}")
