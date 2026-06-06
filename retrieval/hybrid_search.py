import numpy as np
from typing import List, Dict, Any, Optional
from core.config import logger, TOP_K_RETRIEVAL, RRF_CONSTANT
from core.models import QueryResult
from ingestion.embedder import IndexingEngine

class HybridSearcher:
    def __init__(self, indexer: IndexingEngine):
        self.indexer = indexer
        
    def search(self, query: str, doc_ids: Optional[List[str]] = None) -> List[QueryResult]:
        """
        Executes hybrid search using Tantivy BM25 and TurboVec vector search,
        merging results using Reciprocal Rank Fusion (RRF).
        """
        logger.info(f"Executing hybrid search for: '{query}'")
        
        # Resolve target document allowed chunk ids
        allowlist_ids = None
        if doc_ids:
            allowed_chunk_ids = []
            for cid_str, info in self.indexer.chunk_store.items():
                if info["doc_id"] in doc_ids:
                    allowed_chunk_ids.append(int(cid_str))
            
            # Empty allowlist mitigation: if no chunks match doc_ids, return empty results
            if not allowed_chunk_ids:
                logger.info(f"Filtering narrowed target set to 0 chunks. Short-circuiting search.")
                return []
                
            allowlist_ids = np.array(allowed_chunk_ids, dtype=np.uint64)
            logger.info(f"Filtered search active. Allowed chunks: {len(allowlist_ids)}")
            
        dense_results = []
        try:
            query_vector = self.indexer.get_query_embedding(query)
            # Native turbovec search expects a 2D array of queries: shape (1, dim)
            query_vector_2d = query_vector[np.newaxis, :]
            
            # SIMD Allowlist check
            scores_dense, ids_dense = self.indexer.turbovec_index.search(
                query_vector_2d, 
                k=TOP_K_RETRIEVAL, 
                allowlist=allowlist_ids
            )
            
            # If native Rust returns 2D results, extract the first row
            if scores_dense.ndim == 2:
                scores_dense = scores_dense[0]
            if ids_dense.ndim == 2:
                ids_dense = ids_dense[0]
                
            for rank, (score, cid) in enumerate(zip(scores_dense, ids_dense)):
                dense_results.append((int(cid), rank + 1, float(score)))
            logger.info(f"Dense vector search returned {len(dense_results)} candidates.")
        except Exception as e:
            logger.error(f"Dense vector search failed: {e}")
            
        # --- Stage 2: Sparse Search (Tantivy BM25) ---
        sparse_results = []
        try:
            # Parse search query. Search in the "text" (enriched text) field
            query_parser = self.indexer.tantivy_index.parse_query(query, ["text"])
            searcher = self.indexer.tantivy_index.searcher()
            
            # Execute BM25 search
            raw_sparse_hits = searcher.search(query_parser, limit=TOP_K_RETRIEVAL).hits
            
            rank = 1
            for score, doc_addr in raw_sparse_hits:
                doc = searcher.doc(doc_addr)
                cid = int(doc["chunk_id"][0])
                
                # Check if this matches our allowlist filter
                if doc_ids:
                    doc_doc_id = doc["doc_id"][0]
                    if doc_doc_id not in doc_ids:
                        continue
                        
                sparse_results.append((cid, rank, float(score)))
                rank += 1
            logger.info(f"Sparse text search returned {len(sparse_results)} candidates.")
        except Exception as e:
            logger.error(f"Sparse text search failed: {e}")
            
        # --- Stage 3: Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        
        # Add dense ranks
        for cid, rank, score in dense_results:
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_CONSTANT + rank))
            
        # Add sparse ranks
        for cid, rank, score in sparse_results:
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_CONSTANT + rank))
            
        # Sort by RRF score descending
        sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        top_candidates = sorted_rrf[:TOP_K_RETRIEVAL]
        
        logger.info(f"RRF fusion complete. Fused {len(rrf_scores)} unique candidates to Top {len(top_candidates)}.")
        
        # Build list of QueryResult objects
        results = []
        for cid, rrf_score in top_candidates:
            cid_str = str(cid)
            if cid_str in self.indexer.chunk_store:
                info = self.indexer.chunk_store[cid_str]
                results.append(QueryResult(
                    chunk_id=cid,
                    score=rrf_score,
                    text=info["target_text"],  # Send the original raw text
                    metadata=info["metadata"],
                    enriched_text=info["enriched_text"],
                    doc_id=info["doc_id"]
                ))
            else:
                logger.warning(f"Chunk ID {cid} found in index but missing from metadata store.")
                
        return results
