import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from core.config import logger
from core.models import Chunk
from ingestion.resolver import CoreferenceResolver
from ingestion.enricher import enrich_chunk
from ingestion.embedder import IndexingEngine
from retrieval.hybrid_search import HybridSearcher

# Define a set of synthetic test cases for coreference evaluation
EVAL_DATASET = [
    {
        "doc_id": "project_odyssey.pdf",
        "chunks": [
            "Project Odyssey was officially launched by the aerospace division in January 2026. The mission's goal is to construct and deploy a next-generation orbital spacecraft.",
            "It was delayed by six months due to a series of fuel valve thruster leaks. It will miss the crucial Q3 launch window."
        ],
        "queries": [
            {
                "query": "What caused the delay of Project Odyssey?",
                "target_chunk_index": 1  # The second chunk contains the reason for the delay, but lacks the name 'Project Odyssey'
            }
        ]
    },
    {
        "doc_id": "company_financials.pdf",
        "chunks": [
            "NovaTech Inc. reported record-breaking revenue figures for the fiscal year 2025. Total annual earnings surged by 45% compared to the prior period.",
            "This was driven entirely by the rapid adoption of their proprietary cloud integration platform. However, they warning that growth might slow down."
        ],
        "queries": [
            {
                "query": "What drove the revenue growth of NovaTech Inc.?",
                "target_chunk_index": 1  # The second chunk explains the growth driver, but uses 'This' instead of 'NovaTech record revenue'
            }
        ]
    },
    {
        "doc_id": "legal_agreement.pdf",
        "chunks": [
            "The Licensee agrees to pay the Licensor a monthly recurring subscription fee of five thousand dollars ($5,000) for software maintenance services.",
            "They must remit the full payment within five business days of invoice receipt, or incur a 2% late penalty charge."
        ],
        "queries": [
            {
                "query": "Who is subject to late penalty charges under the agreement?",
                "target_chunk_index": 1  # The second chunk specifies the penalty and 'They' refers to 'The Licensee' in chunk 0
            }
        ]
    }
]

def run_evaluation():
    logger.info("Initializing Evaluation Harness...")
    
    # Initialize components
    # We will use in-memory IndexingEngine and HybridSearcher to avoid messing up main indexes
    # But since IndexingEngine persists to disk, we will use a temporary testing directory
    import shutil
    test_index_dir = Path("./data/test_index")
    if test_index_dir.exists():
        shutil.rmtree(test_index_dir)
    test_index_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporarily override INDEX_DIR in config
    import core.config
    old_index_dir = core.config.INDEX_DIR
    core.config.INDEX_DIR = test_index_dir
    
    try:
        resolver = CoreferenceResolver()
        
        # We will test two systems:
        # 1. Standard RAG (Ablated: No reference resolutions appended)
        # 2. ReferentWeave (Full: With high-confidence reference resolutions appended)
        
        # --- System 1 Setup: Standard RAG (Ablated) ---
        logger.info("\n=== Setting up System 1: Standard RAG (Ablation Mode) ===")
        standard_indexer = IndexingEngine()
        standard_indexer.tv_path = test_index_dir / "turbovec_standard.tvim"
        standard_indexer.metadata_path = test_index_dir / "chunks_store_standard.json"
        
        standard_chunks = []
        global_chunk_counter = 5000
        
        for doc in EVAL_DATASET:
            for idx, text in enumerate(doc["chunks"]):
                bg = doc["chunks"][idx-1] if idx > 0 else ""
                chunk = Chunk(
                    chunk_id=global_chunk_counter,
                    doc_id=doc["doc_id"] + "_std",
                    target_text=text,
                    background_context=bg,
                    enriched_text=text,  # No enrichment
                    metadata={"headers": []}
                )
                standard_chunks.append(chunk)
                global_chunk_counter += 1
                
        standard_indexer.ingest_chunks(standard_chunks)
        standard_searcher = HybridSearcher(standard_indexer)
        
        # --- System 2 Setup: ReferentWeave RAG ---
        logger.info("\n=== Setting up System 2: ReferentWeave RAG ===")
        rw_indexer = IndexingEngine()
        rw_indexer.tv_path = test_index_dir / "turbovec_rw.tvim"
        rw_indexer.metadata_path = test_index_dir / "chunks_store_rw.json"
        
        rw_chunks = []
        global_chunk_counter = 7000
        
        for doc in EVAL_DATASET:
            for idx, text in enumerate(doc["chunks"]):
                bg = doc["chunks"][idx-1] if idx > 0 else ""
                chunk = Chunk(
                    chunk_id=global_chunk_counter,
                    doc_id=doc["doc_id"] + "_rw",
                    target_text=text,
                    background_context=bg,
                    enriched_text="",
                    metadata={"headers": []}
                )
                
                # Run resolver and enricher
                res_response = resolver.resolve(chunk.background_context, chunk.target_text)
                enriched = enrich_chunk(chunk, res_response.resolutions)
                rw_chunks.append(enriched)
                global_chunk_counter += 1
                
        rw_indexer.ingest_chunks(rw_chunks)
        rw_searcher = HybridSearcher(rw_indexer)
        
        # --- Run Evaluation Queries ---
        logger.info("\n=== Running Comparative Evaluation Queries ===")
        
        total_queries = 0
        std_success = 0
        rw_success = 0
        
        results_summary = []
        
        for doc_idx, doc in enumerate(EVAL_DATASET):
            doc_id = doc["doc_id"]
            
            for query_info in doc["queries"]:
                total_queries += 1
                query = query_info["query"]
                target_offset = query_info["target_chunk_index"]
                
                # Check Standard RAG top results
                std_results = standard_searcher.search(query, doc_ids=[doc_id + "_std"])
                # The target chunk is the one with offset
                target_std_chunk = standard_chunks[doc_idx * 2 + target_offset]
                
                # Check if target chunk is retrieved in top 1
                std_top_ids = [c.chunk_id for c in std_results[:1]]
                std_hit = target_std_chunk.chunk_id in std_top_ids
                if std_hit:
                    std_success += 1
                    
                # Check ReferentWeave RAG top results
                rw_results = rw_searcher.search(query, doc_ids=[doc_id + "_rw"])
                target_rw_chunk = rw_chunks[doc_idx * 2 + target_offset]
                
                rw_top_ids = [c.chunk_id for c in rw_results[:1]]
                rw_hit = target_rw_chunk.chunk_id in rw_top_ids
                if rw_hit:
                    rw_success += 1
                    
                results_summary.append({
                    "doc": doc_id,
                    "query": query,
                    "std_top_score": std_results[0].score if std_results else 0.0,
                    "std_hit": "YES" if std_hit else "NO",
                    "rw_top_score": rw_results[0].score if rw_results else 0.0,
                    "rw_hit": "YES" if rw_hit else "NO",
                })
                
        # --- Print Comparative Results ---
        print("\n" + "="*80)
        print("REFERENTWEAVE ABLATION COMPARISON (TOP-1 COREFERENCE RECALL)")
        print("="*80)
        print(f"{'Document':<25} | {'Query':<40} | {'Standard RAG':<12} | {'ReferentWeave':<12}")
        print("-"*80)
        for r in results_summary:
            print(f"{r['doc']:<25} | {r['query'][:38]+'...':<40} | {r['std_hit']:<12} | {r['rw_hit']:<12}")
        print("-"*80)
        
        std_recall = (std_success / total_queries) * 100
        rw_recall = (rw_success / total_queries) * 100
        print(f"{'OVERALL RECALL @ 1':<68} | {std_recall:.1f}%      | {rw_recall:.1f}%")
        print("="*80)
        
    finally:
        # Restore configuration
        core.config.INDEX_DIR = old_index_dir
        # Clean up testing dir
        if test_index_dir.exists():
            shutil.rmtree(test_index_dir)

if __name__ == "__main__":
    run_evaluation()
