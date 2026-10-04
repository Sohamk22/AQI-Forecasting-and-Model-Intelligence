"""
End-to-End Policy RAG Orchestration Pipeline.
Integrates Document Loader, Chunker, Hybrid Retriever, Context Builder, Generator & Evaluator.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.rag.schemas import (
    ForecastPolicyContext,
    DocumentChunk,
    RetrievalResult,
    PolicyAnalysis,
    PolicyAnalyzeResponse,
)
from src.rag.loader import PolicyDocumentLoader
from src.rag.chunker import PolicyChunker
from src.rag.retriever import HybridPolicyRetriever
from src.rag.policy_context import PolicyContextBuilder
from src.rag.generator import PolicyGenerator

logger = logging.getLogger(__name__)


class PolicyRAGPipeline:
    """
    Singleton-friendly Policy RAG Pipeline.
    Manages document ingestion, hybrid index persistence, retrieval, and grounded analysis.
    """

    def __init__(
        self,
        doc_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5
    ):
        self.doc_dir = doc_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "policy_documents")
        self.cache_dir = cache_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "cache")
        self.top_k = top_k
        
        self.loader = PolicyDocumentLoader(self.doc_dir)
        self.chunker = PolicyChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = HybridPolicyRetriever(cache_dir=self.cache_dir)
        self.generator = PolicyGenerator()
        self.is_initialized = False

    def initialize(self, force_reindex: bool = False):
        """
        Initializes the retrieval index. Loads cached index if valid,
        or extracts PDFs, chunks documents, fits BM25/Dense models, and persists cache.
        """
        if self.is_initialized and not force_reindex:
            return

        # Attempt to load precomputed cache
        if not force_reindex and self.retriever.load_index():
            self.is_initialized = True
            logger.info(f"Policy RAG Pipeline initialized from cache with {len(self.retriever.chunks)} chunks.")
            return

        # Build index from raw policy documents
        logger.info(f"Building Policy RAG Index from documents in {self.doc_dir}...")
        pages = self.loader.load_all_documents()
        chunks = self.chunker.chunk_documents(pages)
        logger.info(f"Extracted {len(pages)} pages -> generated {len(chunks)} text chunks.")

        self.retriever.fit(chunks)
        self.retriever.save_index()
        self.is_initialized = True
        logger.info("Policy RAG Pipeline indexing complete and persisted.")

    def analyze(
        self,
        forecast_data: Dict[str, Any],
        query_override: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> PolicyAnalyzeResponse:
        """
        Executes end-to-end Policy Intelligence analysis:
        1. Builds leak-free ForecastPolicyContext.
        2. Formulates targeted PolicyQuery.
        3. Executes Hybrid Retrieval (Dense + BM25).
        4. Generates structured, grounded PolicyAnalysis.
        5. Returns structured PolicyAnalyzeResponse.
        """
        if not self.is_initialized:
            self.initialize()

        k = top_k or self.top_k

        # 1. Build prior-only forecast context
        context = PolicyContextBuilder.from_forecast_payload(forecast_data)

        # 2. Formulate retrieval query
        query_obj = PolicyContextBuilder.build_policy_query(context, query_override=query_override)

        # 3. Retrieve relevant chunks
        retrieval_results = self.retriever.search(query_obj.query_text, top_k=k, retrieval_mode="hybrid")
        retrieved_chunks = [r.chunk for r in retrieval_results]

        # 4. Generate grounded analysis
        analysis = self.generator.generate_policy_analysis(
            context=context,
            retrieved_chunks=retrieved_chunks,
            query_override=query_override
        )

        # 5. Format retrieved chunks for API payload inspection
        retrieved_payload = [
            {
                "chunk_id": r.chunk.chunk_id,
                "doc_name": r.chunk.doc_name,
                "page_number": r.chunk.page_number,
                "dense_score": round(r.dense_score, 4),
                "bm25_score": round(r.bm25_score, 4),
                "hybrid_score": round(r.hybrid_score, 4),
                "rank": r.rank,
                "snippet": " ".join(r.chunk.text.split())[:200] + "..."
            }
            for r in retrieval_results
        ]

        return PolicyAnalyzeResponse(
            status="success",
            context=context,
            analysis=analysis,
            retrieved_chunks=retrieved_payload
        )
