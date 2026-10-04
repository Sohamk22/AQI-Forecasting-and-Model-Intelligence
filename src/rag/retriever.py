"""
Hybrid Document Retriever for Policy Documents.
Implements Lab 3 (Dense Retrieval, BM25 Lexical Retrieval, Hybrid Ranking & Caching).
"""
import math
import pickle
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.schemas import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Pure-Python Okapi BM25 implementation for lexical keyword retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[DocumentChunk] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.term_freqs: List[Dict[str, int]] = []
        self.num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Lowercases and extracts alpha-numeric tokens."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())

    def fit(self, chunks: List[DocumentChunk]):
        """Indexes the provided document chunks."""
        self.corpus = chunks
        self.num_docs = len(chunks)
        if self.num_docs == 0:
            return

        self.term_freqs = []
        self.doc_lengths = []
        self.doc_freqs = Counter()

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            for term in tf.keys():
                self.doc_freqs[term] += 1

        self.avg_doc_len = sum(self.doc_lengths) / max(self.num_docs, 1)

        # Compute Robertson-Spärck Jones IDF
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            # Standard smoothed BM25 IDF
            self.idf[term] = math.log(1.0 + (self.num_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Scores all documents against the query using Okapi BM25."""
        if self.num_docs == 0:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores = [0.0] * self.num_docs

        for term in q_tokens:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]
            for doc_idx in range(self.num_docs):
                tf = self.term_freqs[doc_idx].get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1e-6)))
                scores[doc_idx] += term_idf * (numerator / denominator)

        # Rank documents
        scored_indices = sorted(range(self.num_docs), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in scored_indices[:top_k]:
            if scores[idx] > 0.0:
                results.append((self.corpus[idx], float(scores[idx])))
        return results


class DenseVectorRetriever:
    """
    Semantic / Dense Vector Retriever using subword n-gram TF-IDF embeddings
    and cosine similarity for fast, deterministic, reproducible in-memory indexing.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 3)):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english",
            max_features=15000
        )
        self.corpus: List[DocumentChunk] = []
        self.doc_vectors: Optional[np.ndarray] = None

    def fit(self, chunks: List[DocumentChunk]):
        """Fits TF-IDF vectorizer and builds dense document matrix."""
        self.corpus = chunks
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        self.doc_vectors = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Computes cosine similarity of query vector against all document vectors."""
        if self.doc_vectors is None or len(self.corpus) == 0:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()

        ranked_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            score = float(sims[idx])
            if score > 0.0:
                results.append((self.corpus[idx], score))
        return results


class HybridPolicyRetriever:
    """
    Hybrid Retriever combining Dense Vector and BM25 Lexical rankings
    via Reciprocal Rank Fusion (RRF) and linear score normalization.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5
    ):
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.bm25 = BM25Retriever()
        self.dense = DenseVectorRetriever()
        self.chunks: List[DocumentChunk] = []
        self.cache_dir = cache_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "cache")
        self.cache_path = self.cache_dir / "rag_index.pkl"

    def fit(self, chunks: List[DocumentChunk]):
        """Indexes chunks across both BM25 and Dense vector engines."""
        self.chunks = chunks
        self.bm25.fit(chunks)
        self.dense.fit(chunks)

    def save_index(self):
        """Serializes the indexed retrieval engines to disk cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "bm25": self.bm25,
                "dense": self.dense
            }, f)
        logger.info(f"Saved RAG hybrid index cache to {self.cache_path}")

    def load_index(self) -> bool:
        """Loads index from cache if available."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                self.chunks = data["chunks"]
                self.bm25 = data["bm25"]
                self.dense = data["dense"]
                logger.info(f"Loaded {len(self.chunks)} chunks from RAG index cache {self.cache_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load RAG cache: {e}. Rebuilding index.")
        return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_mode: str = "hybrid"
    ) -> List[RetrievalResult]:
        """
        Executes hybrid, dense-only, or lexical-only search.
        Uses Reciprocal Rank Fusion (RRF): RRF_score(d) = sum(1 / (k + rank_i(d))).
        """
        if not self.chunks:
            return []

        # 1. Dense retrieval
        dense_results = self.dense.search(query, top_k=top_k * 3)
        # 2. BM25 retrieval
        bm25_results = self.bm25.search(query, top_k=top_k * 3)

        chunk_map = {c.chunk_id: c for c in self.chunks}
        dense_ranks = {item[0].chunk_id: (rank + 1, item[1]) for rank, item in enumerate(dense_results)}
        bm25_ranks = {item[0].chunk_id: (rank + 1, item[1]) for rank, item in enumerate(bm25_results)}

        all_candidate_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
        if not all_candidate_ids:
            return []

        scored_candidates: List[RetrievalResult] = []

        for cid in all_candidate_ids:
            chunk = chunk_map[cid]
            d_rank, d_score = dense_ranks.get(cid, (999, 0.0))
            b_rank, b_score = bm25_ranks.get(cid, (999, 0.0))

            if retrieval_mode == "dense":
                final_score = d_score
            elif retrieval_mode == "bm25":
                final_score = b_score
            else:
                # Hybrid Reciprocal Rank Fusion (RRF)
                rrf_dense = 1.0 / (self.rrf_k + d_rank) if cid in dense_ranks else 0.0
                rrf_bm25 = 1.0 / (self.rrf_k + b_rank) if cid in bm25_ranks else 0.0
                final_score = self.dense_weight * rrf_dense + self.bm25_weight * rrf_bm25

            scored_candidates.append(
                RetrievalResult(
                    chunk=chunk,
                    dense_score=d_score,
                    bm25_score=b_score,
                    hybrid_score=final_score
                )
            )

        # Sort descending by final score
        scored_candidates.sort(key=lambda r: r.hybrid_score, reverse=True)

        for rank, res in enumerate(scored_candidates[:top_k]):
            res.rank = rank + 1

        return scored_candidates[:top_k]
