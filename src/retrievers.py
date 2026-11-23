import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalHit:
    """A retrieved document chunk with its ordering, score context, and citation."""

    rank: int
    content: str
    score: float | None
    score_label: str
    metadata: dict[str, Any]
    citation: str


def _hit(
    document: Document,
    rank: int,
    score: float | None,
    score_label: str,
) -> RetrievalHit:
    metadata = dict(document.metadata)
    source_file = str(
        metadata.get("source_file")
        or Path(str(metadata.get("source") or "unknown.txt")).name
    )
    chunk_index = metadata.get("chunk_index")
    if chunk_index is None:
        chunk_index = "unknown"

    # Keep ingestion metadata intact while ensuring every result can be cited,
    # even when a caller retrieves documents that were not created by ingestion.
    metadata.setdefault("source_file", source_file)
    metadata.setdefault("chunk_id", None)
    metadata.setdefault("chunk_index", chunk_index)
    metadata.setdefault("document_title", source_file)
    citation = f"[{source_file}#chunk-{chunk_index}]"
    return RetrievalHit(
        rank=rank,
        content=document.page_content,
        score=score,
        score_label=score_label,
        metadata=metadata,
        citation=citation,
    )


def _settings(config: AppConfig | None) -> AppConfig:
    return config if config is not None else load_config()


def _positive_limit(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def retrieve_dense(
    vector_store: FAISS,
    query: str,
    k: int | None = None,
    *,
    config: AppConfig | None = None,
) -> list[RetrievalHit]:
    """Return nearest FAISS results and their raw index scores.

    FAISS score values are not calibrated relevance probabilities. The default
    FAISS index uses L2 distance (lower is closer); custom index metrics can
    change the direction and interpretation of the raw value.
    """
    count = _positive_limit("k", k if k is not None else _settings(config).default_top_k)
    # Keep the vector store's own distance/similarity score instead of
    # presenting it as a normalized confidence value.
    results = vector_store.similarity_search_with_score(query, k=count)
    hits = [
        _hit(document, rank, float(score), "FAISS raw index score (metric-dependent)")
        for rank, (document, score) in enumerate(results, start=1)
    ]
    logger.info("event=dense_retrieval_complete result_count=%d", len(hits))
    return hits


def retrieve_sparse(
    retriever: BM25Retriever,
    query: str,
    k: int | None = None,
    *,
    config: AppConfig | None = None,
) -> list[RetrievalHit]:
    """Rank chunks lexically with BM25 and include its raw relevance score.

    BM25Retriever's normal invoke API returns ordered documents but omits their
    scores. Its configured tokenizer and BM25 vectorizer are used here directly
    so callers receive the actual corpus scores and can request any top-k size.
    BM25 scores are query/corpus-relative and should not be compared to dense
    FAISS scores.
    """
    count = _positive_limit("k", k if k is not None else _settings(config).default_top_k)
    query_tokens = retriever.preprocess_func(query)
    scores = retriever.vectorizer.get_scores(query_tokens)

    # Sorting the vectorizer's scores mirrors BM25Retriever's get_top_n ranking,
    # while retaining the score associated with each original document.
    ranked_indices = sorted(
        range(len(retriever.docs)),
        key=lambda index: scores[index],
        reverse=True,
    )[:count]
    hits = [
        _hit(
            retriever.docs[index],
            rank,
            float(scores[index]),
            "BM25 raw score (higher is better)",
        )
        for rank, index in enumerate(ranked_indices, start=1)
    ]
    logger.info("event=sparse_retrieval_complete result_count=%d", len(hits))
    return hits


def retrieve_mmr(
    vector_store: FAISS,
    query: str,
    k: int | None = None,
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
    *,
    config: AppConfig | None = None,
) -> list[RetrievalHit]:
    """Return relevance/diversity-balanced FAISS results using MMR.

    MMR first fetches a candidate pool, then greedily selects chunks that balance
    query relevance with dissimilarity to chunks already selected. Higher
    lambda_mult favors relevance; lower values favor diversity, which can reduce
    redundant passages but may include less directly relevant evidence.
    """
    settings = _settings(config)
    count = _positive_limit("k", k if k is not None else settings.default_top_k)
    candidate_count = _positive_limit(
        "fetch_k", fetch_k if fetch_k is not None else settings.default_mmr_fetch_k
    )
    relevance_weight = (
        settings.default_mmr_lambda if lambda_mult is None else lambda_mult
    )
    if candidate_count < count:
        raise ValueError("fetch_k must be greater than or equal to k.")
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("lambda_mult must be between 0 and 1.")

    # MMR returns the chosen order but no comparable per-document score; ranking
    # conveys selection order and score stays None instead of implying a metric.
    documents = vector_store.max_marginal_relevance_search(
        query,
        k=count,
        fetch_k=candidate_count,
        lambda_mult=relevance_weight,
    )
    hits = [
        _hit(document, rank, None, "MMR selection rank (score unavailable)")
        for rank, document in enumerate(documents, start=1)
    ]
    logger.info("event=mmr_retrieval_complete result_count=%d", len(hits))
    return hits


def retrieve_reranked(
    vector_store: FAISS,
    query: str,
    k: int | None = None,
    top_n: int | None = None,
    cross_encoder_model_name: str | None = None,
    *,
    config: AppConfig | None = None,
) -> list[RetrievalHit]:
    """Retrieve dense candidates, then re-order them with a Hugging Face cross-encoder.

    The cross-encoder jointly scores each query/chunk pair, which is more precise
    than embedding-only comparison but requires extra inference for every
    candidate. Candidate retrieval stays dense and bounded by top_n to control
    that cost. CrossEncoderReranker returns order but omits its raw scores, so
    score is None rather than a misleading rank-derived numeric value.
    """
    settings = _settings(config)
    result_count = _positive_limit(
        "k", k if k is not None else settings.default_top_k
    )
    candidate_count = _positive_limit(
        "top_n", top_n if top_n is not None else settings.default_rerank_top_n
    )
    if candidate_count < result_count:
        raise ValueError("top_n must be greater than or equal to k.")

    candidates = vector_store.similarity_search(query, k=candidate_count)
    if not candidates:
        return []

    model_name = cross_encoder_model_name or settings.require_cross_encoder_model_name()
    model_kwargs = {"token": settings.hf_token} if settings.hf_token else {}
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )
    # CrossEncoderReranker computes pairwise query/document relevance and sorts
    # descending; only the best k candidates are retained.
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=result_count)
    reranked_documents = reranker.compress_documents(candidates, query)
    hits = [
        _hit(
            document,
            rank,
            None,
            "Cross-encoder reranked order (score unavailable)",
        )
        for rank, document in enumerate(reranked_documents, start=1)
    ]
    logger.info("event=reranked_retrieval_complete result_count=%d", len(hits))
    return hits
