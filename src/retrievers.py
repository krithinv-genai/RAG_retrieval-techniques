import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.utils.math import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_core.cross_encoders import BaseCrossEncoder
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
    query relevance with dissimilarity to chunks already selected. For later
    selections, the score is lambda_mult * query_similarity minus
    (1 - lambda_mult) * maximum_similarity_to_selected. A value of 1 favors
    relevance, 0 favors diversity, and 0.5 balances both. The first chunk is
    always the most query-similar candidate; its displayed score uses zero
    diversity penalty.
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

    query_embedding = vector_store._embed_query(query)
    documents = vector_store.max_marginal_relevance_search_by_vector(
        query_embedding,
        k=count,
        fetch_k=candidate_count,
        lambda_mult=relevance_weight,
    )
    document_embeddings = vector_store._embed_documents(
        [document.page_content for document in documents]
    )
    query_similarities = cosine_similarity(
        [query_embedding],
        document_embeddings,
    )[0]
    selected_embeddings: list[list[float]] = []
    hits: list[RetrievalHit] = []
    for rank, (document, embedding) in enumerate(
        zip(documents, document_embeddings, strict=True),
        start=1,
    ):
        diversity_similarity = (
            float(max(cosine_similarity([embedding], selected_embeddings)[0]))
            if selected_embeddings
            else 0.0
        )
        mmr_score = (
            relevance_weight * query_similarities[rank - 1]
            - (1.0 - relevance_weight) * diversity_similarity
        )
        hits.append(
            _hit(
                document,
                rank,
                float(mmr_score),
                f"MMR objective score (lambda={relevance_weight:.2f})",
            )
        )
        selected_embeddings.append(embedding)
    logger.info("event=mmr_retrieval_complete result_count=%d", len(hits))
    return hits


class _RecordingCrossEncoder(BaseCrossEncoder):
    """Capture scores while letting CrossEncoderReranker perform the sorting."""

    def __init__(self, model: HuggingFaceCrossEncoder) -> None:
        self.model = model
        self.scores: list[float] = []

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        self.scores = [float(score) for score in self.model.score(text_pairs)]
        if len(self.scores) != len(text_pairs):
            raise ValueError("Cross-encoder returned an unexpected number of scores.")
        return self.scores


def _normalize_scores(scores: list[float], *, higher_is_better: bool) -> list[float]:
    if not scores:
        return []
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Retrieval scores must be finite numbers.")

    minimum = min(scores)
    maximum = max(scores)
    if minimum == maximum:
        return [1.0] * len(scores)
    if higher_is_better:
        return [(score - minimum) / (maximum - minimum) for score in scores]
    return [(maximum - score) / (maximum - minimum) for score in scores]


def _document_identity(document: Document) -> tuple[str, ...]:
    chunk_id = document.metadata.get("chunk_id")
    if chunk_id is not None and str(chunk_id):
        return "chunk_id", str(chunk_id)
    source_file = document.metadata.get("source_file", "unknown")
    chunk_index = document.metadata.get("chunk_index", "unknown")
    return "source_file", str(source_file), str(chunk_index)


def retrieve_hybrid_reranked(
    vector_store: FAISS,
    sparse_retriever: BM25Retriever,
    query: str,
    k: int | None = None,
    top_n: int | None = None,
    alpha: float | None = None,
    cross_encoder_model_name: str | None = None,
    *,
    config: AppConfig | None = None,
    cross_encoder_model: HuggingFaceCrossEncoder | None = None,
) -> list[RetrievalHit]:
    """Fuse dense and sparse candidates, then re-rank them with a cross-encoder."""
    settings = _settings(config)
    result_count = _positive_limit(
        "k", k if k is not None else settings.default_top_k
    )
    candidate_count = _positive_limit(
        "top_n", top_n if top_n is not None else settings.default_rerank_top_n
    )
    if candidate_count < result_count:
        raise ValueError("top_n must be greater than or equal to k.")
    alpha_weight = settings.default_hybrid_alpha if alpha is None else alpha
    if not 0.0 <= alpha_weight <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    # Stage 1 — retrieve and normalize both ranked lists, then fuse by chunk.
    dense_results = vector_store.similarity_search_with_score(
        query,
        k=candidate_count,
    )
    dense_scores = _normalize_scores(
        [float(score) for _, score in dense_results],
        higher_is_better=False,
    )
    query_tokens = sparse_retriever.preprocess_func(query)
    sparse_scores = sparse_retriever.vectorizer.get_scores(query_tokens)
    sparse_ranked_indices = sorted(
        range(len(sparse_retriever.docs)),
        key=lambda index: sparse_scores[index],
        reverse=True,
    )[:candidate_count]
    sparse_documents = [
        sparse_retriever.docs[index] for index in sparse_ranked_indices
    ]
    sparse_raw_scores = [
        float(sparse_scores[index]) for index in sparse_ranked_indices
    ]
    normalized_sparse_scores = _normalize_scores(
        sparse_raw_scores,
        higher_is_better=True,
    )

    documents_by_id: dict[tuple[str, ...], Document] = {}
    component_scores: dict[tuple[str, ...], dict[str, float]] = {}
    dense_ranks: dict[tuple[str, ...], int] = {}
    sparse_ranks: dict[tuple[str, ...], int] = {}
    for rank, ((document, _), score) in enumerate(
        zip(dense_results, dense_scores, strict=True),
        start=1,
    ):
        identity = _document_identity(document)
        documents_by_id.setdefault(identity, document)
        dense_ranks.setdefault(identity, rank)
        components = component_scores.setdefault(
            identity,
            {"dense_score": 0.0, "sparse_score": 0.0},
        )
        components["dense_score"] = max(components["dense_score"], score)

    for rank, (document, score) in enumerate(
        zip(sparse_documents, normalized_sparse_scores, strict=True),
        start=1,
    ):
        identity = _document_identity(document)
        documents_by_id.setdefault(identity, document)
        sparse_ranks.setdefault(identity, rank)
        components = component_scores.setdefault(
            identity,
            {"dense_score": 0.0, "sparse_score": 0.0},
        )
        components["sparse_score"] = max(components["sparse_score"], score)

    fused_candidates: list[
        tuple[Document, float, float, float, tuple[str, ...]]
    ] = []
    for identity, document in documents_by_id.items():
        components = component_scores[identity]
        hybrid_score = (
            alpha_weight * components["dense_score"]
            + (1.0 - alpha_weight) * components["sparse_score"]
        )
        fused_candidates.append(
            (
                document,
                hybrid_score,
                components["dense_score"],
                components["sparse_score"],
                identity,
            )
        )

    def fusion_sort_key(
        candidate: tuple[Document, float, float, float, tuple[str, ...]],
    ) -> tuple[float, int]:
        identity = candidate[4]
        if alpha_weight == 0.0:
            return candidate[1], -sparse_ranks.get(identity, candidate_count + 1)
        if alpha_weight == 1.0:
            return candidate[1], -dense_ranks.get(identity, candidate_count + 1)
        return candidate[1], 0

    fused_candidates.sort(key=fusion_sort_key, reverse=True)
    fused_candidates = fused_candidates[:candidate_count]
    if not fused_candidates:
        return []

    # Stage 2 — use the standard reranker ordering and capture its scores in one pass.
    if cross_encoder_model is None:
        model_name = cross_encoder_model_name or settings.require_cross_encoder_model_name()
        model_kwargs = {"token": settings.hf_token} if settings.hf_token else {}
        cross_encoder_model = HuggingFaceCrossEncoder(
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
    recording_model = _RecordingCrossEncoder(cross_encoder_model)
    reranker = CrossEncoderReranker(
        model=recording_model,
        top_n=result_count,
    )
    fused_documents = [candidate[0] for candidate in fused_candidates]
    reranked_documents = reranker.compress_documents(fused_documents, query)
    cross_encoder_scores = {
        id(document): score
        for document, score in zip(
            fused_documents,
            recording_model.scores,
            strict=True,
        )
    }
    candidate_details = {
        id(document): (hybrid_score, dense_score, sparse_score)
        for document, hybrid_score, dense_score, sparse_score, _ in fused_candidates
    }
    hits: list[RetrievalHit] = []
    for rank, document in enumerate(reranked_documents, start=1):
        hit = _hit(
            document,
            rank,
            cross_encoder_scores[id(document)],
            "Cross-Encoder Score",
        )
        hybrid_score, dense_score, sparse_score = candidate_details[id(document)]
        hit.metadata["retrieval_debug"] = {
            "hybrid_score": hybrid_score,
            "dense_score": dense_score,
            "sparse_score": sparse_score,
            "alpha": alpha_weight,
        }
        hits.append(hit)
    logger.info(
        "event=hybrid_reranked_retrieval_complete result_count=%d alpha=%.2f",
        len(hits),
        alpha_weight,
    )
    return hits
