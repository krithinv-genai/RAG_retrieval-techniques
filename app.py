import hashlib
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import streamlit as st
from groq import APIError
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import AppConfig, load_config
from src.grounding import (
    INSUFFICIENT_CONTEXT_ANSWER,
    generate_grounded_answer,
    validate_citations,
)
from src.ingestion import (
    build_bm25_retriever,
    build_faiss_vector_store,
    create_embeddings,
    load_text_files,
    load_uploaded_text_files,
    split_documents,
)
from src.retrievers import (
    RetrievalHit,
    retrieve_dense,
    retrieve_hybrid_reranked,
    retrieve_mmr,
    retrieve_sparse,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PANEL_NAMES = (
    "Dense Search (FAISS)",
    "Sparse Search (BM25)",
    "MMR Search",
    "Hybrid (α) + Re-Ranked",
)
METADATA_FIELDS = ("source_file", "chunk_id", "chunk_index", "document_title")
SourceItem = str | tuple[str, bytes]
SourceItems = tuple[SourceItem, ...]


@st.cache_resource(max_entries=2, show_spinner="Loading Hugging Face embeddings...")
def _cached_embeddings(model_name: str):
    """Keep the expensive embedding model alive across Streamlit reruns."""
    return create_embeddings(model_name)


@st.cache_resource(max_entries=4, show_spinner="Loading and chunking documents...")
def _cached_corpus(
    source_hash: str,
    source_kind: str,
    chunk_size: int,
    chunk_overlap: int,
    _source_items: SourceItems,
) -> tuple[list[Document], BM25Retriever]:
    if source_kind == "local":
        paths = [item for item in _source_items if isinstance(item, str)]
        if len(paths) != len(_source_items):
            raise TypeError("Local source cache received non-path input.")
        documents = load_text_files(paths)
    elif source_kind == "uploaded":
        uploads = [item for item in _source_items if isinstance(item, tuple)]
        if len(uploads) != len(_source_items):
            raise TypeError("Upload cache received non-upload input.")
        documents = load_uploaded_text_files(uploads)
    else:
        raise ValueError(f"Unknown document source kind: {source_kind!r}.")

    chunks = split_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("No text chunks were created from the selected documents.")
    return chunks, build_bm25_retriever(chunks)


@st.cache_resource(max_entries=4, show_spinner="Building the FAISS vector store...")
def _cached_vector_store(
    source_hash: str,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    _chunks: list[Document],
) -> FAISS:
    """Cache a vector store by corpus content, model, and chunk parameters."""
    return build_faiss_vector_store(
        _chunks,
        embeddings=_cached_embeddings(embedding_model_name),
    )


@st.cache_resource(max_entries=2, show_spinner="Loading the cross-encoder...")
def _cached_cross_encoder(
    model_name: str,
    hf_token: str | None,
) -> HuggingFaceCrossEncoder:
    model_kwargs = {"token": hf_token} if hf_token else {}
    return HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)


def _digest_named_files(files: tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256()
    for filename, content in files:
        name_bytes = filename.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _local_source_bundle(paths: tuple[Path, ...]) -> tuple[str, SourceItems, str]:
    named_content = tuple((path.name, path.read_bytes()) for path in paths)
    return "local", tuple(str(path) for path in paths), _digest_named_files(named_content)


def _collect_source_bundle(
    source_mode: str,
    uploaded_files: list[Any],
) -> tuple[str, SourceItems, str] | None:
    if source_mode == "Sample docs/":
        sample_dir = PROJECT_ROOT / "sample_docs"
        paths = tuple(
            sorted(
                path.resolve()
                for path in sample_dir.rglob("*")
                if path.is_file() and path.suffix.lower() == ".txt"
            )
        )
        if not paths:
            raise ValueError(f"No .txt files found in {sample_dir}.")
        return _local_source_bundle(paths)

    if source_mode == "Uploaded .txt files":
        if not uploaded_files:
            return None
        files = tuple((uploaded.name, uploaded.getvalue()) for uploaded in uploaded_files)
        return "uploaded", files, _digest_named_files(files)

    raise ValueError(f"Unknown document source: {source_mode!r}.")


def _render_hit(hit: RetrievalHit, source_number: int) -> None:
    source_file = hit.metadata.get("source_file", "unknown")
    chunk_index = hit.metadata.get("chunk_index", "unknown")
    with st.container(border=True):
        st.markdown(f"**Rank:** {hit.rank}")
        if hit.score_label == "Cross-Encoder Score" and hit.score is not None:
            st.text(f"Cross-Encoder Score: {hit.score:.4f}")
        elif hit.score is None:
            st.text(f"Similarity / Relevance Score: {hit.score_label}")
        else:
            st.text(
                f"Similarity / Relevance Score: {hit.score:.4f} "
                f"({hit.score_label})"
            )
        retrieval_debug = hit.metadata.get("retrieval_debug")
        if isinstance(retrieval_debug, dict):
            st.caption(
                "Hybrid fusion score: "
                f"{retrieval_debug['hybrid_score']:.4f} "
                f"(α={retrieval_debug['alpha']:.2f})"
            )
        st.text(
            f"Citation Reference: [SRC-{source_number}] | "
            f"{source_file} | chunk {chunk_index}"
        )
        st.text(hit.content)
        with st.expander("Metadata"):
            metadata = {
                field: hit.metadata.get(field) for field in METADATA_FIELDS
            }
            if isinstance(retrieval_debug, dict):
                metadata["retrieval_debug"] = retrieval_debug
            st.json(metadata)


def _render_retrieval_panels(
    results: dict[str, list[RetrievalHit]],
    errors: dict[str, str],
) -> None:
    columns = st.columns(4)
    for column, panel_name in zip(columns, PANEL_NAMES):
        with column:
            st.subheader(panel_name)
            if panel_name in errors:
                st.warning(errors[panel_name])
                continue
            panel_hits = results.get(panel_name, [])
            if not panel_hits:
                st.info("No chunks were returned for this query.")
                continue
            for source_number, hit in enumerate(panel_hits, start=1):
                _render_hit(hit, source_number)


def _render_grounded_synthesis(
    results: dict[str, list[RetrievalHit]],
    config: AppConfig,
    query: str,
) -> None:
    st.subheader("Grounded Synthesis & Citation Audit")
    if config.groq_api_key and config.groq_model_name:
        st.caption("Synthesis uses Groq with temperature 0.")
    else:
        st.caption("Groq settings are incomplete; using the local extractive fallback.")
    synthesis_columns = st.columns(4)
    for column, panel_name in zip(synthesis_columns, PANEL_NAMES):
        with column:
            st.markdown(f"**{panel_name}**")
            hits = results.get(panel_name, [])
            try:
                answer = generate_grounded_answer(hits, query, config=config)
            except APIError as exc:
                st.error(f"Groq synthesis failed: {exc}")
                continue

            validated_answer = validate_citations(answer, hits)
            audit_passed = validated_answer == answer
            st.write(validated_answer)
            if audit_passed:
                st.success(
                    "Citation audit passed: zero uncited factual claims were retained."
                )
            else:
                st.error("Citation audit changed the answer to remove ungrounded text.")

            used_citations = list(dict.fromkeys(re.findall(r"\[SRC-(\d+)\]", validated_answer)))
            if used_citations:
                citation_details = []
                for number in used_citations:
                    hit_index = int(number) - 1
                    if 0 <= hit_index < len(hits):
                        hit = hits[hit_index]
                        citation_details.append(
                            f"[SRC-{number}] {hit.metadata.get('source_file', 'unknown')} "
                            f"(chunk {hit.metadata.get('chunk_index', 'unknown')})"
                        )
                st.caption("Citations used: " + "; ".join(citation_details))
            elif validated_answer == INSUFFICIENT_CONTEXT_ANSWER:
                st.caption("No evidence citations apply to the exact insufficient-context fallback.")


def main() -> None:
    st.set_page_config(page_title="Grounded RAG Comparison", layout="wide")
    config = load_config()
    st.title("Grounded RAG Retrieval Comparison")

    with st.sidebar:
        st.title("RAG Controls")
        source_mode = st.radio(
            "Document Source",
            ("Sample docs/", "Uploaded .txt files"),
        )
        uploaded_files = st.file_uploader(
            "Upload one or Upto 3 .txt files",
            accept_multiple_files=True,
            type=["txt"],
        )

        st.subheader("Models & API")
        embedding_model_name = st.text_input(
            "EMBEDDING_MODEL_NAME",
            value=config.embedding_model_name or "",
            placeholder="Set in .env or enter a Hugging Face model name",
        ).strip()
        cross_encoder_model_name = st.text_input(
            "CROSS_ENCODER_MODEL_NAME",
            value=config.cross_encoder_model_name or "",
            placeholder="Set in .env or enter a Hugging Face model name",
        ).strip()
        groq_api_key = st.text_input(
            "GROQ_API_KEY (optional)",
            value=config.groq_api_key or "",
            type="password",
            placeholder="Set in .env or paste a Groq API key",
        ).strip()
        groq_model_name = st.text_input(
            "GROQ_MODEL_NAME (optional)",
            value=config.groq_model_name or "",
            placeholder="Set in .env or enter a Groq model name",
        ).strip()

        st.subheader("Retrieval Hyperparameters")
        top_k = int(
            st.number_input(
                "Top-K",
                min_value=1,
                max_value=50,
                value=min(max(config.default_top_k, 1), 50),
                step=1,
            )
        )
        chunk_size = int(
            st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=4000,
                value=min(max(config.default_chunk_size, 100), 4000),
                step=50,
            )
        )
        chunk_overlap_default = max(
            0,
            min(config.default_chunk_overlap, chunk_size - 1),
        )
        chunk_overlap = int(
            st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=chunk_size - 1,
                value=chunk_overlap_default,
                step=10,
            )
        )
        mmr_lambda = st.slider(
            "MMR Lambda (relevance ↔ diversity)",
            min_value=0.0,
            max_value=1.0,
            value=min(max(config.default_mmr_lambda, 0.0), 1.0),
            step=0.05,
        )
        rerank_top_n = int(
            st.number_input(
                "Re-rank Top-N",
                min_value=top_k,
                max_value=100,
                value=max(top_k, min(config.default_rerank_top_n, 100)),
                step=1,
            )
        )
        hybrid_alpha = st.slider(
            "Alpha Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=min(max(config.default_hybrid_alpha, 0.0), 1.0),
            step=0.05,
        )
        st.caption(
            "1.0 = Pure Dense (Semantic) | 0.5 = Balanced Hybrid | "
            "0.0 = Pure Sparse (BM25)"
        )

    try:
        source_bundle = _collect_source_bundle(
            source_mode,
            uploaded_files or [],
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        source_bundle = None
        st.sidebar.error(f"Could not use the selected document source: {exc}")

    with st.form("query_form"):
        query = st.text_input("Ask a question about the selected documents")
        compare_clicked = st.form_submit_button("Compare Retrievers", type="primary")

    if not compare_clicked:
        st.info("Choose a document source, enter a query, then compare the four retrieval strategies.")
        return
    if not query.strip():
        st.error("Enter a query before comparing retrievers.")
        return
    if source_bundle is None:
        st.error("Select or provide at least one document before running retrieval.")
        return

    source_kind, source_items, source_hash = source_bundle
    try:
        chunks, bm25_retriever = _cached_corpus(
            source_hash,
            source_kind,
            chunk_size,
            chunk_overlap,
            source_items,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        st.error(f"Document ingestion failed: {exc}")
        return

    st.caption(f"Loaded {len(chunks)} chunks from the selected source.")
    active_config = replace(
        config,
        embedding_model_name=embedding_model_name or None,
        cross_encoder_model_name=cross_encoder_model_name or None,
        groq_api_key=groq_api_key or None,
        groq_model_name=groq_model_name or None,
        default_top_k=top_k,
        default_chunk_size=chunk_size,
        default_chunk_overlap=chunk_overlap,
        default_mmr_fetch_k=max(config.default_mmr_fetch_k, top_k),
        default_mmr_lambda=mmr_lambda,
        default_rerank_top_n=rerank_top_n,
        default_hybrid_alpha=hybrid_alpha,
    )

    results: dict[str, list[RetrievalHit]] = {name: [] for name in PANEL_NAMES}
    errors: dict[str, str] = {}
    inference_started = time.perf_counter()
    results["Sparse Search (BM25)"] = retrieve_sparse(
        bm25_retriever,
        query,
        k=top_k,
        config=active_config,
    )

    vector_store: FAISS | None = None
    vector_store_error: str | None = None
    if embedding_model_name:
        try:
            vector_store = _cached_vector_store(
                source_hash,
                embedding_model_name,
                chunk_size,
                chunk_overlap,
                chunks,
            )
        except (ImportError, OSError, ValueError) as exc:
            vector_store_error = f"{type(exc).__name__}: {exc}"
    else:
        vector_store_error = "Enter EMBEDDING_MODEL_NAME to enable dense retrieval."

    if vector_store is not None:
        results["Dense Search (FAISS)"] = retrieve_dense(
            vector_store,
            query,
            k=top_k,
            config=active_config,
        )
        results["MMR Search"] = retrieve_mmr(
            vector_store,
            query,
            k=top_k,
            fetch_k=max(config.default_mmr_fetch_k, top_k),
            lambda_mult=mmr_lambda,
            config=active_config,
        )
        if cross_encoder_model_name:
            try:
                cross_encoder = _cached_cross_encoder(
                    cross_encoder_model_name,
                    config.hf_token,
                )
                results["Hybrid (α) + Re-Ranked"] = retrieve_hybrid_reranked(
                    vector_store,
                    bm25_retriever,
                    query,
                    k=top_k,
                    top_n=rerank_top_n,
                    alpha=hybrid_alpha,
                    config=active_config,
                    cross_encoder_model=cross_encoder,
                )
            except (ImportError, OSError, ValueError) as exc:
                errors["Hybrid (α) + Re-Ranked"] = (
                    f"Cross-encoder unavailable: {type(exc).__name__}: {exc}"
                )
        else:
            errors["Hybrid (α) + Re-Ranked"] = (
                "Enter CROSS_ENCODER_MODEL_NAME to enable re-ranking."
            )
    else:
        errors["Dense Search (FAISS)"] = vector_store_error or "FAISS is unavailable."
        errors["MMR Search"] = vector_store_error or "FAISS is unavailable."
        errors["Hybrid (α) + Re-Ranked"] = (
            vector_store_error or "FAISS is unavailable."
        )

    _render_retrieval_panels(results, errors)
    st.divider()
    _render_grounded_synthesis(results, active_config, query)
    elapsed_seconds = time.perf_counter() - inference_started
    st.caption(
        f"Inference completed in {elapsed_seconds:.2f} seconds "
        f"({elapsed_seconds / 60:.2f} minutes)."
    )


if __name__ == "__main__":
    main()
