import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO, Iterable, Protocol, TextIO

from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)


class NamedUploadedFile(Protocol):
    """A file-uploader object exposing a filename and readable content."""

    name: str

    def read(self, size: int = -1) -> bytes | str: ...


UploadPayload = bytes | BinaryIO | TextIO
UploadedTextInput = NamedUploadedFile | tuple[str, UploadPayload]


def _document_title(text: str, filename: str) -> str:
    for line in text.splitlines():
        title = line.strip()
        if title:
            return title
    return filename


def _attach_source_metadata(document: Document, filename: str) -> Document:
    document.metadata["source_file"] = filename
    document.metadata["document_title"] = _document_title(document.page_content, filename)
    return document


def _expand_text_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    candidates = [paths] if isinstance(paths, (str, Path)) else list(paths)
    expanded: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.txt")))
        else:
            if path.suffix.lower() != ".txt":
                raise ValueError(f"Expected a .txt file, received {str(path)!r}.")
            if not path.is_file():
                raise FileNotFoundError(f"Text file does not exist: {path}")
            expanded.append(path)
    if not expanded:
        raise ValueError("No .txt files were found in the provided paths.")
    return expanded


def load_text_files(paths: str | Path | Iterable[str | Path]) -> list[Document]:
    """Load one or more local .txt files or directories containing .txt files."""
    documents: list[Document] = []
    for path in _expand_text_paths(paths):
        loaded = TextLoader(str(path), encoding="utf-8").load()
        documents.extend(_attach_source_metadata(doc, path.name) for doc in loaded)
        logger.info("event=text_file_loaded source_file=%s", path.name)
    return documents


def _upload_name_and_payload(upload: UploadedTextInput) -> tuple[str, UploadPayload]:
    if isinstance(upload, tuple):
        filename, payload = upload
        return filename, payload

    filename = upload.name
    getvalue = getattr(upload, "getvalue", None)
    if callable(getvalue):
        return filename, getvalue()
    return filename, upload


def _read_upload_payload(payload: UploadPayload) -> bytes:
    if isinstance(payload, bytes):
        return payload

    tell = getattr(payload, "tell", None)
    seek = getattr(payload, "seek", None)
    position = tell() if callable(tell) else None
    try:
        content = payload.read()
    finally:
        if position is not None and callable(seek):
            seek(position)
    return content.encode("utf-8") if isinstance(content, str) else content


def load_uploaded_text_files(uploads: Iterable[UploadedTextInput]) -> list[Document]:
    """Load uploaded .txt files, preserving their filenames and cleaning temp files."""
    documents: list[Document] = []
    for upload in uploads:
        original_name, payload = _upload_name_and_payload(upload)
        filename = Path(original_name).name
        if not filename or Path(filename).suffix.lower() != ".txt":
            raise ValueError(f"Uploaded file must have a .txt filename, received {original_name!r}.")

        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temporary_file:
                temporary_path = temporary_file.name
                temporary_file.write(_read_upload_payload(payload))
            loaded = TextLoader(temporary_path, encoding="utf-8").load()
            for document in loaded:
                document.metadata["source"] = filename
                documents.append(_attach_source_metadata(document, filename))
            logger.info("event=uploaded_text_file_loaded source_file=%s", filename)
        finally:
            if temporary_path is not None:
                os.unlink(temporary_path)
    return documents


def split_documents(
    documents: Iterable[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    *,
    config: AppConfig | None = None,
) -> list[Document]:
    """Split documents and attach stable chunk metadata fields."""
    settings = config or load_config()
    chunk_size = settings.default_chunk_size if chunk_size is None else chunk_size
    chunk_overlap = settings.default_chunk_overlap if chunk_overlap is None else chunk_overlap
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Document] = []
    for document in documents:
        source = document.metadata.get("source") or "unknown.txt"
        filename = str(document.metadata.get("source_file") or Path(str(source)).name)
        title = str(
            document.metadata.get("document_title")
            or _document_title(document.page_content, filename)
        )
        for index, chunk in enumerate(splitter.split_documents([document])):
            chunk.metadata.update(
                source_file=filename,
                document_title=title,
                chunk_id=uuid.uuid4().hex,
                chunk_index=index,
            )
            chunks.append(chunk)
    logger.info("event=documents_split chunk_count=%d", len(chunks))
    return chunks


def create_embeddings(
    model_name: str | None = None,
    *,
    config: AppConfig | None = None,
) -> HuggingFaceEmbeddings:
    """Initialize Hugging Face embeddings using the configured model name."""
    settings = config or load_config()
    resolved_model_name = model_name or settings.require_embedding_model_name()
    logger.info("event=embedding_model_initialized model_name=%s", resolved_model_name)
    return HuggingFaceEmbeddings(model_name=resolved_model_name)


def build_faiss_vector_store(
    documents: Iterable[Document],
    embeddings: Embeddings | None = None,
    *,
    model_name: str | None = None,
    config: AppConfig | None = None,
) -> FAISS:
    """Build a FAISS vector store over document chunks."""
    chunk_list = list(documents)
    if not chunk_list:
        raise ValueError("At least one document chunk is required to build a FAISS index.")
    embedding_function = (
        embeddings if embeddings is not None else create_embeddings(model_name, config=config)
    )
    logger.info("event=faiss_index_build_started document_count=%d", len(chunk_list))
    return FAISS.from_documents(chunk_list, embedding_function)


def build_bm25_retriever(documents: Iterable[Document]) -> BM25Retriever:
    """Build a BM25 retriever over document chunks."""
    chunk_list = list(documents)
    if not chunk_list:
        raise ValueError("At least one document chunk is required to build a BM25 retriever.")
    logger.info("event=bm25_retriever_build_started document_count=%d", len(chunk_list))
    return BM25Retriever.from_documents(chunk_list)
