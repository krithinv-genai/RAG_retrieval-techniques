import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s level=%(levelname)s logger=%(name)s message=%(message)s",
)
logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """Raised when an environment setting is invalid or required but missing."""


@dataclass(frozen=True)
class AppConfig:
    embedding_model_name: str | None
    cross_encoder_model_name: str | None
    default_chunk_size: int
    default_chunk_overlap: int
    default_top_k: int
    default_mmr_fetch_k: int
    default_mmr_lambda: float
    default_rerank_top_n: int
    hf_token: str | None = None
    groq_api_key: str | None = None
    groq_model_name: str | None = None

    def require_embedding_model_name(self) -> str:
        return _require_model_name("EMBEDDING_MODEL_NAME", self.embedding_model_name)

    def require_cross_encoder_model_name(self) -> str:
        return _require_model_name("CROSS_ENCODER_MODEL_NAME", self.cross_encoder_model_name)

    def require_groq_model_name(self) -> str:
        return _require_model_name("GROQ_MODEL_NAME", self.groq_model_name)

    def require_groq_api_key(self) -> str:
        if self.groq_api_key is None:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Set it in .env before initializing the Groq client."
            )
        return self.groq_api_key


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer; received {value!r}.") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number; received {value!r}.") from exc


def _require_model_name(name: str, value: str | None) -> str:
    if value is None:
        raise ConfigurationError(
            f"{name} is not set. Copy .env.example to .env and set {name} "
            "before initializing the corresponding model."
        )
    return value


def load_config() -> AppConfig:
    load_dotenv(_PROJECT_ROOT / ".env")
    config = AppConfig(
        embedding_model_name=_optional_env("EMBEDDING_MODEL_NAME"),
        cross_encoder_model_name=_optional_env("CROSS_ENCODER_MODEL_NAME"),
        default_chunk_size=_int_env("DEFAULT_CHUNK_SIZE", 500),
        default_chunk_overlap=_int_env("DEFAULT_CHUNK_OVERLAP", 100),
        default_top_k=_int_env("DEFAULT_TOP_K", 4),
        default_mmr_fetch_k=_int_env("DEFAULT_MMR_FETCH_K", 12),
        default_mmr_lambda=_float_env("DEFAULT_MMR_LAMBDA", 0.5),
        default_rerank_top_n=_int_env("DEFAULT_RERANK_TOP_N", 15),
        hf_token=_optional_env("HF_TOKEN"),
        groq_api_key=_optional_env("GROQ_API_KEY"),
        groq_model_name=_optional_env("GROQ_MODEL_NAME"),
    )
    logger.info("event=config_loaded")
    return config
