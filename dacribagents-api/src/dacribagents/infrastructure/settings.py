"""Application settings using Pydantic Settings for configuration management."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Woodcreek Agents"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Milvus Vector Database
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "woodcreek_documents"

    # PostgreSQL (LangGraph Checkpoints)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: SecretStr = Field(default=SecretStr("postgres"))
    postgres_db: str = "woodcreek_agents"

    # LLM Configuration
    llm_provider: Literal["groq", "openai", "anthropic", "local"] = "groq"
    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # Local vLLM (for local inference)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "gpt-oss-20b"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Observability
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "woodcreek-agents"
    honeyhive_api_key: SecretStr | None = None

    # Grafana Cloud (OTLP)
    otel_exporter_otlp_endpoint: str | None = None
    otel_exporter_otlp_headers: str | None = None

    @computed_field
    @property
    def postgres_dsn(self) -> str:
        """Construct PostgreSQL connection string."""
        password = self.postgres_password.get_secret_value()
        return f"postgresql://{self.postgres_user}:{password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @computed_field
    @property
    def milvus_uri(self) -> str:
        """Construct Milvus connection URI."""
        return f"http://{self.milvus_host}:{self.milvus_port}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()