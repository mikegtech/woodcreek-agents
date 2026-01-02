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

    # =========================================================================
    # Application
    # =========================================================================
    app_name: str = "Woodcreek Agents"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # =========================================================================
    # API Server
    # =========================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # =========================================================================
    # Milvus Vector Database (Local - Woodcreek)
    # =========================================================================
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "woodcreek_documents"

    # =========================================================================
    # PostgreSQL (LangGraph Checkpoints)
    # =========================================================================
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: SecretStr = Field(default=SecretStr("postgres"))
    postgres_db: str = "woodcreek_agents"

    # =========================================================================
    # Kafka (Confluent Cloud) - For publishing to Lattice
    # =========================================================================
    kafka_brokers: str = ""
    kafka_sasl_username: SecretStr | None = None
    kafka_sasl_password: SecretStr | None = None
    kafka_security_protocol: str = "SASL_SSL"
    kafka_sasl_mechanism: str = "PLAIN"
    kafka_topic_raw: str = "lattice.mail.raw.v1"

    # =========================================================================
    # Lattice Integration
    # =========================================================================
    lattice_tenant_id: str = "woodcreek"
    
    # Lattice Milvus (for RAG retrieval)
    lattice_milvus_uri: str = "http://localhost:19530"
    lattice_milvus_collection: str = "email_chunks_v1"
    
    # Lattice Postgres (system of record)
    lattice_postgres_host: str = "localhost"
    lattice_postgres_port: int = 5432
    lattice_postgres_user: str = "lattice"
    lattice_postgres_password: SecretStr = Field(default=SecretStr("lattice"))
    lattice_postgres_db: str = "lattice"

    # =========================================================================
    # RAG Configuration
    # =========================================================================
    rag_top_k: int = 5
    rag_min_score: float = 0.25
    rag_use_fts: bool = True
    rag_fts_boost: float = 0.2
    rag_max_context_chars: int = 3000

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    llm_provider: Literal["groq", "openai", "anthropic", "local"] = "groq"
    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # Local vLLM (for local inference)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "gpt-oss-20b"

    # =========================================================================
    # Embeddings
    # =========================================================================
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dimension: int = 768

    # =========================================================================
    # Email Worker
    # =========================================================================
    email_poll_minutes: int = 5
    workmail_mailboxes: str = "agents"  # comma-separated: agents,solar,hoa

    # =========================================================================
    # Observability
    # =========================================================================
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "woodcreek-agents"
    honeyhive_api_key: SecretStr | None = None

    # Grafana Cloud (OTLP)
    otel_exporter_otlp_endpoint: str | None = None
    otel_exporter_otlp_headers: str | None = None

    # =========================================================================
    # Computed Properties
    # =========================================================================

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

    @computed_field
    @property
    def lattice_postgres_dsn(self) -> str:
        """Construct Lattice PostgreSQL connection string."""
        password = self.lattice_postgres_password.get_secret_value()
        return f"postgresql://{self.lattice_postgres_user}:{password}@{self.lattice_postgres_host}:{self.lattice_postgres_port}/{self.lattice_postgres_db}"

    @computed_field
    @property
    def kafka_enabled(self) -> bool:
        """Check if Kafka is configured."""
        return bool(self.kafka_brokers and self.kafka_sasl_username and self.kafka_sasl_password)

    def get_kafka_config(self) -> dict:
        """Get Kafka producer configuration dict."""
        if not self.kafka_enabled:
            raise ValueError("Kafka not configured. Set KAFKA_BROKERS, KAFKA_SASL_USERNAME, KAFKA_SASL_PASSWORD")
        
        return {
            "bootstrap.servers": self.kafka_brokers,
            "security.protocol": self.kafka_security_protocol,
            "sasl.mechanisms": self.kafka_sasl_mechanism,
            "sasl.username": self.kafka_sasl_username.get_secret_value(),
            "sasl.password": self.kafka_sasl_password.get_secret_value(),
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 1000,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()