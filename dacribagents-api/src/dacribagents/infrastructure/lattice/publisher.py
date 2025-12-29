"""
Lattice Kafka Publisher with Claim Check Pattern.

Publishes raw RFC822 emails to Lattice pipeline via Kafka.
Large emails (>256KB) are stored in MinIO/S3 and only a reference
is published to Kafka.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger


# Claim check thresholds (matching Lattice)
INLINE_THRESHOLD_BYTES = 256 * 1024  # 256 KB
MAX_EMAIL_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


@dataclass
class LatticeEnvelope:
    """Lattice Kafka message envelope."""
    
    message_id: str
    schema_version: str = "v1"
    domain: str = "mail"
    stage: str = "raw"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    tenant_id: str = ""
    account_id: str = ""
    alias: Optional[str] = None
    
    source_service: str = "woodcreek-email-worker"
    source_version: str = "1.0.0"
    
    data_classification: str = "internal"
    pii: bool = True
    
    trace_id: Optional[str] = None
    
    payload: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert envelope to dictionary for JSON serialization."""
        return {
            "message_id": self.message_id,
            "schema_version": self.schema_version,
            "domain": self.domain,
            "stage": self.stage,
            "created_at": self.created_at,
            "tenant_id": self.tenant_id,
            "account_id": self.account_id,
            "alias": self.alias,
            "source": {
                "service": self.source_service,
                "version": self.source_version,
            },
            "data_classification": self.data_classification,
            "pii": {
                "contains_pii": self.pii,
            },
            "trace_id": self.trace_id or str(uuid.uuid4()),
            "payload": self.payload,
        }


class ObjectStore:
    """S3-compatible object storage client for MinIO/S3."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key: str = "",
        secret_key: str = "",
        region: str = "us-east-1",
        bucket: str = "lattice-raw",
    ):
        """Initialize object store.
        
        Args:
            endpoint_url: MinIO endpoint (None for AWS S3)
            access_key: AWS/MinIO access key
            secret_key: AWS/MinIO secret key
            region: AWS region
            bucket: Default bucket name
        """
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version="s3v4"),
        )
        
        # Ensure bucket exists
        self._ensure_bucket()
    
    def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchBucket"):
                logger.info(f"Creating bucket {self.bucket}")
                self._client.create_bucket(Bucket=self.bucket)
            else:
                raise

    def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "message/rfc822",
    ) -> str:
        """
        Upload bytes to object storage.

        Args:
            key: Object key (path)
            data: Raw bytes to upload
            content_type: MIME content type

        Returns:
            URI in format s3://{bucket}/{key}
        """
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        uri = f"s3://{self.bucket}/{key}"
        logger.debug(f"Uploaded {len(data)} bytes to {uri}")
        return uri


def build_message_key(
    tenant_id: str,
    account_id: str,
    alias: str,
    provider: str,
    provider_message_id: str,
    filename: str = "message.eml",
) -> str:
    """
    Build object key for a message.

    Pattern: {tenant_id}/{account_id}/{alias}/{provider}/{provider_message_id}/{filename}
    """
    # Sanitize provider_message_id (may contain special chars)
    safe_msg_id = provider_message_id.replace(":", "_").replace("/", "_")
    return f"{tenant_id}/{account_id}/{alias}/{provider}/{safe_msg_id}/{filename}"


class LatticeKafkaPublisher:
    """Publishes emails to Lattice via Kafka with claim check pattern."""
    
    def __init__(
        self,
        brokers: str,
        username: str,
        password: str,
        topic: str = "lattice.mail.raw.v1",
        tenant_id: str = "woodcreek",
        source_service: str = "woodcreek-email-worker",
        source_version: str = "1.0.0",
        object_store: Optional[ObjectStore] = None,
    ):
        from confluent_kafka import Producer
        
        self.topic = topic
        self.tenant_id = tenant_id
        self.source_service = source_service
        self.source_version = source_version
        self.object_store = object_store
        
        config = {
            "bootstrap.servers": brokers,
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "PLAIN",
            "sasl.username": username,
            "sasl.password": password,
            # Reliability settings
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 1000,
            # Batching for efficiency
            "batch.size": 65536,
            "linger.ms": 100,
            # Increase max message size (but we'll use claim check anyway)
            "message.max.bytes": 1048576,  # 1 MB
        }
        
        self.producer = Producer(config)
        self._delivery_errors: list[str] = []
        
        logger.info(f"Kafka publisher initialized for topic {topic}")
        if object_store:
            logger.info(f"Claim check enabled with bucket {object_store.bucket}")
    
    def _delivery_callback(self, err, msg):
        """Handle delivery confirmation."""
        if err:
            logger.error(f"Delivery failed: {err}")
            self._delivery_errors.append(str(err))
        else:
            logger.debug(f"Delivered to {msg.topic()}[{msg.partition()}] @ {msg.offset()}")
    
    def publish(
        self,
        rfc822_bytes: bytes,
        account_id: str,
        alias: str,
        provider_message_id: str,
        folder: str = "INBOX",
        received_at: Optional[datetime] = None,
    ) -> str:
        """
        Publish a raw email to Lattice.
        
        Uses claim check pattern:
        - Always stores raw bytes in object storage
        - Only includes inline payload if ≤ 256 KB
        - Large emails (> 256 KB) only have raw_object_uri
        
        Args:
            rfc822_bytes: Raw RFC822 email bytes
            account_id: Lattice account ID (e.g., "workmail-agents")
            alias: Email alias for routing (e.g., "solar", "hoa", "agents")
            provider_message_id: Unique ID from provider (e.g., IMAP UID)
            folder: Source folder name
            received_at: When email was received
        
        Returns:
            message_id of the envelope
        
        Raises:
            ValueError: If email exceeds MAX_EMAIL_SIZE_BYTES (25 MB)
        """
        size_bytes = len(rfc822_bytes)
        
        # Reject oversized emails
        if size_bytes > MAX_EMAIL_SIZE_BYTES:
            raise ValueError(
                f"Email size {size_bytes:,} bytes exceeds maximum {MAX_EMAIL_SIZE_BYTES:,} bytes"
            )
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(rfc822_bytes).hexdigest()
        
        # Generate deterministic email_id
        email_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"lattice:{self.tenant_id}:{content_hash}"))
        
        # ===== CLAIM CHECK PATTERN =====
        raw_object_uri: str | None = None
        raw_payload: str | None = None
        
        if self.object_store:
            # Always store in object storage
            object_key = build_message_key(
                tenant_id=self.tenant_id,
                account_id=account_id,
                alias=alias,
                provider="imap",
                provider_message_id=provider_message_id,
            )
            raw_object_uri = self.object_store.put_bytes(object_key, rfc822_bytes)
            logger.debug(f"Stored email in {raw_object_uri} ({size_bytes:,} bytes)")
            
            # Only inline small emails
            if size_bytes <= INLINE_THRESHOLD_BYTES:
                raw_payload = base64.urlsafe_b64encode(rfc822_bytes).decode("ascii")
                logger.debug(f"Email under threshold ({size_bytes:,} ≤ {INLINE_THRESHOLD_BYTES:,}), including inline")
            else:
                logger.info(f"Large email ({size_bytes:,} bytes), using claim check (no inline payload)")
        else:
            # No object store configured - must inline (will fail for large emails)
            raw_payload = base64.urlsafe_b64encode(rfc822_bytes).decode("ascii")
            logger.warning("No object store configured, inlining payload (may fail for large emails)")
        
        # Build payload matching lattice.mail.raw.v1 schema
        payload = {
            "provider": "imap",
            "provider_message_id": provider_message_id,
            "email_id": email_id,
            "folder": folder,
            "received_at": (received_at or datetime.now(timezone.utc)).isoformat(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "raw_format": "rfc822",
            # Claim check fields
            "raw_object_uri": raw_object_uri,
            "raw_payload": raw_payload,
            # IMAP metadata
            "imap_metadata": {
                "folder": folder,
                "provider_message_id": provider_message_id,
            },
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        # Create envelope
        envelope = LatticeEnvelope(
            message_id=str(uuid.uuid4()),
            tenant_id=self.tenant_id,
            account_id=account_id,
            alias=alias,
            source_service=self.source_service,
            source_version=self.source_version,
            payload=payload,
        )
        
        # Kafka key for partitioning (same email always goes to same partition)
        key = f"{self.tenant_id}:{account_id}:{provider_message_id}".encode()
        
        # Publish
        self.producer.produce(
            topic=self.topic,
            key=key,
            value=json.dumps(envelope.to_dict()).encode("utf-8"),
            callback=self._delivery_callback,
        )
        
        logger.debug(f"Published message {envelope.message_id} to {self.topic}")
        return envelope.message_id
    
    def flush(self, timeout: float = 30.0) -> int:
        """
        Flush pending messages.
        
        Returns:
            Number of messages still in queue after timeout
        """
        return self.producer.flush(timeout)
    
    def get_errors(self) -> list[str]:
        """Get delivery errors since last check and clear the list."""
        errors = self._delivery_errors.copy()
        self._delivery_errors.clear()
        return errors
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        return False


def get_object_store_from_env() -> ObjectStore | None:
    """
    Create ObjectStore from environment variables.
    
    Environment variables:
        MINIO_ENDPOINT: MinIO endpoint URL (e.g., http://minio:9000)
        MINIO_ACCESS_KEY: MinIO access key
        MINIO_SECRET_KEY: MinIO secret key
        MINIO_BUCKET: Bucket name (default: lattice-raw)
        
    Or for AWS S3:
        AWS_ACCESS_KEY_ID: AWS access key
        AWS_SECRET_ACCESS_KEY: AWS secret key
        AWS_REGION: AWS region (default: us-east-1)
        S3_BUCKET: Bucket name (default: lattice-raw)
    
    Returns:
        ObjectStore if configured, None otherwise
    """
    # Try MinIO first
    minio_endpoint = os.environ.get("MINIO_ENDPOINT")
    minio_access = os.environ.get("MINIO_ACCESS_KEY")
    minio_secret = os.environ.get("MINIO_SECRET_KEY")
    
    if minio_endpoint and minio_access and minio_secret:
        return ObjectStore(
            endpoint_url=minio_endpoint,
            access_key=minio_access,
            secret_key=minio_secret,
            bucket=os.environ.get("MINIO_BUCKET", "lattice-raw"),
        )
    
    # Try AWS S3
    aws_access = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    if aws_access and aws_secret:
        return ObjectStore(
            endpoint_url=None,  # Use AWS S3
            access_key=aws_access,
            secret_key=aws_secret,
            region=os.environ.get("AWS_REGION", "us-east-1"),
            bucket=os.environ.get("S3_BUCKET", "lattice-raw"),
        )
    
    logger.warning("No object storage configured - claim check disabled")
    return None


def get_lattice_publisher(
    tenant_id: str | None = None,
    topic: str | None = None,
) -> LatticeKafkaPublisher:
    """
    Create a Lattice Kafka publisher from environment variables.
    
    Environment variables:
        KAFKA_BROKERS: Kafka bootstrap servers
        KAFKA_SASL_USERNAME: SASL username (API key)
        KAFKA_SASL_PASSWORD: SASL password (API secret)
        KAFKA_TOPIC_RAW: Topic name (default: lattice.mail.raw.v1)
        LATTICE_TENANT_ID: Tenant ID (default: woodcreek)
        
        Plus object storage vars (see get_object_store_from_env)
    
    Args:
        tenant_id: Override tenant ID from env
        topic: Override topic from env
    
    Returns:
        Configured LatticeKafkaPublisher
    """
    brokers = os.environ.get("KAFKA_BROKERS")
    username = os.environ.get("KAFKA_SASL_USERNAME")
    password = os.environ.get("KAFKA_SASL_PASSWORD")
    
    if not all([brokers, username, password]):
        raise ValueError(
            "Kafka not configured. Set KAFKA_BROKERS, KAFKA_SASL_USERNAME, KAFKA_SASL_PASSWORD"
        )
    
    # Get object store for claim check
    object_store = get_object_store_from_env()
    
    return LatticeKafkaPublisher(
        brokers=brokers,
        username=username,
        password=password,
        topic=topic or os.environ.get("KAFKA_TOPIC_RAW", "lattice.mail.raw.v1"),
        tenant_id=tenant_id or os.environ.get("LATTICE_TENANT_ID", "woodcreek"),
        object_store=object_store,
    )