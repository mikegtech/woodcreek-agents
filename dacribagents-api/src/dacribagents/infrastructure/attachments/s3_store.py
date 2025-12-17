from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
import boto3
from botocore.config import Config

from dacribagents.domain.entities.attachment import AttachmentRef

@dataclass(frozen=True)
class S3StoreConfig:
    endpoint: str
    region: str
    access_key: str
    secret_key: str
    bucket: str
    prefix: str = "email-attachments"
    use_ssl: bool = False
    force_path_style: bool = True

class S3AttachmentStore:
    def __init__(self, cfg: S3StoreConfig) -> None:
        self.cfg = cfg
        s3_cfg = Config(s3={"addressing_style": "path"} if cfg.force_path_style else {})
        self.client = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint,
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
            region_name=cfg.region,
            use_ssl=cfg.use_ssl,
            config=s3_cfg,
        )

    def put(self, *, account: str, message_id: str, filename: str, content_type: str, data: bytes) -> AttachmentRef:
        # Stable-ish key: prefix/account/message_id/sha256_filename.ext
        h = hashlib.sha256(data).hexdigest()[:16]
        safe_name = filename.replace("/", "_")
        key = f"{self.cfg.prefix}/{account}/{message_id}/{h}-{safe_name}"

        resp = self.client.put_object(
            Bucket=self.cfg.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        etag = resp.get("ETag")
        return AttachmentRef(
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            bucket=self.cfg.bucket,
            key=key,
            etag=etag.strip('"') if isinstance(etag, str) else None,
        )

def s3_store_from_env() -> S3AttachmentStore:
    cfg = S3StoreConfig(
        endpoint=os.environ["S3_ENDPOINT"],
        region=os.getenv("S3_REGION", "us-east-1"),
        access_key=os.environ["S3_ACCESS_KEY"],
        secret_key=os.environ["S3_SECRET_KEY"],
        bucket=os.environ["S3_BUCKET"],
        prefix=os.getenv("S3_PREFIX", "email-attachments"),
        use_ssl=os.getenv("S3_USE_SSL", "false").lower() == "true",
        force_path_style=os.getenv("S3_FORCE_PATH_STYLE", "true").lower() == "true",
    )
    return S3AttachmentStore(cfg)
