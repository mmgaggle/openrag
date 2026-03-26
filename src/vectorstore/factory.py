"""
VectorStore factory.

Reads VECTOR_BACKEND to decide which implementation to return.
"""

from __future__ import annotations

import os
from typing import Optional

from utils.logging_config import get_logger

from .base import VectorStore

logger = get_logger(__name__)

# Supported backend names
BACKEND_OPENSEARCH = "opensearch"
BACKEND_S3VECTORS = "s3vectors"


def get_vector_backend() -> str:
    return os.getenv("VECTOR_BACKEND", BACKEND_OPENSEARCH).lower()


async def create_vector_store() -> VectorStore:
    """Create and initialize the configured VectorStore implementation."""
    backend = get_vector_backend()

    if backend == BACKEND_S3VECTORS:
        from .s3vectors_store import S3VectorsStore

        bucket = os.getenv("S3_VECTORS_BUCKET_NAME", "openrag-vectors")
        region = os.getenv("S3_VECTORS_REGION", os.getenv("AWS_REGION", "us-east-1"))
        role_arn = os.getenv("S3_VECTORS_ROLE_ARN", "")

        if not role_arn:
            raise ValueError(
                "S3_VECTORS_ROLE_ARN must be set when VECTOR_BACKEND=s3vectors"
            )

        store = S3VectorsStore(
            bucket_name=bucket,
            region=region,
            role_arn=role_arn,
        )
        await store.initialize()
        logger.info(
            "Initialized S3 Vectors store",
            bucket=bucket,
            region=region,
        )
        return store

    elif backend == BACKEND_OPENSEARCH:
        # OpenSearch path — no VectorStore wrapper yet, returns None so
        # callers fall back to the existing direct-OpenSearch code paths.
        logger.info("VECTOR_BACKEND=opensearch; using existing OpenSearch code paths")
        return None  # type: ignore[return-value]

    else:
        raise ValueError(f"Unknown VECTOR_BACKEND: {backend!r}")
