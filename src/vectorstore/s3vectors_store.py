"""
Amazon S3 Vectors implementation of the VectorStore protocol.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

import boto3
from botocore.exceptions import ClientError

from utils.embedding_fields import normalize_model_name
from utils.logging_config import get_logger

from .base import VectorRecord, VectorResult

logger = get_logger(__name__)

# S3 Vectors API limits
_PUT_BATCH_SIZE = 500
_GET_BATCH_SIZE = 100
_DELETE_BATCH_SIZE = 500
_MAX_TOP_K = 100

# STS credential cache TTL (seconds). Credentials last 1 hour; refresh early.
_STS_CACHE_TTL = 50 * 60  # 50 minutes


def _user_id_hash(user_id: str) -> str:
    """Return a short, filesystem-safe hash of a user ID."""
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]


def _index_name(user_id: str, model_name: str, dimensions: int) -> str:
    """Build a deterministic index name for a user + model + dimension triple."""
    uid = _user_id_hash(user_id)
    model = normalize_model_name(model_name)
    return f"user-{uid}-{model}-{dimensions}"


class S3VectorsStore:
    """VectorStore backed by Amazon S3 Vectors.

    Uses per-user indexes with STS-scoped credentials for access control.
    """

    def __init__(
        self,
        bucket_name: str,
        region: str,
        role_arn: str,
    ):
        self.bucket_name = bucket_name
        self.region = region
        self.role_arn = role_arn

        # Admin client — used for bucket/index management only
        self._admin_client: Any = None
        self._sts_client: Any = None

        # Credential cache: (user_id, scope) -> (client, expiry)
        self._credential_cache: Dict[Tuple[str, str], Tuple[Any, float]] = {}

        # Known indexes cache: set of index names confirmed to exist
        self._known_indexes: Set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        loop = asyncio.get_event_loop()
        self._admin_client = await loop.run_in_executor(
            None,
            lambda: boto3.client("s3vectors", region_name=self.region),
        )
        self._sts_client = await loop.run_in_executor(
            None,
            lambda: boto3.client("sts", region_name=self.region),
        )

        # Ensure the vector bucket exists
        await self._ensure_bucket()

    async def cleanup(self) -> None:
        self._credential_cache.clear()
        self._known_indexes.clear()

    async def health_check(self) -> bool:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._admin_client.get_vector_bucket(
                    vectorBucketName=self.bucket_name,
                ),
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Bucket / index management
    # ------------------------------------------------------------------

    async def _ensure_bucket(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._admin_client.get_vector_bucket(
                    vectorBucketName=self.bucket_name,
                ),
            )
            logger.info("S3 Vectors bucket exists", bucket=self.bucket_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info("Creating S3 Vectors bucket", bucket=self.bucket_name)
                await loop.run_in_executor(
                    None,
                    lambda: self._admin_client.create_vector_bucket(
                        vectorBucketName=self.bucket_name,
                    ),
                )
            else:
                raise

    async def ensure_index(
        self,
        user_id: str,
        embedding_model: str,
        dimensions: int,
    ) -> None:
        name = _index_name(user_id, embedding_model, dimensions)
        if name in self._known_indexes:
            return

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._admin_client.get_index(
                    vectorBucketName=self.bucket_name,
                    indexName=name,
                ),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info(
                    "Creating S3 Vectors index",
                    index=name,
                    dimensions=dimensions,
                )
                await loop.run_in_executor(
                    None,
                    lambda: self._admin_client.create_index(
                        vectorBucketName=self.bucket_name,
                        indexName=name,
                        dimension=dimensions,
                        distanceMetric="cosine",
                    ),
                )
            else:
                raise

        self._known_indexes.add(name)

    # ------------------------------------------------------------------
    # STS credential scoping
    # ------------------------------------------------------------------

    def _get_scoped_client(self, user_id: str, scope: str = "readwrite") -> Any:
        """Return a boto3 s3vectors client with credentials scoped to the user's indexes."""
        cache_key = (user_id, scope)
        cached = self._credential_cache.get(cache_key)
        if cached and cached[1] > time.time():
            return cached[0]

        uid = _user_id_hash(user_id)
        resource_arn = (
            f"arn:aws:s3vectors:{self.region}:*:"
            f"bucket/{self.bucket_name}/index/user-{uid}-*"
        )

        actions_map = {
            "read": [
                "s3vectors:GetVectors",
                "s3vectors:QueryVectors",
                "s3vectors:ListVectors",
            ],
            "write": [
                "s3vectors:PutVectors",
                "s3vectors:DeleteVectors",
            ],
            "readwrite": [
                "s3vectors:GetVectors",
                "s3vectors:QueryVectors",
                "s3vectors:ListVectors",
                "s3vectors:PutVectors",
                "s3vectors:DeleteVectors",
            ],
        }

        policy = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": actions_map[scope],
                        "Resource": resource_arn,
                    }
                ],
            }
        )

        response = self._sts_client.assume_role(
            RoleArn=self.role_arn,
            RoleSessionName=f"openrag-{uid}-{scope}",
            Policy=policy,
            DurationSeconds=3600,
        )

        creds = response["Credentials"]
        client = boto3.client(
            "s3vectors",
            region_name=self.region,
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )

        self._credential_cache[cache_key] = (client, time.time() + _STS_CACHE_TTL)
        return client

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    async def put_vectors(
        self,
        vectors: List[VectorRecord],
        user_id: str,
        embedding_model: str,
    ) -> None:
        if not vectors:
            return

        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(
            None, lambda: self._get_scoped_client(user_id, "write")
        )

        # Determine dimensions from first vector and ensure index exists
        dimensions = len(vectors[0].data)
        await self.ensure_index(user_id, embedding_model, dimensions)
        idx = _index_name(user_id, embedding_model, dimensions)

        # Batch into groups of 500
        for start in range(0, len(vectors), _PUT_BATCH_SIZE):
            batch = vectors[start : start + _PUT_BATCH_SIZE]
            payload = [
                {
                    "key": v.key,
                    "data": {"float32": v.data},
                    "metadata": v.metadata,
                }
                for v in batch
            ]

            await loop.run_in_executor(
                None,
                lambda p=payload: client.put_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=idx,
                    vectors=p,
                ),
            )

        logger.info(
            "Put vectors",
            count=len(vectors),
            index=idx,
        )

    async def get_vectors(
        self,
        keys: List[str],
        user_id: str,
        embedding_model: str,
    ) -> List[VectorRecord]:
        if not keys:
            return []

        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(
            None, lambda: self._get_scoped_client(user_id, "read")
        )

        # We need to know dimensions to resolve the index name.
        # List the user's indexes and find the one for this model.
        idx = await self._resolve_index(user_id, embedding_model)
        if not idx:
            return []

        results: List[VectorRecord] = []
        for start in range(0, len(keys), _GET_BATCH_SIZE):
            batch = keys[start : start + _GET_BATCH_SIZE]
            response = await loop.run_in_executor(
                None,
                lambda b=batch: client.get_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=idx,
                    keys=b,
                    returnData=True,
                    returnMetadata=True,
                ),
            )
            for v in response.get("vectors", []):
                results.append(
                    VectorRecord(
                        key=v["key"],
                        data=v.get("data", {}).get("float32", []),
                        metadata=v.get("metadata", {}),
                    )
                )
        return results

    async def delete_vectors(
        self,
        keys: List[str],
        user_id: str,
        embedding_model: str,
    ) -> int:
        if not keys:
            return 0

        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(
            None, lambda: self._get_scoped_client(user_id, "write")
        )

        idx = await self._resolve_index(user_id, embedding_model)
        if not idx:
            return 0

        deleted = 0
        for start in range(0, len(keys), _DELETE_BATCH_SIZE):
            batch = keys[start : start + _DELETE_BATCH_SIZE]
            await loop.run_in_executor(
                None,
                lambda b=batch: client.delete_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=idx,
                    keys=b,
                ),
            )
            deleted += len(batch)

        logger.info("Deleted vectors", count=deleted, index=idx)
        return deleted

    async def query_vectors(
        self,
        embedding: List[float],
        user_id: str,
        embedding_model: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[VectorResult]:
        top_k = min(top_k, _MAX_TOP_K)

        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(
            None, lambda: self._get_scoped_client(user_id, "read")
        )

        idx = await self._resolve_index(user_id, embedding_model)
        if not idx:
            return []

        kwargs: Dict[str, Any] = {
            "vectorBucketName": self.bucket_name,
            "indexName": idx,
            "queryVector": {"float32": embedding},
            "topK": top_k,
            "returnMetadata": True,
            "returnDistance": True,
        }
        if metadata_filter:
            kwargs["filter"] = metadata_filter

        response = await loop.run_in_executor(
            None,
            lambda: client.query_vectors(**kwargs),
        )

        results: List[VectorResult] = []
        for v in response.get("vectors", []):
            results.append(
                VectorResult(
                    key=v["key"],
                    score=v.get("distance", 0.0),
                    metadata=v.get("metadata", {}),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Document-level helpers
    # ------------------------------------------------------------------

    async def list_document_keys(
        self,
        user_id: str,
        embedding_model: str,
        document_id: str,
    ) -> List[str]:
        """List vector keys belonging to a document by iterating the index.

        Keys follow the ``{document_id}_{n}`` convention, so we filter
        by the ``document_id`` metadata field.
        """
        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(
            None, lambda: self._get_scoped_client(user_id, "read")
        )

        idx = await self._resolve_index(user_id, embedding_model)
        if not idx:
            return []

        keys: List[str] = []
        next_token: Optional[str] = None
        while True:
            kwargs: Dict[str, Any] = {
                "vectorBucketName": self.bucket_name,
                "indexName": idx,
                "maxResults": 1000,
            }
            if next_token:
                kwargs["nextToken"] = next_token

            response = await loop.run_in_executor(
                None,
                lambda kw=kwargs: client.list_vectors(**kw),
            )

            for v in response.get("vectors", []):
                # Keys follow {document_id}_{chunk_index}
                if v["key"].startswith(f"{document_id}_"):
                    keys.append(v["key"])

            next_token = response.get("nextToken")
            if not next_token:
                break

        return keys

    async def document_exists(
        self,
        document_id: str,
        user_id: str,
        embedding_model: str,
    ) -> bool:
        # Try to get the first chunk directly by key convention
        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(
            None, lambda: self._get_scoped_client(user_id, "read")
        )

        idx = await self._resolve_index(user_id, embedding_model)
        if not idx:
            return False

        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.get_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=idx,
                    keys=[f"{document_id}_0"],
                    returnData=False,
                    returnMetadata=False,
                ),
            )
            return len(response.get("vectors", [])) > 0
        except ClientError:
            return False

    async def delete_document(
        self,
        document_id: str,
        user_id: str,
        embedding_model: str,
    ) -> int:
        keys = await self.list_document_keys(user_id, embedding_model, document_id)
        if not keys:
            return 0
        return await self.delete_vectors(keys, user_id, embedding_model)

    # ------------------------------------------------------------------
    # Multi-model helpers
    # ------------------------------------------------------------------

    async def list_user_models(self, user_id: str) -> List[str]:
        """List embedding models for which the user has indexes."""
        uid = _user_id_hash(user_id)
        prefix = f"user-{uid}-"

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._admin_client.list_indexes(
                vectorBucketName=self.bucket_name,
            ),
        )

        models: List[str] = []
        for idx in response.get("indexes", []):
            name = idx.get("indexName", "")
            if name.startswith(prefix):
                # Parse model from: user-{hash}-{model}-{dim}
                suffix = name[len(prefix) :]
                # Last segment after final dash is dimensions
                parts = suffix.rsplit("-", 1)
                if len(parts) == 2:
                    models.append(parts[0])
        return models

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_index(
        self, user_id: str, embedding_model: str
    ) -> Optional[str]:
        """Find the index name for a user + model, discovering dimensions from existing indexes."""
        uid = _user_id_hash(user_id)
        model = normalize_model_name(embedding_model)
        prefix = f"user-{uid}-{model}-"

        # Check cache first
        for name in self._known_indexes:
            if name.startswith(prefix):
                return name

        # Query the API
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._admin_client.list_indexes(
                vectorBucketName=self.bucket_name,
            ),
        )

        for idx in response.get("indexes", []):
            name = idx.get("indexName", "")
            if name.startswith(prefix):
                self._known_indexes.add(name)
                return name

        return None
