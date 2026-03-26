"""
VectorStore protocol and data types.

Defines the interface that all vector store backends must implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class VectorRecord:
    """A vector with its key, embedding data, and metadata."""

    key: str
    data: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorResult:
    """A search result with score and metadata."""

    key: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector store backends.

    Each implementation handles its own authentication and index management.
    Methods that operate on vectors take a ``user_id`` so the backend can
    route to the correct per-user index or apply access control.
    """

    async def initialize(self) -> None:
        """Perform any setup required before first use (e.g. create buckets, verify connectivity)."""
        ...

    async def ensure_index(
        self,
        user_id: str,
        embedding_model: str,
        dimensions: int,
    ) -> None:
        """Ensure a vector index exists for the given user and embedding model.

        Implementations should be idempotent — calling this multiple times
        with the same arguments is a no-op.
        """
        ...

    async def put_vectors(
        self,
        vectors: List[VectorRecord],
        user_id: str,
        embedding_model: str,
    ) -> None:
        """Insert or upsert vectors into the user's index for the given model.

        Implementations must handle batching internally if the backend
        imposes per-call limits.
        """
        ...

    async def get_vectors(
        self,
        keys: List[str],
        user_id: str,
        embedding_model: str,
    ) -> List[VectorRecord]:
        """Retrieve vectors by key from the user's index."""
        ...

    async def delete_vectors(
        self,
        keys: List[str],
        user_id: str,
        embedding_model: str,
    ) -> int:
        """Delete vectors by key. Returns the number of vectors deleted."""
        ...

    async def query_vectors(
        self,
        embedding: List[float],
        user_id: str,
        embedding_model: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[VectorResult]:
        """Run a similarity search and return the top-k results."""
        ...

    async def list_document_keys(
        self,
        user_id: str,
        embedding_model: str,
        document_id: str,
    ) -> List[str]:
        """List all vector keys belonging to a given document.

        Keys follow the convention ``{document_id}_{chunk_index}``.
        """
        ...

    async def document_exists(
        self,
        document_id: str,
        user_id: str,
        embedding_model: str,
    ) -> bool:
        """Check whether any vectors exist for the given document."""
        ...

    async def delete_document(
        self,
        document_id: str,
        user_id: str,
        embedding_model: str,
    ) -> int:
        """Delete all vectors for a document. Returns the count of deleted vectors."""
        ...

    async def list_user_models(self, user_id: str) -> List[str]:
        """Return the embedding model names for which the user has indexes."""
        ...

    async def health_check(self) -> bool:
        """Return True if the backend is reachable and healthy."""
        ...

    async def cleanup(self) -> None:
        """Release resources (clients, connections). Called on shutdown."""
        ...
