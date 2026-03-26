"""
VectorStore abstraction layer for OpenRAG.

Provides a backend-agnostic interface for vector storage and search,
with implementations for OpenSearch and Amazon S3 Vectors.
"""

from .base import VectorStore, VectorRecord, VectorResult

__all__ = ["VectorStore", "VectorRecord", "VectorResult"]
