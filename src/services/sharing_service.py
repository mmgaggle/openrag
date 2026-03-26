"""
Document sharing service for S3 Vectors backend.

Implements the "embed once, copy vectors" pattern: when a document is shared
with another user, the pre-computed vectors are copied from the owner's index
into the recipient's index.
"""

from __future__ import annotations

from typing import List

from utils.logging_config import get_logger

logger = get_logger(__name__)


class SharingService:
    """Handles cross-user document sharing via vector copy/delete."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    async def share_document(
        self,
        owner_user_id: str,
        recipient_user_id: str,
        document_id: str,
        embedding_model: str,
    ) -> int:
        """Copy all vectors for a document from the owner's index to the recipient's.

        Returns the number of vectors copied.
        """
        # Read vectors from the owner's index
        vectors = await self.vector_store.get_vectors(
            keys=[],  # We don't know all keys yet — discover via list
            user_id=owner_user_id,
            embedding_model=embedding_model,
        )

        # We need to list document keys first, then fetch them
        keys = await self.vector_store.list_document_keys(
            user_id=owner_user_id,
            embedding_model=embedding_model,
            document_id=document_id,
        )

        if not keys:
            logger.warning(
                "No vectors found for document to share",
                document_id=document_id,
                owner=owner_user_id,
            )
            return 0

        vectors = await self.vector_store.get_vectors(
            keys=keys,
            user_id=owner_user_id,
            embedding_model=embedding_model,
        )

        if not vectors:
            return 0

        # Write them into the recipient's index (put_vectors handles ensure_index)
        await self.vector_store.put_vectors(
            vectors=vectors,
            user_id=recipient_user_id,
            embedding_model=embedding_model,
        )

        logger.info(
            "Shared document vectors",
            document_id=document_id,
            owner=owner_user_id,
            recipient=recipient_user_id,
            vector_count=len(vectors),
        )
        return len(vectors)

    async def unshare_document(
        self,
        recipient_user_id: str,
        document_id: str,
        embedding_model: str,
    ) -> int:
        """Remove all vectors for a document from the recipient's index.

        Returns the number of vectors deleted.
        """
        keys = await self.vector_store.list_document_keys(
            user_id=recipient_user_id,
            embedding_model=embedding_model,
            document_id=document_id,
        )

        if not keys:
            return 0

        deleted = await self.vector_store.delete_vectors(
            keys=keys,
            user_id=recipient_user_id,
            embedding_model=embedding_model,
        )

        logger.info(
            "Unshared document vectors",
            document_id=document_id,
            recipient=recipient_user_id,
            deleted_count=deleted,
        )
        return deleted
