"""Unit tests for S3VectorsStore."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vectorstore.s3vectors_store import (
    S3VectorsStore,
    _user_id_hash,
    _index_name,
    _PUT_BATCH_SIZE,
    _GET_BATCH_SIZE,
    _DELETE_BATCH_SIZE,
)
from vectorstore.base import VectorRecord, VectorResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store():
    """Create a store with mocked AWS clients."""
    store = S3VectorsStore(
        bucket_name="test-bucket",
        region="us-east-1",
        role_arn="arn:aws:iam::123456:role/TestRole",
    )
    store._admin_client = MagicMock()
    store._sts_client = MagicMock()
    return store


def _mock_sts_response():
    return {
        "Credentials": {
            "AccessKeyId": "AKIA_TEST",
            "SecretAccessKey": "secret_test",
            "SessionToken": "token_test",
        }
    }


# ---------------------------------------------------------------------------
# Tests: naming helpers
# ---------------------------------------------------------------------------

class TestNamingHelpers:
    def test_user_id_hash_deterministic(self):
        h1 = _user_id_hash("user123")
        h2 = _user_id_hash("user123")
        assert h1 == h2
        assert len(h1) == 12

    def test_user_id_hash_different_users(self):
        assert _user_id_hash("alice") != _user_id_hash("bob")

    def test_index_name_format(self):
        name = _index_name("user1", "text-embedding-3-small", 1536)
        uid = _user_id_hash("user1")
        assert name == f"user-{uid}-text_embedding_3_small-1536"

    def test_index_name_different_dimensions(self):
        a = _index_name("user1", "model-a", 1536)
        b = _index_name("user1", "model-a", 3072)
        assert a != b


# ---------------------------------------------------------------------------
# Tests: ensure_index
# ---------------------------------------------------------------------------

class TestEnsureIndex:
    @pytest.mark.asyncio
    async def test_creates_index_on_404(self):
        from botocore.exceptions import ClientError

        store = _make_store()
        store._admin_client.get_index.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not found"}}, "GetIndex"
        )
        store._admin_client.create_index.return_value = {}

        await store.ensure_index("user1", "text-embedding-3-small", 1536)

        store._admin_client.create_index.assert_called_once()
        call_kwargs = store._admin_client.create_index.call_args
        assert call_kwargs.kwargs["dimension"] == 1536
        assert call_kwargs.kwargs["distanceMetric"] == "cosine"

    @pytest.mark.asyncio
    async def test_skips_existing_index(self):
        store = _make_store()
        store._admin_client.get_index.return_value = {"indexName": "test"}

        await store.ensure_index("user1", "text-embedding-3-small", 1536)
        store._admin_client.create_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_caches_known_index(self):
        store = _make_store()
        store._admin_client.get_index.return_value = {"indexName": "test"}

        await store.ensure_index("user1", "text-embedding-3-small", 1536)
        await store.ensure_index("user1", "text-embedding-3-small", 1536)

        # Second call should not hit the API
        assert store._admin_client.get_index.call_count == 1


# ---------------------------------------------------------------------------
# Tests: credential scoping
# ---------------------------------------------------------------------------

class TestCredentialScoping:
    def test_scoped_client_calls_assume_role(self):
        store = _make_store()
        store._sts_client.assume_role.return_value = _mock_sts_response()

        with patch("vectorstore.s3vectors_store.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            client = store._get_scoped_client("user1", "read")

        store._sts_client.assume_role.assert_called_once()
        call_kwargs = store._sts_client.assume_role.call_args.kwargs
        assert call_kwargs["RoleArn"] == store.role_arn
        assert "user1" not in call_kwargs["RoleSessionName"]  # Uses hash
        assert _user_id_hash("user1") in call_kwargs["RoleSessionName"]

    def test_scoped_client_caches(self):
        store = _make_store()
        store._sts_client.assume_role.return_value = _mock_sts_response()

        with patch("vectorstore.s3vectors_store.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            c1 = store._get_scoped_client("user1", "read")
            c2 = store._get_scoped_client("user1", "read")

        # Should only call assume_role once (cached)
        assert store._sts_client.assume_role.call_count == 1
        assert c1 is c2


# ---------------------------------------------------------------------------
# Tests: put_vectors batching
# ---------------------------------------------------------------------------

class TestPutVectors:
    @pytest.mark.asyncio
    async def test_batches_large_input(self):
        store = _make_store()
        store._sts_client.assume_role.return_value = _mock_sts_response()

        mock_client = MagicMock()
        mock_client.put_vectors.return_value = {}

        with patch("vectorstore.s3vectors_store.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            # Create 501 vectors to force 2 batches
            vectors = [
                VectorRecord(key=f"k{i}", data=[0.1] * 4, metadata={})
                for i in range(501)
            ]

            # Pre-cache the index as known
            idx = _index_name("user1", "model-a", 4)
            store._known_indexes.add(idx)

            await store.put_vectors(vectors, "user1", "model-a")

        assert mock_client.put_vectors.call_count == 2


# ---------------------------------------------------------------------------
# Tests: query_vectors
# ---------------------------------------------------------------------------

class TestQueryVectors:
    @pytest.mark.asyncio
    async def test_returns_results(self):
        store = _make_store()
        store._sts_client.assume_role.return_value = _mock_sts_response()

        idx = _index_name("user1", "model-a", 4)
        store._known_indexes.add(idx)

        mock_client = MagicMock()
        mock_client.query_vectors.return_value = {
            "vectors": [
                {
                    "key": "doc1_0",
                    "distance": 0.15,
                    "metadata": {"filename": "test.pdf", "text": "hello"},
                }
            ]
        }

        with patch("vectorstore.s3vectors_store.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            results = await store.query_vectors(
                embedding=[0.1] * 4,
                user_id="user1",
                embedding_model="model-a",
                top_k=10,
            )

        assert len(results) == 1
        assert results[0].key == "doc1_0"
        assert results[0].score == 0.15
        assert results[0].metadata["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_clamps_top_k(self):
        store = _make_store()
        store._sts_client.assume_role.return_value = _mock_sts_response()

        idx = _index_name("user1", "model-a", 4)
        store._known_indexes.add(idx)

        mock_client = MagicMock()
        mock_client.query_vectors.return_value = {"vectors": []}

        with patch("vectorstore.s3vectors_store.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            await store.query_vectors(
                embedding=[0.1] * 4,
                user_id="user1",
                embedding_model="model-a",
                top_k=500,
            )

        call_kwargs = mock_client.query_vectors.call_args.kwargs
        assert call_kwargs["topK"] == 100  # clamped to _MAX_TOP_K


# ---------------------------------------------------------------------------
# Tests: document_exists
# ---------------------------------------------------------------------------

class TestDocumentExists:
    @pytest.mark.asyncio
    async def test_returns_true_when_chunk_0_exists(self):
        store = _make_store()
        store._sts_client.assume_role.return_value = _mock_sts_response()

        idx = _index_name("user1", "model-a", 4)
        store._known_indexes.add(idx)

        mock_client = MagicMock()
        mock_client.get_vectors.return_value = {
            "vectors": [{"key": "abc123_0"}]
        }

        with patch("vectorstore.s3vectors_store.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            exists = await store.document_exists("abc123", "user1", "model-a")

        assert exists is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_index(self):
        store = _make_store()
        # No indexes in known_indexes, and list_indexes returns empty
        store._admin_client.list_indexes.return_value = {"indexes": []}

        exists = await store.document_exists("abc123", "user1", "model-a")
        assert exists is False


# ---------------------------------------------------------------------------
# Tests: list_user_models
# ---------------------------------------------------------------------------

class TestListUserModels:
    @pytest.mark.asyncio
    async def test_parses_models_from_index_names(self):
        store = _make_store()
        uid = _user_id_hash("user1")

        store._admin_client.list_indexes.return_value = {
            "indexes": [
                {"indexName": f"user-{uid}-text_embedding_3_small-1536"},
                {"indexName": f"user-{uid}-granite_embedding_107m-384"},
                {"indexName": "user-other-model-1536"},  # different user
            ]
        }

        models = await store.list_user_models("user1")
        assert sorted(models) == ["granite_embedding_107m", "text_embedding_3_small"]


# ---------------------------------------------------------------------------
# Tests: health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self):
        store = _make_store()
        store._admin_client.get_vector_bucket.return_value = {}

        assert await store.health_check() is True

    @pytest.mark.asyncio
    async def test_unhealthy(self):
        store = _make_store()
        store._admin_client.get_vector_bucket.side_effect = Exception("unreachable")

        assert await store.health_check() is False
