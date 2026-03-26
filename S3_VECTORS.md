# S3 Vectors Backend for OpenRAG

OpenRAG supports Amazon S3 Vectors as an alternative to OpenSearch for vector storage and similarity search. This provides a serverless, zero-infrastructure option with per-user IAM-scoped access control.

## Architecture

### How it differs from OpenSearch

| Aspect | OpenSearch | S3 Vectors |
|---|---|---|
| **Index model** | Single shared index with Document-Level Security (DLS) | Per-user indexes with IAM-scoped credentials |
| **Access control** | OpenSearch DLS role filters queries by `owner`/`allowed_users` fields | STS `AssumeRole` with session policies scoped to `user-{hash}-*` indexes |
| **Search** | Hybrid: 70% KNN semantic + 30% BM25 keyword | Pure semantic similarity search |
| **Aggregations** | Native OpenSearch aggregations for faceted filtering | Not supported (application-layer if needed) |
| **Multi-model embeddings** | Dynamic fields in a single index (`chunk_embedding_{model}`) | Separate index per model (fixed dimensions per index) |
| **Infrastructure** | Managed OpenSearch cluster | Serverless, pay-per-use |
| **Latency** | 10-100ms | 100-800ms (acceptable for RAG where LLM inference dominates) |

### Per-user index naming

Each user gets one index per embedding model, named:

```
user-{user_id_hash}-{normalized_model_name}-{dimensions}
```

For example:
```
user-a1b2c3d4e5f6-text_embedding_3_small-1536
user-a1b2c3d4e5f6-granite_embedding_107m-384
```

S3 Vectors supports up to 10,000 indexes per bucket.

### Access control via STS

Instead of OpenSearch's DLS, access is enforced at the IAM level:

1. The backend assumes a role (`S3_VECTORS_ROLE_ARN`) with an inline session policy scoped to the user's index prefix
2. The scoped credentials can only access `user-{hash}-*` indexes
3. Credentials are cached for 50 minutes (STS sessions last 1 hour)

This mirrors the existing `SessionManager.get_user_opensearch_client()` pattern but uses STS instead of JWTs.

### Document sharing

Since access is structural (per-user indexes) rather than field-based (DLS `allowed_users`), sharing works by copying vectors:

1. **Share**: Read vectors from owner's index via `get_vectors`, write to recipient's index via `put_vectors`
2. **Unshare**: Delete the document's vectors from recipient's index via `delete_vectors`

Vector keys follow the `{document_id}_{chunk_index}` convention, making them predictable and idempotent (re-sharing is a no-op due to upsert semantics).

Embeddings are only generated once — sharing copies pre-computed float arrays, not re-embeds.

## Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `VECTOR_BACKEND` | `opensearch` | Set to `s3vectors` to use S3 Vectors |
| `S3_VECTORS_BUCKET_NAME` | `openrag-vectors` | Name of the S3 Vectors vector bucket |
| `S3_VECTORS_REGION` | `us-east-1` | AWS region for S3 Vectors |
| `S3_VECTORS_ROLE_ARN` | (required) | IAM role ARN that the backend assumes for per-user credential scoping |
| `AWS_ACCESS_KEY_ID` | | AWS credentials for the backend process |
| `AWS_SECRET_ACCESS_KEY` | | AWS credentials for the backend process |
| `AWS_REGION` | `us-east-1` | Default AWS region |

### Docker Compose

Use the S3 Vectors override file to disable OpenSearch and configure the backend:

```bash
docker compose -f docker-compose.yml -f docker-compose.s3vectors.yml up
```

This override:
- Disables the `opensearch` and `dashboards` services
- Sets `VECTOR_BACKEND=s3vectors` on the backend
- Forces `DISABLE_INGEST_WITH_LANGFLOW=true` (the Langflow OpenSearch component is not yet ported)

### IAM setup

The backend needs:

1. **An IAM role** (`S3_VECTORS_ROLE_ARN`) that the backend can assume. This role needs:
   - `s3vectors:CreateVectorBucket`, `s3vectors:GetVectorBucket` on the bucket
   - `s3vectors:CreateIndex`, `s3vectors:GetIndex`, `s3vectors:ListIndexes` on all indexes
   - `s3vectors:PutVectors`, `s3vectors:GetVectors`, `s3vectors:DeleteVectors`, `s3vectors:QueryVectors`, `s3vectors:ListVectors` on all indexes

2. **A trust policy** on the role allowing the backend's identity to assume it

3. **The backend's own credentials** (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or an instance profile) must have `sts:AssumeRole` permission on the role ARN

Example IAM policy for the role:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3vectors:CreateVectorBucket",
                "s3vectors:GetVectorBucket",
                "s3vectors:ListVectorBuckets"
            ],
            "Resource": "arn:aws:s3vectors:*:*:bucket/openrag-vectors"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3vectors:CreateIndex",
                "s3vectors:DeleteIndex",
                "s3vectors:GetIndex",
                "s3vectors:ListIndexes",
                "s3vectors:PutVectors",
                "s3vectors:GetVectors",
                "s3vectors:DeleteVectors",
                "s3vectors:QueryVectors",
                "s3vectors:ListVectors"
            ],
            "Resource": "arn:aws:s3vectors:*:*:bucket/openrag-vectors/index/*"
        }
    ]
}
```

Per-user credential scoping is handled automatically by the backend — each user's STS session is restricted to their own `user-{hash}-*` indexes via inline session policies.

## Migrating from OpenSearch

A migration script is provided to copy existing data:

```bash
# Dry run — see what would be migrated
python scripts/migrate_opensearch_to_s3vectors.py \
    --opensearch-host localhost \
    --opensearch-port 9200 \
    --opensearch-password YOUR_PASSWORD \
    --index documents \
    --s3v-bucket openrag-vectors \
    --s3v-region us-east-1 \
    --s3v-role-arn arn:aws:iam::ACCOUNT:role/openrag-vectors \
    --dry-run

# Run for real
python scripts/migrate_opensearch_to_s3vectors.py \
    --opensearch-host localhost \
    --opensearch-port 9200 \
    --opensearch-password YOUR_PASSWORD \
    --index documents \
    --s3v-bucket openrag-vectors \
    --s3v-region us-east-1 \
    --s3v-role-arn arn:aws:iam::ACCOUNT:role/openrag-vectors
```

The script:
- Scrolls through all document chunks in the OpenSearch index
- Groups them by `owner` and `embedding_model`
- Writes each group to the appropriate per-user S3 Vectors index
- Drops ACL fields (`allowed_users`, `allowed_groups`) since access is now structural
- Preserves all other metadata

## Limitations

- **No keyword search**: The hybrid search (BM25 + KNN) is replaced with pure semantic search. For RAG workloads, this typically produces equivalent or better results.
- **No aggregations**: Faceted filtering (by filename, mimetype, owner, etc.) is not available from S3 Vectors. The search API returns empty aggregations.
- **Top-K capped at 100**: S3 Vectors limits query results to 100 per call. The current default is 10, so this is rarely an issue.
- **Langflow ingest not supported**: The Langflow OpenSearch component (`opensearch_multimodal.py`) is not ported. Ingestion must go through the OpenRAG backend's traditional processor (`DISABLE_INGEST_WITH_LANGFLOW=true`).
- **Auxiliary services**: API key storage, knowledge filters, and monitors still use OpenSearch if configured. These are non-vector CRUD operations and work independently of the vector backend.

## File inventory

### New files

| File | Purpose |
|---|---|
| `src/vectorstore/__init__.py` | Package exports |
| `src/vectorstore/base.py` | `VectorStore` protocol, `VectorRecord`, `VectorResult` data types |
| `src/vectorstore/s3vectors_store.py` | S3 Vectors implementation: STS scoping, per-user indexes, batching, search |
| `src/vectorstore/factory.py` | Creates the right `VectorStore` based on `VECTOR_BACKEND` env var |
| `src/services/sharing_service.py` | Document sharing via vector copy between user indexes |
| `docker-compose.s3vectors.yml` | Compose override: disables OpenSearch, configures S3 Vectors |
| `tests/unit/test_s3vectors_store.py` | Unit tests |
| `scripts/migrate_opensearch_to_s3vectors.py` | Data migration script |

### Modified files

| File | Change |
|---|---|
| `src/config/settings.py` | `AppClients.vector_store`, initialization and cleanup |
| `src/services/search_service.py` | Routes to `_search_s3vectors()` when active |
| `src/services/document_service.py` | Backend-aware existence check |
| `src/models/processors.py` | Split into `_process_document_s3vectors` / `_process_document_opensearch` |
| `src/api/documents.py` | Split delete-by-filename into S3 Vectors / OpenSearch paths |
| `.env.example` | New `VECTOR_BACKEND`, `S3_VECTORS_*`, `AWS_REGION` variables |
| `pyproject.toml` | `boto3>=1.38.0` for S3 Vectors client support |
