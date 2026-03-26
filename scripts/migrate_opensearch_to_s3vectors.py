#!/usr/bin/env python3
"""
Migrate vector data from OpenSearch to Amazon S3 Vectors.

Reads all documents from the OpenSearch index, groups them by owner and
embedding model, and writes them to per-user S3 Vectors indexes.

Usage:
    python scripts/migrate_opensearch_to_s3vectors.py \
        --opensearch-host localhost \
        --opensearch-port 9200 \
        --opensearch-password <PASSWORD> \
        --index documents \
        --s3v-bucket openrag-vectors \
        --s3v-region us-east-1 \
        --s3v-role-arn arn:aws:iam::123456:role/openrag-vectors \
        [--batch-size 100] \
        [--dry-run]
"""

import argparse
import asyncio
import sys
import os

# Add src/ to path so we can import from the project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def main():
    parser = argparse.ArgumentParser(
        description="Migrate OpenSearch vectors to S3 Vectors"
    )
    parser.add_argument("--opensearch-host", default="localhost")
    parser.add_argument("--opensearch-port", type=int, default=9200)
    parser.add_argument("--opensearch-username", default="admin")
    parser.add_argument("--opensearch-password", required=True)
    parser.add_argument("--index", default="documents")
    parser.add_argument("--s3v-bucket", required=True)
    parser.add_argument("--s3v-region", default="us-east-1")
    parser.add_argument("--s3v-role-arn", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from opensearchpy import AsyncOpenSearch
    from opensearchpy._async.http_aiohttp import AIOHttpConnection
    from vectorstore.s3vectors_store import S3VectorsStore
    from vectorstore.base import VectorRecord
    from utils.embedding_fields import get_embedding_field_name

    # Connect to OpenSearch
    os_client = AsyncOpenSearch(
        hosts=[{"host": args.opensearch_host, "port": args.opensearch_port}],
        connection_class=AIOHttpConnection,
        scheme="https",
        use_ssl=True,
        verify_certs=False,
        http_auth=(args.opensearch_username, args.opensearch_password),
    )

    # Initialize S3 Vectors store
    s3v_store = S3VectorsStore(
        bucket_name=args.s3v_bucket,
        region=args.s3v_region,
        role_arn=args.s3v_role_arn,
    )

    if not args.dry_run:
        await s3v_store.initialize()

    # Scroll through all documents in the OpenSearch index
    print(f"Scanning OpenSearch index '{args.index}'...")

    scroll_body = {
        "query": {"match_all": {}},
        "size": args.batch_size,
    }

    try:
        response = await os_client.search(
            index=args.index,
            body=scroll_body,
            scroll="5m",
        )
    except Exception as e:
        print(f"Error connecting to OpenSearch: {e}")
        return

    scroll_id = response.get("_scroll_id")
    total = response["hits"]["total"]
    total_count = total["value"] if isinstance(total, dict) else total
    print(f"Found {total_count} document chunks to migrate")

    migrated = 0
    skipped = 0
    errors = 0

    while True:
        hits = response["hits"]["hits"]
        if not hits:
            break

        # Group chunks by (owner, embedding_model)
        groups = {}
        for hit in hits:
            source = hit["_source"]
            owner = source.get("owner", "anonymous")
            model = source.get("embedding_model")

            if not model:
                skipped += 1
                continue

            # Find the embedding vector in the source
            field_name = get_embedding_field_name(model)
            vector_data = source.get(field_name)

            if not vector_data:
                # Try legacy field
                vector_data = source.get("chunk_embedding")

            if not vector_data:
                print(f"  Skipping {hit['_id']}: no embedding found")
                skipped += 1
                continue

            key = (owner, model)
            if key not in groups:
                groups[key] = []

            # Build metadata (exclude the embedding vector itself)
            metadata = {}
            for k, v in source.items():
                if k.startswith("chunk_embedding"):
                    continue
                if k in ("allowed_users", "allowed_groups", "user_permissions", "group_permissions"):
                    continue  # ACL fields not needed in per-user indexes
                if v is not None:
                    metadata[k] = v

            groups[key].append(
                VectorRecord(
                    key=hit["_id"],
                    data=vector_data,
                    metadata=metadata,
                )
            )

        # Write each group to S3 Vectors
        for (owner, model), vectors in groups.items():
            if args.dry_run:
                print(f"  [DRY RUN] Would write {len(vectors)} vectors for owner={owner}, model={model}")
                migrated += len(vectors)
            else:
                try:
                    await s3v_store.put_vectors(vectors, owner, model)
                    migrated += len(vectors)
                    print(f"  Wrote {len(vectors)} vectors for owner={owner}, model={model}")
                except Exception as e:
                    errors += len(vectors)
                    print(f"  ERROR writing vectors for owner={owner}, model={model}: {e}")

        # Continue scrolling
        response = await os_client.scroll(scroll_id=scroll_id, scroll="5m")

    # Cleanup
    try:
        await os_client.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass
    await os_client.close()

    if not args.dry_run:
        await s3v_store.cleanup()

    print(f"\nMigration complete: {migrated} migrated, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    asyncio.run(main())
