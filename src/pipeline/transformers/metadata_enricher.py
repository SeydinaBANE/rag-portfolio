from datetime import datetime, timezone


def enrich(
    base_metadata: dict,
    source_uri: str,
    doc_type: str,
    chunk_index: int,
    total_chunks: int,
) -> dict:
    return {
        **base_metadata,
        "source_uri": source_uri,
        "doc_type": doc_type,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
