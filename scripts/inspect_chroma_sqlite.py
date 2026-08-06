#!/usr/bin/env python3
"""Read-only inspection helper for a persistent Chroma SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    parser.add_argument("collections", nargs="+")
    parser.add_argument("--no-samples", action="store_true")
    args = parser.parse_args()

    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    print("TABLES", json.dumps(sorted(tables)))
    for table in ("collections", "segments", "embeddings", "embedding_metadata"):
        if table in tables:
            columns = [(row[1], row[2]) for row in connection.execute(f'PRAGMA table_info("{table}")')]
            print("COLUMNS", table, json.dumps(columns))

    for name in args.collections:
        collection = connection.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
        print("COLLECTION", name, json.dumps(dict(collection) if collection else None, default=str))
        if not collection:
            continue
        segments = list(connection.execute("SELECT * FROM segments WHERE collection = ?", (collection["id"],)))
        print("SEGMENTS", name, json.dumps([dict(row) for row in segments], default=str))
        segment_ids = [row["id"] for row in segments]
        for segment_id in segment_ids:
            count = connection.execute("SELECT count(*) FROM embeddings WHERE segment_id = ?", (segment_id,)).fetchone()[0]
            print("EMBEDDING_COUNT", name, segment_id, count)
            if not count:
                continue
            doc_id_summary = connection.execute(
                "SELECT count(*), count(DISTINCT m.string_value) "
                "FROM embeddings e JOIN embedding_metadata m ON m.id = e.id "
                "WHERE e.segment_id = ? AND m.key = 'doc_id'",
                (segment_id,),
            ).fetchone()
            print("DOC_ID_SUMMARY", name, segment_id, doc_id_summary[0], doc_id_summary[1])
            if args.no_samples:
                continue
            sample = list(connection.execute("SELECT id, embedding_id FROM embeddings WHERE segment_id = ? LIMIT 5", (segment_id,)))
            print("EMBEDDING_SAMPLE", name, segment_id, json.dumps([dict(row) for row in sample]))
            if "embedding_metadata" in tables:
                for row in sample:
                    metadata = list(
                        connection.execute(
                            "SELECT key, string_value, int_value, float_value, bool_value "
                            "FROM embedding_metadata WHERE id = ? ORDER BY key",
                            (row["id"],),
                        )
                    )
                    print("METADATA_SAMPLE", row["embedding_id"], json.dumps([dict(item) for item in metadata], ensure_ascii=False))


if __name__ == "__main__":
    main()
