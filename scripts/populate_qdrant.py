"""
scripts/populate_qdrant.py — Phase 3: embed job descriptions and store in Qdrant.

What this does
--------------
1. Reads data/processed/job_descriptions.csv
2. Encodes every jd_text with the fine-tuned SBERT model
3. Upserts all vectors + payload into a Qdrant collection named "job_descriptions"

The collection is (re)created each run so the script is safe to re-run.

Usage
-----
    python scripts/populate_qdrant.py
    python scripts/populate_qdrant.py --model models/resume-jd-sbert-v2
    python scripts/populate_qdrant.py --corpus data/processed/job_descriptions.csv
    python scripts/populate_qdrant.py --qdrant-url http://localhost:6333
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_COLLECTION_NAME = "job_descriptions"
_BATCH_SIZE = 64


def populate(model_path: str, corpus_path: str, storage_path: str | None = None, qdrant_url: str | None = None) -> None:
    # --- Load corpus ---
    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_file}")

    df = pd.read_csv(corpus_path)
    required = {"jd_id", "jd_text", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")

    if df["jd_id"].duplicated().any():
        dups = df.loc[df["jd_id"].duplicated(), "jd_id"].unique().tolist()
        raise ValueError(f"Corpus has duplicate jd_id values: {dups[:5]}")

    logger.info("Corpus loaded: %d job descriptions from %s", len(df), corpus_path)

    # --- Load model ---
    logger.info("Loading model: %s", model_path)
    model = SentenceTransformer(model_path, trust_remote_code=True)

    # --- Encode ---
    logger.info("Encoding %d job descriptions (batch_size=%d)...", len(df), _BATCH_SIZE)
    jd_texts = df["jd_text"].tolist()
    vectors = model.encode(
        jd_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=_BATCH_SIZE,
    )
    vector_dim = vectors.shape[1]
    logger.info("Encoding complete. Vector dimension: %d", vector_dim)

    # --- Connect to Qdrant ---
    if qdrant_url:
        logger.info("Connecting to Qdrant server at %s", qdrant_url)
        client = QdrantClient(url=qdrant_url)
    else:
        path = storage_path or "data/qdrant_storage"
        logger.info("Using local Qdrant disk storage at %s", path)
        client = QdrantClient(path=path)

    try:
        # Recreate the collection each run (idempotent for dev).
        existing = {c.name for c in client.get_collections().collections}
        if _COLLECTION_NAME in existing:
            logger.info("Deleting existing collection '%s'...", _COLLECTION_NAME)
            client.delete_collection(_COLLECTION_NAME)

        logger.info("Creating collection '%s' (dim=%d, cosine)...", _COLLECTION_NAME, vector_dim)
        client.create_collection(
            collection_name=_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

        # --- Upsert in batches ---
        jd_ids = df["jd_id"].tolist()
        categories = df["category"].tolist()
        titles = df["title"].tolist() if "title" in df.columns else [""] * len(df)

        # Qdrant point IDs must be integers or UUIDs; map jd_id strings to sequential ints.
        total = 0
        for start in range(0, len(df), _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, len(df))
            points = [
                PointStruct(
                    id=start + i,
                    vector=vectors[start + i].tolist(),
                    payload={
                        "jd_id": jd_ids[start + i],
                        "category": categories[start + i],
                        "title": titles[start + i],
                        "jd_text": jd_texts[start + i],
                    },
                )
                for i in range(end - start)
            ]
            client.upsert(collection_name=_COLLECTION_NAME, points=points)
            total += len(points)
            logger.info("Upserted %d / %d points", total, len(df))

        logger.info("Done. Collection '%s' contains %d points.", _COLLECTION_NAME, total)
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed job descriptions and load them into Qdrant."
    )
    parser.add_argument(
        "--model",
        default="jinaai/jina-embeddings-v2-base-en",
        help="Path or name of the SentenceTransformer model.",
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/job_descriptions.csv",
        help="Path to job_descriptions.csv.",
    )
    parser.add_argument(
        "--storage-path",
        default="data/qdrant_storage_jina",
        help="Path to local Qdrant disk storage directory.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Optional URL of a running Qdrant instance (overrides --storage-path).",
    )
    args = parser.parse_args()

    try:
        populate(args.model, args.corpus, args.storage_path, args.qdrant_url)
    except Exception as exc:
        logger.error("Populate failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
