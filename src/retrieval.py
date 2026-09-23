"""
src/retrieval.py — retrieval module.

Public API
----------
    match_resume(resume_text, top_k, model_path, qdrant_url) -> list[dict]

Each returned dict has:
    jd_id    : str   — unique job description identifier
    category : str   — job category label
    jd_text  : str   — full job description text
    score    : float — cosine similarity in [0, 1]

The model and Qdrant client are lazy-loaded on first call and cached at
module level.  Importing this module has no side effects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level caches — populated on first call to match_resume().
_model: Any = None
_model_path_loaded: str | None = None

_DEFAULT_MODEL_PATH = "models/resume-jd-jina" if Path("models/resume-jd-jina").exists() else "jinaai/jina-embeddings-v2-base-en"
_DEFAULT_QDRANT_PATH = "data/qdrant_storage_jina"
_COLLECTION_NAME = "job_descriptions"


def _get_model(model_path: str):
    """Return a cached SentenceTransformer, loading it if necessary."""
    global _model, _model_path_loaded
    if _model is None or _model_path_loaded != model_path:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        logger.info("Loading model: %s", model_path)
        _model = SentenceTransformer(model_path, trust_remote_code=True)
        _model_path_loaded = model_path
        logger.info("Model loaded.")
    return _model


def match_resume(
    resume_text: str,
    top_k: int = 3,
    model_path: str = _DEFAULT_MODEL_PATH,
    qdrant_path: str | None = _DEFAULT_QDRANT_PATH,
    qdrant_url: str | None = None,
) -> list[dict]:
    """
    Embed *resume_text* and return the top-k most similar job descriptions
    from the Qdrant collection.

    Parameters
    ----------
    resume_text : str
        Raw text of the resume to match.
    top_k : int
        Number of job descriptions to return (default 3).
    model_path : str
        Path to the SentenceTransformer model directory.
    qdrant_path : str | None
        Path to local Qdrant disk storage directory (used if qdrant_url is None).
    qdrant_url : str | None
        URL of a running Qdrant instance (e.g. http://localhost:6333).

    Returns
    -------
    list[dict]
        Each dict: {"jd_id": str, "category": str, "jd_text": str, "score": float}
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text must be a non-empty string.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    from qdrant_client import QdrantClient  # noqa: PLC0415

    model = _get_model(model_path)

    # Encode the resume into a normalized embedding vector.
    vector = model.encode(
        resume_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    if qdrant_url:
        client = QdrantClient(url=qdrant_url)
    else:
        client = QdrantClient(path=qdrant_path or _DEFAULT_QDRANT_PATH)

    try:
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=_COLLECTION_NAME,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = client.search(
                collection_name=_COLLECTION_NAME,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "jd_id": payload.get("jd_id", ""),
                    "category": payload.get("category", ""),
                    "jd_text": payload.get("jd_text", ""),
                    "score": round(float(hit.score), 6),
                }
            )

        return results
    finally:
        client.close()

