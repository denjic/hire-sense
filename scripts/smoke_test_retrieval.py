"""
scripts/smoke_test_retrieval.py — smoke test.

Picks one resume per category from data/processed/resumes.csv,
calls match_resume(), and prints the top-3 JD matches.

Done when: the top-3 JD categories are visually plausible for the resume's category.

Usage
-----
    python scripts/smoke_test_retrieval.py
    python scripts/smoke_test_retrieval.py --resume-idx 5   # use the 6th resume
    python scripts/smoke_test_retrieval.py --top-k 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import match_resume  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_smoke_test(
    resumes_csv: str,
    resume_idx: int,
    top_k: int,
    model_path: str,
    storage_path: str | None = None,
    qdrant_url: str | None = None,
) -> None:
    resumes_file = Path(resumes_csv)
    if not resumes_file.exists():
        raise FileNotFoundError(f"Resumes file not found: {resumes_file}")

    df = pd.read_csv(resumes_csv, encoding="utf-8", encoding_errors="replace")
    required = {"category", "resume_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Resumes file is missing required columns: {sorted(missing)}")

    if resume_idx >= len(df):
        raise ValueError(
            f"--resume-idx {resume_idx} is out of range (file has {len(df)} rows)."
        )

    row = df.iloc[resume_idx]
    resume_category = row["category"]
    resume_text = str(row["resume_text"])

    print(f"\n{'='*60}")
    print(f"Resume index   : {resume_idx}")
    print(f"Resume category: {resume_category}")
    print(f"Resume snippet : {resume_text[:200].strip()}...")
    print(f"{'='*60}\n")

    logger.info("Calling match_resume(top_k=%d)...", top_k)
    matches = match_resume(
        resume_text=resume_text,
        top_k=top_k,
        model_path=model_path,
        qdrant_path=storage_path,
        qdrant_url=qdrant_url,
    )

    print(f"\nTop-{top_k} matches for a '{resume_category}' resume:\n")
    for rank, match in enumerate(matches, start=1):
        print(f"  Rank {rank}")
        print(f"    jd_id     : {match['jd_id']}")
        print(f"    category  : {match['category']}")
        print(f"    score     : {match['score']:.4f}")
        print(f"    JD snippet: {match['jd_text'][:120].strip()}...")
        print()

    correct = sum(1 for m in matches if m["category"] == resume_category)
    print(f"Category match: {correct}/{top_k} top results match resume category '{resume_category}'")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for Phase 3 match_resume()."
    )
    parser.add_argument(
        "--resumes",
        default="data/processed/resumes.csv",
        help="Path to resumes.csv.",
    )
    parser.add_argument(
        "--resume-idx",
        type=int,
        default=0,
        help="Row index of the resume to test (default 0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top JDs to retrieve (default 3).",
    )
    parser.add_argument(
        "--model",
        default="models/resume-jd-jina" if Path("models/resume-jd-jina").exists() else "jinaai/jina-embeddings-v2-base-en",
        help="Path to the SentenceTransformer model.",
    )
    parser.add_argument(
        "--storage-path",
        default="data/qdrant_storage_jina",
        help="Path to local Qdrant disk storage directory.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Optional URL of the Qdrant instance (overrides --storage-path).",
    )
    args = parser.parse_args()

    try:
        run_smoke_test(
            resumes_csv=args.resumes,
            resume_idx=args.resume_idx,
            top_k=args.top_k,
            model_path=args.model,
            storage_path=args.storage_path,
            qdrant_url=args.qdrant_url,
        )
    except Exception as exc:
        logger.error("Smoke test failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

