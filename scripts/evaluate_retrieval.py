"""
Retrieval evaluation for HireSense — resume query -> job corpus.

What this script checks
------------------------
This script answers one question: does the correct job land near the
top of the ranked list, when a resume is the query?

This is different from the Spearman check in train.py.
Spearman checks pair order. This script checks real retrieval rank.

Query direction
---------------
    query  = one resume
    corpus = all unique job descriptions in the split
    task   = rank every job in the corpus for that resume

Metrics
-------
    Recall@1 / Recall@3 / Recall@5
        Did at least one correct job land in the top k? (1 or 0 per query)
    MRR (Mean Reciprocal Rank)
        1 / rank of the first correct job. Higher is better.
    NDCG@5
        Rewards correct jobs found near the top more than near the bottom.

Expected input schema
----------------------
The pairs CSV must have these columns:
    jd_id        - unique id for each job description
    jd_text      - job description text
    resume_text  - resume text
    pair_type    - "positive", "negative_hard", or "negative_easy"
    label        - 1 for positive pairs, 0 for negative pairs

An optional "resume_id" column can be used to identify each resume.
If it is missing, this script builds a stable id from the resume text
itself, so the same resume is never counted twice.

    The required "resume_category" column enables a per-category breakdown.

Usage
-----
    python evaluate_retrieval.py --model models/resume-jd-sbert \\
        --pairs data/processed/val_pairs.csv \\
        --output results/val_retrieval_results.csv

    # Test against a held-out, never-before-seen job corpus:
    python evaluate_retrieval.py --model models/resume-jd-sbert \\
        --pairs data/processed/test_pairs.csv \\
        --unseen-corpus data/processed/unseen_jds.csv \\
        --output results/test_unseen_retrieval_results.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalQuery:
    """One resume, and the set of correct job ids for that resume."""

    resume_id: str
    resume_text: str
    category: str | None
    relevant_jd_ids: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_resume_id(resume_text: str) -> str:
    """Build a short, stable id from resume text when no id column exists."""
    return hashlib.sha1(resume_text.encode("utf-8")).hexdigest()[:12]


def load_corpus(df: pd.DataFrame, jd_split: str | None = None) -> dict[str, str]:
    """Return the authoritative jd_id -> jd_text corpus, optionally filtered."""
    if jd_split is not None:
        if "jd_split" not in df.columns:
            raise ValueError("Corpus is missing required jd_split column")
        df = df[df["jd_split"] == jd_split]
        if df.empty:
            raise ValueError(f"Corpus has no JDs for jd_split={jd_split!r}")
    if df["jd_id"].duplicated().any():
        duplicate_ids = df.loc[df["jd_id"].duplicated(),
                               "jd_id"].unique().tolist()
        raise ValueError(
            f"Corpus contains duplicate jd_id values: {duplicate_ids[:5]}")
    corpus = df.set_index("jd_id")["jd_text"].to_dict()
    logger.info("Corpus built: %d unique job descriptions.", len(corpus))
    return corpus


def load_queries(df: pd.DataFrame) -> list[RetrievalQuery]:
    """
    Build one query per unique resume, using only positive pairs as the
    ground-truth relevant jobs for that resume.
    """
    has_resume_id = "resume_id" in df.columns
    if "resume_category" not in df.columns:
        raise ValueError(
            "Pairs file is missing required resume_category column")

    positives = df[df["pair_type"] == "positive"].copy()
    if positives.empty:
        raise ValueError(
            "No rows with pair_type == 'positive' found. "
            "Cannot build ground truth without at least one positive per resume."
        )

    if not has_resume_id:
        positives["resume_id"] = positives["resume_text"].apply(make_resume_id)

    queries: dict[str, RetrievalQuery] = {}
    for _, row in positives.iterrows():
        rid = row["resume_id"]
        if rid not in queries:
            queries[rid] = RetrievalQuery(
                resume_id=rid,
                resume_text=str(row["resume_text"]),
                category=str(row["resume_category"]),
            )
        queries[rid].relevant_jd_ids.add(row["jd_id"])

    logger.info("Queries built: %d unique resumes.", len(queries))
    return list(queries.values())


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode a list of texts into normalized embeddings."""
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )


def rank_corpus_for_query(
    query_vec: np.ndarray, corpus_vecs: np.ndarray, corpus_ids: list[str]
) -> list[str]:
    """Return corpus_ids sorted from most similar to least similar."""
    scores = corpus_vecs @ query_vec
    order = np.lexsort((np.array(corpus_ids), -scores))
    return [corpus_ids[i] for i in order]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> int:
    """1 if any relevant job appears in the top k, else 0."""
    return int(any(jid in relevant for jid in ranked_ids[:k]))


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant job. 0 if none is found."""
    for i, jid in enumerate(ranked_ids, start=1):
        if jid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """
    NDCG@k with binary relevance.

    Each relevant job counts as 1. Jobs found near the top count more
    than jobs found near the bottom of the top-k window.
    """
    dcg = 0.0
    for i, jid in enumerate(ranked_ids[:k], start=1):
        if jid in relevant:
            dcg += 1.0 / np.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    pairs_csv: str,
    corpus_csv: str,
    jd_split: str | None,
    output_csv: str,
    device: str = "cpu",
    max_seq_length: int = 3072,
) -> None:
    df = pd.read_csv(pairs_csv)
    required = {
        "jd_id", "resume_id", "resume_text", "resume_category", "pair_type", "label"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pairs file is missing required columns: {missing}")
    if not set(df["label"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Pairs label must contain only 0 and 1")
    expected_pair_types = {"positive", "negative_hard", "negative_easy"}
    unknown_pair_types = set(df["pair_type"].dropna()) - expected_pair_types
    if unknown_pair_types:
        raise ValueError(
            f"Pairs contain unknown pair_type values: {sorted(unknown_pair_types)}")
    if not (
        ((df["pair_type"] == "positive") == (df["label"] == 1))
        & ((df["pair_type"] != "positive") == (df["label"] == 0))
    ).all():
        raise ValueError(
            "Pairs contain inconsistent pair_type and label values")

    queries = load_queries(df)
    corpus_df = pd.read_csv(corpus_csv)
    corpus_required = {"jd_id", "jd_text"}
    corpus_missing = corpus_required - set(corpus_df.columns)
    if corpus_missing:
        raise ValueError(
            f"Corpus is missing required columns: {sorted(corpus_missing)}")
    corpus = load_corpus(corpus_df, jd_split=jd_split)
    setting = f"jd-split:{jd_split}" if jd_split else "full-job-corpus"
    corpus_id_set = set(corpus.keys())
    pair_ids = set(df["jd_id"])
    if not pair_ids.issubset(corpus_id_set):
        missing_pair_ids = sorted(pair_ids - corpus_id_set)
        raise ValueError(
            f"Pairs reference JDs absent from the selected corpus: {missing_pair_ids[:10]}"
        )
    for q in queries:
        missing_gt = q.relevant_jd_ids - corpus_id_set
        if missing_gt:
            raise ValueError(
                f"Ground-truth JDs for resume {q.resume_id} are absent from the selected corpus: "
                f"{sorted(missing_gt)}"
            )

    logger.info("Evaluation setting: %s", setting)

    logger.info("Loading model: %s (device=%s)", model_path, device)
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    if hasattr(model, "max_seq_length") and max_seq_length:
        model.max_seq_length = max_seq_length
        logger.info("Set model.max_seq_length to %d", model.max_seq_length)

    corpus_ids = sorted(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    corpus_vecs = encode_texts(model, corpus_texts)

    queries = sorted(queries, key=lambda query: query.resume_id)
    query_texts = [q.resume_text for q in queries]

    # Measure per-query encoding latency (single document inference) and collect embeddings
    import time
    latencies: list[float] = []
    query_vec_list = []
    if query_texts:
        _ = model.encode([query_texts[0]], convert_to_numpy=True, normalize_embeddings=True)
        for qt in query_texts:
            t0 = time.perf_counter()
            vec = model.encode([qt], convert_to_numpy=True, normalize_embeddings=True)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            query_vec_list.append(vec[0])

    query_vecs = np.array(query_vec_list)

    rows = []
    for q, qvec in zip(queries, query_vecs):
        ranked_ids = rank_corpus_for_query(qvec, corpus_vecs, corpus_ids)
        top_5_ids = ranked_ids[:5]
        corpus_index = {jd_id: index for index, jd_id in enumerate(corpus_ids)}
        top_5_scores = [float(corpus_vecs[corpus_index[jd_id]] @ qvec)
                        for jd_id in top_5_ids]
        rows.append(
            {
                "resume_id": q.resume_id,
                "category": q.category,
                "num_relevant": len(q.relevant_jd_ids),
                "first_relevant_rank": next(
                    (rank for rank, jd_id in enumerate(
                        ranked_ids, start=1) if jd_id in q.relevant_jd_ids),
                    0,
                ),
                "top_5_jd_ids": "|".join(top_5_ids),
                "top_5_similarity_scores": "|".join(f"{score:.8f}" for score in top_5_scores),
                "recall_at_1": hit_at_k(ranked_ids, q.relevant_jd_ids, 1),
                "recall_at_3": hit_at_k(ranked_ids, q.relevant_jd_ids, 3),
                "recall_at_5": hit_at_k(ranked_ids, q.relevant_jd_ids, 5),
                "reciprocal_rank": reciprocal_rank(ranked_ids, q.relevant_jd_ids),
                "ndcg_at_5": ndcg_at_k(ranked_ids, q.relevant_jd_ids, 5),
            }
        )

    results_df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    logger.info("Per-query results saved to: %s", output_csv)

    summary = {
        "model": model_path,
        "pairs": pairs_csv,
        "corpus": corpus_csv,
        "jd_split": jd_split,
        "setting": setting,
        "num_queries": len(results_df),
        "corpus_size": len(corpus_ids),
        "Recall@1": results_df["recall_at_1"].mean(),
        "Recall@3": results_df["recall_at_3"].mean(),
        "Recall@5": results_df["recall_at_5"].mean(),
        "MRR": results_df["reciprocal_rank"].mean(),
        "NDCG@5": results_df["ndcg_at_5"].mean(),
        "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "per_category_recall_at_3": {
            str(category): float(value)
            for category, value in results_df.groupby("category")["recall_at_3"].mean().items()
        },
        "top_ranked_failures": results_df[
            results_df["recall_at_3"] == 0
        ].sort_values(["first_relevant_rank", "resume_id"]).head(10).to_dict("records"),
    }
    summary_path = Path(output_csv).with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Aggregate summary saved to: %s", summary_path)

    logger.info("=== Retrieval summary (%s) ===", setting)
    for key, val in summary.items():
        if isinstance(val, float):
            logger.info("  %-10s %.4f", key, val)
        else:
            logger.info("  %-10s %s", key, val)

    if "category" in results_df.columns and results_df["category"].notna().any():
        logger.info("=== Per-category Recall@3 ===")
        by_cat = results_df.groupby(
            "category")["recall_at_3"].mean().sort_values()
        for cat, val in by_cat.items():
            logger.info("  %-30s %.4f", cat, val)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-to-job retrieval evaluation (Recall@k, MRR, NDCG@5)."
    )
    parser.add_argument("--model", required=True,
                        help="Path or name of the SentenceTransformer model.")
    parser.add_argument("--pairs", required=True,
                        help="Path to the pairs CSV (val or test split).")
    parser.add_argument("--corpus", required=True,
                        help="Authoritative job-description corpus CSV.")
    parser.add_argument(
        "--jd-split",
        choices=("train_val", "test"),
        default=None,
        help="Optional JD split to evaluate, such as train_val or test.",
    )
    parser.add_argument(
        "--output",
        default="retrieval_results.csv",
        help="Where to save the per-query results CSV.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run evaluation on ('cpu' or 'cuda', default: cpu to avoid 4GB VRAM limit).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=3072,
        help="Maximum sequence length (default: 3072).",
    )
    args = parser.parse_args()

    try:
        evaluate(
            args.model,
            args.pairs,
            args.corpus,
            args.jd_split,
            args.output,
            device=args.device,
            max_seq_length=args.max_seq_length,
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
