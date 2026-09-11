"""Validate the retrieval evaluator's corpus and pair-file contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_RESUMES = {"val": 25, "test": 26}
EXPECTED_CORPORA = {"train_val": 150, "test": 75}


def require_columns(frame: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def validate(args: argparse.Namespace) -> None:
    corpus = pd.read_csv(args.corpus)
    require_columns(corpus, args.corpus, {
                    "jd_id", "jd_text", "category", "jd_split"})
    if corpus["jd_id"].duplicated().any():
        raise ValueError("Corpus contains duplicate jd_id values")

    corpus_ids = {
        split: set(corpus.loc[corpus["jd_split"] == split, "jd_id"])
        for split in EXPECTED_CORPORA
    }
    for split, expected_count in EXPECTED_CORPORA.items():
        actual_count = len(corpus_ids[split])
        if actual_count != expected_count:
            raise ValueError(
                f"{split} corpus has {actual_count} unique JDs; expected {expected_count}"
            )

    overlap = corpus_ids["train_val"] & corpus_ids["test"]
    if overlap:
        raise ValueError(f"Test JDs overlap train_val: {sorted(overlap)[:10]}")

    pair_paths = {"val": args.val_pairs, "test": args.test_pairs}
    for split, path in pair_paths.items():
        pairs = pd.read_csv(path)
        require_columns(
            pairs,
            path,
            {
                "resume_id",
                "resume_text",
                "resume_category",
                "jd_id",
                "label",
                "pair_type",
            },
        )
        resume_count = pairs["resume_id"].nunique()
        if resume_count != EXPECTED_RESUMES[split]:
            raise ValueError(
                f"{split} pairs contain {resume_count} unique resumes; "
                f"expected {EXPECTED_RESUMES[split]}"
            )
        selected_ids = corpus_ids["train_val" if split == "val" else "test"]
        positive_ids = set(pairs.loc[pairs["label"] == 1, "jd_id"])
        missing = positive_ids - selected_ids
        if missing:
            raise ValueError(
                f"{split} positives absent from selected corpus: {sorted(missing)[:10]}"
            )
        if pairs["resume_category"].isna().any():
            raise ValueError(
                f"{split} pairs contain missing resume_category values")

    print("Retrieval data contract passed")
    print(f"Validation resumes: {EXPECTED_RESUMES['val']}")
    print(f"Test resumes: {EXPECTED_RESUMES['test']}")
    print(f"train_val corpus: {len(corpus_ids['train_val'])}")
    print(f"test corpus: {len(corpus_ids['test'])}")
    print("Test/train_val JD overlap: 0")
    print("Positive JD coverage: passed")
    print("Category source: resume_category")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=Path("data/processed/job_descriptions.csv"))
    parser.add_argument("--val-pairs", type=Path,
                        default=Path("data/processed/val_pairs.csv"))
    parser.add_argument("--test-pairs", type=Path,
                        default=Path("data/processed/test_pairs.csv"))
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
