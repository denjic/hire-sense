"""Resume-level 80/10/10 split with no resume ID leaking across splits."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def _counts_for_category(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    """Allocate at least one resume to train; give val/test a slot when n allows."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    if n == 3:
        return 1, 1, 1

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train + n_val + n_test > n:
            n_val = max(1, n_val - 1)
        if n_train + n_val + n_test > n:
            n_train = n - n_val - n_test
    if n_val < 1:
        n_val = 1
        n_train = n - n_val - n_test
    n_train = max(1, n_train)
    # Fix rounding overflow
    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    while n_train + n_val + n_test < n:
        n_train += 1
    return n_train, n_val, n_test


def assign_resume_splits(
    resumes: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify: bool,
) -> pd.DataFrame:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    assigned: dict[str, str] = {}

    if stratify:
        groups = list(resumes.groupby("category", sort=True))
    else:
        groups = [("all", resumes)]

    for _, group in groups:
        ids = group["resume_id"].drop_duplicates().tolist()
        rng.shuffle(ids)
        n_train, n_val, n_test = _counts_for_category(len(ids), train_ratio, val_ratio)
        split_ids = (
            [("train", ids[:n_train])]
            + [("val", ids[n_train : n_train + n_val])]
            + [("test", ids[n_train + n_val : n_train + n_val + n_test])]
        )
        for split_name, chunk in split_ids:
            for resume_id in chunk:
                assigned[resume_id] = split_name

    out = resumes.copy()
    out["split"] = out["resume_id"].map(assigned)
    missing = out["split"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} resumes were not assigned a split")

    # Hard guarantee: no resume_id in more than one split
    check = out.groupby("resume_id")["split"].nunique()
    leaked = check[check > 1]
    if not leaked.empty:
        raise RuntimeError(f"Resume IDs leaked across splits: {leaked.index.tolist()[:10]}")

    counts = out.drop_duplicates("resume_id")["split"].value_counts().to_dict()
    logger.info("Resume split counts: %s", {k: counts.get(k, 0) for k in SPLITS})
    return out


def split_manifest(resumes_with_split: pd.DataFrame) -> dict:
    by_split: dict[str, list[str]] = defaultdict(list)
    unique = resumes_with_split[["resume_id", "category", "split"]].drop_duplicates()
    for _, row in unique.iterrows():
        by_split[str(row["split"])].append(str(row["resume_id"]))
    category_split = (
        unique.groupby(["category", "split"]).size().unstack(fill_value=0).to_dict()
    )
    return {
        "resume_counts": {split: len(by_split[split]) for split in SPLITS},
        "resume_ids": {split: sorted(by_split[split]) for split in SPLITS},
        "category_by_split": category_split,
    }
