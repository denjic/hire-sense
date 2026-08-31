"""Build labeled resume–JD pairs with an approximately 1:3 positive:negative ratio."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _sample_without_replacement(
    rng: np.random.Generator,
    population: np.ndarray,
    k: int,
) -> np.ndarray:
    if len(population) == 0:
        return np.array([], dtype=object)
    if k <= len(population):
        idx = rng.choice(len(population), size=k, replace=False)
        return population[idx]
    # Not enough unique items: take all, then fill with replacement
    extra = rng.choice(len(population), size=k - len(population), replace=True)
    return np.concatenate([population, population[extra]])


def build_pairs_for_split(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    rng: np.random.Generator,
    positives_per_resume: int,
    negative_ratio: int,
    hard_negative_fraction: float = 0.35,
) -> pd.DataFrame:
    if positives_per_resume < 1:
        raise ValueError("positives_per_resume must be >= 1")
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be >= 1")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be between 0 and 1")

    jobs_by_cat: dict[str, pd.DataFrame] = {
        cat: frame.reset_index(drop=True) for cat, frame in jobs.groupby("category")
    }
    all_cats = list(jobs_by_cat.keys())
    rows: list[dict] = []

    for _, resume in resumes.iterrows():
        category = str(resume["category"])
        same = jobs_by_cat.get(category)
        if same is None or same.empty:
            logger.warning("No JDs for category %s; skipping resume %s", category, resume["resume_id"])
            continue

        n_pos = positives_per_resume
        n_neg = positives_per_resume * negative_ratio
        pos_idx = _sample_without_replacement(rng, np.arange(len(same)), n_pos)
        pos_jd_ids = set()
        for loc in pos_idx:
            jd = same.iloc[int(loc)]
            pos_jd_ids.add(jd["jd_id"])
            rows.append(_pair_row(resume, jd, label=1, pair_type="positive"))

        # Hard negatives: same category, but not one of this resume's positive JDs.
        # These force the model to key off actual fit signals rather than category
        # keyword overlap, which is the one thing pure cross-category negatives can't teach.
        n_hard_target = int(round(n_neg * hard_negative_fraction))
        same_pool = same[~same["jd_id"].isin(pos_jd_ids)]
        n_hard = min(n_hard_target, len(same_pool)) if not same_pool.empty else 0
        if n_hard_target > 0 and n_hard < n_hard_target:
            logger.warning(
                "Only %d/%d hard-negative JDs available in category %s for resume %s; "
                "shortfall will be backfilled with easy negatives",
                n_hard, n_hard_target, category, resume["resume_id"],
            )
        hard_idx = _sample_without_replacement(rng, np.arange(len(same_pool)), n_hard) if n_hard else np.array([])
        for loc in hard_idx:
            jd = same_pool.iloc[int(loc)]
            rows.append(_pair_row(resume, jd, label=0, pair_type="negative_hard"))

        # Easy negatives: cross-category, fill the remainder of the negative budget.
        n_easy = n_neg - n_hard
        other_frames = [jobs_by_cat[c] for c in all_cats if c != category and not jobs_by_cat[c].empty]
        if not other_frames:
            logger.warning("No cross-category JDs available for resume %s", resume["resume_id"])
            continue
        others = pd.concat(other_frames, ignore_index=True)
        easy_idx = _sample_without_replacement(rng, np.arange(len(others)), n_easy) if n_easy else np.array([])
        for loc in easy_idx:
            jd = others.iloc[int(loc)]
            rows.append(_pair_row(resume, jd, label=0, pair_type="negative_easy"))

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise RuntimeError("Pair builder produced zero rows")
    return pairs


def _pair_row(resume: pd.Series, jd: pd.Series, label: int, pair_type: str) -> dict:
    return {
        "resume_id": resume["resume_id"],
        "resume_category": resume["category"],
        "resume_text": resume["resume_text"],
        "jd_id": jd["jd_id"],
        "jd_category": jd["category"],
        "jd_title": jd.get("title", ""),
        "jd_text": jd["jd_text"],
        "label": int(label),
        "pair_type": pair_type,
        "split": resume["split"],
    }


def validate_pairs(
    pairs: pd.DataFrame,
    expected_ratio: float,
    tolerance: float = 0.35,
    min_hard_negative_share: float = 0.15,
) -> None:
    """Fail loudly if labels or leakage look wrong. Ratio may flex slightly due to sampling."""
    pos = int((pairs["label"] == 1).sum())
    neg = int((pairs["label"] == 0).sum())
    if pos == 0:
        raise ValueError("No positive pairs")
    ratio = neg / pos
    logger.info("Pair counts: positives=%d negatives=%d neg/pos=%.2f (target %.2f)", pos, neg, ratio, expected_ratio)
    if abs(ratio - expected_ratio) > tolerance:
        raise ValueError(
            f"Negative:positive ratio {ratio:.2f} is outside {expected_ratio} ± {tolerance}"
        )

    bad_pos = pairs[(pairs["label"] == 1) & (pairs["resume_category"] != pairs["jd_category"])]
    if not bad_pos.empty:
        raise ValueError(f"Label/category mismatch: {len(bad_pos)} positives point at the wrong category")

    is_neg = pairs["label"] == 0
    hard_mask = is_neg & (pairs["pair_type"] == "negative_hard")
    easy_mask = is_neg & (pairs["pair_type"] == "negative_easy")
    legacy_mask = is_neg & ~hard_mask & ~easy_mask
    if int(legacy_mask.sum()):
        raise ValueError(
            f"{int(legacy_mask.sum())} negative rows have an unrecognized pair_type "
            "(expected 'negative_hard' or 'negative_easy')"
        )

    bad_hard = pairs[hard_mask & (pairs["resume_category"] != pairs["jd_category"])]
    bad_easy = pairs[easy_mask & (pairs["resume_category"] == pairs["jd_category"])]
    if not bad_hard.empty or not bad_easy.empty:
        raise ValueError(
            f"Label/category mismatch: {len(bad_hard)} hard negatives are cross-category, "
            f"{len(bad_easy)} easy negatives are same-category"
        )

    hard_share = int(hard_mask.sum()) / neg if neg else 0.0
    logger.info("Hard-negative share: %.2f (%d/%d negatives)", hard_share, int(hard_mask.sum()), neg)
    if hard_share < min_hard_negative_share:
        raise ValueError(
            f"Hard-negative share {hard_share:.2f} is below the minimum {min_hard_negative_share:.2f}. "
            "Check whether categories have enough JDs to support hard negatives without reusing positives."
        )


def pair_summary(pairs: pd.DataFrame) -> dict:
    pos = int((pairs["label"] == 1).sum())
    neg = int((pairs["label"] == 0).sum())
    hard_neg = int((pairs["pair_type"] == "negative_hard").sum())
    easy_neg = int((pairs["pair_type"] == "negative_easy").sum())
    return {
        "rows": int(len(pairs)),
        "unique_resumes": int(pairs["resume_id"].nunique()),
        "unique_jds": int(pairs["jd_id"].nunique()),
        "positives": pos,
        "negatives": neg,
        "negatives_hard": hard_neg,
        "negatives_easy": easy_neg,
        "hard_negative_share": round(hard_neg / neg, 3) if neg else None,
        "neg_to_pos_ratio": round(neg / pos, 3) if pos else None,
        "categories": int(pairs["resume_category"].nunique()),
    }