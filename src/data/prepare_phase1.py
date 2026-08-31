"""Phase 1 orchestrator: resumes + JDs + labeled pairs + 80/10/10 split."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.data.build_pairs import build_pairs_for_split, pair_summary, validate_pairs
from src.data.download_resumes import ResumeDatasetError, acquire_resume_csv, load_resume_dataframe
from src.data.generate_job_descriptions import write_job_descriptions
from src.data.io_utils import load_config, resolve_path, write_json
from src.data.split_dataset import assign_resume_splits, split_manifest
from src.data.synthetic_resumes import generate_synthetic_resumes

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_or_synthesize_resumes(config: dict) -> tuple[pd.DataFrame, str]:
    min_chars = int(config["pairs"]["min_resume_chars"])
    try:
        csv_path = acquire_resume_csv(config)
        resumes = load_resume_dataframe(csv_path, min_chars=min_chars)
        return resumes, f"kaggle_or_local:{csv_path}"
    except ResumeDatasetError as exc:
        logger.warning("Real resume dataset unavailable (%s). Falling back to synthetic resumes.", exc)
        n = int(config["synthetic_fallback"]["resumes_per_category"])
        resumes = generate_synthetic_resumes(n)
        short = resumes["resume_text"].str.len() < min_chars
        if short.any():
            raise RuntimeError(f"{int(short.sum())} synthetic resumes are shorter than {min_chars} chars")
        return resumes, "synthetic_fallback"


def assert_no_resume_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    sets = {
        "train": set(train["resume_id"].astype(str)),
        "val": set(val["resume_id"].astype(str)),
        "test": set(test["resume_id"].astype(str)),
    }
    leaks = {
        "train∩val": sorted(sets["train"] & sets["val"]),
        "train∩test": sorted(sets["train"] & sets["test"]),
        "val∩test": sorted(sets["val"] & sets["test"]),
    }
    leaked = {k: v for k, v in leaks.items() if v}
    if leaked:
        raise RuntimeError(f"Resume IDs leaked across splits: { {k: v[:5] for k, v in leaked.items()} }")
    logger.info(
        "No resume leakage. Unique resumes: train=%d val=%d test=%d",
        len(sets["train"]),
        len(sets["val"]),
        len(sets["test"]),
    )


def write_pairs_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [
        "resume_id",
        "jd_id",
        "label",
        "pair_type",
        "split",
        "resume_category",
        "jd_category",
        "jd_title",
        "resume_text",
        "jd_text",
    ]
    df[ordered].to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(df))


def run(config_path: str | None, force_synthetic: bool) -> dict:
    load_dotenv()
    config = load_config(config_path)
    seed = int(config["seed"])
    ratio = float(config["pairs"]["negative_to_positive_ratio"])

    jobs = write_job_descriptions(config)
    if force_synthetic:
        n = int(config["synthetic_fallback"]["resumes_per_category"])
        resumes = generate_synthetic_resumes(n)
        source = "synthetic_forced"
    else:
        resumes, source = load_or_synthesize_resumes(config)

    split_cfg = config["split"]
    resumes = assign_resume_splits(
        resumes,
        seed=seed,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        stratify=bool(split_cfg["stratify_by_category"]),
    )

    processed_resumes_path = resolve_path(config["paths"]["cleaned_resumes_csv"])
    processed_resumes_path.parent.mkdir(parents=True, exist_ok=True)
    resumes.to_csv(processed_resumes_path, index=False)
    jobs.to_csv(resolve_path(config["paths"]["cleaned_jds_csv"]), index=False)

    rng = np.random.default_rng(seed)
    pair_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        subset = resumes[resumes["split"] == split_name]
        if subset.empty:
            raise RuntimeError(f"Split '{split_name}' has zero resumes")
        pairs = build_pairs_for_split(
            subset,
            jobs,
            rng=rng,
            positives_per_resume=int(config["pairs"]["positives_per_resume"]),
            negative_ratio=int(config["pairs"]["negative_to_positive_ratio"]),
            hard_negative_fraction=float(config["pairs"].get("hard_negative_fraction", 0.35)),
        )
        validate_pairs(pairs, expected_ratio=ratio)
        pair_frames[split_name] = pairs

    train, val, test = pair_frames["train"], pair_frames["val"], pair_frames["test"]
    assert_no_resume_leakage(train, val, test)

    train_path = resolve_path(config["paths"]["train_pairs_csv"])
    val_path = resolve_path(config["paths"]["val_pairs_csv"])
    test_path = resolve_path(config["paths"]["test_pairs_csv"])
    write_pairs_csv(train, train_path)
    write_pairs_csv(val, val_path)
    write_pairs_csv(test, test_path)

    manifest = split_manifest(resumes)
    write_json(resolve_path(config["paths"]["split_manifest"]), manifest)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_source": source,
        "seed": seed,
        "job_descriptions": int(len(jobs)),
        "resumes_total": int(resumes["resume_id"].nunique()),
        "resume_split_counts": manifest["resume_counts"],
        "pairs": {
            "train": pair_summary(train),
            "val": pair_summary(val),
            "test": pair_summary(test),
        },
        "outputs": {
            "train_pairs": str(train_path),
            "val_pairs": str(val_path),
            "test_pairs": str(test_path),
        },
    }
    write_json(resolve_path(config["paths"]["phase1_summary"]), summary)
    logger.info("Phase 1 complete.")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: build train/val/test resume–JD pair CSVs."
    )
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to data.yaml (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Skip Kaggle and generate synthetic resumes (useful for a smoke test).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)
    run(config_path=args.config, force_synthetic=args.force_synthetic)


if __name__ == "__main__":
    main()