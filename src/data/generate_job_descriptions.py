"""Materialize curated job descriptions to CSV."""

from __future__ import annotations

import logging
import pandas as pd

from src.data.io_utils import ensure_dir, resolve_path
from src.data.jd_templates import iter_job_descriptions

logger = logging.getLogger(__name__)


def build_job_descriptions_frame() -> pd.DataFrame:
    df = pd.DataFrame(iter_job_descriptions())
    if df["jd_id"].duplicated().any():
        raise ValueError("Duplicate jd_id values in JD templates")
    logger.info(
        "Generated %d job descriptions across %d categories",
        len(df),
        df["category"].nunique(),
    )
    return df[["jd_id", "category", "title", "jd_text"]]


def write_job_descriptions(config: dict) -> pd.DataFrame:
    df = build_job_descriptions_frame()
    min_chars = int(config["pairs"]["min_jd_chars"])
    too_short = df[df["jd_text"].str.len() < min_chars]
    if not too_short.empty:
        raise ValueError(
            f"{len(too_short)} job descriptions are shorter than {min_chars} characters"
        )
    out_path = resolve_path(config["paths"]["job_descriptions_csv"])
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    logger.info("Wrote job descriptions to %s", out_path)
    return df


def load_job_descriptions(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["job_descriptions_csv"])
    if path.exists():
        df = pd.read_csv(path)
        logger.info("Loaded job descriptions from %s (%d rows)", path, len(df))
        return df
    return write_job_descriptions(config)
