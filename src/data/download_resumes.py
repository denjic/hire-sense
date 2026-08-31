"""Download or locate the Kaggle Resume Dataset."""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.data.constants import canonicalize_category
from src.data.io_utils import ensure_dir, resolve_path

logger = logging.getLogger(__name__)

RESUME_COLUMN_ALIASES = ("Resume", "Resume_str", "resume", "resume_text", "Resume Text")
CATEGORY_COLUMN_ALIASES = ("Category", "category", "label", "Job Category")


class ResumeDatasetError(RuntimeError):
    """Raised when resume data cannot be obtained or parsed."""


def _find_existing_csv(candidates: list[str]) -> Path | None:
    for relative in candidates:
        path = resolve_path(relative)
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _try_kaggle_download(raw_dir: Path, dataset: str) -> None:
    load_dotenv()
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    key = os.getenv("KAGGLE_KEY", "").strip()
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists() and not (username and key):
        raise ResumeDatasetError(
            "Kaggle credentials not found. Put kaggle.json in ~/.kaggle/ "
            "or set KAGGLE_USERNAME and KAGGLE_KEY in .env."
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ResumeDatasetError("The 'kaggle' package is not installed.") from exc

    api = KaggleApi()
    api.authenticate()
    ensure_dir(raw_dir)
    logger.info("Downloading Kaggle dataset %s into %s", dataset, raw_dir)
    api.dataset_download_files(dataset, path=str(raw_dir), unzip=True)

    # Some Kaggle downloads leave a zip even when unzip=True fails on Windows
    for zip_path in raw_dir.glob("*.zip"):
        logger.info("Unzipping leftover archive %s", zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(raw_dir)


def _detect_column(df: pd.DataFrame, aliases: tuple[str, ...], kind: str) -> str:
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    raise ResumeDatasetError(
        f"Could not find a {kind} column. Looked for {aliases}. "
        f"Found columns: {list(df.columns)}"
    )


def load_resume_dataframe(csv_path: Path, min_chars: int) -> pd.DataFrame:
    logger.info("Loading resumes from %s", csv_path)
    df = pd.read_csv(csv_path)
    resume_col = _detect_column(df, RESUME_COLUMN_ALIASES, "resume text")
    category_col = _detect_column(df, CATEGORY_COLUMN_ALIASES, "category")

    cleaned = pd.DataFrame(
        {
            "resume_text": df[resume_col].astype(str),
            "category_raw": df[category_col].astype(str),
        }
    )
    cleaned["resume_text"] = (
        cleaned["resume_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    cleaned["category"] = cleaned["category_raw"].map(canonicalize_category)
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["category"])
    cleaned = cleaned[cleaned["resume_text"].str.len() >= min_chars]
    cleaned = cleaned.drop_duplicates(subset=["resume_text"]).reset_index(drop=True)
    cleaned.insert(0, "resume_id", [f"R{i:05d}" for i in range(1, len(cleaned) + 1)])
    cleaned = cleaned[["resume_id", "category", "resume_text"]]
    logger.info(
        "Loaded %d resumes (%d dropped for missing category, short text, or duplicates)",
        len(cleaned),
        before - len(cleaned),
    )
    if cleaned.empty:
        raise ResumeDatasetError(f"No usable resumes after cleaning {csv_path}")
    return cleaned


def acquire_resume_csv(config: dict) -> Path:
    """Return a path to a resume CSV, downloading from Kaggle if needed."""
    candidates: list[str] = list(config["paths"]["resume_csv_candidates"])
    existing = _find_existing_csv(candidates)
    if existing is not None:
        logger.info("Using existing resume file: %s", existing)
        return existing

    raw_dir = resolve_path(config["paths"]["raw_dir"])
    dataset = config["kaggle"]["dataset"]
    try:
        _try_kaggle_download(raw_dir, dataset)
    except Exception as exc:
        logger.warning("Kaggle download failed: %s", exc)
        raise ResumeDatasetError(str(exc)) from exc

    existing = _find_existing_csv(candidates)
    if existing is not None:
        return existing

    csvs = sorted(raw_dir.glob("*.csv"))
    if csvs:
        logger.info("Using discovered CSV after Kaggle download: %s", csvs[0])
        return csvs[0]

    raise ResumeDatasetError(
        f"Kaggle download completed but no CSV was found in {raw_dir}"
    )
