"""
Phase 2 — Fine-tune all-MiniLM-L6-v2 with MultipleNegativesRankingLoss.

Training strategy
-----------------
        * Build (resume_text, positive_jd_text, hard_negative_jd_text) triplets.
    - Anchor   = resume_text     (the query at inference time)
    - Positive = matching JD
    - Negative = hard negative JD (same category, wrong match)
* Feed triplets into MNRL.  The explicit hard negative is added on top of the
  in-batch negatives that MNRL synthesises automatically.
* Easy negatives are excluded from training — in-batch negatives already
  provide that trivial-rejection signal for free.
* Evaluation uses EmbeddingSimilarityEvaluator on the val split (positives +
  hard negatives).  The best checkpoint (highest Spearman cosine) is saved.

Usage
-----
    python -m src.training.train
    python -m src.training.train --config configs/training_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Set UTF-8 stream handler for Windows console compatibility
console_handler = logging.StreamHandler(sys.stdout)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        console_handler,
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyper-parameters for Phase 2 training."""

    base_model: str = "all-MiniLM-L6-v2"
    train_csv: str = "data/processed/train_pairs.csv"
    val_csv: str = "data/processed/val_pairs.csv"
    output_dir: str = "models/resume-jd-sbert"
    epochs: int = 5
    batch_size: int = 16
    warmup_ratio: float = 0.1
    evaluation_steps: int = 50
    seed: int = 42
    fp16: bool = True


def load_config(config_path: str) -> TrainingConfig:
    """Load a YAML config file and return a TrainingConfig dataclass."""
    with open(config_path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)
    return TrainingConfig(**raw)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Pin all random-number generators for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed set to %d", seed)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_triplets(csv_path: str, seed: int = 42) -> list[InputExample]:
    """
    Build MNRL triplets from the training CSV.

    Strategy
    --------
    For every positive (jd, resume) pair, look up hard negatives that share
    the same jd_id and randomly draw one.  Easy negatives are skipped —
    in-batch negatives already cover that signal.

    Returns
    -------
    list[InputExample]
        Each example has texts = [resume_text, positive_jd_text, hard_negative_jd_text].
    """
    random.seed(seed)
    df = pd.read_csv(csv_path)

    positives = df[df["pair_type"] == "positive"].copy()
    hard_negs = df[df["pair_type"] == "negative_hard"].copy()

    # resume_id → list of hard-negative JD texts
    hard_neg_map: dict[str, list[str]] = (
        hard_negs.groupby("resume_id")["jd_text"].apply(list).to_dict()
    )

    examples: list[InputExample] = []
    skipped = 0

    for _, row in positives.iterrows():
        resume_id: str = row["resume_id"]
        candidates = hard_neg_map.get(resume_id, [])
        if not candidates:
            skipped += 1
            continue
        examples.append(
            InputExample(
                texts=[
                    str(row["resume_text"]),
                    str(row["jd_text"]),
                    random.choice(candidates),
                ]
            )
        )

    logger.info(
        "Triplets built: %d from %d positives (%d skipped — no hard neg for that JD).",
        len(examples),
        len(positives),
        skipped,
    )
    return examples


def build_val_evaluator(csv_path: str) -> EmbeddingSimilarityEvaluator:
    """
    Build an EmbeddingSimilarityEvaluator on the validation split.

    Uses positives (label=1) and hard negatives (label=0).
    Easy negatives are excluded to keep the evaluator aligned with the
    triplet training objective.
    """
    df = pd.read_csv(csv_path)
    eval_df = df[df["pair_type"].isin(["positive", "negative_hard"])].copy()

    logger.info(
        "Val evaluator: %d pairs (%d positive, %d hard negative).",
        len(eval_df),
        (eval_df["label"] == 1).sum(),
        (eval_df["label"] == 0).sum(),
    )

    return EmbeddingSimilarityEvaluator(
        sentences1=eval_df["resume_text"].tolist(),
        sentences2=eval_df["jd_text"].tolist(),
        scores=eval_df["label"].astype(float).tolist(),
        name="val",
        show_progress_bar=True,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(output_dir: str) -> None:
    """
    Reload the saved model and assert positive similarity > cross-domain
    similarity on 3 hand-crafted pairs.

    Raises
    ------
    AssertionError
        If any sanity check fails.
    """
    logger.info("=== Smoke test: reloading model from %s ===", output_dir)
    model = SentenceTransformer(output_dir)

    it_resume = (
        "Python developer with 5 years of experience in FastAPI, PostgreSQL, "
        "Docker, and machine learning pipelines."
    )
    finance_resume = (
        "Senior Financial Analyst with expertise in GAAP, SAP, Excel modelling, "
        "budget forecasting, and financial reporting for Fortune 500 companies."
    )
    it_jd = (
        "We are hiring a Backend Engineer with Python, REST APIs, SQL, and "
        "cloud deployment experience."
    )
    finance_jd = (
        "Seeking a Finance Manager with strong accounting, forecasting, and "
        "stakeholder communication skills."
    )

    sentences = [it_resume, finance_resume, it_jd, finance_jd]
    vecs = model.encode(sentences, convert_to_tensor=True,
                        normalize_embeddings=True)

    from torch.nn.functional import cosine_similarity

    # cosine_similarity expects (1, d) tensors
    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    pos_it = cos(vecs[0], vecs[2])   # IT resume   ↔ IT JD
    pos_fin = cos(vecs[1], vecs[3])   # Fin resume  ↔ Fin JD
    cross_it = cos(vecs[0], vecs[3])   # IT resume   ↔ Fin JD
    cross_fin = cos(vecs[1], vecs[2])   # Fin resume  ↔ IT JD

    logger.info("Smoke-test similarity scores:")
    logger.info("  IT resume  ↔ IT JD      (expect HIGH):  %.4f", pos_it)
    logger.info("  Fin resume ↔ Fin JD     (expect HIGH):  %.4f", pos_fin)
    logger.info("  IT resume  ↔ Fin JD     (expect LOW):   %.4f", cross_it)
    logger.info("  Fin resume ↔ IT JD      (expect LOW):   %.4f", cross_fin)

    assert pos_it > cross_it, (
        f"FAIL — IT positive ({pos_it:.4f}) should exceed IT×Finance cross ({cross_it:.4f})"
    )
    assert pos_fin > cross_fin, (
        f"FAIL — Finance positive ({pos_fin:.4f}) should exceed Finance×IT cross ({cross_fin:.4f})"
    )
    logger.info("✓ All smoke-test assertions passed.")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: TrainingConfig) -> None:
    """
    Fine-tune all-MiniLM-L6-v2 using MultipleNegativesRankingLoss with
    hard-negative triplets.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=== Phase 2: Fine-tuning %s ===", config.base_model)
    logger.info("Device: %s | fp16: %s", device,
                config.fp16 and device == "cuda")

    set_seed(config.seed)

    # ── Data ──────────────────────────────────────────────────────────────
    triplets = build_triplets(config.train_csv, seed=config.seed)
    if not triplets:
        raise RuntimeError(
            "No training triplets could be built.  Check that train_pairs.csv "
            "contains both 'positive' and 'negative_hard' pair_type rows."
        )

    train_dataloader = DataLoader(
        triplets, shuffle=True, batch_size=config.batch_size)
    evaluator = build_val_evaluator(config.val_csv)

    # ── Model & loss ───────────────────────────────────────────────────────
    model = SentenceTransformer(config.base_model, device=device)
    loss = MultipleNegativesRankingLoss(model=model)

    # ── Warmup ────────────────────────────────────────────────────────────
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    logger.info(
        "Training: %d triplets | %d epochs | batch %d | %d steps | %d warmup",
        len(triplets),
        config.epochs,
        config.batch_size,
        total_steps,
        warmup_steps,
    )

    # ── Output dir ────────────────────────────────────────────────────────
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Fine-tune ─────────────────────────────────────────────────────────
    # model.fit() accepts InputExample triplets for MNRL directly.
    # The third text in each triplet becomes an explicit hard negative on
    # top of the in-batch negatives synthesised by the loss.
    # NOTE: model.fit() is deprecated in sentence-transformers ≥ 3.x in
    # favour of SentenceTransformerTrainer, but remains fully functional.
    # Migration is straightforward if needed for Phase 6+.
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        evaluator=evaluator,
        epochs=config.epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=config.evaluation_steps,
        output_path=config.output_dir,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=(config.fp16 and device == "cuda"),
        checkpoint_path=None,
    )

    logger.info("Training complete.  Best model saved to: %s",
                config.output_dir)

    # ── Verify ────────────────────────────────────────────────────────────
    run_smoke_test(config.output_dir)

    logger.info("=== Phase 2 done. ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 — Fine-tune Sentence-BERT on resume–JD pairs."
    )
    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="Path to YAML training config (default: configs/training_config.yaml)",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    config = load_config(args.config)
    logger.info("Loaded config: %s", config)
    train(config)


if __name__ == "__main__":
    main()
