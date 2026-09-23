"""
scripts/check_truncation.py
----------------------------
Diagnose how many resumes are truncated by the MiniLM tokenizer.

Reports truncation at THREE relevant thresholds:
  - 128  tokens  -> effective max_length used during training (tokenizer_config.json)
  - 256  tokens  -> model_max_length from tokenizer_config.json
  - 512  tokens  -> hard BERT positional embedding limit (config.json)

Usage:
    python scripts/check_truncation.py
    python scripts/check_truncation.py --csv data/processed/resumes.csv --col resume_text
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH  = str(ROOT / "models" / "resume-jd-sbert-v2")
RESUME_CSV  = str(ROOT / "data" / "processed" / "resumes.csv")
TEXT_COL    = "resume_text"
THRESHOLDS  = [128, 256, 512]


def load_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print(f"[INFO] Loaded tokenizer from: {model_path}")
        print(f"[INFO] model_max_length reported by tokenizer : {tok.model_max_length}")
        return tok
    except Exception as exc:
        print(f"[ERROR] Could not load tokenizer: {exc}")
        sys.exit(1)


def load_resumes(csv_path: str, text_col: str):
    import csv
    texts = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t = row.get(text_col, "").strip()
            if t:
                texts.append(t)
    print(f"[INFO] Loaded {len(texts)} resumes from {csv_path}")
    return texts


def analyse(texts, tokenizer, thresholds):
    print("\n[INFO] Tokenising resumes (no truncation) ...")
    token_lengths = []
    for i, text in enumerate(texts, 1):
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        token_lengths.append(len(ids))
        if i % 50 == 0 or i == len(texts):
            print(f"  processed {i}/{len(texts)}", end="\r", flush=True)
    print()

    n = len(token_lengths)
    sorted_lengths = sorted(token_lengths)

    print("\n" + "=" * 60)
    print("  TOKEN-LENGTH STATISTICS")
    print("=" * 60)
    print(f"  Resumes analysed : {n}")
    print(f"  Min tokens       : {sorted_lengths[0]}")
    print(f"  Max tokens       : {sorted_lengths[-1]}")
    print(f"  Mean tokens      : {sum(token_lengths)/n:.1f}")
    print(f"  Median tokens    : {sorted_lengths[n//2]}")
    pct = lambda p: sorted_lengths[int(p/100 * n)]
    print(f"  p75              : {pct(75)}")
    print(f"  p90              : {pct(90)}")
    print(f"  p95              : {pct(95)}")
    print(f"  p99              : {pct(99)}")

    print("\n" + "=" * 60)
    print("  TRUNCATION RATES")
    print("=" * 60)
    print(f"  {'Threshold':>10}  {'Truncated':>10}  {'Rate':>8}  Explanation")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*42}")
    explanations = {
        128: "max_length in tokenizer_config.json (training cutoff)",
        256: "model_max_length in tokenizer_config.json",
        512: "BERT hard limit (max_position_embeddings)",
    }
    for t in thresholds:
        truncated = sum(1 for l in token_lengths if l > t)
        rate = truncated / n * 100
        flag = "WARN" if rate > 10 else ("HIGH" if rate > 50 else "OK  ")
        print(f"  {t:>10}  {truncated:>10}  {rate:>7.1f}%  [{flag}] {explanations.get(t, '')}")

    print("=" * 60)

    indexed = sorted(enumerate(token_lengths), key=lambda x: -x[1])[:5]
    print(f"\n  TOP 5 LONGEST RESUMES (by token count):")
    for idx, length in indexed:
        preview = texts[idx][:80].replace("\n", " ")
        print(f"    [{idx:04d}]  {length} tokens  \"{preview}...\"")
    print()


def main():
    parser = argparse.ArgumentParser(description="Check resume truncation rate")
    parser.add_argument("--csv",   default=RESUME_CSV,  help="Path to resumes CSV")
    parser.add_argument("--col",   default=TEXT_COL,    help="Column name for resume text")
    parser.add_argument("--model", default=MODEL_PATH,  help="Path to model/tokenizer")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    texts     = load_resumes(args.csv, args.col)
    analyse(texts, tokenizer, thresholds=THRESHOLDS)


if __name__ == "__main__":
    main()
