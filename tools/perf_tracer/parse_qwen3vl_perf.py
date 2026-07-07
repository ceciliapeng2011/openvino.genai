#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
import statistics
from pathlib import Path

METRIC_RE = re.compile(r"^\s*([^:]+):\s*([-+]?\d+(?:\.\d+)?)\s*(?:±\s*([-+]?\d+(?:\.\d+)?))?\s*([A-Za-z/]+)?\s*$")
IMG_PROMPT_RE = re.compile(r"Number of images:\s*(\d+),\s*Prompt token size:\s*(\d+)")
OUT_TOKEN_RE = re.compile(r"Output token size:\s*(\d+)")
CASE_RE = re.compile(r"trace\.(.*?)\.plain\.log$")

CASE_ORDER_AND_LABELS = [
    (
        "448x448x2.512_cn.cm_path.cfg_none",
        "1. ViT/TPS/TTFT: 2 IMG 448x448 + 512 tokens",
    ),
    (
        "512x384x2.512_cn.cm_path.cfg_none",
        "2. ViT (ms) @2IMG 512x384",
    ),
    (
        "1024x512x2.512_cn.cm_path.cfg_none",
        "3. ViT (ms) @2IMG 1024x512",
    ),
    (
        "1260x700x2.512_cn.cm_path.cfg_none",
        "4. ViT (ms) @2IMG 1260x700",
    ),
    (
        "1260x700x2.833_cn.mt512.cm_path.cfg_none",
        "5. Prefill&Decode Rate (tokens/s) @2560 context",
    ),
    (
        "1280x768x15.512_cn.cm_path.cfg_none",
        "6. ViT (ms) @15IMG 1280x768",
    ),
]

CASE_ORDER = {case: idx for idx, (case, _) in enumerate(CASE_ORDER_AND_LABELS, start=1)}
CASE_LABEL = {case: label for case, label in CASE_ORDER_AND_LABELS}


def normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = key.replace("[", "").replace("]", "")
    key = key.replace("(", "").replace(")", "")
    key = key.replace(" ", "_")
    key = key.replace("-", "_")
    key = key.replace("/", "_per_")
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def parse_log(path: str) -> dict:
    row = {
        "file": os.path.basename(path),
        "case": os.path.basename(path).replace("trace.", "").replace(".plain.log", ""),
    }

    m = CASE_RE.search(os.path.basename(path))
    if m:
        row["case"] = m.group(1)
    row["case_order"] = CASE_ORDER.get(row["case"], 999)
    row["case_label"] = CASE_LABEL.get(row["case"], row["case"])

    for line in Path(path).read_text(errors="ignore").splitlines():
        m = IMG_PROMPT_RE.search(line)
        if m:
            row["num_images"] = int(m.group(1))
            row["prompt_token_size"] = int(m.group(2))
            continue

        m = OUT_TOKEN_RE.search(line)
        if m:
            row["output_token_size"] = int(m.group(1))
            continue

        m = METRIC_RE.match(line)
        if not m:
            continue

        key = normalize_key(m.group(1))
        row[key] = float(m.group(2))

        if m.group(3) is not None:
            row[f"{key}_std"] = float(m.group(3))

        unit = (m.group(4) or "").strip()
        if unit:
            row[f"{key}_unit"] = unit

    return row


def write_per_case(rows: list, out_csv: str) -> None:
    rows = sorted(rows, key=lambda r: (int(r.get("case_order", 999)), str(r.get("case", ""))))

    fixed = [
        "case",
        "ttft",
        "embeddings_preparation_time",
        "vision_embeddings_merger",
        "lm_prefill",
        "tpot",
        "throughput",
        "file",
        "num_images",
        "prompt_token_size",
        "output_token_size",
    ]
    keys = set()
    for r in rows:
        keys.update(r.keys())
    hidden = {"case_order", "case_label"}
    fieldnames = fixed + sorted(k for k in keys if k not in fixed and k not in hidden)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_summary(rows: list, out_csv: str) -> None:
    metrics = set()
    skip = {"num_images", "prompt_token_size", "output_token_size"}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in skip and not k.endswith("_std"):
                metrics.add(k)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "count", "mean", "min", "max", "std"])
        for metric in sorted(metrics):
            vals = [float(r[metric]) for r in rows if metric in r]
            if not vals:
                continue
            writer.writerow([
                metric,
                len(vals),
                f"{statistics.mean(vals):.6f}",
                f"{min(vals):.6f}",
                f"{max(vals):.6f}",
                f"{statistics.pstdev(vals) if len(vals) > 1 else 0.0:.6f}",
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse qwen3-vl perf logs into CSV reports")
    parser.add_argument("--input-glob", default="/home/intel/ceciliapeng/VM/qwen3-vl/log_profiling/*.plain.log")
    parser.add_argument("--out-per-case", default="/home/intel/ceciliapeng/VM/qwen3-vl/perf_report_per_case.csv")
    parser.add_argument("--out-summary", default="/home/intel/ceciliapeng/VM/qwen3-vl/perf_report_summary.csv")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No logs matched: {args.input_glob}")

    rows = [parse_log(p) for p in paths]
    write_per_case(rows, args.out_per_case)
    write_summary(rows, args.out_summary)

    print(f"Parsed {len(rows)} logs")
    print(f"Per-case CSV: {args.out_per_case}")
    print(f"Summary CSV: {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
