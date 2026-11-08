#!/usr/bin/env python3
"""Aggregate reference metrics and rubric scores by model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.artifacts import load_json, save_json_atomic
from src.judge import LABEL_SCORE, RUBRICS, normalize_label


def _load_items(path: Path) -> list[dict]:
    return load_json(path)


def _mean(values: list) -> "float | None":
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def aggregate(items: list, reference: dict | None, judge: dict | None) -> tuple[list[dict], list[str]]:
    targets = sorted({t for item in items for t in item.get("predictions", {})})
    if judge and judge["ground_truth"]:
        targets.append("ground_truth")
    rubric_keys = [r.key for r in RUBRICS] if judge else []
    metric_fields = []
    if reference is not None:
        metric_fields += ["pred_len_chars", "pred_len_words"]
        if reference["rouge_bleu"]:
            metric_fields += ["rouge1", "rouge2", "rougeL", "bleu"]
        if reference["bertscore"] is not None:
            metric_fields += ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
        metric_fields += [f"{key}_cosine" for key in (reference["embeddings"] or {})]

    rows = []
    for target in targets:
        row: dict = {"target": target}
        per_field: dict[str, list] = {f: [] for f in metric_fields}
        rubric_scores: dict[str, list] = {key: [] for key in rubric_keys}
        n = 0

        for item in items:
            text = item.get("ground_truth") if target == "ground_truth" else item.get("predictions", {}).get(target)
            ev = item.get("evaluations", {}).get(target)
            if text is None or not isinstance(ev, dict):
                continue
            n += 1
            for f in metric_fields:
                if isinstance(ev.get(f), (int, float)):
                    per_field[f].append(ev[f])
            for rubric_key, scores in rubric_scores.items():
                label = normalize_label((ev.get("rubrics", {}).get(rubric_key) or {}).get("label"))
                if label in LABEL_SCORE:
                    scores.append(LABEL_SCORE[label])

        texts = [item.get("ground_truth") if target == "ground_truth"
                 else item.get("predictions", {}).get(target) for item in items]
        row["expected"] = len(items)
        row["predictions"] = sum(text is not None for text in texts)
        row["empty"] = sum(text is not None and not text.strip() for text in texts)
        row["n"] = n
        for f in metric_fields:
            row[f] = _mean(per_field[f])
            row[f"{f}_n"] = len(per_field[f])
        for rubric_key, scores in rubric_scores.items():
            row[rubric_key] = _mean(scores)
            row[f"{rubric_key}_n"] = len(scores)
        complete = row["predictions"] == len(items)
        complete &= all(row[f"{key}_n"] == len(items) for key in rubric_keys)
        if target != "ground_truth":
            for field in metric_fields:
                expected = len(items) - row["empty"] if field.endswith("_cosine") else len(items)
                complete &= row[f"{field}_n"] == expected
        row["status"] = "complete" if complete else "incomplete"
        rows.append(row)

    columns = ["target", "status", "expected", "predictions", "empty"]
    for field in metric_fields + rubric_keys:
        columns.extend([field, f"{field}_n"])
    return rows, columns


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="configs/eval.yaml")
    args = parser.parse_args()
    settings = config.load_config(args.config)
    out = config.prepare_output_dir(settings)
    # Without a judge, only eval_reference.py has written metrics.
    items = _load_items(out / ("evaluations.json" if settings["judge"] else "predictions.json"))
    keys = [(item["call_id"], item["turn_id"]) for item in items]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate (call_id, turn_id) in evaluation input")
    rows, columns = aggregate(items, settings["reference"], settings["judge"])
    if not rows:
        print("No evaluations in the input.")
        return

    save_json_atomic(rows, out / "metrics.json")
    metric_fields = columns[5::2]
    for row in rows:
        print(f"\n{row['target']} ({row['status']})")
        print(f"Predictions: {row['predictions']}/{row['expected']}; empty: {row['empty']}")
        print(f"{'Metric':52} {'Mean':>10} {'Scored':>10}")
        for metric in metric_fields:
            if row["target"] == "ground_truth" and row[f"{metric}_n"] == 0:
                continue
            expected = row["expected"] - row["empty"] if metric.endswith("_cosine") else row["expected"]
            coverage = f"{row[f'{metric}_n']}/{expected}"
            print(f"{metric:52} {_fmt(row[metric]):>10} {coverage:>10}")

if __name__ == "__main__":
    main()
