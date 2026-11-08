#!/usr/bin/env python3
"""Add reference metrics to <output_dir>/predictions.json, in place."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, metrics, predictions
from src.artifacts import load_json, save_json_atomic


def _eval_master_file(items: list, out_path: Path, reference: dict) -> None:
    model_names = sorted({m for item in items for m in item.get("predictions", {})})
    incomplete = False
    for model_name in model_names:
        rows = [
            {"prediction": item.get("predictions", {}).get(model_name),
             "ground_truth": item.get("ground_truth")}
            for item in items
        ]
        summary = metrics.evaluate_prediction_rows(
            rows, reference)
        incomplete |= summary["scored"] != summary["expected"]
        _, _, valid_idx = predictions.pred_ref_lists(rows)
        corpus_scores = {k: summary[k] for k in ("rouge1", "rouge2", "rougeL", "bleu") if k in summary}
        for row in rows:
            row.update(dict.fromkeys(corpus_scores))
        for i in valid_idx:
            rows[i].update(corpus_scores)
        for item, row in zip(items, rows):
            evaluations = item.setdefault("evaluations", {})
            if not isinstance(evaluations.get(model_name), dict):
                evaluations[model_name] = {}
            evaluations[model_name].update({
                k: v for k, v in row.items()
                if k not in ("prediction", "ground_truth")
            })
        print(f"{model_name}:", summary)

    save_json_atomic(items, out_path)
    print("\nSaved to:", out_path)
    if incomplete:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="configs/eval.yaml")
    args = parser.parse_args()
    cfg = config.load_config(args.config)
    path = config.prepare_output_dir(cfg) / "predictions.json"
    _eval_master_file(load_json(path), path, cfg["reference"])


if __name__ == "__main__":
    main()
