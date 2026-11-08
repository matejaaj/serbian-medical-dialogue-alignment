#!/usr/bin/env python3
"""Merge predictions.json files of several models by call_id and turn_id into <output_dir>/predictions.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.artifacts import load_json, save_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="configs/eval.yaml")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    out = config.prepare_output_dir(config.load_config(args.config))
    inputs = args.inputs
    master: dict[tuple, dict] = {}

    for in_path in inputs:
        data = load_json(in_path)

        try:
            rows = data["predictions"]
            model_key = data["model"]
        except (KeyError, TypeError) as e:
            raise ValueError(f"{in_path}: invalid predictions file (missing {e!s}); expected keys 'model' and 'predictions'") from e
        if any(model_key in item["predictions"] for item in master.values()):
            raise ValueError(f"{in_path}: model {model_key!r} was already merged from another input")

        for row in rows:
            key = (row["call_id"], row["turn_id"])
            if key not in master:
                master[key] = {
                    "call_id": row["call_id"],
                    "turn_id": row["turn_id"],
                    "input_messages": row["input_messages"],
                    "ground_truth": row["ground_truth"],
                    "predictions": {},
                }
            elif (master[key]["input_messages"], master[key]["ground_truth"]) != (row["input_messages"], row["ground_truth"]):
                # The judge sees one context per item; runs with another system prompt need their own eval.
                raise ValueError(f"{in_path}: {key} has a different context or reference than an earlier input")
            master[key]["predictions"][model_key] = row.get("prediction")

        print(f"Loaded '{model_key}' ({len(rows)} rows) from {in_path}")

    items = [master[k] for k in sorted(master, key=lambda k: (str(k[0]), k[1]))]

    out_path = out / "predictions.json"
    save_json_atomic(items, out_path)

    print(f"\nMerged {len(items)} items, {len(inputs)} models -> {out_path}")


if __name__ == "__main__":
    main()
