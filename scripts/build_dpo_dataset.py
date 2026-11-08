#!/usr/bin/env python3
"""Build DPO pairs from candidates with decisive head-to-head preferences."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, judge as judge_mod
from src.artifacts import load_json, save_json_atomic, utc_timestamp


def _load_master(path: Path) -> list[dict]:
    return load_json(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = config.load_config(args.config)

    out = config.prepare_output_dir(cfg)
    source = cfg["sft_dir"]
    master_path = out / "candidates.json"
    output_path = out / "dpo_pairs.json"

    items = _load_master(master_path)

    pairs: list[dict] = []
    skipped_no_ranking = 0
    skipped_identical = 0
    skipped_empty_chosen = 0
    skipped_tie = 0

    for item in items:
        src_evals = (item.get("evaluations") or {}).get(source) or {}
        ranking = src_evals.get("pair_ranking")
        candidates = (item.get("candidates") or {}).get(source) or []
        if not ranking or len(candidates) < cfg["candidates"]["count"]:
            skipped_no_ranking += 1
            continue

        best_idx, worst_idx = ranking[0], ranking[-1]
        best_text = (candidates[best_idx].get("text") or "").strip()
        worst_text = (candidates[worst_idx].get("text") or "").strip()
        # An empty rejected answer is a useful negative; an empty chosen one would teach silence.
        if not best_text:
            skipped_empty_chosen += 1
            continue
        if best_text == worst_text:
            skipped_identical += 1
            continue
        # Require a direct win in both A/B orders, not just a higher tournament rank.
        judgements = {j["key"]: j for j in src_evals["pair_judgements"]}
        if judge_mod.pair_winner(judgements, (best_idx, worst_idx)) != best_idx:
            skipped_tie += 1
            continue

        pairs.append({
            "call_id": item["call_id"],
            "turn_id": item["turn_id"],
            "input_messages": item["input_messages"],
            "chosen": best_text,
            "rejected": worst_text,
            "chosen_candidate_idx": best_idx,
            "rejected_candidate_idx": worst_idx,
            "chosen_seed": candidates[best_idx].get("seed"),
            "rejected_seed": candidates[worst_idx].get("seed"),
        })

    first_candidates = next(item["candidates"][source] for item in items if item.get("candidates", {}).get(source))
    payload = {
        "created_at": utc_timestamp(),
        "sft_dir": source,
        "candidate_decoding_params": {
            "temperature": first_candidates[0].get("temperature"),
            "top_p": first_candidates[0].get("top_p"),
            "num_candidates": len(first_candidates),
            "seeds": [c.get("seed") for c in first_candidates],
        },
        "ranking_method": "copeland_with_swap",
        "num_pairs": len(pairs),
        "num_skipped_no_ranking": skipped_no_ranking,
        "num_skipped_identical": skipped_identical,
        "num_skipped_empty_chosen": skipped_empty_chosen,
        "num_skipped_tie": skipped_tie,
        "pairs": pairs,
    }

    # DPO training resumes from its last checkpoint, which was trained on the saved pairs.
    if any(out.glob("checkpoint-*")) and _load_master(output_path)["pairs"] != pairs:
        raise ValueError(f"DPO training in {out} started on different pairs; use a new output_dir")
    save_json_atomic(payload, output_path)

    print(f"[build_dpo_dataset] {source}: kept={len(pairs)}  "
          f"skip(no_ranking)={skipped_no_ranking}  "
          f"skip(identical)={skipped_identical}  "
          f"skip(empty_chosen)={skipped_empty_chosen}  "
          f"skip(tie)={skipped_tie}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
