#!/usr/bin/env python3
"""Generate DPO candidates on the training split, resuming completed examples."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, conversation, dataset as ds_mod, generation
from src.artifacts import load_json, save_json_atomic, utc_timestamp


def _load_master(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return load_json(path)


def _item_key(call_id, turn_id) -> tuple:
    return (str(call_id), int(turn_id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = config.load_config(args.config)

    out = config.prepare_output_dir(cfg)
    setup = config.model_setup(cfg)
    source = cfg["sft_dir"]
    output_path = out / "candidates.json"
    candidate_settings = cfg["candidates"]

    items = _load_master(output_path)
    by_key: dict[tuple, dict] = {
        _item_key(it["call_id"], it["turn_id"]): it for it in items
    }

    train, _ = ds_mod.load_split()
    todo: list[int] = []
    for idx, example in enumerate(train):
        key = _item_key(example["call_id"], example["turn_id"])
        existing = by_key.get(key)
        candidates_for_source = (existing or {}).get("candidates", {}).get(source, [])
        if existing and existing["input_messages"] != conversation.with_system(example["input"], setup["system_prompt"]):
            raise ValueError("Candidate context or system prompt changed; use a new output_dir")
        if len(candidates_for_source) >= candidate_settings["count"]:
            continue
        todo.append(idx)

    print(f"[generate_candidates] source={source}  total={len(train)}  to_do={len(todo)}  "
          f"already_done={len(train) - len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    # Candidates come from the SFT adapter, not from this run's DPO adapter.
    generator = generation.build_generator(setup, seed=cfg["seed"], adapter=config.REPO_ROOT / source)

    for n_done, idx in enumerate(todo, start=1):
        example = train[idx]
        input_messages = conversation.with_system(example["input"], setup["system_prompt"])
        ground_truth = example["output"]["content"]
        key = _item_key(example["call_id"], example["turn_id"])

        candidates = generator.generate_candidates(
            input_messages,
            seeds=[cfg["seed"] + i for i in range(candidate_settings["count"])],
            temperature=candidate_settings["temperature"],
            top_p=candidate_settings["top_p"],
        )
        gen_ts = utc_timestamp()
        for c in candidates:
            c["gen_timestamp"] = gen_ts

        existing = by_key.get(key)
        if existing is None:
            existing = {
                "call_id": example["call_id"],
                "turn_id": example["turn_id"],
                "input_messages": input_messages,
                "ground_truth": ground_truth,
                "candidates": {},
                "evaluations": {},
            }
            by_key[key] = existing
            items.append(existing)
        existing.setdefault("candidates", {})[source] = candidates
        existing.setdefault("evaluations", {}).pop(source, None)

        # Save each example so an interrupted run loses at most one generation.
        save_json_atomic(items, output_path)
        if n_done % 10 == 0 or n_done == len(todo):
            print(f"  [{n_done}/{len(todo)}] saved (last call_id={example['call_id']}, "
                  f"turn_id={example['turn_id']})")

    print(f"Done. Wrote {output_path}")


if __name__ == "__main__":
    main()
