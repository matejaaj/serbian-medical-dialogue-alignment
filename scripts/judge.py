#!/usr/bin/env python3
"""Score <output_dir>/predictions.json into evaluations.json, reusing saved scores with the same signature."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, judge as judge_mod
from src.artifacts import load_json, save_json_atomic


def _load_items(path: Path) -> list[dict]:
    return load_json(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="configs/eval.yaml")
    args = parser.parse_args()
    cfg = config.load_config(args.config)
    out = config.prepare_output_dir(cfg)
    settings = cfg["judge"]
    output_path = out / "evaluations.json"
    items = _load_items(out / "predictions.json")
    previous = _load_items(output_path) if output_path.exists() else []
    saved = {(item["call_id"], item["turn_id"]): item for item in previous}
    templates = judge_mod.load_rubric_prompts(config.REPO_ROOT / settings["rubric_prompts"])
    backend = judge_mod.OllamaBackend(model_name=settings["model"], url=settings["url"], options=settings["options"])

    model_names = {name for item in items for name in item.get("predictions", {})}
    tasks = []
    reused = missing = 0
    for item_idx, item in enumerate(items):
        evaluations = item.setdefault("evaluations", {})
        old = saved.get((item["call_id"], item["turn_id"]), {})
        targets = {name: item.get("predictions", {}).get(name) for name in model_names}
        if settings["ground_truth"]:
            targets["ground_truth"] = item.get("ground_truth")
        for target, text in targets.items():
            entry = evaluations.setdefault(target, {})
            scores = entry.setdefault("rubrics", {})
            old_scores = old.get("evaluations", {}).get(target, {}).get("rubrics", {})
            if text is None:
                scores.clear()
                missing += 1
                continue
            for rubric in judge_mod.RUBRICS:
                prompt = judge_mod.build_prompt(templates[rubric.key], item["input_messages"], text)
                cached = old_scores.get(rubric.key, scores.get(rubric.key, {}))
                if cached.get("signature") == backend.rubric_signature(prompt):
                    scores[rubric.key] = cached
                    reused += 1
                    continue
                scores.pop(rubric.key, None)
                tasks.append((item_idx, target, rubric.key, prompt))

    def run_one(task):
        return backend.score(task[3])

    completed = failed = 0
    save_json_atomic(items, output_path)
    print(f"[judge] reused={reused} pending={len(tasks)} missing_predictions={missing}")
    with ThreadPoolExecutor(max_workers=max(1, settings["parallel"])) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for future in as_completed(futures):
            item_idx, target, rubric_key, _ = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failed += 1
                print(f"[error] item={item_idx} target={target} rubric={rubric_key}: {error}", flush=True)
                continue
            items[item_idx]["evaluations"][target]["rubrics"][rubric_key] = result
            completed += 1
            if completed % settings["save_every"] == 0:
                save_json_atomic(items, output_path)
    save_json_atomic(items, output_path)
    print(f"[judge] reused={reused} completed={completed} failed={failed} missing_predictions={missing}")
    print("Saved to:", output_path)
    if failed or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
