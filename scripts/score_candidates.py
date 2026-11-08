#!/usr/bin/env python3
"""Score each DPO candidate by rubric, reusing saved scores with the same signature."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, judge as judge_mod
from src.artifacts import load_json, save_json_atomic, utc_timestamp


def _load_master(path: Path) -> list[dict]:
    return load_json(path)


def _ensure_eval_skeleton(item: dict, source: str) -> dict:
    evals = item.setdefault("evaluations", {})
    src_evals = evals.setdefault(source, {})
    return src_evals.setdefault("rubrics", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = config.load_config(args.config)

    out = config.prepare_output_dir(cfg)
    source = cfg["sft_dir"]
    master_path = out / "candidates.json"
    settings = cfg["judge"]

    items = _load_master(master_path)
    if not items:
        sys.exit(f"No candidates at {master_path}; run generate_candidates first.")

    templates = judge_mod.load_rubric_prompts(config.REPO_ROOT / settings["rubric_prompts"])
    backend = judge_mod.OllamaBackend(model_name=settings["model"], url=settings["url"], options=settings["options"])

    tasks: list[tuple[int, int, str, str]] = []
    for item_idx, item in enumerate(items):
        candidates = (item.get("candidates") or {}).get(source) or []
        if not candidates:
            continue
        rubrics_node = _ensure_eval_skeleton(item, source)
        for cand_idx, cand in enumerate(candidates):
            cand_node = rubrics_node.get(str(cand_idx), {})
            for rubric in judge_mod.RUBRICS:
                prompt = judge_mod.build_prompt(templates[rubric.key], item["input_messages"], cand.get("text", ""))
                if (cand_node.get(rubric.key) or {}).get("signature") == backend.rubric_signature(prompt):
                    continue
                # A stale score must not reach the ranking if rescoring fails.
                cand_node.pop(rubric.key, None)
                tasks.append((item_idx, cand_idx, rubric.key, prompt))

    total_examples = sum(
        1 for it in items if (it.get("candidates") or {}).get(source)
    )
    print(f"[score_candidates] source={source}  examples_with_candidates={total_examples}  "
          f"tasks_to_run={len(tasks)}  parallel={settings['parallel']}")
    missing = len(items) - total_examples
    if not tasks:
        print(f"Nothing to do. Missing candidates={missing}")
        if missing:
            raise SystemExit(1)
        return

    completed_counter = 0
    failed = 0

    def run_one(task):
        item_idx, cand_idx, rubric_key, prompt = task
        return item_idx, cand_idx, rubric_key, {**backend.score(prompt), "judge_timestamp": utc_timestamp()}

    with ThreadPoolExecutor(max_workers=max(1, settings["parallel"])) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                item_idx, cand_idx, rubric_key, record = fut.result()
            except Exception as e:
                failed += 1
                print(f"  [error] task={futures[fut][:3]}: {e}", flush=True)
                continue

            item = items[item_idx]
            rubrics_node = item["evaluations"][source]["rubrics"]
            cand_node = rubrics_node.setdefault(str(cand_idx), {})
            cand_node[rubric_key] = record

            completed_counter += 1
            if (completed_counter % settings["save_every"] == 0
                    or completed_counter == len(tasks)):
                save_json_atomic(items, master_path)
                print(f"  [{completed_counter}/{len(tasks)}] saved", flush=True)

    save_json_atomic(items, master_path)
    print(f"Saved to {master_path}. Completed={completed_counter}, failed={failed}")
    if failed or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
