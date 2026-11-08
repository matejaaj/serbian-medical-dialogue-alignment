#!/usr/bin/env python3
"""Rank DPO candidates using pairwise comparisons in both A/B orders."""
from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, judge as judge_mod
from src.artifacts import load_json, save_json_atomic, utc_timestamp


RANKING_METHOD = "copeland_with_swap"
CHARS_PER_TOKEN = 4  # Approximation, not tokenizer-based counting.


def _load_master(path: Path) -> list[dict]:
    return load_json(path)


def _pair_ids(count: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(count), 2))


def _truncate_messages(input_messages, budget_chars: int) -> tuple[list[dict], bool]:
    """Drop oldest non-system messages; return messages and a truncation flag."""
    total = sum(len(m.get("content", "")) for m in input_messages)
    if total <= budget_chars:
        return input_messages, False

    system_msgs = [m for m in input_messages if m.get("role") == "system"]
    other_msgs = [m for m in input_messages if m.get("role") != "system"]
    while other_msgs and sum(len(m.get("content", "")) for m in (system_msgs + other_msgs)) > budget_chars:
        other_msgs.pop(0)
    return system_msgs + other_msgs, True


def _compute_ranking(candidates: list[dict], rubric_node: dict, judgements: list[dict]) -> list[int]:
    """Rank by Copeland score, then rubric sum, then lowest seed."""
    count = len(candidates)
    by_key = {j["key"]: j for j in judgements}
    copeland: dict[int, float] = {i: 0.0 for i in range(count)}
    for a, b in _pair_ids(count):
        winner = judge_mod.pair_winner(by_key, (a, b))
        if winner is None:
            copeland[a] += 0.5
            copeland[b] += 0.5
        else:
            copeland[winner] += 1.0

    def rubric_sum(cand_idx: int) -> int:
        cand_rubric = rubric_node.get(str(cand_idx)) or {}
        total = 0
        for rubric in judge_mod.RUBRICS:
            entry = cand_rubric.get(rubric.key) or {}
            label = judge_mod.normalize_label(entry.get("label"))
            if label is None:
                raise ValueError(f"Missing or invalid rubric {rubric.key} for candidate {cand_idx}")
            total += judge_mod.LABEL_SCORE[label]
        return total

    def seed_of(cand_idx: int) -> int:
        return int(candidates[cand_idx].get("seed", 0))

    ordered = sorted(
        range(count),
        key=lambda i: (-copeland[i], -rubric_sum(i), seed_of(i)),
    )
    return ordered


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

    template = judge_mod.load_pairwise_prompt(config.REPO_ROOT / settings["pairwise_prompt"])
    backend = judge_mod.OllamaBackend(model_name=settings["model"], url=settings["url"], options=settings["options"])

    count = cfg["candidates"]["count"]
    pairs = _pair_ids(count)
    context_budget = settings["context_token_cap"] * CHARS_PER_TOKEN
    tasks: list[tuple] = []
    skipped_no_candidates = 0
    skipped_no_rubrics = 0
    for item_idx, item in enumerate(items):
        candidates = (item.get("candidates") or {}).get(source) or []
        if len(candidates) != count:
            skipped_no_candidates += 1
            continue
        src_evals = item.setdefault("evaluations", {}).setdefault(source, {})
        rubric_node = src_evals.get("rubrics") or {}
        all_rubrics_complete = all(
            isinstance(rubric_node.get(str(c)), dict)
            and all(judge_mod.normalize_label((rubric_node[str(c)].get(r.key) or {}).get("label")) is not None
                    for r in judge_mod.RUBRICS)
            for c in range(count)
        )
        if not all_rubrics_complete:
            skipped_no_rubrics += 1
            continue
        messages, truncated = _truncate_messages(item["input_messages"], context_budget)
        saved = {j["key"]: j for j in src_evals.get("pair_judgements") or []}
        current = []
        for pair in pairs:
            for round_id in (1, 2):
                a_idx, b_idx = pair if round_id == 1 else pair[::-1]
                prompt = judge_mod.build_pairwise_prompt(
                    template,
                    messages,
                    response_a=candidates[a_idx].get("text", ""),
                    response_b=candidates[b_idx].get("text", ""),
                    rubric_summaries_a=judge_mod.format_rubric_summary(rubric_node[str(a_idx)]),
                    rubric_summaries_b=judge_mod.format_rubric_summary(rubric_node[str(b_idx)]),
                )
                key = judge_mod.judgement_key(pair, round_id)
                if saved.get(key, {}).get("signature") == backend.pairwise_signature(prompt):
                    current.append(saved[key])
                    continue
                tasks.append((item_idx, key, pair, round_id, a_idx, b_idx, prompt, truncated))
        # Judgements of another request, e.g. made with older rubric scores, are redone.
        src_evals["pair_judgements"] = current

    print(f"[rank_candidates] source={source}  items={len(items)}  "
          f"skipped(missing candidates)={skipped_no_candidates}  "
          f"skipped(missing rubric scores)={skipped_no_rubrics}  "
          f"tasks_to_run={len(tasks)}  parallel={settings['parallel']}")

    completed = 0
    failed = 0

    def run_one(task):
        item_idx, key, pair, round_id, a_idx, b_idx, prompt, truncated = task
        return item_idx, {
            "key": key,
            "pair": list(pair),
            "round": round_id,
            "presented_as": {"A": a_idx, "B": b_idx},
            **backend.compare(prompt),
            "context_truncated": truncated,
            "judge_timestamp": utc_timestamp(),
        }

    items_finalised: set[int] = set()

    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, settings["parallel"])) as pool:
            futures = {pool.submit(run_one, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    item_idx, record = fut.result()
                except Exception as e:
                    failed += 1
                    print(f"  [error] task={futures[fut][:2]}: {e}", flush=True)
                    continue
                items[item_idx]["evaluations"][source]["pair_judgements"].append(record)

                completed += 1
                if (completed % settings["save_every"] == 0
                        or completed == len(tasks)):
                    save_json_atomic(items, master_path)
                    print(f"  [{completed}/{len(tasks)}] saved", flush=True)

    # Rankings are cheap to recompute; saved judge calls remain reusable.
    for item_idx, item in enumerate(items):
        src_evals = (item.get("evaluations") or {}).get(source)
        if not src_evals:
            continue
        judgements = src_evals.get("pair_judgements") or []
        src_evals.pop("pair_ranking", None)
        src_evals.pop("ranking_method", None)
        if len(judgements) >= len(pairs) * 2:
            try:
                ranking = _compute_ranking(
                    item["candidates"][source],
                    src_evals.get("rubrics") or {},
                    judgements,
                )
                src_evals["pair_ranking"] = ranking
                src_evals["ranking_method"] = RANKING_METHOD
                items_finalised.add(item_idx)
            except Exception as e:
                failed += 1
                print(f"  [ranking warn] item_idx={item_idx}: {e}", flush=True)

    save_json_atomic(items, master_path)
    print(f"Done. Wrote {master_path}. Items with final ranking: {len(items_finalised)}")

    print(f"Completed={completed}, failed={failed}")
    if failed or skipped_no_candidates or skipped_no_rubrics:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
