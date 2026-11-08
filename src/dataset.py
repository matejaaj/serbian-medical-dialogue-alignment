from __future__ import annotations

import random
from pathlib import Path

from . import config
from .artifacts import load_json


SPEAKER_ROLES = {"POZIVALAC": "user", "SESTRA": "assistant"}


def to_chat_format(row: dict) -> dict:
    new_transcription = []
    for turn in row["transcription"]:
        if turn["speaker"] not in SPEAKER_ROLES:
            raise ValueError(f"{row['call_id']}: unknown speaker {turn['speaker']!r}")
        new_transcription.append(dict(role=SPEAKER_ROLES[turn["speaker"]], content=turn["text"]))
    row["transcription"] = new_transcription
    return row


def ensure_user_starts(row: dict) -> dict:
    """Prepend an empty user turn when the assistant speaks first."""
    empty_msg = dict(role="user", content="")
    if row["transcription"][0]["role"] != "user":
        row["transcription"].insert(0, empty_msg)
    return row


def create_assistant_turns_dataset(dataset) -> "Dataset":
    """Create one example per assistant turn, using preceding turns as context."""
    from datasets import Dataset

    all_assistant_turns = []
    for conversation in dataset:
        call_id = conversation.get("call_id")
        transcription = conversation["transcription"]
        for turn_index, turn in enumerate(transcription):
            if turn["role"] != "assistant":
                continue
            all_assistant_turns.append({
                "input": transcription[:turn_index],
                "output": turn,
                "call_id": call_id,
                "turn_id": turn_index,
            })
    return Dataset.from_list(all_assistant_turns)


def validate_role_order(row: dict) -> dict:
    """Require the user/assistant alternation that the Gemma chat template enforces."""
    for i, turn in enumerate(row["transcription"]):
        expected = "user" if i % 2 == 0 else "assistant"
        if turn["role"] != expected:
            raise ValueError(f"{row['call_id']}: turn {i} is {turn['role']}, expected {expected}")
    return row


def build_hf_dataset(raw_dir: str | Path, out_dir: str | Path) -> "Dataset":
    from datasets import load_dataset

    dataset = load_dataset(
        "json", split="train", data_files=str(Path(raw_dir) / "*.json"), encoding="utf-8"
    )
    dataset = dataset.map(to_chat_format)
    dataset = dataset.map(ensure_user_starts)
    dataset = dataset.map(validate_role_order)
    dataset = create_assistant_turns_dataset(dataset)

    dataset.save_to_disk(str(out_dir))
    print(f"Saved {len(dataset)} examples to: {out_dir}")
    return dataset


def split_call_ids(call_ids, *, seed: int, train_fraction: float) -> dict:
    ids = sorted(set(call_ids))
    random.Random(seed).shuffle(ids)
    boundary = int(len(ids) * train_fraction)
    return {"train": ids[:boundary], "validation": ids[boundary:]}


def load_split():
    """Train and validation examples; one saved split by call_id is shared by every run."""
    from datasets import load_from_disk

    dataset = load_from_disk(str(config.DATASET_DIR))
    split = load_json(config.SPLIT_PATH)
    train_ids, val_ids = set(split["train"]), set(split["validation"])
    if train_ids & val_ids or train_ids | val_ids != set(dataset["call_id"]):
        raise ValueError("Saved split does not match the dataset or contains overlapping calls")
    return (dataset.filter(lambda row: row["call_id"] in train_ids),
            dataset.filter(lambda row: row["call_id"] in val_ids))
