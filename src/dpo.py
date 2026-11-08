from __future__ import annotations

from pathlib import Path

from . import config, training as training_mod
from .artifacts import load_json


def prepare_dpo_dataset(pairs_path: Path, validation_fraction: float, split_seed: int):
    """Return train and eval pairs as conversations; TRL applies the chat template."""
    from datasets import Dataset

    payload = load_json(pairs_path)
    rows = []
    for pair in payload["pairs"]:
        rows.append({
            # Saved input_messages already include the system prompt.
            "prompt": pair["input_messages"],
            "chosen": [{"role": "assistant", "content": pair["chosen"]}],
            "rejected": [{"role": "assistant", "content": pair["rejected"]}],
        })

    split = Dataset.from_list(rows).train_test_split(test_size=validation_fraction, seed=split_seed)
    print(f"[dpo] loaded {len(rows)} pairs from {pairs_path} -> train={len(split['train'])} eval={len(split['test'])}")
    return split["train"], split["test"]


def _load_sft_model_for_dpo(model: dict, sft_dir: Path, gradient_checkpointing: bool):
    """Load the SFT adapter with gradients enabled for quantized, checkpointed training."""
    from peft import PeftModel

    base = training_mod.load_base_model(model)
    if model["quantization"] == "4bit":
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=gradient_checkpointing)
    trainable = PeftModel.from_pretrained(base, str(sft_dir), is_trainable=True)
    trainable.enable_input_require_grads()
    return trainable


def _load_sft_reference(model: dict, sft_dir: Path):
    from peft import PeftModel

    base = training_mod.load_base_model(model, for_training=False)
    ref = PeftModel.from_pretrained(base, str(sft_dir))
    ref.eval()
    return ref


def train_dpo(cfg: dict) -> None:
    from transformers import set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import DPOConfig, DPOTrainer

    if cfg["reference"] not in ("sft", "base"):
        raise ValueError(f"reference must be 'sft' or 'base', not {cfg['reference']!r}")
    out = config.prepare_output_dir(cfg)
    setup = config.model_setup(cfg)
    sft_dir = config.REPO_ROOT / cfg["sft_dir"]
    set_seed(cfg["seed"])
    tokenizer = training_mod.load_tokenizer(setup["model"]["tokenizer"])
    train_ds, eval_ds = prepare_dpo_dataset(out / "dpo_pairs.json", cfg["validation_fraction"], cfg["split_seed"])
    model = _load_sft_model_for_dpo(setup["model"], sft_dir, cfg["training"]["gradient_checkpointing"])
    # "sft" uses a frozen copy of the SFT adapter as reference; "base" uses the adapter-disabled base model.
    ref_model = _load_sft_reference(setup["model"], sft_dir) if cfg["reference"] == "sft" else None
    # rpo_alpha weights the chosen-answer NLL added to the DPO loss; 0 disables it.
    alpha = cfg["rpo_alpha"]
    loss = {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, alpha]} if alpha else {"loss_type": ["sigmoid"]}
    trainer = DPOTrainer(
        model=model, ref_model=ref_model,
        args=DPOConfig(**cfg["training"], **loss, seed=cfg["seed"], output_dir=str(out)),
        train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(out)))
    trainer.save_model()
    tokenizer.save_pretrained(str(out))
    print("Saved to:", out)
