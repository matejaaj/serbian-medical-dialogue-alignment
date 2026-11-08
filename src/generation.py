from __future__ import annotations

from . import conversation, dataset as ds_mod, training


def _check_prompt_length(num_tokens: int, max_input_length: int) -> None:
    if num_tokens > max_input_length:
        raise ValueError(f"Prompt has {num_tokens} tokens, more than max_input_length={max_input_length}")


class HFGenerator:

    def __init__(self, model: dict, generation: dict, *, seed: int, adapter=None):
        from transformers import AutoTokenizer

        self.generation = generation
        self.seed = seed
        self.model_dir = str(adapter) if adapter else None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir or model["tokenizer"], use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stop_token_ids = [self.tokenizer.eos_token_id]
        eot = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot >= 0 and eot not in self.stop_token_ids:
            self.stop_token_ids.append(eot)

        base_model = training.load_base_model(model, for_training=False)
        if self.model_dir:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, self.model_dir)
        else:
            self.model = base_model
        self.model.eval()

    def generate(self, input_messages: list[dict]) -> str:
        return self._generate_one(
            input_messages,
            do_sample=self.generation["do_sample"],
            temperature=self.generation["temperature"],
            top_p=self.generation["top_p"],
            seed=self.seed,
        )["text"]

    def _generate_one(
        self,
        input_messages: list[dict],
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> dict:
        import torch

        prompt = conversation.messages_to_text(
            input_messages, self.tokenizer, add_generation_prompt=True)
        # The chat template already starts with BOS.
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        _check_prompt_length(inputs["input_ids"].shape[1], self.generation["max_input_length"])

        gen_kwargs = {
            "max_new_tokens": self.generation["max_new_tokens"],
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.stop_token_ids,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            from transformers import set_seed
            set_seed(seed)

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        gen = output[:, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return {"text": text, "generated_tokens": int(gen.shape[1])}

    def generate_candidates(
        self,
        input_messages: list[dict],
        seeds: list[int],
        temperature: float,
        top_p: float,
    ) -> list[dict]:
        """Sample candidates with separate seeds, returning text, token count and seed."""
        results = []
        for seed in seeds:
            out = self._generate_one(
                input_messages, do_sample=True,
                temperature=temperature, top_p=top_p, seed=seed,
            )
            results.append({
                "text": out["text"],
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "generated_tokens": out["generated_tokens"],
            })
        return results


class OllamaGenerator:

    def __init__(self, model: dict, generation: dict, *, seed: int):
        import requests
        from transformers import AutoTokenizer

        self.model_name = model["weights"]
        self.generation = generation
        self.seed = seed
        self.url = model.get("url") or generation.get("ollama_url") or "http://127.0.0.1:11434/api/generate"
        # Prompts are rendered with the HF chat template, as for HF models, and sent raw.
        self.tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        self.session = requests.Session()
        self.session.trust_env = False

    def generate(self, input_messages: list[dict]) -> str:
        prompt = conversation.messages_to_text(input_messages, self.tokenizer, add_generation_prompt=True)
        _check_prompt_length(len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                             self.generation["max_input_length"])
        payload = {
            "model": self.model_name,
            # Ollama adds BOS when it tokenizes a raw prompt.
            "prompt": prompt.removeprefix(self.tokenizer.bos_token),
            "raw": True,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": self.generation["temperature"] if self.generation["do_sample"] else 0.0,
                "top_p": self.generation["top_p"],
                "num_predict": self.generation["max_new_tokens"],
                "seed": self.seed,
                # Ollama silently drops the start of a prompt that exceeds its context window.
                "num_ctx": self.generation["max_input_length"] + self.generation["max_new_tokens"],
            },
        }
        response = self.session.post(self.url, json=payload, timeout=600)
        response.raise_for_status()
        return response.json()["response"].strip()


def build_generator(setup: dict, *, seed: int, adapter=None):
    """Ollama models are listed with backend: ollama; everything else loads through HF."""
    if setup["model"].get("backend") == "ollama":
        return OllamaGenerator(setup["model"], setup["generation"], seed=seed)
    return HFGenerator(setup["model"], setup["generation"], seed=seed, adapter=adapter)


def run_generation(setup: dict, seed: int) -> list[dict]:
    from tqdm.auto import tqdm

    _, val = ds_mod.load_split()
    generator = build_generator(setup, seed=seed, adapter=setup["adapter"])

    rows = []
    for sample_idx, example in enumerate(tqdm(val, desc="Generating")):
        input_messages = conversation.with_system(example["input"], setup["system_prompt"])
        rows.append({
            "sample_idx": sample_idx,
            "call_id": example["call_id"],
            "turn_id": example["turn_id"],
            "input_messages": input_messages,
            "ground_truth": example["output"]["content"],
            "prediction": generator.generate(input_messages),
        })
    return rows
