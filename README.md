# Aligning Gemma 3 for Serbian Medical Conversations

Can SFT and DPO bring smaller Gemma 3 models closer to a 27B instruction-tuned
baseline on a Serbian medical dialogue task? This project trains models to collect
information from parents of newborns before a physician takes over. Given the
conversation history, the assistant should ask relevant questions without offering
medical advice or diagnoses.

## Scope and safety

This repository contains research code, prompts, and evaluation settings for a
medical-dialogue intake assistant. It is **not** a clinical product and must not
be used for diagnosis, triage decisions, medical advice, or autonomous patient
care. The assistant prompt explicitly instructs the model to collect information
only and to defer medical decisions to qualified professionals. Human review by
qualified clinicians is required before any real-world or patient-facing use.

The dataset consists of sensitive Serbian call-center conversations and is not
currently public. Handle any local copy of `dataset/`, `data/`, `experiments/`,
and generated artifacts as sensitive data; do not commit them.

## License

No repository license file is included at the time of this audit. Before reuse,
redistribution, or downstream deployment, confirm the rights and permissions for
the code, prompts, model weights referenced by configs, and any private dataset.

## Experiments

The dataset consists of Serbian call-center conversations, with recorded nurse
responses as training targets. An 85/15 split by conversation keeps turns from the
same call together; all models use the same validation set. The dataset is not
currently public and is not included in this repository.

The main comparison starts from Gemma 3 instruction-tuned (IT) checkpoints:

| Starting checkpoint | Baseline config | SFT config | DPO config |
| --- | --- | --- | --- |
| Gemma 3 1B IT | [1B](configs/base/1b.yaml) | [LoRA](configs/sft/1b.yaml) | [DPO](configs/dpo/1b.yaml) |
| Gemma 3 12B IT | [12B](configs/base/12b.yaml) | [QLoRA](configs/sft/12b-it.yaml) | [DPO](configs/dpo/12b-it.yaml) |
| Gemma 3 27B IT | [27B](configs/base/27b.yaml) | — | — |

Baselines use the same [system prompt](prompts/assistant/baseline.md) without
training on this corpus. An additional 12B pretrained (PT) branch has
[SFT](configs/sft/12b-pt.yaml) and [DPO](configs/dpo/12b-pt.yaml) configs for
comparison with the IT starting point; it is not included in the tables below.

SFT learns the recorded responses using LoRA at 1B and QLoRA at 12B. Each SFT model
then generates four candidates per training example. Qwen3.5-122B scores the
candidates against task-specific rubrics and compares them pairwise, swapping
response order to reduce position bias. The resulting preference pairs train the
SFT adapter with DPO, using a frozen SFT reference model and an additional
supervised loss on preferred responses.

Evaluation combines reference-based metrics with the same judge's scores for
[four rubrics](prompts/judge/rubrics): safety and role adherence, clinical
usefulness for handoff, contextual relevance, and clarity and conciseness.
[Evaluation settings](configs/eval.yaml) specify ROUGE, BLEU, multilingual
BERTScore, and E5 and ModernBERTić embedding similarity.

## Results

On 315 validation examples, SFT improves both reference-based metrics and judge
scores. DPO further improves judge scores, with a small decrease in reference-based
metrics relative to SFT. The overall judge score averages the four rubrics (1–3).

| Gemma 3 variant | ROUGE-1 | BERTScore F1 | Judge score (1–3) |
| --- | ---: | ---: | ---: |
| 1B IT baseline | 0.150 | 0.682 | 1.31 |
| 1B IT → SFT | 0.287 | 0.738 | 1.94 |
| 1B IT → SFT → DPO | 0.265 | 0.735 | 2.12 |
| 12B IT baseline | 0.153 | 0.670 | 2.47 |
| 12B IT → SFT | **0.306** | **0.749** | 2.57 |
| 12B IT → SFT → DPO | 0.286 | 0.741 | **2.77** |
| 27B IT baseline | 0.160 | 0.678 | 2.68 |

Mean scores for each rubric (1–3; higher is better):

| Gemma 3 variant | Safety and role adherence | Clinical usefulness | Contextual relevance | Clarity and conciseness |
| --- | ---: | ---: | ---: | ---: |
| 1B IT baseline | 1.40 | 1.29 | 1.38 | 1.18 |
| 1B IT → SFT | 2.16 | 1.86 | 1.98 | 1.76 |
| 1B IT → SFT → DPO | 2.35 | 1.95 | 2.10 | 2.09 |
| 12B IT baseline | 2.69 | 2.57 | 2.47 | 2.14 |
| 12B IT → SFT | 2.70 | 2.43 | 2.49 | 2.68 |
| 12B IT → SFT → DPO | **2.84** | 2.68 | **2.72** | **2.82** |
| 27B IT baseline | 2.79 | **2.72** | 2.68 | 2.53 |

The 12B SFT model approaches the 27B IT baseline's overall judge score (2.57 vs.
2.68). After DPO, it scores highest overall (2.77), while the 27B IT baseline
retains the highest clinical-usefulness score (2.72 vs. 2.68).

These are validation results: the same split also guides SFT checkpoint selection.
The judge supplies both training preferences and evaluation scores, so the DPO
gains need independent human evaluation before drawing broader conclusions.

Analysis: [model evaluation](notebooks/results.ipynb) and
[metric correlations](notebooks/metric_correlations.ipynb).
Response examples: [1B](notebooks/overview_1b.ipynb),
[12B PT](notebooks/overview_12b.ipynb), and
[cross-model comparison](notebooks/overview_1b_12b_27b.ipynb).

## Project structure

```text
configs/     Model, training, and evaluation settings
prompts/     Assistant system prompt and judge rubrics
scripts/     Commands for data preparation, training, generation, and evaluation
src/         Shared implementation used by the scripts
notebooks/   Results analysis and plots
run.sh       Runs the full experiment pipeline
```

Private data and generated outputs live in `dataset/`, `data/`, and
`experiments/`; these directories are not tracked in Git.

## Running

You need access to the private dataset and the Hugging Face models in the configs,
a CUDA training environment, and a local Ollama server for Qwen3.5-122B and the
27B IT baseline. The documented hardware setup uses A100 for training and H200
for the judge. Training logs to W&B; authenticate or set `report_to: none` in the
training configs.

Install dependencies in an isolated environment (virtualenv/conda) after reviewing
the version ranges in `requirements.txt`. Pinning exact versions in a lock file is
recommended for reproducible runs. Authenticate with Hugging Face after obtaining
model access, and download the Ollama models:

```bash
python3 -m pip install -r requirements.txt
hf auth login
ollama pull qwen3.5:122b
ollama pull gemma3:27b-it-qat
```

Place the conversations in `dataset/json_ser/` and start Ollama with
`ollama serve`. Local Ollama endpoints are hardcoded to `127.0.0.1` in code; if
your setup differs, update the endpoint in source/config consistently.
[run.sh](run.sh) runs data preparation, all configured training and generation
runs, and joint evaluation:

```bash
bash run.sh
```

To run only some experiments, name them by their config under `configs/`. Runs
execute in the given order, so list an SFT run before its DPO run. For example,
1B SFT and its evaluation:

```bash
bash run.sh sft/1b
```

Outputs are saved under `experiments/`, with joint metrics in `experiments/eval/`.
Training resumes from saved checkpoints. Use a new `output_dir` when changing
configs or prompts.

## Verification status

Local static checks in this audit verified Python syntax (`python -m py_compile`)
for modified modules and `src.artifacts` import in the local environment. End-to-end
runs were **not** executed here because the private dataset, CUDA training stack,
Ollama models, and heavyweight ML dependencies were unavailable.
