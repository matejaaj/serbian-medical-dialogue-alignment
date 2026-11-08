# Serbian Medical Dialogue Alignment — Reverse Engineering / Audit

## Scope
- **Repository:** https://github.com/matejaaj/serbian-medical-dialogue-alignment
- **Base (Phase 1):** `a214013 Initial commit` (`main`), local clone `C:\Users\User\audit-repos\serbian-medical-dialogue-alignment`.
- **Mode:** Phase 1 static audit (read/inspect only, no execution of training/judging pipelines); Phase 2 fixes implemented with explicit user authorization (“implement all proposals found”).
- **Constraints:** Private dataset, CUDA/GPU stack, Ollama models, Hugging Face weights, and full ML dependencies were not available in this environment. Findings are static + limited local syntax/import checks.

## Phase 1 Findings (summary)
| ID | Sev | Status | Area | Summary |
|---|---|---|---|---|
| REA-001 | P2 | CONFIRMED | Legal/reuse | No `LICENSE` file; reuse/redistribution/deployment rights unclear. |
| REA-002 | P2 | CONFIRMED | Reproducibility | `requirements.txt` had no lower bounds/pins (except `trl>=0.24.0`) and no lock file. |
| REA-003 | P3 | CONFIRMED | Robustness | `scripts/merge_predictions.py` raised low-diagnostic `KeyError`/type errors on malformed prediction files. |
| REA-004 | P3 | IMPROVED | Resume/diagnostics | Several scripts loaded JSON artifacts with bare `json.load`; corrupt/incompatible files produced low-context errors. |
| REA-005 | P3 | CONFIRMED (partial fix) | Configurability | Ollama generation endpoint was hardcoded in `src/generation.py` while judge endpoint was configurable; config/docs did not state this clearly. |
| REA-006 | P3 | CONFIRMED (mitigated) | Secrets hygiene | `.env`/variants ignored and `.env.example` present, but example included `export` and no explicit “do not commit secrets” note. |
| REA-007 | P2 | CONFIRMED (documented) | Verification | No automated tests/CI; pipeline requires private data + heavyweight infra. |
| REA-008 | — | DESCARTADO (risk) | YAML parsing | `src/config.py` uses `yaml.safe_load` (not unsafe load). |
| REA-009 | P2 | CONFIRMED (documented) | Medical domain | Research medical-dialogue code lacks explicit non-clinical/disclaimer/sensitive-data handling guidance in README. |
| REA-010 | P3 | PROBABLE (hardened) | Judge parsing | Judge JSON parsing used regex fallbacks; first quoted-field matches could be brittle when text mentions schema keys. Malformed/unexpected outputs still fail closed for invalid labels/choices. |

No P0/P1 security vulnerabilities were proven from static inspection.

## Phase 2 Changes Implemented
- **`src/artifacts.py`**: added `load_json(path)` with path-aware error context for OSError/JSON decode/value errors.
- **Scripts hardened to use `load_json`**: `merge_predictions.py`, `generate_candidates.py`, `score_candidates.py`, `rank_candidates.py`, `build_dpo_dataset.py`, `judge.py`, `aggregate_results.py`, `eval_reference.py`.
- **`src/dataset.py`, `src/dpo.py`**: use `load_json` for split/pairs loading.
- **`scripts/merge_predictions.py`**: explicit validation for required `model`/`predictions` keys with actionable `ValueError`.
- **`src/generation.py`**: `OllamaGenerator` accepts `model.url` or `generation.ollama_url`, defaulting to `http://127.0.0.1:11434/api/generate`.
- **`src/judge.py`**: added last-match quoted-field fallback for malformed JSON parsing (`ocena`/`izbor`/`analiza`) to reduce false extraction from explanatory text; behavior remains fail-closed for unrecognized normalized labels/choices.
- **`README.md`**: added “Scope and safety” (non-clinical, sensitive data handling), “License” status warning, dependency reproducibility note, Ollama endpoint note, and “Verification status”.
- **`requirements.txt`**: added conservative lower bounds for previously unbounded packages and a reproducibility note (still not a lock file).
- **`.env.example`**: clarified key/value format and “do not commit secrets”.

## Validation Performed
- `python -m py_compile` passed for modified modules: `src/artifacts.py`, `src/dataset.py`, `src/dpo.py`, `src/judge.py`, `scripts/merge_predictions.py`, `scripts/generate_candidates.py`, `scripts/score_candidates.py`, `scripts/rank_candidates.py`, `scripts/build_dpo_dataset.py`, `scripts/judge.py`, `scripts/aggregate_results.py`, `scripts/eval_reference.py`.
- Local import smoke for `src.artifacts` in the available Python environment succeeded only after path setup; full package/runtime imports were not exercised due to missing ML dependencies.

## Not Executed / Remaining Risks
- No end-to-end training, candidate generation, DPO, or judge runs (missing dataset, GPU/CUDA, Ollama, HF models, dependencies).
- No addition of a LICENSE (legal/owner decision required).
- No lock file generation (would require dependency resolution in target environment).
- Judge parsing remains best-effort for malformed model output; strict schema-conformant responses are still preferred.
- Medical safety requires human clinical review and governance beyond code changes.
