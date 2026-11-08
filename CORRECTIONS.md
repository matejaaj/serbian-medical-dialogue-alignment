# Corrections — serbian-medical-dialogue-alignment

Base: `matejaaj/serbian-medical-dialogue-alignment@a214013` (main). User authorized implementation of all audit proposals.

## Implemented

1. **Centralized artifact JSON loading with diagnostics** (`src/artifacts.py`, scripts, `src/dataset.py`, `src/dpo.py`)
   - Added `load_json(path)` to report file path + underlying error.
   - Replaced bare `json.load`/`json.loads(path.read_text())` in audit-targeted pipeline scripts and core loaders.

2. **Merge predictions validation** (`scripts/merge_predictions.py`)
   - Validates required `model` and `predictions` keys; raises actionable `ValueError` naming the input file.

3. **Configurable Ollama generation endpoint** (`src/generation.py`)
   - `OllamaGenerator` now reads `model.url` or `generation.ollama_url`, default `http://127.0.0.1:11434/api/generate`.

4. **Hardened malformed judge JSON recovery** (`src/judge.py`)
   - Added last-match quoted-field fallback for `ocena`, `izbor`, `analiza` when JSON parsing fails.
   - Preserves fail-closed behavior: unknown normalized rubric labels / pairwise choices still raise `ValueError`.

5. **Documentation and safety guidance** (`README.md`)
   - Added Scope and safety (non-clinical, no diagnosis/triage/advice, sensitive data handling).
   - Added License status warning (no LICENSE file found).
   - Added dependency reproducibility/lock-file recommendation and Ollama endpoint note.
   - Added verification status (static checks only; no E2E runs).

6. **Dependency metadata** (`requirements.txt`)
   - Added conservative lower bounds for previously unpinned packages and a reproducibility note.

7. **Secrets hygiene** (`.env.example`)
   - Added explicit “do not commit secrets” and key/value guidance.

## Not changed (owner/legal/infra decisions)

- **LICENSE file not added** (REA-001): requires rights/owner decision.
- **Exact dependency lock file not generated** (REA-002): requires target environment resolution.
- **No CI/tests added** (REA-007): would require fixtures/mocks for private data and heavyweight ML/Ollama dependencies; documented instead.
- **Hardcoded Ollama generation URL not moved into all configs**: code now supports config override; existing configs left unchanged to avoid behavior drift.
