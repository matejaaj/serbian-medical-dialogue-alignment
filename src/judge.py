from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

Label = Literal["nezadovoljavajuce", "zadovoljavajuce", "veoma_dobro"]

PairwiseChoice = Literal["A", "B", "tie"]


@dataclass(frozen=True)
class Rubric:
    key: str
    prompt_filename: str


RUBRICS: list[Rubric] = [
    Rubric("bezbednost_i_pridrzavanje_uloge", "bezbednost_i_pridrzavanje_uloge.md"),
    Rubric("klinicka_korisnost_za_prosledjivanje", "klinicka_korisnost_za_prosledjivanje.md"),
    Rubric("uskladjenost_sa_kontekstom_i_fazom_razgovora", "uskladjenost_sa_kontekstom_i_fazom_razgovora.md"),
    Rubric("jasnoca_i_ekonomicnost_komunikacije", "jasnoca_i_ekonomicnost_komunikacije.md"),
]


def load_rubric_prompts(prompts_dir: str | Path) -> dict[str, str]:
    prompts_dir = Path(prompts_dir)
    templates = {}
    for rubric in RUBRICS:
        templates[rubric.key] = (prompts_dir / rubric.prompt_filename).read_text(encoding="utf-8")
    return templates


def load_pairwise_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_prompt(template: str, input_messages, prediction: str) -> str:
    import json
    context_str = json.dumps(input_messages, ensure_ascii=False, indent=2)
    return template.replace("{input_messages}", context_str).replace("{prediction}", prediction)


def build_pairwise_prompt(
    template: str,
    input_messages,
    response_a: str,
    response_b: str,
    rubric_summaries_a: str = "",
    rubric_summaries_b: str = "",
) -> str:
    import json
    context_str = json.dumps(input_messages, ensure_ascii=False, indent=2)
    return (
        template
        .replace("{input_messages}", context_str)
        .replace("{response_a}", response_a)
        .replace("{response_b}", response_b)
        .replace("{rubric_summaries_a}", rubric_summaries_a)
        .replace("{rubric_summaries_b}", rubric_summaries_b)
    )


def prompt_hash(prompt: str) -> str:
    import hashlib
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]


def judgement_key(pair: tuple[int, int], round_id: int) -> str:
    a, b = pair
    return f"{a}-{b}:r{round_id}"


def pair_winner(judgements: dict[str, dict], pair: tuple[int, int]) -> int | None:
    """Candidate that won the pair in both A/B orders, else None; a missing round raises KeyError."""
    a, b = sorted(pair)
    # Round 1 shows a as A, round 2 shows b as A.
    choices = (judgements[judgement_key((a, b), 1)]["choice"], judgements[judgement_key((a, b), 2)]["choice"])
    if choices == ("A", "B"):
        return a
    if choices == ("B", "A"):
        return b
    return None


# Ordinal labels from lowest to highest.
LABELS_ORDERED: list[str] = list(get_args(Label))
LABEL_SCORE: dict[str, int] = {label: i + 1 for i, label in enumerate(LABELS_ORDERED)}

# Normalize Cyrillic lookalikes in otherwise Latin labels.
_LABEL_CYR2LAT = str.maketrans({
    "а": "a", "е": "e", "о": "o", "с": "c", "ј": "j", "у": "u", "р": "p",
    "х": "x", "к": "k", "в": "v", "н": "n", "м": "m", "т": "t", "і": "i",
})


def normalize_label(raw) -> str | None:
    """Normalize spelling and homoglyphs; return None for unknown labels."""
    if not isinstance(raw, str):
        return None
    import difflib
    s = raw.strip().lower().translate(_LABEL_CYR2LAT).replace(" ", "_").replace("-", "_")
    if s in LABELS_ORDERED:
        return s
    match = difflib.get_close_matches(s, LABELS_ORDERED, n=1, cutoff=0.6)
    return match[0] if match else None


def format_rubric_summary(rubric_scores: dict) -> str:
    if not rubric_scores:
        return "(nema rubrik-ocena)"
    lines = []
    for rubric in RUBRICS:
        entry = rubric_scores.get(rubric.key) or {}
        label = entry.get("label") or "?"
        reason = entry.get("reason") or ""
        lines.append(f"- {rubric.key}: {label} ({reason})")
    return "\n".join(lines)


class OllamaBackend:

    RUBRIC_SCHEMA = {
        "type": "object",
        "properties": {
            "ocena": {"type": "string", "enum": list(get_args(Label))},
            "obrazlozenje": {"type": "string"},
        },
        "required": ["ocena", "obrazlozenje"],
    }
    PAIRWISE_SCHEMA = {
        "type": "object",
        "properties": {
            # Request analysis before the verdict.
            "analiza": {"type": "string"},
            "izbor": {"type": "string", "enum": list(get_args(PairwiseChoice))},
        },
        "required": ["analiza", "izbor"],
    }

    def __init__(
        self,
        model_name: str,
        url: str,
        options: dict,
        max_retries: int = 3,
        timeout: int = 600,
    ):
        import requests

        self.model_name = model_name
        self.url = url
        self.max_retries = max_retries
        self.timeout = timeout
        self.options = options

        self.session = requests.Session()
        self.session.trust_env = False

    def _build_payload(self, prompt: str, schema: dict) -> dict:
        # Use /api/chat so think=False disables reasoning tokens.
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema,
            "think": False,
            "keep_alive": "30m",
            "options": self.options,
        }

    @staticmethod
    def _ns_to_ms(ns) -> int | None:
        if ns is None:
            return None
        try:
            return int(round(int(ns) / 1_000_000))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_text(raw: dict) -> str:
        message = raw.get("message") or {}
        return message.get("content") or ""

    def _clean_response(self, raw: dict) -> dict:
        return {
            "model": raw.get("model"),
            "created_at": raw.get("created_at"),
            "response": self._extract_text(raw),
            "done_reason": raw.get("done_reason"),
            "total_duration_ms": self._ns_to_ms(raw.get("total_duration")),
            "prompt_eval_count": raw.get("prompt_eval_count"),
            "prompt_eval_duration_ms": self._ns_to_ms(raw.get("prompt_eval_duration")),
            "eval_count": raw.get("eval_count"),
            "eval_duration_ms": self._ns_to_ms(raw.get("eval_duration")),
        }

    def _call(self, prompt: str, schema: dict) -> dict:
        """Retry failed HTTP or JSON responses with exponential backoff."""
        import time
        from requests import RequestException

        payload = self._build_payload(prompt, schema)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    sleep_s = 2 ** attempt
                    print(f"[judge] attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                          f"retrying in {sleep_s}s...", flush=True)
                    time.sleep(sleep_s)
        raise RuntimeError(
            f"Judge call failed after {self.max_retries} attempts. Last error: {last_err}"
        )

    @staticmethod
    def _extract_field(raw_text: str, field: str) -> str | None:
        """Recover a quoted field from malformed JSON."""
        import re
        m = re.search(r'"' + re.escape(field) + r'"\s*:\s*"([^"]*)"', raw_text)
        return m.group(1) if m else None

    @staticmethod
    def _extract_obrazlozenje(raw_text: str) -> str:
        """Recover the trailing explanation when its inner quotes are unescaped."""
        import re
        m = re.search(r'"obrazlozenje"\s*:\s*"(.*)"\s*\}?\s*$', raw_text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_last_quoted_field(raw_text: str, field: str) -> str | None:
        """Recover a quoted field by taking the last match (useful when text mentions the key)."""
        import re
        matches = re.findall(r'"' + re.escape(field) + r'"\s*:\s*"([^"]*)"', raw_text)
        return matches[-1] if matches else None

    def _parse_response(self, raw_text: str) -> tuple[str, str]:
        """Return raw label and reason, falling back to regex for malformed JSON."""
        import json

        try:
            parsed = json.loads(raw_text)
            label = parsed.get("ocena") if isinstance(parsed, dict) else None
            reason = parsed.get("obrazlozenje", "") if isinstance(parsed, dict) else ""
        except json.JSONDecodeError:
            label = self._extract_last_quoted_field(raw_text, "ocena") or self._extract_field(raw_text, "ocena")
            reason = self._extract_obrazlozenje(raw_text)

        label = label.strip() if isinstance(label, str) else ""
        reason = reason if isinstance(reason, str) else str(reason)
        return label, reason

    def _signature(self, prompt: str, schema: dict) -> str:
        """Hash of the full request; a saved result is reused only for the same request."""
        import json

        payload = self._build_payload(prompt, schema)
        payload.pop("keep_alive")
        return prompt_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def rubric_signature(self, prompt: str) -> str:
        return self._signature(prompt, self.RUBRIC_SCHEMA)

    def pairwise_signature(self, prompt: str) -> str:
        return self._signature(prompt, self.PAIRWISE_SCHEMA)

    def score(self, prompt: str) -> dict:
        """Rubric result with the normalized label; the raw answer stays in raw_response."""
        raw = self._call(prompt, self.RUBRIC_SCHEMA)
        label, reason = self._parse_response(self._extract_text(raw))
        normalized = normalize_label(label)
        if normalized is None:
            raise ValueError(f"Judge returned an unrecognized rubric label: {label!r}")
        return {"label": normalized, "reason": reason, "signature": self.rubric_signature(prompt),
                "raw_response": self._clean_response(raw)}

    def _parse_pairwise(self, raw_text: str) -> tuple[str, str]:
        """Return raw choice and analysis, falling back to regex for malformed JSON."""
        import json

        try:
            parsed = json.loads(raw_text)
            choice = parsed.get("izbor") if isinstance(parsed, dict) else None
            reason = parsed.get("analiza", "") if isinstance(parsed, dict) else ""
        except json.JSONDecodeError:
            choice = self._extract_last_quoted_field(raw_text, "izbor") or self._extract_field(raw_text, "izbor")
            reason = self._extract_last_quoted_field(raw_text, "analiza") or self._extract_field(raw_text, "analiza") or ""

        choice = choice.strip() if isinstance(choice, str) else ""
        reason = reason if isinstance(reason, str) else str(reason)
        return choice, reason

    def compare(self, prompt: str) -> dict:
        raw = self._call(prompt, self.PAIRWISE_SCHEMA)
        choice, reason = self._parse_pairwise(self._extract_text(raw))
        normalized = {"A": "A", "B": "B", "TIE": "tie"}.get(choice.upper())
        if normalized is None:
            raise ValueError(f"Judge returned an unrecognized pairwise choice: {choice!r}")
        return {"choice": normalized, "reason": reason, "signature": self.pairwise_signature(prompt),
                "raw_response": self._clean_response(raw)}
