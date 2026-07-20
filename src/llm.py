"""Gemini via Vertex AI (Express mode), plus token/cost accounting.

Called over raw HTTP rather than through a client library: Express-mode keys
authenticate with `x-goog-api-key` against the publisher-model endpoint, which
is a narrower surface than the full Vertex SDK expects (no project/location,
no ADC). One small module is clearer than fighting an SDK into that shape.

Thinking is controlled per call. Every Gemini 2.5+ model reasons before
answering — a trivial "reply ok" prompt burned 98 thought tokens on
`gemini-3.5-flash` — so leaving it dynamic on cheap classifier steps would
dominate their cost. Grading and routing run with thinking off; synthesis
runs with it dynamic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

VERTEX_BASE = "https://aiplatform.googleapis.com/v1beta1/publishers/google/models"

# gemini-3.5-flash is the most capable Gemini currently released — 3.5-pro
# does not exist yet (it 404s on this project). One model everywhere; the
# cost dial is the thinking budget, not a cheaper model. Fewer moving parts,
# and no reason to drop to an older tier when the newest flash is the ceiling.
MODEL_SYNTHESIS = "gemini-3.5-flash"
MODEL_CHEAP = "gemini-3.5-flash"

# Thinking budgets. 0 disables reasoning on flash models; -1 lets the model
# decide. These are the analogue of an "effort" dial.
THINK_OFF = 0
THINK_DYNAMIC = -1

# Rates are configuration, not measurement. Token counts below are exact
# (reported by the API); the money figure is only as right as these numbers,
# so they live in one place and are labelled everywhere they surface.
RATES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "gemini-3.5-flash": {"input": 0.30, "output": 2.50, "cached": 0.075},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached": 0.075},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached": 0.31},
}
DEFAULT_RATE = {"input": 0.30, "output": 2.50, "cached": 0.075}


class GeminiError(RuntimeError):
    pass


@dataclass
class Usage:
    """Token accounting accumulated across a request.

    `thought_tokens` is tracked separately because it is the line item that
    surprises people: it is billed as output but never appears in the answer.
    """

    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    per_step: list[dict[str, Any]] = field(default_factory=list)

    def add(self, label: str, model: str, meta: dict[str, Any]) -> None:
        prompt = int(meta.get("promptTokenCount") or 0)
        out = int(meta.get("candidatesTokenCount") or 0)
        thoughts = int(meta.get("thoughtsTokenCount") or 0)
        cached = int(meta.get("cachedContentTokenCount") or 0)

        rate = RATES_USD_PER_MTOK.get(model, DEFAULT_RATE)
        fresh_prompt = max(prompt - cached, 0)
        cost = (
            fresh_prompt * rate["input"]
            + cached * rate["cached"]
            # Thought tokens bill at the output rate.
            + (out + thoughts) * rate["output"]
        ) / 1_000_000

        self.prompt_tokens += prompt
        self.output_tokens += out
        self.thought_tokens += thoughts
        self.cached_tokens += cached
        self.calls += 1
        self.cost_usd += cost
        self.per_step.append(
            {
                "step": label,
                "model": model,
                "prompt_tokens": prompt,
                "output_tokens": out,
                "thought_tokens": thoughts,
                "cached_tokens": cached,
                "cost_usd": round(cost, 6),
            }
        )

    @property
    def cost_cents(self) -> float:
        return self.cost_usd * 100

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thought_tokens": self.thought_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cost_note": "at configured rates; see RATES_USD_PER_MTOK",
            "per_step": self.per_step,
        }


def _post(api_key: str, model: str, body: dict[str, Any], timeout: float = 120.0) -> dict:
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{VERTEX_BASE}/{model}:generateContent",
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
            json=body,
        )
    if response.status_code != 200:
        try:
            message = response.json().get("error", {}).get("message", response.text)
        except Exception:
            message = response.text
        raise GeminiError(f"{model}: HTTP {response.status_code}: {message[:400]}")
    return response.json()


def _extract_text(payload: dict) -> str:
    """Join text parts, skipping thought parts.

    A thinking model can return a candidate whose only parts are thoughts —
    typically when the output budget was consumed before it wrote anything.
    That must read as empty, not crash.
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(
        p.get("text", "") for p in parts if not p.get("thought")
    ).strip()


def generate(
    *,
    api_key: str,
    system: str | None,
    prompt: str,
    model: str = MODEL_SYNTHESIS,
    max_tokens: int = 4096,
    thinking_budget: int = THINK_DYNAMIC,
    schema: dict[str, Any] | None = None,
    label: str = "call",
    usage: Usage | None = None,
    cached_content: str | None = None,
) -> tuple[str, dict]:
    """One model call. Returns (text, raw_payload)."""
    generation_config: dict[str, Any] = {
        "maxOutputTokens": max_tokens,
        "thinkingConfig": {"thinkingBudget": thinking_budget},
    }
    if schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = schema

    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if cached_content:
        body["cachedContent"] = cached_content

    payload = _post(api_key, model, body)
    if usage is not None:
        usage.add(label, model, payload.get("usageMetadata") or {})
    return _extract_text(payload), payload


def generate_json(
    *,
    api_key: str,
    system: str | None,
    prompt: str,
    schema: dict[str, Any],
    model: str = MODEL_CHEAP,
    max_tokens: int = 1024,
    thinking_budget: int = THINK_OFF,
    label: str = "call",
    usage: Usage | None = None,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured call. Falls back rather than raising, so one malformed
    classifier response cannot take down an otherwise answerable request."""
    try:
        text, _ = generate(
            api_key=api_key,
            system=system,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            schema=schema,
            label=label,
            usage=usage,
        )
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, GeminiError):
        if fallback is not None:
            return fallback
        raise


def count_tokens(api_key: str, model: str, text: str) -> int:
    """Exact token count from the API — never a third-party tokenizer."""
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{VERTEX_BASE}/{model}:countTokens",
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": text}]}]},
        )
    if response.status_code != 200:
        raise GeminiError(f"countTokens {model}: HTTP {response.status_code}: {response.text[:200]}")
    return int(response.json().get("totalTokens") or 0)
