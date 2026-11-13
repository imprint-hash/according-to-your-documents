"""Minimal, dependency-free clients for the providers used in the study.

Every call returns (text, meta). Keys are read from ../.env and never printed.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

# model key -> (provider, provider model id)
MODELS = {
    "claude-sonnet-5.5": ("anthropic", "claude-sonnet-5-5"),
    "claude-haiku-5.5": ("anthropic", "claude-haiku-5-5"),
    "gemini-3-flash": ("gemini", "gemini-3-flash-preview"),
    "gemini-3.1-flash-lite": ("gemini", "gemini-3.1-flash-lite"),
    "gpt-oss-120b": ("groq", "openai/gpt-oss-120b"),
    "qwen3.8-27b": ("groq", "qwen/qwen3.8-27b"),
    "qwen3.5-397b-adaption": ("adaption", "qwen3.5-397b"),
    "deepseek-v4-flash": ("adaption", "deepseek-v4-flash"),
    "glm-5.2": ("adaption", "glm-5.2"),
    "gemma-4-31b": ("adaption", "gemma-4-31b-it"),
}

# Minimum seconds between calls per provider, to stay inside free-tier limits.
PACE = {"anthropic": 0.0, "gemini": 4.5, "groq": 2.0, "adaption": 1.0}
# Per-model overrides (free-tier limits): Gemini 3 Flash ~5 req/min; Groq ~8k tokens/min per model.
MODEL_PACE = {"gemini-3-flash": 13.0, "gpt-oss-120b": 19.0, "qwen3.8-27b": 19.0}
_last: dict[str, float] = {}


def _post(url: str, headers: dict, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "according-docs-study/0.1", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(model_key: str, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 700,
         history: list[tuple[str, str]] | None = None) -> tuple[str, dict]:
    """history: earlier turns as (role, text) with role 'user' or 'assistant'; `user` is the final turn."""
    turns = [{"role": r, "content": t} for r, t in (history or [])] + [{"role": "user", "content": user}]
    provider, model = MODELS[model_key]
    pace = MODEL_PACE.get(model_key, PACE[provider])
    wait = pace - (time.time() - _last.get(provider, 0))
    if wait > 0:
        time.sleep(wait)
    for attempt in range(6):
        try:
            _last[provider] = time.time()
            if provider == "anthropic":
                d = _post("https://api.anthropic.com/v1/messages",
                          {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                          {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                           "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                           "messages": turns})
                text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
                return text, {"usage": d.get("usage"), "stop": d.get("stop_reason")}
            if provider in ("gemini", "groq"):
                url = ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" if provider == "gemini"
                       else "https://api.groq.com/openai/v1/chat/completions")
                key = os.environ["GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"]
                body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                        "messages": [{"role": "system", "content": system}] + turns}
                d = _post(url, {"Authorization": f"Bearer {key}"}, body)
                msg = d["choices"][0]["message"]
                return (msg.get("content") or ""), {"usage": d.get("usage"), "stop": d["choices"][0].get("finish_reason")}
            if provider == "adaption":
                d = _post("https://api.prod.adaptionlabs.ai/api/v1/responses",
                          {"Authorization": f"Bearer {os.environ['ADAPTION_API_KEY']}"},
                          {"model": model, "instructions": system, "input": turns if history else user,
                           "temperature": temperature, "max_output_tokens": max(max_tokens, 4000)})
                # only the final message the user would see; skip private reasoning items
                text = "".join(c.get("text", "") for o in d.get("output", []) if o.get("type") == "message"
                               for c in (o.get("content") or []) if isinstance(c, dict) and c.get("type") in ("output_text", "text"))
                if not text and d.get("status") == "incomplete":
                    raise RuntimeError("incomplete response (token budget used up by reasoning)")
                return text, {"usage": d.get("usage"), "stop": d.get("status")}
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(min(90, 8 * (attempt + 1)))
                continue
            raise RuntimeError(f"{model_key} HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError) as e:
            time.sleep(5 * (attempt + 1))
            continue
    raise RuntimeError(f"{model_key}: gave up after retries")
