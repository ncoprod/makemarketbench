"""OpenAI-compatible LLM dealer with strict JSON parsing, one repair retry,
429/5xx backoff and an on-disk response cache keyed by the exact request."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .market import Observation, validate_quote
from .prompts import OUTPUT_SPEC, messages_for

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Return the last JSON object in ``text`` that has a "bid" key.

    Reasoning blocks (<think>...</think>) and markdown fences are ignored."""
    if text is None:
        raise ValueError("empty response")
    text = _THINK_RE.sub("", text)
    if "<think>" in text.lower():  # unterminated reasoning block
        raise ValueError("response ended inside a <think> block")
    candidates = []
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : i + 1])
    for c in reversed(candidates):
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "bid" in obj:
            return obj
    raise ValueError("no JSON object with a 'bid' field found")


def parse_quote(text: str, max_size: int):
    return validate_quote(extract_json(text), max_size)


@dataclass
class LLMConfig:
    model: str
    base_url: str = NIM_BASE_URL
    api_key_env: str = "NVIDIA_API_KEY"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: float = 240.0
    extra_body: dict = field(default_factory=dict)
    label: str | None = None  # name used in results (defaults to model)

    @property
    def name(self) -> str:
        return self.label or self.model


class Throttle:
    """Process-wide concurrency limit shared by all clients (rate-limit hygiene)."""

    def __init__(self, n: int) -> None:
        self.sem = threading.Semaphore(n)


class ChatClient:
    def __init__(self, cfg: LLMConfig, cache_dir: str | Path = "cache/llm",
                 throttle: Throttle | None = None, max_retries: int = 8) -> None:
        from openai import OpenAI

        self.cfg = cfg
        key = os.environ.get(cfg.api_key_env) or "not-needed"
        self.client = OpenAI(base_url=cfg.base_url, api_key=key, timeout=cfg.timeout, max_retries=0)
        self.cache_dir = Path(cache_dir) / re.sub(r"[^A-Za-z0-9_.-]", "_", cfg.name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.throttle = throttle or Throttle(2)
        self.max_retries = max_retries

    def _key(self, messages: list[dict]) -> str:
        payload = json.dumps({"model": self.cfg.model, "messages": messages,
                              "temperature": self.cfg.temperature, "max_tokens": self.cfg.max_tokens,
                              "extra": self.cfg.extra_body}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def chat(self, messages: list[dict]) -> dict:
        path = self.cache_dir / f"{self._key(messages)}.json"
        if path.exists():
            out = json.loads(path.read_text())
            out["cached"] = True
            return out
        delay = 2.0
        last_err = None
        for _attempt in range(self.max_retries):
            try:
                with self.throttle.sem:
                    t0 = time.time()
                    r = self.client.chat.completions.create(
                        model=self.cfg.model, messages=messages, temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_tokens, extra_body=self.cfg.extra_body or None)
                    latency = time.time() - t0
                msg = r.choices[0].message
                extra = msg.model_extra or {}
                out = {
                    "content": msg.content or "",
                    "reasoning": extra.get("reasoning_content") or extra.get("reasoning") or "",
                    "finish_reason": r.choices[0].finish_reason,
                    "completion_tokens": getattr(r.usage, "completion_tokens", None),
                    "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
                    "served_model": r.model,
                    "system_fingerprint": getattr(r, "system_fingerprint", None),
                    "latency": latency,
                }
                path.write_text(json.dumps(out))
                out["cached"] = False
                return out
            except Exception as e:  # rate limits, 5xx, timeouts, connection resets
                status = getattr(e, "status_code", None)
                last_err = f"{type(e).__name__}({status}): {str(e)[:200]}"
                # Listed NIM models can return 404 between successful generations.
                retry_404 = status == 404 and self.cfg.base_url.rstrip("/") == NIM_BASE_URL
                if status is not None and status < 500 and status not in (408, 409, 429) and not retry_404:
                    raise RuntimeError(last_err) from e
                time.sleep(delay + random.random())
                delay = min(delay * 2, 90.0)
        raise RuntimeError(f"giving up after {self.max_retries} attempts: {last_err}")


class LLMQuoter:
    """Dealer backed by a chat model. One repair retry on an invalid answer;
    after two invalid answers the round is played with no quote."""

    def __init__(self, client: ChatClient, hint: bool = False) -> None:
        self.client = client
        self.hint = hint  # ablation: include the public posterior mean/sd in the prompt
        self.name = client.cfg.name

    def reset(self, episode_seed: int) -> None:
        pass

    def quote(self, obs: Observation):
        msgs = messages_for(obs, self.hint)
        max_size = obs.game.params.max_size
        attempts = []
        for attempt in range(2):
            out = self.client.chat(msgs)
            text = out["content"]
            attempts.append({k: out.get(k) for k in ("content", "finish_reason", "completion_tokens",
                                                     "prompt_tokens", "served_model", "system_fingerprint",
                                                     "latency", "cached")})
            try:
                q = parse_quote(text, max_size)
                return q, {"attempts": attempts, "format_error": attempt > 0, "failed": False}
            except ValueError as e:
                attempts[-1]["error"] = str(e)
                if attempt == 0:
                    msgs = msgs + [
                        {"role": "assistant", "content": text[-2000:] if text else ""},
                        {"role": "user", "content": f"Your answer was invalid: {e}. {OUTPUT_SPEC}"},
                    ]
        return None, {"attempts": attempts, "format_error": True, "failed": True}
