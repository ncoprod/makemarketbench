"""Sequential, resumable NIM expansion; run under scripts/with_resource.sh nim.

Long-timeout probes and thinking use distinct labels and caches. No local
workers are used, so one held NIM slot equals at most one request in flight.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from makemarket.cli import slug
from makemarket.evaluate import run_player
from makemarket.games import make_game
from makemarket.llm import ChatClient, LLMConfig, LLMQuoter, Throttle, parse_quote
from makemarket.market import Observation
from makemarket.prompts import messages_for
from makemarket.provenance import file_sha256, receipt

OUT = Path("results/expansion")
MODELS = [
    ("z-ai/glm-5.3-flash", {}, 8192, "long-timeout"),
    ("deepseek-ai/deepseek-v4.1-flash", {}, 8192, "long-timeout"),
    ("nvidia/nemotron-3-ultra-550b-a55b", {"chat_template_kwargs": {"enable_thinking": True}}, 16384, "thinking"),
    ("meta/llama-3.3-70b-instruct", {}, 4096, ""),
    ("qwen/qwen3-235b-a22b", {"chat_template_kwargs": {"enable_thinking": False}}, 4096, "no-think"),
    ("mistralai/mistral-small-3.1-24b-instruct-2503", {}, 4096, ""),
    ("meta/llama-3.1-8b-instruct", {}, 4096, ""),
    ("openai/gpt-oss-20b", {"reasoning_effort": "low"}, 4096, ""),
    ("nvidia/nemotron-3-ultra-550b-a55b", {"chat_template_kwargs": {"enable_thinking": False}}, 4096, "no-think"),
    ("nvidia/nemotron-3.5-bolt-e4b", {}, 4096, ""),
    ("nvidia/nemotron-3.5-lightning-30b-a3b", {"chat_template_kwargs": {"enable_thinking": False}}, 4096, "no-think"),
]
# Explicit --models batches handle these, avoiding writes shared with the default batch.
SUPPLEMENT = [
    ("mistralai/mistral-7b-instruct-v0.3", {}, 4096, ""),
    ("nv-mistralai/mistral-nemo-12b-instruct", {}, 4096, ""),
    ("meta/llama-3.2-11b-vision-instruct", {}, 4096, "text-only"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--protocol-file", help="separate receipt filename for an independent resume batch")
    retry = ap.add_mutually_exclusive_group()
    retry.add_argument("--reduced-reasoning", action="store_true",
                    help="retry GLM at low effort and DeepSeek with thinking disabled")
    retry.add_argument("--ultra-medium", action="store_true",
                       help="retry Ultra with the documented medium-effort thinking mode")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    models = list(MODELS + SUPPLEMENT if args.models else MODELS)
    batch = receipt({"args": vars(args)}, "NIM immutable deployment revision not exposed")
    batch["source_sha256"] = {str(p): file_sha256(p) for p in
                              [Path(__file__), *sorted(Path("src/makemarket").rglob("*.py"))]}
    filename = "protocol-supplement.json" if args.models else "protocol.json"
    if args.reduced_reasoning:
        filename = "protocol-reasoning.json"
        models = [
            ("z-ai/glm-5.3-flash", {"reasoning_effort": "low",
                                  "chat_template_kwargs": {"clear_thinking": True}}, 8192, "reasoning-low"),
            ("deepseek-ai/deepseek-v4.1-flash",
             {"chat_template_kwargs": {"enable_thinking": False}}, 8192, "no-think"),
        ]
    if args.ultra_medium:
        filename = "protocol-ultra-medium.json"
        models = [("nvidia/nemotron-3-ultra-550b-a55b",
                   {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}},
                   16384, "thinking-medium")]
    if args.protocol_file:
        filename = args.protocol_file
    path = OUT / filename
    protocol = json.loads(path.read_text()) if path.exists() else {}
    discovery = OUT / "models.json"
    if not discovery.exists():
        client = ChatClient(LLMConfig(model="discovery", timeout=60), throttle=Throttle(1))
        try:
            data = client.client.models.list().model_dump()
            discovery.write_text(json.dumps({"provenance": receipt({"method": "GET /v1/models"}),
                                             "data": data}, indent=2) + "\n")
        except Exception as exc:
            discovery.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
    catalog = json.loads(discovery.read_text()).get("data", {}).get("data", [])
    if not args.models and not args.reduced_reasoning and not args.ultra_medium:
        for model in catalog:
            if "qwen3.8" in model["id"].lower():
                models.append((model["id"], {"chat_template_kwargs": {"enable_thinking": False}}, 8192, "no-think"))
    g = make_game("dice")
    ep = g.episode(80000)
    obs = Observation(g, 2, 6, ep.reveals[:3], [], 0, 0.0, ep.public_beliefs[2], ep.public_beliefs[2].mean)
    ready = []
    for model, extra, tokens, suffix in models:
        if args.models and model not in args.models:
            continue
        label = model + (f" ({suffix})" if suffix else "")
        cfg = LLMConfig(model=model, label=label, timeout=600, max_tokens=tokens,
                        temperature=0.3, extra_body=extra)
        provenance = {**batch, "config": {**dataclasses.asdict(cfg), "batch": vars(args)}}
        client = ChatClient(cfg, cache_dir="cache/expansion", throttle=Throttle(1), max_retries=3)
        entry = protocol.get(label)
        if entry is None:
            entry = {"model": model, "label": label, "timeout_seconds": 600,
                     "probe_seed": 80000, "probe_round": 2,
                     "max_tokens": tokens, "temperature": cfg.temperature, "extra_body": extra,
                     "provenance": provenance}
            started = time.time()
            try:
                response = client.chat(messages_for(obs))
                entry["probe"] = response
                parse_quote(response["content"], g.params.max_size)
                entry["usable"] = True
            except Exception as exc:
                entry.update(usable=False, error=f"{type(exc).__name__}: {exc}")
            entry["probe_wall_seconds"] = time.time() - started
            protocol[label] = entry
            path.write_text(json.dumps(protocol, indent=2) + "\n")
        print(f"[probe] {label}: usable={entry['usable']}", flush=True)
        if entry["usable"]:
            ready.append((label, client, entry, provenance))
    # Establish availability first; finish faster cohorts before slower ones.
    for label, client, entry, provenance in sorted(ready, key=lambda item: item[2]["probe_wall_seconds"]):
        for name in ("dice", "cards"):
            started = time.time()
            try:
                eps = run_player(make_game(name), lambda c=client: LLMQuoter(c),
                                 list(range(args.seed, args.seed + args.episodes)),
                                 OUT / "raw" / name / f"{slug(label)}.jsonl", workers=1,
                                 provenance=provenance)
                entry[name] = {"episodes": len(eps)}
            except Exception as exc:
                entry[name] = {"error": f"{type(exc).__name__}: {exc}"}
            entry[name]["wall_seconds"] = time.time() - started
            entry[name]["provenance"] = provenance
            entry.setdefault("evaluation_attempts", []).append({"game": name, **entry[name]})
            path.write_text(json.dumps(protocol, indent=2) + "\n")
            print(f"[eval] {label} {name}: {entry[name]}", flush=True)


if __name__ == "__main__":
    main()
