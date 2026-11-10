#!/usr/bin/env python3
"""
Cross-provider sweep driver (MISSION3 Workstream D)
===================================================
Runs the FROZEN protocols — hot2_v5, berg_test, ae1_ladder2 — against
non-Anthropic models by monkeypatching ONLY each module's transport function
onto providers.call(). Prompts, generators, parsers, scoring, analysis, and
file handling are the frozen modules' own code, byte-identical; the swap is
transport-level (the same role providers.py was built for).

Per run it records a REGIME TALLY — (regime_requested, regime_observed)
counts across every call — plus token usage and failure counts, saved to
sweepmeta_<script>_<model>.json next to the protocol's own results JSON.

Seeds match the frozen Claude runs: hot2_v5 seed 71 (core, fixedlen 30),
ae1_ladder2 seed 90 (core, 24 episodes), berg_test seed 401 / n 100. NOTE
(review finding): only ae1_ladder2's per-episode reseeding makes worlds
IDENTICAL across labs; berg/hot2_v5 problems are distribution-matched, not
item-identical, because the global RNG path depends on each model's adaptive
calibration walk.

Stop-reason mapping: OpenAI-compat finish_reason 'length' is mapped to the
Anthropic-style 'max_tokens' that the frozen scripts' truncation accounting
expects; 'stop' maps to 'end_turn'.

Usage:
  python3 sweep.py --selftest                      (offline, no network)
  python3 sweep.py --script hot2v5 --model gemini-3.5-flash
  python3 sweep.py --script berg   --model grok-4.3
  python3 sweep.py --script ladder2 --model gpt-5.6-sol
"""

import os
import sys
import json
import argparse
from collections import Counter

import providers as P

REGIME = Counter()
USAGE = Counter()
PARAMS = Counter()
ERRORS = Counter()

def _tally(r):
    REGIME[f"{r.regime} | observed: {r.regime_observed}"] += 1
    PARAMS[json.dumps(r.params_used, sort_keys=True, default=str)] += 1
    USAGE["calls"] += 1
    USAGE["in_tokens"] += (r.usage or {}).get("input_tokens") or 0
    USAGE["out_tokens"] += (r.usage or {}).get("output_tokens") or 0
    if not r.ok:
        USAGE["failed"] += 1
        if len(ERRORS) < 20:
            ERRORS[r.error[:160]] += 1
        # a fatal adapter error (bad key/model/permission) mid-run must not
        # silently become thousands of drop-rows in a paid results file
        if r.error.startswith("fatal") or r.regime == "skipped":
            print(f"\nFATAL adapter error — aborting run: {r.error[:200]}",
                  flush=True)
            raise SystemExit(1)

def _map_stop(stop):
    s = (stop or "").lower()
    return {"length": "max_tokens", "stop": "end_turn",
            "content_filter": "refusal"}.get(s, s)

# ----------------------------- TRANSPORT PATCHES -----------------------------
# Each replicates the frozen function's exact signature and return contract.

def patched_hot2v5_api_chat(client, model, messages, max_tokens,
                            thinking="disabled"):
    r = P.call(model, messages, max_tokens=max_tokens, temperature=1.0)
    _tally(r)
    if not r.ok:
        return None, None
    return r.text, _map_stop(r.stop_reason)

def patched_ladder2_call_model2(client, model, history, thinking="disabled"):
    r = P.call(model, history, max_tokens=300, temperature=1.0)
    _tally(r)
    return r.text if r.ok else None

def make_patched_berg_ask(berg):
    def ask(client, model, body):
        prompt = berg.PROMPT.format(body=body)
        for _ in range(berg.RETRIES):
            r = P.call(model, [{"role": "user", "content": prompt}],
                       max_tokens=berg.MAX_TOKENS, temperature=1.0)
            _tally(r)
            if not r.ok:
                continue
            parsed = berg.parse_reply(r.text)
            if parsed is not None:
                return parsed, r.text
        return None, None
    return ask

# ----------------------------- RUN ORCHESTRATION -----------------------------

FROZEN_ARGS = {
    "hot2v5":  ["--tier", "core", "--seed", "71", "--fixedlen", "30"],
    "ladder2": ["--tier", "core", "--seed", "90", "--episodes", "24"],
    "berg":    ["--seed", "401", "--n", "100"],
}

def preflight(model):
    """Reviewer critical #1: OpenAI-compat max_completion_tokens INCLUDES
    reasoning tokens. If reasoning cannot truly be zeroed, the frozen
    protocols' tiny caps (bet=12, conf=20) return empty text and the whole
    paid run becomes drops. Three cheap live calls mimic the tightest-cap
    turns and must come back parseable, or we refuse to spend."""
    checks = [
        ("Reply with ONLY the word KEEP.", 12,
         lambda t: "KEEP" in t.upper()),
        ("Reply with ONLY the integer 85.", 20, lambda t: "85" in t),
        ("What is 340 + 217? Reply with ONLY the integer answer.", 150,
         lambda t: "557" in t),
    ]
    for prompt, cap, ok_fn in checks:
        r = P.call(model, [{"role": "user", "content": prompt}],
                   max_tokens=cap, temperature=1.0)
        if not (r.ok and r.text and ok_fn(r.text)):
            sys.exit(f"PRE-FLIGHT FAILED for {model}: cap={cap} "
                     f"text={r.text[:60]!r} stop={r.stop_reason} "
                     f"err={r.error[:120]} — reasoning may be eating the "
                     f"token caps; refusing to spend on a full run.")
        print(f"  preflight ok: cap={cap} -> {r.text[:20]!r} "
              f"(stop={r.stop_reason}, observed={r.regime_observed})")


def run_one(script, model):
    prov = P.provider_of(model)
    if prov == "anthropic":
        sys.exit("Claude models run through the frozen scripts directly "
                 "(they carry the fable max_tokens branch the sweep patches "
                 "drop) — refusing (review finding #4).")
    if not P.key_present(prov):
        sys.exit(f"{P.PROVIDERS[prov]['env']} not set — refusing to start a "
                 f"run that would record garbage (review finding #8).")
    print(f"pre-flight gate for {model}:")
    preflight(model)
    REGIME.clear()
    USAGE.clear()
    PARAMS.clear()
    ERRORS.clear()
    if script == "hot2v5":
        import hot2_v5 as m
        m.api_chat = patched_hot2v5_api_chat
        m.THINKING_SENT["value"] = "adapter:minimize_reasoning"
        argv = ["hot2_v5.py", "--model", model] + FROZEN_ARGS[script]
    elif script == "ladder2":
        import ae1_ladder2 as m
        m.call_model2 = patched_ladder2_call_model2
        argv = ["ae1_ladder2.py", "--model", model] + FROZEN_ARGS[script]
    elif script == "berg":
        import berg_test as m
        m.ask = make_patched_berg_ask(m)
        argv = ["berg_test.py", "--model", model] + FROZEN_ARGS[script]
    else:
        sys.exit(f"unknown script {script}")
    old_argv = sys.argv
    sys.argv = argv
    try:
        m.main()
    finally:
        sys.argv = old_argv
        meta = {"script": script, "model": model,
                "provider": P.provider_of(model),
                "frozen_args": FROZEN_ARGS[script],
                "regime_tally": dict(REGIME), "usage": dict(USAGE),
                "params_tally": dict(PARAMS),
                "errors": dict(ERRORS),
                "est_cost_usd": P.estimate_cost(
                    model, 1, USAGE.get("in_tokens", 0),
                    USAGE.get("out_tokens", 0))}
        path = f"sweepmeta_{script}_{model.replace('/', '_')}.json"
        k = 2
        while os.path.exists(path):
            path = f"sweepmeta_{script}_{model.replace('/', '_')}_run{k}.json"
            k += 1
        with open(path, "w") as f:
            json.dump(meta, f, indent=1)
        print(f"\nREGIME TALLY: {dict(REGIME)}")
        print(f"USAGE: {dict(USAGE)}  est_cost="
              f"{meta['est_cost_usd']}")
        print(f"sweep meta -> {path}")

# ----------------------------- SELFTEST (offline) ----------------------------

def selftest():
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("SWEEP SELFTEST: offline — providers.call monkeypatched\n")
    canned = {}
    def fake_call(model, messages, max_tokens=300, provider=None,
                  minimize_reasoning=True, temperature=None):
        canned["last"] = {"model": model, "max_tokens": max_tokens,
                          "temperature": temperature,
                          "n_messages": len(messages)}
        return P.LLMResponse(
            text=canned.get("text", "42"), provider="test", model=model,
            regime="reasoning-minimal", regime_observed="reasoning-tokens=0",
            usage={"input_tokens": 10, "output_tokens": 3},
            stop_reason=canned.get("stop", "stop"))
    real_call = P.call
    P.call = fake_call
    try:
        REGIME.clear(); USAGE.clear()
        # hot2v5 transport contract: (text, mapped stop_reason)
        canned["stop"] = "length"
        t, s = patched_hot2v5_api_chat(None, "x-model", [{"role": "user",
                                                          "content": "q"}],
                                       150)
        check(t == "42" and s == "max_tokens",
              "hot2v5 transport maps finish_reason length->max_tokens")
        check(canned["last"]["temperature"] == 1.0
              and canned["last"]["max_tokens"] == 150,
              "hot2v5 transport sends temperature=1.0 and the caller's cap")
        canned["stop"] = "stop"
        t, s = patched_hot2v5_api_chat(None, "x", [{"role": "user",
                                                    "content": "q"}], 20)
        check(s == "end_turn", "stop->end_turn mapping")
        # ladder2 transport contract: text or None
        t = patched_ladder2_call_model2(None, "x", [{"role": "user",
                                                     "content": "q"}])
        check(t == "42" and canned["last"]["max_tokens"] == 300,
              "ladder2 transport returns text, 300-token cap")
        # berg transport contract: ((pred, conf, mode), raw)
        import berg_test as berg
        canned["text"] = '{"answer": 7, "confidence": 55}'
        ask = make_patched_berg_ask(berg)
        out, raw = ask(None, "x", "2+5?")
        check(out is not None and out[0] == 7 and out[1] == 55.0,
              f"berg transport parses through frozen parse_reply (got {out})")
        # failure path: ok=False propagates as protocol-visible failure
        def dead_call(*a, **k):
            r = fake_call(*a, **k)
            r.ok = False
            return r
        P.call = dead_call
        t, s = patched_hot2v5_api_chat(None, "x", [{"role": "user",
                                                    "content": "q"}], 150)
        check(t is None and s is None, "adapter failure -> (None, None), "
                                       "counted as transport failure upstream")
        P.call = fake_call
        # regime tally accumulated
        check(USAGE["calls"] >= 5 and len(REGIME) == 1,
              f"regime tally accumulates ({USAGE['calls']} calls)")
        check(_map_stop("CONTENT_FILTER") == "refusal"
              and _map_stop(None) == "",
              "stop mapping: content_filter->refusal, None->''")
        check(len(PARAMS) >= 1, "params_used recorded per call")
        # fatal adapter error aborts instead of continuing
        def fatal_call(*a, **k):
            r = fake_call(*a, **k)
            r.ok = False
            r.error = "fatal: AuthenticationError: revoked"
            return r
        P.call = fatal_call
        try:
            patched_ladder2_call_model2(None, "x", [{"role": "user",
                                                     "content": "q"}])
            check(False, "fatal error aborts the run")
        except SystemExit:
            check(True, "fatal error aborts the run (SystemExit)")
        P.call = fake_call
    finally:
        P.call = real_call
    print("\nSWEEP SELFTEST", "PASSED" if ok else "FAILED")
    return ok

# ----------------------------- MAIN -----------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--script", choices=["hot2v5", "berg", "ladder2"])
    ap.add_argument("--model")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not (args.script and args.model):
        sys.exit("need --script and --model (or --selftest)")
    run_one(args.script, args.model)
