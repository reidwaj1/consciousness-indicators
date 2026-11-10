#!/usr/bin/env python3
"""
Cross-provider adapter (Mission 3, Workstream D — hardened in the
bulletproofing round; this module has produced no frozen scientific results,
so it is upgraded in place)
==========================================================================
One call interface across four labs so the FROZEN protocols (HOT-2 v5,
berg_test, AE-1 ladder v2) can run the same code against every model. HOT-3
stays Claude-only.

Design goals:
  - Anthropic native SDK for Claude; OpenAI-compatible /v1/chat/completions
    for OpenAI, xAI, and Gemini.
  - Minimize each provider's reasoning mode WHERE THE API ALLOWS. Two regime
    fields are recorded on every response:
      regime          what we REQUESTED (e.g. "reasoning-minimal")
      regime_observed what the RESPONSE shows actually happened (thinking
                      blocks / reasoning-token usage), because a provider can
                      silently ignore a knob — the observed field is the
                      ground truth for the paper trail.
  - Parameter-fallback ladder: if a provider 400s on a reasoning knob or on
    max_completion_tokens, the call retries once per downgrade
    (minimal -> low -> omit; max_completion_tokens -> max_tokens) and records
    what was finally sent. Auth/permission/not-found errors fail fast.
  - call() NEVER raises and never exits: missing key, dead endpoint, or
    exhausted retries all return LLMResponse(ok=False) so a sweep skips
    gracefully.
  - The default selftest is genuinely OFFLINE (transports are monkeypatched;
    fake keys are injected; no network I/O of any kind). A --live flag sends
    ONE tiny real call per provider whose key is present, and says so first.
  - Error strings are scrubbed of anything shaped like an API key before
    being stored (the SDKs shouldn't echo keys, but results JSONs get shared;
    defense in depth).

Endpoints (verify model-ID strings at run time via each provider's docs;
these base URLs are stable):
  anthropic  native anthropic SDK
  openai     https://api.openai.com/v1                              (native)
  xai        https://api.x.ai/v1                                    (OpenAI-compatible)
  gemini     https://generativelanguage.googleapis.com/v1beta/openai/ (Google's OpenAI-compat shim)

Keys come from env: ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY,
GEMINI_API_KEY. They are read by the SDKs and never printed or stored.
"""

import os
import re
import sys
import time
import json
from dataclasses import dataclass, field, asdict

# ----------------------------- CONFIG ---------------------------------------

PROVIDERS = {
    "anthropic": {"env": "ANTHROPIC_API_KEY", "kind": "anthropic"},
    "openai":    {"env": "OPENAI_API_KEY",
                  "kind": "openai", "base_url": "https://api.openai.com/v1"},
    "xai":       {"env": "XAI_API_KEY",
                  "kind": "openai", "base_url": "https://api.x.ai/v1"},
    "gemini":    {"env": "GEMINI_API_KEY", "kind": "openai",
                  "base_url":
                  "https://generativelanguage.googleapis.com/v1beta/openai/"},
}

# approximate per-1M-token USD pricing for the cost estimator; callers should
# treat as rough. Keyed by substring of the model id (first match wins).
PRICING = [
    ("claude-opus",    (5.0, 25.0)),
    ("claude-sonnet",  (3.0, 15.0)),
    ("claude-fable",   (10.0, 50.0)),
    ("claude-haiku",   (1.0, 5.0)),
    ("gpt-5.5",        (5.0, 15.0)),        # placeholder; verify at run time
    ("gpt-5.4-mini",   (0.6, 2.4)),
    ("gpt-5",          (5.0, 15.0)),
    ("gemini-3.1-pro", (2.5, 15.0)),
    ("gemini-3.5-flash", (0.3, 2.5)),
    ("gemini",         (1.0, 5.0)),
    ("grok-4.3",       (5.0, 15.0)),
    ("grok-4.1-fast",  (0.5, 2.0)),
    ("grok",           (3.0, 15.0)),
]

TRANSIENT_RETRIES = 4        # for 429/5xx/network only
# reasoning-knob downgrade ladders, tried in order on a knob-specific 400
KNOB_LADDER = {
    "openai": ["minimal", "low", None],
    "xai":    ["low", None],
    "gemini": [0, None],          # thinking_budget 0 -> omit config entirely
}

# error-string scrubbing: anything shaped like a bearer credential
_KEY_SHAPES = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|AIza[A-Za-z0-9_\-]{10,}|xai-[A-Za-z0-9_\-]{8,}"
    r"|Bearer\s+\S{8,})")

def _scrub(s):
    return _KEY_SHAPES.sub("[REDACTED]", str(s))


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    regime: str                      # reasoning regime we REQUESTED
    regime_observed: str = ""        # what the response shows happened
    ok: bool = True
    error: str = ""
    usage: dict = field(default_factory=dict)
    params_used: dict = field(default_factory=dict)   # knobs actually sent
    stop_reason: str = ""            # anthropic stop_reason / openai finish_reason


def provider_of(name_or_model):
    """Explicit provider name, or infer from a model id prefix."""
    if name_or_model in PROVIDERS:
        return name_or_model
    m = name_or_model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("grok"):
        return "xai"
    raise ValueError(f"cannot infer provider for model '{name_or_model}'")


def key_present(provider):
    return bool(os.environ.get(PROVIDERS[provider]["env"]))


def pricing_for(model):
    for frag, price in PRICING:
        if frag in model:
            return price
    return (None, None)


def estimate_cost(model, n_calls, in_tokens, out_tokens):
    """Rough USD estimate for a planned run."""
    pin, pout = pricing_for(model)
    if pin is None:
        return None
    return n_calls * (in_tokens * pin + out_tokens * pout) / 1_000_000


# ----------------------------- CLIENTS --------------------------------------

_clients = {}

def _anthropic_client():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic()
    return _clients["anthropic"]

def _openai_client(provider):
    if provider not in _clients:
        from openai import OpenAI
        cfg = PROVIDERS[provider]
        _clients[provider] = OpenAI(api_key=os.environ[cfg["env"]],
                                    base_url=cfg["base_url"])
    return _clients[provider]


# ----------------------------- ERROR CLASSIFICATION -------------------------

def _status_of(exc):
    """HTTP status of an SDK exception, or None for network/unknown."""
    return getattr(exc, "status_code", None)

def _is_transient(exc):
    if isinstance(exc, (ImportError, AttributeError, TypeError)):
        return False                    # environment/code errors: fail fast
    st = _status_of(exc)
    if st is None:                      # timeouts, connection errors
        return True
    return st == 429 or st >= 500

def _is_knob_400(exc, knob_words):
    if _status_of(exc) != 400:
        return False
    msg = str(exc).lower()
    return any(w in msg for w in knob_words)


# ----------------------------- TRANSPORTS -----------------------------------
# Kept as module-level functions so the offline selftest can monkeypatch them.
# Each does exactly one HTTP round-trip and raises SDK errors unmodified.

def _transport_anthropic(model, messages, max_tokens, thinking_param,
                         temperature=None):
    client = _anthropic_client()
    kwargs = dict(model=model, max_tokens=max_tokens, messages=list(messages))
    if temperature is not None:
        kwargs["temperature"] = temperature
    if thinking_param is not None:
        kwargs["thinking"] = thinking_param
    resp = client.messages.create(**kwargs)
    text = "\n".join(b.text for b in resp.content
                     if getattr(b, "type", "") == "text")
    thinking_blocks = sum(1 for b in resp.content
                          if getattr(b, "type", "") == "thinking")
    usage = {"input_tokens": getattr(resp.usage, "input_tokens", None),
             "output_tokens": getattr(resp.usage, "output_tokens", None)}
    return text, usage, {"thinking_blocks": thinking_blocks,
                         "stop_reason": getattr(resp, "stop_reason", None)}


def _transport_openai(provider, model, messages, kwargs, extra_body):
    client = _openai_client(provider)
    if extra_body:
        resp = client.chat.completions.create(model=model, messages=messages,
                                              extra_body=extra_body, **kwargs)
    else:
        resp = client.chat.completions.create(model=model, messages=messages,
                                              **kwargs)
    if not getattr(resp, "choices", None):
        return "", {}, {"reasoning_tokens": None,
                        "finish_reason": "empty_choices"}
    text = resp.choices[0].message.content or ""
    u = getattr(resp, "usage", None)
    usage = {}
    reasoning_tokens = None
    if u is not None:
        usage = {"input_tokens": getattr(u, "prompt_tokens", None),
                 "output_tokens": getattr(u, "completion_tokens", None)}
        det = getattr(u, "completion_tokens_details", None)
        if det is not None:
            reasoning_tokens = getattr(det, "reasoning_tokens", None)
    fin = getattr(resp.choices[0], "finish_reason", None)
    return text, usage, {"reasoning_tokens": reasoning_tokens,
                         "finish_reason": fin}


# ----------------------------- CALL -----------------------------------------

def _messages_norm(messages):
    """Anthropic-style [{role, content}] passes through to OpenAI unchanged;
    this guard also strips any extra keys callers may carry."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def _anthropic_requested_regime(model, minimize):
    if "fable" in model or "mythos" in model:
        return "always-on", None            # explicit disable 400s on fable
    if minimize:
        return "thinking-disabled", {"type": "disabled"}
    return "default", None


def _observed_anthropic(meta):
    n = meta.get("thinking_blocks", 0)
    return "thinking-observed" if n else "no-thinking-observed"


def _observed_openai(meta):
    rt = meta.get("reasoning_tokens")
    if rt is None:
        return "reasoning-usage-unreported"
    return f"reasoning-tokens={rt}"


def _call_anthropic_full(model, messages, max_tokens, minimize,
                         temperature=None):
    regime, thinking_param = _anthropic_requested_regime(model, minimize)
    last = None
    for attempt in range(TRANSIENT_RETRIES):
        try:
            text, usage, meta = _transport_anthropic(
                model, messages, max_tokens, thinking_param, temperature)
            return LLMResponse(
                text=text, provider="anthropic", model=model, regime=regime,
                regime_observed=_observed_anthropic(meta), usage=usage,
                stop_reason=str(meta.get("stop_reason") or ""),
                params_used={"thinking": thinking_param,
                             "temperature": temperature,
                             "max_tokens": max_tokens})
        except Exception as e:
            if _is_knob_400(e, ("thinking",)) and thinking_param is not None:
                # model rejects the disable knob: omit it, record honestly
                thinking_param = None
                regime = "default(knob-rejected)"
                continue
            if not _is_transient(e):
                return LLMResponse(text="", provider="anthropic", model=model,
                                   regime=regime, ok=False,
                                   error=_scrub(f"fatal: {type(e).__name__}: {e}"))
            last = e
            time.sleep(1.5 * (attempt + 1))
    return LLMResponse(text="", provider="anthropic", model=model,
                       regime=regime, ok=False,
                       error=_scrub(f"exhausted retries: {type(last).__name__}: {last}"))


def _call_openai_full(provider, model, messages, max_tokens, minimize,
                      temperature=None):
    ladder = list(KNOB_LADDER[provider]) if minimize else [None]
    knob_i = 0
    token_param = "max_completion_tokens"
    temp = temperature
    last = None
    attempt = 0
    while attempt < TRANSIENT_RETRIES:
        knob = ladder[knob_i]
        kwargs = {token_param: max_tokens}
        if temp is not None:
            kwargs["temperature"] = temp
        extra_body = None
        if provider in ("openai", "xai") and knob is not None:
            kwargs["reasoning_effort"] = knob
            regime = f"reasoning-{knob}"
        elif provider == "gemini" and knob is not None:
            # Google's OpenAI-compat shim expects a body FIELD literally
            # named extra_body (the SDK kwarg merges its keys into the top
            # level, so the double nesting is required — audit finding,
            # verified against ai.google.dev docs). If this is ever wrong,
            # regime_observed still records the ground truth from usage.
            extra_body = {"extra_body": {"google": {"thinking_config":
                          {"thinking_budget": knob}}}}
            regime = ("thinking-off" if "flash" in model else "thinking-min")
        else:
            regime = "default" if knob is None and minimize is False \
                else "default(knob-omitted)" if minimize else "default"
        try:
            text, usage, meta = _transport_openai(
                provider, model, _messages_norm(messages), kwargs, extra_body)
            return LLMResponse(
                text=text, provider=provider, model=model, regime=regime,
                regime_observed=_observed_openai(meta), usage=usage,
                stop_reason=str(meta.get("finish_reason") or ""),
                params_used={"knob": knob, "token_param": token_param,
                             "temperature": temp,
                             "max_tokens": max_tokens})
        except Exception as e:
            if _is_knob_400(e, ("reasoning_effort", "reasoning", "thinking")) \
                    and knob is not None and knob_i + 1 < len(ladder):
                knob_i += 1          # downgrade knob; not a transport retry
                continue
            if _is_knob_400(e, ("max_completion_tokens",)) \
                    and token_param == "max_completion_tokens":
                token_param = "max_tokens"    # older shims; retry once
                continue
            if _is_knob_400(e, ("temperature",)) and temp is not None:
                temp = None                   # reasoning models reject it
                continue
            if _is_knob_400(e, ("output limit", "max_tokens")):
                # OpenAI-compat reasoning models THROW when reasoning wants
                # more than the cap, where Anthropic would return truncated
                # text with stop_reason=max_tokens. Normalize to that
                # semantics: empty text + 'length', so the frozen protocols
                # count a truncation and retry (found live: gpt-5.6-luna).
                return LLMResponse(
                    text="", provider=provider, model=model, regime=regime,
                    regime_observed="reasoning-consumed-cap",
                    stop_reason="length",
                    params_used={"knob": knob, "token_param": token_param,
                                 "temperature": temp,
                                 "max_tokens": max_tokens})
            if not _is_transient(e):
                return LLMResponse(text="", provider=provider, model=model,
                                   regime=regime, ok=False,
                                   error=_scrub(f"fatal: {type(e).__name__}: {e}"))
            last = e
            attempt += 1
            time.sleep(1.5 * attempt)
    return LLMResponse(text="", provider=provider, model=model, regime="error",
                       ok=False,
                       error=_scrub(f"exhausted retries: {type(last).__name__}: {last}"))


def call(model, messages, max_tokens=300, provider=None,
         minimize_reasoning=True, temperature=None):
    """Unified call. Returns an LLMResponse; ok=False (never raises) on a
    missing key, fatal API error, or exhausted retries, so a sweep can skip
    gracefully. regime_observed is the ground truth of what the model did."""
    try:
        prov = provider or provider_of(model)
    except ValueError as e:
        return LLMResponse(text="", provider="?", model=model, regime="error",
                           ok=False, error=str(e))
    if not key_present(prov):
        return LLMResponse(
            text="", provider=prov, model=model, regime="skipped", ok=False,
            error=f"{PROVIDERS[prov]['env']} not set; skipping {prov}/{model}")
    try:
        if PROVIDERS[prov]["kind"] == "anthropic":
            return _call_anthropic_full(model, messages, max_tokens,
                                        minimize_reasoning, temperature)
        return _call_openai_full(prov, model, messages, max_tokens,
                                 minimize_reasoning, temperature)
    except Exception as e:          # belt and braces: call() must never raise
        return LLMResponse(text="", provider=prov, model=model, regime="error",
                           ok=False,
                           error=_scrub(f"unexpected: {type(e).__name__}: {e}"))


# ----------------------------- SELFTEST -------------------------------------

class _FakeStatusError(Exception):
    def __init__(self, msg, status):
        super().__init__(msg)
        self.status_code = status


def selftest():
    """OFFLINE. Transports are monkeypatched and fake keys injected, so no
    network I/O can occur regardless of which real keys are present."""
    # patch the EXECUTING module: when run as a script this module is
    # __main__, and `import providers` would load a second copy whose
    # functions nothing here calls — patching that copy would silently leave
    # the real transports live (found the hard way: the first version of this
    # selftest made real API calls while claiming to be offline)
    P = sys.modules[__name__]

    print("SELFTEST: offline — transports monkeypatched, no network\n")
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    # 1. routing
    targets = {"claude-sonnet-5": "anthropic", "claude-opus-4-8": "anthropic",
               "claude-fable-5": "anthropic", "gpt-5.5": "openai",
               "o3-mini": "openai", "gemini-3.1-pro": "gemini",
               "gemini-3.5-flash": "gemini", "grok-4.3": "xai"}
    check(all(provider_of(m) == p for m, p in targets.items()),
          "provider inference for all model families")
    try:
        provider_of("llama-99")
        check(False, "unknown model raises ValueError")
    except ValueError:
        check(True, "unknown model raises ValueError")
    check(not call("llama-99", []).ok, "call() on unknown model returns "
                                       "ok=False instead of raising")

    # 2. graceful skip without keys (scoped env wipe)
    saved = {p: os.environ.pop(PROVIDERS[p]["env"], None) for p in PROVIDERS}
    try:
        r = call("gpt-5.5", [{"role": "user", "content": "hi"}])
        check(not r.ok and r.regime == "skipped" and "not set" in r.error,
              "missing key -> graceful skip, no exception")
    finally:
        for p, v in saved.items():
            if v is not None:
                os.environ[PROVIDERS[p]["env"]] = v

    # 3. offline transport fakes + fake keys
    for p in PROVIDERS:
        os.environ.setdefault(PROVIDERS[p]["env"], "SELFTEST-FAKE-KEY")
    calls_log = []

    def fake_anthropic(model, messages, max_tokens, thinking_param,
                       temperature=None):
        calls_log.append(("anthropic", thinking_param, temperature))
        blocks = 1 if ("fable" in model or thinking_param is None) else 0
        return "42", {"input_tokens": 10, "output_tokens": 2}, \
               {"thinking_blocks": blocks, "stop_reason": "end_turn"}

    rejects_minimal = {"on": True}
    def fake_openai(provider, model, messages, kwargs, extra_body):
        calls_log.append((provider, kwargs.get("reasoning_effort"),
                          "max_completion_tokens" in kwargs))
        if provider == "openai" and rejects_minimal["on"] \
                and kwargs.get("reasoning_effort") == "minimal":
            raise _FakeStatusError(
                "invalid value for reasoning_effort: 'minimal'", 400)
        if provider == "xai" and "max_completion_tokens" in kwargs:
            raise _FakeStatusError(
                "unknown parameter max_completion_tokens", 400)
        if provider == "gemini" and kwargs.get("temperature") == 0.77:
            raise _FakeStatusError(
                "temperature is not supported for this model", 400)
        rt = 0 if kwargs.get("reasoning_effort") else None
        return "7", {"input_tokens": 5, "output_tokens": 1}, \
               {"reasoning_tokens": rt, "finish_reason": "stop"}

    real_a, real_o = P._transport_anthropic, P._transport_openai
    real_ca, real_co = P._anthropic_client, P._openai_client
    P._transport_anthropic, P._transport_openai = fake_anthropic, fake_openai
    # belt and braces: if any code path evades the transport fakes and tries
    # to build a real client, fail LOUDLY instead of touching the network
    def _no_net(*a, **k):
        raise RuntimeError("OFFLINE SELFTEST attempted real network access")
    P._anthropic_client = _no_net
    P._openai_client = _no_net
    try:
        # anthropic: knob sent for sonnet, omitted for fable; observed recorded
        r = call("claude-sonnet-5", [{"role": "user", "content": "x"}])
        check(r.ok and r.regime == "thinking-disabled"
              and r.regime_observed == "no-thinking-observed",
              f"sonnet: requested={r.regime} observed={r.regime_observed}")
        r = call("claude-fable-5", [{"role": "user", "content": "x"}])
        check(r.ok and r.regime == "always-on"
              and r.regime_observed == "thinking-observed",
              f"fable: requested={r.regime} observed={r.regime_observed}")
        # openai: 'minimal' 400s -> ladder downgrades to 'low', records it
        r = call("gpt-5.5", [{"role": "user", "content": "x"}])
        check(r.ok and r.regime == "reasoning-low"
              and r.params_used["knob"] == "low",
              f"openai knob ladder minimal->low on 400 (got {r.regime})")
        # xai: max_completion_tokens 400s -> falls back to max_tokens
        r = call("grok-4.3", [{"role": "user", "content": "x"}])
        check(r.ok and r.params_used["token_param"] == "max_tokens",
              "xai token-param fallback max_completion_tokens->max_tokens")
        # gemini: budget knob requested, observed field populated
        r = call("gemini-3.5-flash", [{"role": "user", "content": "x"}])
        check(r.ok and r.regime == "thinking-off",
              f"gemini flash regime requested (got {r.regime})")
        # fatal fast-fail: 401 must not retry
        def fake_auth_fail(provider, model, messages, kwargs, extra_body):
            calls_log.append(("authfail",))
            raise _FakeStatusError("invalid api key", 401)
        P._transport_openai = fake_auth_fail
        n0 = len(calls_log)
        r = call("gpt-5.5", [{"role": "user", "content": "x"}])
        check(not r.ok and r.error.startswith("fatal")
              and len(calls_log) == n0 + 1,
              "401 fails fast (exactly one transport attempt, no backoff)")
        # scrubbing
        check("[REDACTED]" in _scrub("boom sk-abcdefghijk1234567890"),
              "key-shaped strings scrubbed from error text")
        # OpenAI 'output limit reached' 400 -> truncation semantics, not fatal
        def fake_outlimit(provider, model, messages, kwargs, extra_body):
            raise _FakeStatusError(
                "Could not finish the message because max_tokens or model "
                "output limit was reached.", 400)
        P._transport_openai = fake_outlimit
        r = call("gpt-5.5", [{"role": "user", "content": "x"}])
        check(r.ok and r.text == "" and r.stop_reason == "length",
              "output-limit 400 normalized to empty text + stop=length")
        P._transport_openai = fake_openai
        # NEW: stop_reason surfaced + temperature passthrough & rejection
        P._transport_openai = fake_openai
        r = call("claude-sonnet-5", [{"role": "user", "content": "x"}],
                 temperature=1.0)
        check(r.ok and r.stop_reason == "end_turn"
              and r.params_used.get("temperature") == 1.0,
              "anthropic path surfaces stop_reason and sends temperature")
        r = call("gemini-3.5-flash", [{"role": "user", "content": "x"}],
                 temperature=0.77)
        check(r.ok and r.params_used.get("temperature") is None
              and r.stop_reason == "stop",
              "temperature-rejecting model -> dropped, recorded, call still ok")
    finally:
        P._transport_anthropic, P._transport_openai = real_a, real_o
        P._anthropic_client, P._openai_client = real_ca, real_co
        for p in PROVIDERS:
            if os.environ.get(PROVIDERS[p]["env"]) == "SELFTEST-FAKE-KEY":
                del os.environ[PROVIDERS[p]["env"]]

    # 4. cost estimator sanity
    c = estimate_cost("claude-sonnet-5", 2800, 250, 10)
    check(c is not None and 0 < c < 100, f"cost estimate sane (~${c:.2f})")

    print("\nSELFTEST", "PASSED (fully offline)" if ok else
          "FAILED — do not run a sweep")
    return ok


def live_ping():
    """ONE tiny real call per provider whose key is present. Costs real money
    (fractions of a cent). Only runs with an explicit --live flag."""
    targets = ["claude-sonnet-5", "gpt-5.5", "gemini-3.5-flash", "grok-4.3"]
    print("LIVE PING: one minimal real call per provider with a key present\n")
    for t in targets:
        prov = provider_of(t)
        if not key_present(prov):
            print(f"  {t:20s} SKIP ({PROVIDERS[prov]['env']} not set)")
            continue
        r = call(t, [{"role": "user", "content": "Reply with the word OK."}],
                 max_tokens=16)
        status = "ok" if r.ok else f"FAILED: {r.error[:120]}"
        print(f"  {t:20s} {status}  regime={r.regime} "
              f"observed={r.regime_observed} usage={r.usage}")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    elif "--live" in sys.argv:
        live_ping()
    else:
        print("provider key availability:")
        for p in PROVIDERS:
            print(f"  {p:10s} {PROVIDERS[p]['env']:20s} "
                  f"{'present' if key_present(p) else 'ABSENT'}")
        print("\nusage: providers.py --selftest (offline) | --live (1 real "
              "call per available provider)")
