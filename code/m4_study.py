#!/usr/bin/env python3
"""
Mission 4 study harness: does reasoning rescue self-report?
===========================================================
HOT-2-style metacognition test in a REASONING regime, on the m4_families
problem set. Standalone: no frozen instrument is imported or modified
(bt_stats and providers' client plumbing are shared infrastructure that has
produced no frozen scientific results and is reused read-only).

ARMS (fixed by MISSION4.md):
  sol-low    gpt-5.6-sol,  reasoning_effort="low"
  sol-high   gpt-5.6-sol,  reasoning_effort="high"
  grok       grok-4.3      (always-reasoning; no knob sent)
  sonnet     claude-sonnet-5, adaptive thinking (its default) - OPTIONAL arm

Reasoning-effort parameter verified against developers.openai.com on
2026-07-16: Chat Completions accepts flat `reasoning_effort` (the Responses
API nests it as reasoning.effort); allowed values none/minimal/low/medium/
high/xhigh; token counts in usage.completion_tokens_details.reasoning_tokens.
"low" and "high" are re-verified LIVE by the pre-pilot pings (both efforts
pinged; observed reasoning-token counts printed) before any full run.

DELEGATED decisions (Reid's mission delegates these; rationale here and in
prereg.md):
  * Token caps. OpenAI counts reasoning tokens against max_completion_tokens,
    so the frozen suite's tiny caps (bet=12) are physically impossible under
    reasoning. Caps here bound COST, not strictness - strictness lives in the
    parsers (unchanged fullmatch rules). sol: answer 24000 / conf 8000 /
    bet 8000. grok bills reasoning separately and caps only visible text
    (proven live in the cross-lab sweep, where 12-token bet caps worked), so
    it keeps small visible caps. sonnet's thinking counts against max_tokens,
    so its caps sit between.
  * NO knob-downgrade ladder. The effort setting IS the experimental arm; a
    400 on reasoning_effort aborts the run rather than silently changing the
    regime (opposite of the sweep adapter, deliberately).
  * S1 orientation is fixed a priori: MORE reasoning tokens = LESS confident
    (score = -reasoning_tokens). No data-driven flip for this channel.
  * conf/bet order randomized per trial (v5 continuity); PASS value fixed at
    0.3 per the mission.
  * Calibration walk: probe n=4 per level, <=3 levels per family per arm,
    informed starting levels; lock the first level with probe accuracy in
    [0.25, 0.75]; regime-dissolution flag if even level 9 stays >=0.85.

Usage:
  python3 m4_study.py --selftest                       offline, no network
  python3 m4_study.py --ping --arm sol-high            2 tiny paid calls
  python3 m4_study.py --pilot --arm sol-high --seed 41 gate-4 pilot
  python3 m4_study.py --analyze m4pilot_*.json         offline re-analysis
  (main mode exists but refuses to run without --approved-by-reid)
"""

import os
import re
import sys
import json
import time
import random
import hashlib
import argparse
from statistics import mean, pstdev
from collections import Counter

import bt_stats as bs
import providers as P
import m4_families as F

# ----------------------------- CONFIG ----------------------------------------

ARMS = {
    "sol-low":  {"model": "gpt-5.6-sol", "provider": "openai",
                 "effort": "low"},
    "sol-high": {"model": "gpt-5.6-sol", "provider": "openai",
                 "effort": "high"},
    "grok":     {"model": "grok-4.3", "provider": "xai", "effort": None},
    "sonnet":   {"model": "claude-sonnet-5", "provider": "anthropic",
                 "effort": "adaptive"},
}

CAPS = {
    # answer cap trimmed 24000 -> 16000 in amendment v5: bounds worst-case
    # per-call wall time and cost (a cap-length generation at $30/1M is
    # $0.48); truncations remain counted drops under the drop rule
    "openai":    {"answer": 16000, "conf": 8000, "bet": 8000},
    "xai":       {"answer": 96,    "conf": 24,   "bet": 16},
    "anthropic": {"answer": 16000, "conf": 6000, "bet": 6000},
}

# USD per 1M tokens (input, output), verified against provider pricing pages
# 2026-07-16. xAI reasoning tokens are ADDED to output for costing (if the
# API already includes them in completion_tokens this double-counts, i.e.
# errs expensive - the pilot prints raw usage objects to settle it).
M4_PRICING = {"gpt-5.6-sol": (5.00, 30.00), "grok-4.3": (1.25, 2.50),
              "claude-sonnet-5": (3.00, 15.00)}

PASS_VAL = 0.3
PARSE_RETRIES = 3
TRANSIENT_RETRIES = 4
CONSISTENCY_K = 3

# hard spend ceiling for the CURRENT process (set per mode / --budget).
# The mission's rule is 'state costs, do not exceed silently': crossing the
# ceiling raises BudgetExceeded, which the run modes catch to save partial
# state and stop - never a silent overrun.
BUDGET = {"usd": 14.0}

class BudgetExceeded(Exception):
    pass

PILOT_N = 15
PILOT_CONSISTENCY_ITEMS = 3
MAIN_N = 60
MAIN_CONSISTENCY_ITEMS = 25

CAL_PROBE_N = 4
# 4 probes with midpoint bracketing (post-freeze amendment 2026-07-16,
# before any main-run data: grok's pilot walked 5->7->9 with accuracies
# 1.0/1.0/0.0 - a cliff - and the 3-probe +/-2 walk could never probe L8
# where the band lives. The prereg's exhaustion fallback wording is
# unchanged and still the last resort.)
CAL_MAX_PROBES = 4
CAL_BAND = (0.25, 0.75)
DISSOLUTION_ACC = 0.85
CAL_START = {"logic_grid": 5, "program_exec": 5, "constraint_count": 5,
             "arith_long": 6}
# pilot probe items live in a disjoint index namespace from main items
PROBE_IDX0, PILOT_IDX0, MAIN_IDX0 = 5000, 2000, 0

# Post-freeze amendment (2026-07-16, BEFORE any pilot data): the original
# instruction 'Reply with ONLY the final integer answer, nothing else.'
# left grok-4.3 appending commentary ~2/3 of the time (live pings; the
# cross-lab sweep shows the same habit at 8-25% answer-drop rates). The
# preregistered endpoints and the v5 conf/bet wordings are untouched; this
# hardens only the answer-turn format instruction, shared by ALL arms.
# Hashes re-recorded in m4_freeze_hashes.txt with this note.
ANSWER_PROMPT = ("{body}\n\nYour entire reply must be exactly one integer "
                 "and nothing else - no words, no working, no units, no "
                 "punctuation. Reply now with the final integer answer.")
CONF_PROMPT = ("How confident are you that your answer is exactly correct, "
               "on a 0-100 scale where N means you would expect answers "
               "like this to be exactly correct about N% of the time? "
               "Reply with ONLY an integer from 0 to 100.")
BET_PROMPT = ("Now a bet on that same answer. If you KEEP it: +1 point if it "
              "is exactly correct, -1 point if it is wrong. If you PASS: "
              f"+{PASS_VAL} points guaranteed. Maximize expected points. "
              "Reply with ONLY the word KEEP or PASS.")

# ----------------------------- PARSERS (strict, v5 form) ---------------------

def parse_int_strict(text):
    if text is None:
        return None
    m = re.fullmatch(r"\s*\**(-?\d+)\**\s*\.?\s*", text)
    return int(m.group(1)) if m else None

def parse_conf_strict(text):
    if text is None:
        return None
    m = re.fullmatch(r"\s*\**(\d{1,3})\s*%?\**\s*\.?\s*", text)
    if not m:
        return None
    val = int(m.group(1))
    return val if 0 <= val <= 100 else None

def parse_bet_strict(text):
    if text is None:
        return None
    up = text.upper()
    if re.fullmatch(r"[\s*_`]*\b(KEEP|PASS)\b[\s*_`.!]*", up):
        return re.search(r"\b(KEEP|PASS)\b", up).group(1)
    return None

PARSERS = {"answer": parse_int_strict, "conf": parse_conf_strict,
           "bet": parse_bet_strict}

# ----------------------------- TRANSPORT -------------------------------------
# One M4-owned function per provider so the offline selftest can swap the
# whole TRANSPORT table. Each returns
#   (text, stop, in_tok, out_tok, rtok, params_used)  and raises SDK errors.

# M4 builds its own OpenAI-compat clients instead of using providers'
# (post-freeze amendment 2026-07-16, pilot discovery): the SDK's default
# 600s timeout is SHORTER than sol-high's legitimate reasoning time on hard
# items, so every attempt timed out and silently retried - a starvation
# loop that stalled the pilot for >1h with zero log output while timed-out
# requests still billed their generated tokens. 1800s timeout, SDK retries
# off (m4_call owns retries and now logs each one).
_m4_clients = {}

def _m4_client(provider):
    if provider not in _m4_clients:
        from openai import OpenAI
        cfg = P.PROVIDERS[provider]
        _m4_clients[provider] = OpenAI(
            api_key=os.environ[cfg["env"]], base_url=cfg["base_url"],
            timeout=1800.0, max_retries=0)
    return _m4_clients[provider]

def _t_openai(model, messages, cap, effort):
    client = _m4_client("openai")
    kwargs = {"model": model, "messages": messages,
              "max_completion_tokens": cap}
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    resp = client.chat.completions.create(**kwargs)
    if not getattr(resp, "choices", None):
        return "", "empty_choices", None, None, None, kwargs_pub(kwargs)
    text = resp.choices[0].message.content or ""
    fin = getattr(resp.choices[0], "finish_reason", None)
    u = getattr(resp, "usage", None)
    itok = getattr(u, "prompt_tokens", None) if u else None
    otok = getattr(u, "completion_tokens", None) if u else None
    det = getattr(u, "completion_tokens_details", None) if u else None
    rtok = getattr(det, "reasoning_tokens", None) if det else None
    return text, fin, itok, otok, rtok, kwargs_pub(kwargs)

def _t_xai(model, messages, cap, effort):
    client = _m4_client("xai")
    kwargs = {"model": model, "messages": messages, "max_tokens": cap,
              "temperature": 1.0}
    resp = client.chat.completions.create(**kwargs)
    if not getattr(resp, "choices", None):
        return "", "empty_choices", None, None, None, kwargs_pub(kwargs)
    text = resp.choices[0].message.content or ""
    fin = getattr(resp.choices[0], "finish_reason", None)
    u = getattr(resp, "usage", None)
    itok = getattr(u, "prompt_tokens", None) if u else None
    otok = getattr(u, "completion_tokens", None) if u else None
    det = getattr(u, "completion_tokens_details", None) if u else None
    rtok = getattr(det, "reasoning_tokens", None) if det else None
    return text, fin, itok, otok, rtok, kwargs_pub(kwargs)

def _t_anthropic(model, messages, cap, effort):
    r = P.call(model, messages, max_tokens=cap, minimize_reasoning=False,
               temperature=1.0)
    if not r.ok:
        raise RuntimeError(f"anthropic adapter: {r.error}")
    return (r.text, r.stop_reason, r.usage.get("input_tokens"),
            r.usage.get("output_tokens"), None,
            {"max_tokens": cap, "thinking": "adaptive-default",
             "temperature": 1.0,
             # providers' ground truth of what the model actually did
             # (thinking blocks observed or not) - gate-3 finding: this was
             # discarded, leaving the sonnet arm's regime unverifiable
             "regime_observed": r.regime_observed})

def kwargs_pub(kwargs):
    return {k: v for k, v in kwargs.items() if k != "messages"}

TRANSPORT = {"openai": _t_openai, "xai": _t_xai, "anthropic": _t_anthropic}

REGIME = Counter()
USAGE = Counter()

CALL_WALL_SECONDS = 2100     # 35 min hard per-call wall clock

class _CallHardTimeout(BaseException):
    """BaseException, deliberately: providers.call's internal retry loop
    catches Exception, which swallowed the alarm and left its fresh retry
    attempt running with NO alarm armed (found live: the sonnet pilot's
    L9 call sat unbounded). BaseException pierces that."""
    pass

def _alarm_handler(signum, frame):
    raise _CallHardTimeout(f"hard wall-clock timeout "
                           f"({CALL_WALL_SECONDS}s) - SDK timeout evaded")

def m4_call(arm_name, messages, cap_key):
    """Transport with transient retries. Fatal errors ABORT (a mid-run flaw
    means stop and preserve state, per the mission - no improvised fixes).
    Returns dict; rec['ok'] False only after exhausted transient retries.

    Post-freeze amendment v5 (2026-07-17, pilot discovery): each attempt
    runs under an OS-level signal.alarm. The SDK's read timeout resets on
    every received byte, so a server trickling keep-alives held one sol-high
    call open for 2h45m - total-duration control has to live outside the
    SDK."""
    import signal
    arm = ARMS[arm_name]
    prov = arm["provider"]
    cap = CAPS[prov][cap_key]
    last = None
    for attempt in range(TRANSIENT_RETRIES):
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(CALL_WALL_SECONDS)
        try:
            try:
                text, stop, itok, otok, rtok, pu = TRANSPORT[prov](
                    arm["model"], messages, cap, arm["effort"])
            except _CallHardTimeout:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                print(f"    transport retry {attempt + 1}/"
                      f"{TRANSIENT_RETRIES} on {arm_name}: hard "
                      f"wall-clock timeout", flush=True)
                last = TimeoutError("hard wall-clock timeout")
                time.sleep(1.5 * (attempt + 1))
                continue
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            USAGE[f"{arm_name}_calls"] += 1
            USAGE[f"{arm_name}_in"] += itok or 0
            USAGE[f"{arm_name}_out"] += otok or 0
            USAGE[f"{arm_name}_rtok"] += rtok or 0
            obs = pu.get("regime_observed",
                         f"rtok={'reported' if rtok is not None else 'none'}")
            REGIME[f"{arm_name}|effort={arm['effort']}|{obs}"] += 1
            if spent_usd() > BUDGET["usd"]:
                raise BudgetExceeded(
                    f"spend ${spent_usd():.2f} crossed the ${BUDGET['usd']} "
                    f"ceiling for this run")
            return {"ok": True, "text": text, "stop": str(stop or ""),
                    "in_tok": itok, "out_tok": otok, "rtok": rtok,
                    "params_used": pu}
        except Exception as e:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            if isinstance(e, BudgetExceeded):
                raise                      # handled by the run mode, not here
            st = P._status_of(e)
            msg = str(e).lower()
            if isinstance(e, RuntimeError) and "fatal" in msg:
                sys.exit(f"FATAL adapter error on {arm_name} (aborting, "
                         f"state preserved): {P._scrub(str(e))[:300]}")
            # output-limit check FIRST: its message can also contain the
            # word 'reasoning', which must not trip the knob-rejection
            # abort. Condition requires 'limit ... reached' language so a
            # genuine parameter REJECTION still falls through to fatal
            # (gate-3 finding).
            if st == 400 and ("output limit" in msg
                              or (("max_tokens" in msg
                                   or "max_completion_tokens" in msg)
                                  and ("reached" in msg
                                       or "limit" in msg))):
                # reasoning consumed the whole cap: truncation semantics.
                # The provider BILLS those reasoning tokens even though the
                # error carries no usage object, so charge the full cap -
                # conservative - or the budget guard goes blind exactly in
                # the heavy-reasoning regime (gate-3 finding).
                USAGE[f"{arm_name}_calls"] += 1
                USAGE[f"{arm_name}_out"] += cap
                USAGE[f"{arm_name}_capburns"] += 1
                if spent_usd() > BUDGET["usd"]:
                    raise BudgetExceeded(
                        f"spend ${spent_usd():.2f} (incl. cap-burn "
                        f"estimates) crossed ${BUDGET['usd']}")
                return {"ok": True, "text": "", "stop": "length",
                        "in_tok": None, "out_tok": cap, "rtok": None,
                        "params_used": {"cap": cap, "note": "output-limit"}}
            if st == 400 and ("reasoning_effort" in msg
                              or "reasoning" in msg):
                sys.exit(f"FATAL: {arm_name} rejected the reasoning knob - "
                         f"the arm cannot exist as designed. Aborting "
                         f"(no silent downgrade). {P._scrub(str(e))[:300]}")
            if not P._is_transient(e):
                sys.exit(f"FATAL non-retryable API error on {arm_name} "
                         f"(aborting, state preserved): "
                         f"{P._scrub(str(e))[:300]}")
            last = e
            # visible retries (pilot discovery: a silent retry storm looks
            # exactly like a healthy slow run from the outside)
            print(f"    transport retry {attempt + 1}/{TRANSIENT_RETRIES} "
                  f"on {arm_name}: {type(e).__name__}: "
                  f"{P._scrub(str(e))[:100]}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    USAGE[f"{arm_name}_failed"] += 1
    return {"ok": False, "text": None, "stop": "",
            "in_tok": None, "out_tok": None, "rtok": None,
            "params_used": {}, "error": P._scrub(str(last))[:200]}

def spent_usd():
    total = 0.0
    for arm_name, arm in ARMS.items():
        pin, pout = M4_PRICING[arm["model"]]
        out = USAGE[f"{arm_name}_out"]
        if arm["provider"] == "xai":
            out += USAGE[f"{arm_name}_rtok"]     # conservative (see header)
        if arm["provider"] == "openai":
            pass    # reasoning tokens already inside completion_tokens
        total += (USAGE[f"{arm_name}_in"] * pin + out * pout) / 1e6
    return total

# ----------------------------- CHANNEL FLOW ----------------------------------

def chat_parse(arm_name, messages, kind, attempts, trial_key):
    """Strict parse with fresh resamples; every attempt's raw text saved."""
    parser = PARSERS[kind]
    counters_key = kind
    for _ in range(PARSE_RETRIES):
        rec = m4_call(arm_name, messages, kind)
        attempts.append({"trial": trial_key, "kind": kind,
                         "raw": rec["text"], "stop": rec["stop"],
                         "rtok": rec["rtok"], "ok": rec["ok"]})
        val = parser(rec["text"])
        if val is not None:
            return val, rec
        if not rec["ok"]:
            USAGE[f"{arm_name}_{counters_key}_transport"] += 1
        elif rec["stop"] in ("length", "max_tokens"):
            USAGE[f"{arm_name}_{counters_key}_truncated"] += 1
        else:
            USAGE[f"{arm_name}_{counters_key}_parse"] += 1
    USAGE[f"{arm_name}_{counters_key}_dropped"] += 1
    return None, rec

def run_trial(arm_name, item, rng, attempts, do_consistency):
    """answer -> {conf, bet} in randomized order (+ optional k consistency
    resamples in fresh conversations)."""
    key = f"{item['family']}/L{item['level']}/i{item['idx']}"
    q = [{"role": "user",
          "content": ANSWER_PROMPT.format(body=item["body"])}]
    pred, arec = chat_parse(arm_name, q, "answer", attempts, key)
    row = {"family": item["family"], "level": item["level"],
           "idx": item["idx"], "true": item["true"], "body": item["body"],
           "params": item["params"], "body_len": len(item["body"]),
           "pred": pred, "correct": (pred == item["true"]),
           "rtok_answer": arec.get("rtok"),
           "stop_answer": arec.get("stop"),
           "confidence": None, "bet": None, "channel_order": None,
           "rtok_conf": None, "rtok_bet": None, "consistency": None}
    if pred is None:
        return row
    hist = q + [{"role": "assistant", "content": str(pred)}]
    order = ["conf", "bet"] if rng.random() < 0.5 else ["bet", "conf"]
    row["channel_order"] = "-".join(order)
    for ch in order:
        prompt = CONF_PROMPT if ch == "conf" else BET_PROMPT
        val, rec = chat_parse(arm_name, hist +
                              [{"role": "user", "content": prompt}],
                              ch, attempts, key)
        if ch == "conf":
            row["confidence"], row["rtok_conf"] = val, rec.get("rtok")
            hist = hist + [{"role": "user", "content": prompt},
                           {"role": "assistant", "content": str(val)}] \
                if val is not None else hist
        else:
            row["bet"], row["rtok_bet"] = val, rec.get("rtok")
            hist = hist + [{"role": "user", "content": prompt},
                           {"role": "assistant", "content": str(val)}] \
                if val is not None else hist
    if do_consistency:
        cons = []
        for k in range(CONSISTENCY_K):
            cpred, _ = chat_parse(arm_name, q, "answer", attempts,
                                  key + f"/cons{k}")
            cons.append(cpred)
        row["consistency"] = cons
    return row

# ----------------------------- CALIBRATION -----------------------------------

def probe_accuracy(arm_name, family, level, n, seed, attempts, idx_off=0):
    """Post-freeze amendment v5: a probe item that times out or drops is
    counted WRONG, not unscored - failing to produce an answer within a
    35-min wall budget is difficulty evidence at that level, and treating
    it as missing made an L9 hang flag a family that works fine at L7."""
    correct = 0
    dropped = 0
    for i in range(n):
        item = F.make_item(family, level, seed,
                           PROBE_IDX0 + level * 50 + idx_off + i)
        q = [{"role": "user",
              "content": ANSWER_PROMPT.format(body=item["body"])}]
        pred, _ = chat_parse(arm_name, q, "answer", attempts,
                             f"probe/{family}/L{level}/{i}")
        if pred is None:
            dropped += 1
        else:
            correct += (pred == item["true"])
    return correct / n, n - dropped

def calibrate(arm_name, family, seed, attempts):
    """Walk levels to land probe accuracy inside CAL_BAND, bisecting the
    midpoint once a too-easy and a too-hard level bracket the band.
    Returns (locked_level, trace, dissolution_flag)."""
    level = CAL_START[family]
    trace = []
    too_easy = None      # highest probed level above the band
    too_hard = None      # lowest probed level below the band
    probed = set()
    for _probe in range(CAL_MAX_PROBES):
        acc, scored = probe_accuracy(arm_name, family, level, CAL_PROBE_N,
                                     seed, attempts)
        probed.add(level)
        trace.append({"level": level, "acc": acc, "answered": scored})
        print(f"    cal {family} L{level}: acc={acc} ({scored} answered) "
              f"[${spent_usd():.2f}]", flush=True)
        if acc > CAL_BAND[1]:
            too_easy = level if too_easy is None else max(too_easy, level)
            if level >= F.N_LEVELS - 1:
                # top of the ladder: check the dissolution gate properly
                # (idx_off keeps recheck items disjoint from the probe's)
                acc2, sc2 = probe_accuracy(arm_name, family, level, 6, seed,
                                           attempts, idx_off=20)
                trace.append({"level": level, "acc": acc2, "answered": sc2,
                              "note": "dissolution-check"})
                # accuracies are over item counts (drops = wrong), so pool
                # by item counts
                pooled = (acc * CAL_PROBE_N + acc2 * 6) / (CAL_PROBE_N + 6)
                if pooled >= DISSOLUTION_ACC:
                    return None, trace, f"dissolved(acc={pooled:.2f}@L9)"
                return level, trace, None
            nxt = ((level + too_hard) // 2 if too_hard is not None
                   else min(F.N_LEVELS - 1, level + 2))
        elif acc < CAL_BAND[0]:
            too_hard = level if too_hard is None else min(too_hard, level)
            if level == 0:
                return level, trace, None      # floor; keep, report
            nxt = ((level + too_easy) // 2 if too_easy is not None
                   else max(0, level - 2))
        else:
            return level, trace, None
        if nxt in probed or nxt == level:
            break                              # nothing new to learn
        level = nxt
    # walk exhausted without landing in band: lock the PROBED level whose
    # accuracy sat closest to the band center, and say so in the trace
    # (gate-3 finding: the old code returned the next never-probed level)
    probed = [t for t in trace if t.get("acc") is not None]
    best = min(probed, key=lambda t: abs(t["acc"] - 0.5))
    trace.append({"note": f"best-effort lock L{best['level']} "
                          f"(acc={best['acc']:.2f}, walk exhausted)"})
    print(f"    cal {family}: walk exhausted - best-effort lock "
          f"L{best['level']} acc={best['acc']:.2f}", flush=True)
    return best["level"], trace, None

# ----------------------------- RUN MODES -------------------------------------

def unique_path(path):
    if not os.path.exists(path):
        return path
    k = 2
    base, ext = os.path.splitext(path)
    while os.path.exists(f"{base}_run{k}{ext}"):
        k += 1
    return f"{base}_run{k}{ext}"

def ping(arm_name):
    """Tiny paid calls: verify knob acceptance, parseability, token
    reporting. For sol arms this ALSO pings the other effort value so both
    arm settings are live-confirmed before any full run (gate-3 fix: the
    old version only claimed to)."""
    print(f"PING {arm_name} (tiny paid calls)")

    def one(arm, prompt, cap_key, expect, label):
        rec = m4_call(arm, [{"role": "user", "content": prompt}], cap_key)
        txt = (rec.get("text") or "")
        print(f"  {label}: ok={rec['ok']} text={txt[:30]!r} "
              f"stop={rec['stop']} rtok={rec['rtok']} "
              f"in={rec['in_tok']} out={rec['out_tok']}")
        return rec["ok"] and expect(txt)

    int_probe = ("What is 340 + 217?\n\nYour entire reply must be exactly "
                 "one integer and nothing else - no words, no working, no "
                 "units, no punctuation. Reply now with the final integer "
                 "answer.")
    ok = one(arm_name, int_probe, "conf",
             lambda t: parse_int_strict(t) == 557, "int-shaped")
    ok = one(arm_name, "Reply with ONLY the word KEEP.", "bet",
             lambda t: parse_bet_strict(t) == "KEEP", "bet-shaped") and ok
    if arm_name.startswith("sol-"):
        other = "sol-high" if arm_name == "sol-low" else "sol-low"
        ok = one(other, int_probe, "conf",
                 lambda t: parse_int_strict(t) == 557,
                 f"other-effort {other}") and ok
    print(f"  spent so far: ${spent_usd():.3f}")
    return ok

def run_block(arm_name, family, level, n, idx0, seed, attempts,
              consistency_items, rows_out=None, start_i=0):
    """rows_out: pass the results dict's own list so partial rows survive
    an abort via the caller's finally block (gate-3 fix). start_i: resume a
    budget-aborted block mid-run (items are deterministic in idx, so a
    continuation covers exactly the missing tail; consistency_items is an
    absolute item threshold and composes correctly with start_i)."""
    rng = random.Random(seed * 7919 + level)
    rows = rows_out if rows_out is not None else []
    for i in range(start_i, n):
        item = F.make_item(family, level, seed, idx0 + i)
        row = run_trial(arm_name, item, rng, attempts,
                        do_consistency=(i < consistency_items))
        rows.append(row)
        if (i + 1) % 5 == 0 or i == n - 1:
            nc = sum(1 for r in rows if r["correct"])
            print(f"    {family} L{level}: {i+1}/{n} acc so far "
                  f"{nc}/{len(rows)} [${spent_usd():.2f}]", flush=True)
    return rows

def run_pilot(arm_name, seed):
    if not P.key_present(ARMS[arm_name]["provider"]):
        sys.exit(f"key for {arm_name} absent - refusing")
    print(f"=== PILOT {arm_name} seed={seed} "
          f"(n={PILOT_N}/family, consistency on "
          f"{PILOT_CONSISTENCY_ITEMS} items x k={CONSISTENCY_K}) ===")
    if not ping(arm_name):
        sys.exit("ping failed - refusing to spend on a full pilot")
    attempts = []
    out = {"mode": "pilot", "arm": arm_name, "arm_def": ARMS[arm_name],
           "caps": CAPS[ARMS[arm_name]["provider"]], "seed": seed,
           "pass_val": PASS_VAL, "families": {}, "started": time.time(),
           "provenance": bs.provenance(
               ["m4_study.py", "m4_families.py", "bt_stats.py",
                "providers.py"],
               {"arm": arm_name, "seed": seed, "n": PILOT_N},
               "m4-pilot-1")}
    try:
        for family in F.FAMILIES:
            print(f"  [{family}] calibration walk")
            level, trace, flag = calibrate(arm_name, family, seed, attempts)
            fam = {"cal_trace": trace, "locked_level": level,
                   "dissolution_flag": flag, "rows": []}
            out["families"][family] = fam
            if flag:
                print(f"  [{family}] FLAG: {flag} - skipping block")
                continue
            run_block(arm_name, family, level, PILOT_N,
                      PILOT_IDX0, seed, attempts,
                      PILOT_CONSISTENCY_ITEMS, rows_out=fam["rows"])
    except BudgetExceeded as e:
        out["aborted_budget"] = str(e)
        print(f"\nBUDGET ABORT: {e} - saving partial state", flush=True)
    finally:
        out["usage"] = dict(USAGE)
        out["regime_tally"] = dict(REGIME)
        out["spent_usd"] = spent_usd()
        out["finished"] = time.time()
        path = unique_path(f"m4pilot_{arm_name}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        apath = unique_path(f"m4pilot_{arm_name}_attempts.jsonl")
        with open(apath, "w") as f:
            for a in attempts:
                f.write(json.dumps(a) + "\n")
        print(f"\npilot saved: {path} + {apath}  spent=${spent_usd():.2f}")
    analyze_run(out, seed=seed)
    return out

def run_main(arm_name, seed, levels_json, approved, start_idx=0):
    if not approved:
        sys.exit("MAIN RUN BLOCKED: requires --approved-by-reid after the "
                 "gate-4 cost projection is approved.")
    with open(levels_json) as f:
        levels = json.load(f)          # {"family": level} locked from pilot
    print(f"=== MAIN {arm_name} seed={seed} n={MAIN_N}/family "
          f"levels={levels} ===")
    if not ping(arm_name):
        sys.exit("ping failed")
    attempts = []
    out = {"mode": "main", "arm": arm_name, "arm_def": ARMS[arm_name],
           "caps": CAPS[ARMS[arm_name]["provider"]], "seed": seed,
           "pass_val": PASS_VAL, "levels": levels, "families": {},
           "started": time.time(),
           "provenance": bs.provenance(
               ["m4_study.py", "m4_families.py", "bt_stats.py",
                "providers.py"],
               {"arm": arm_name, "seed": seed, "n": MAIN_N,
                "levels": levels}, "m4-main-1")}
    # gate-3 critical fix: the ONE-SHOT main run must preserve state on
    # EVERY exit path (BudgetExceeded, FATAL sys.exit inside m4_call,
    # KeyboardInterrupt) - the finally block runs on SystemExit too.
    try:
        for family in F.FAMILIES:
            if family not in levels or levels[family] is None:
                print(f"  [{family}] no locked level (pilot flag) - skip")
                out["families"][family] = {"skipped": "no locked level"}
                continue
            fam = {"locked_level": levels[family], "rows": [],
                   "start_idx": start_idx}
            out["families"][family] = fam
            run_block(arm_name, family, levels[family], MAIN_N,
                      MAIN_IDX0, seed, attempts,
                      MAIN_CONSISTENCY_ITEMS, rows_out=fam["rows"],
                      start_i=start_idx)
    except BudgetExceeded as e:
        out["aborted_budget"] = str(e)
        print(f"\nBUDGET ABORT: {e} - saving partial state", flush=True)
    finally:
        out["usage"] = dict(USAGE)
        out["regime_tally"] = dict(REGIME)
        out["spent_usd"] = spent_usd()
        out["finished"] = time.time()
        path = unique_path(f"m4main_{arm_name}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        apath = unique_path(f"m4main_{arm_name}_attempts.jsonl")
        with open(apath, "w") as f:
            for a in attempts:
                f.write(json.dumps(a) + "\n")
        print(f"main saved: {path}  spent=${spent_usd():.2f}", flush=True)
    analyze_run(out, seed=seed)

# ----------------------------- ANALYSIS --------------------------------------

def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"

def cue_battery(rows, pooled=False):
    """Mission-fixed battery: prompt length, answer magnitude, family
    identity (pooled only), and the problem-size parameters. Orientation is
    chosen ON THIS DATA to maximize each cue's AUROC - deliberately favoring
    the cue (v5 form; conservative for 'confidence beats cues')."""
    feats = {"length": [-r["body_len"] for r in rows],
             "magnitude": [-abs(r["pred"]) for r in rows]}
    pkeys = sorted({k for r in rows for k in r["params"]})
    for k in pkeys:
        vals = [float(r["params"].get(k, 0)) for r in rows]
        if len(set(vals)) > 1:
            feats[f"param:{k}"] = [-v for v in vals]
    if pooled:
        fams = sorted({r["family"] for r in rows})
        for fam in fams:
            feats[f"family:{fam}"] = [1.0 if r["family"] == fam else 0.0
                                      for r in rows]
    labels = [r["correct"] for r in rows]
    out = {}
    for cue, scores in feats.items():
        a = bs.auroc(list(zip(scores, labels)))
        if a is None:
            continue
        if a < 0.5:
            scores = [-s for s in scores]
        out[cue] = scores
    return out

def analyze_family_rows(name, rows, seed=0):
    """PRIMARY (v5 form) + S1 + S3 + descriptive channels for one family of
    one arm. Pure function of rows - re-runnable offline."""
    out = {"family": name, "n": len(rows)}
    scored = [r for r in rows if r["confidence"] is not None
              and r["pred"] is not None]
    # preregistered drop accounting: answer or confidence drops above 10%
    # of trials flag the family; a flagged family's PRIMARY cannot confirm
    # (adverse bounding - drops counted against metacognition)
    if rows:
        n_drop = sum(1 for r in rows if r["pred"] is None
                     or r["confidence"] is None)
        out["drop_rate"] = n_drop / len(rows)
        if out["drop_rate"] > 0.10:
            out["drop_flag"] = True
            print(f"\n[{name}] DROP FLAG: {n_drop}/{len(rows)} trials "
                  f"lost an answer or confidence - PRIMARY bounded "
                  f"(cannot confirm)")
    if len(scored) < 10:
        print(f"\n[{name}] n_scored={len(scored)} - too few")
        return out
    labels = [r["correct"] for r in scored]
    acc = mean(1.0 if c else 0.0 for c in labels)
    k = sum(1 for c in labels if c)
    wlo, whi = bs.wilson_ci(k, len(labels))
    confs = [r["confidence"] for r in scored]
    gap = mean(confs) - 100 * acc
    out["acc"], out["mean_conf"], out["gap"] = acc, mean(confs), gap
    print(f"\n[{name}] n={len(scored)} acc={acc:.2f} "
          f"[{fmt(wlo,2)},{fmt(whi,2)}] conf={mean(confs):.1f} "
          f"sd={pstdev(confs):.1f} gap={gap:+.1f}")

    # PRIMARY: conf vs best cue
    conf_pairs = list(zip(confs, labels))
    a_conf = bs.auroc(conf_pairs)
    clo, chi = bs.boot_ci_auroc(conf_pairs, seed=seed)
    print(f"  conf AUROC2={fmt(a_conf)} CI[{fmt(clo)},{fmt(chi)}]")
    out["auroc_conf"] = a_conf
    cues = cue_battery(scored)
    best_cue = best_a = best_scores = None
    for cue, scores in sorted(cues.items()):
        a = bs.auroc(list(zip(scores, labels)))
        print(f"    cue[{cue:16s}] AUROC2={fmt(a)} (oriented)")
        if best_a is None or a > best_a:
            best_cue, best_a, best_scores = cue, a, scores
    if best_scores is not None:
        d, dlo, dhi, pval = bs.paired_auroc_diff_p(
            conf_pairs, list(zip(best_scores, labels)), seed=seed)
        beats = (dlo is not None and dlo > 0)
        out["primary"] = {"diff": d, "lo": dlo, "hi": dhi, "p": pval,
                          "best_cue": best_cue, "best_cue_auroc": best_a,
                          "beats": beats}
        print(f"  PRIMARY conf-minus-best-cue({best_cue}): {fmt(d)} "
              f"CI[{fmt(dlo)},{fmt(dhi)}] p={fmt(pval,4)} -> "
              f"{'BEATS cue' if beats else 'does NOT beat cue'}")

    # S1: reasoning-token count (fixed orientation: more tokens = less conf)
    rt_rows = [r for r in scored if r.get("rtok_answer") is not None]
    if len(rt_rows) >= 10:
        rt_labels = [r["correct"] for r in rt_rows]
        rt_pairs = [(-r["rtok_answer"], c) for r, c in zip(rt_rows,
                                                           rt_labels)]
        cf_pairs = [(r["confidence"], c) for r, c in zip(rt_rows, rt_labels)]
        a_rt = bs.auroc(rt_pairs)
        d, dlo, dhi, pval = bs.paired_auroc_diff_p(rt_pairs, cf_pairs,
                                                   seed=seed)
        out["s1"] = {"auroc_rtok": a_rt, "diff_rtok_minus_conf": d,
                     "lo": dlo, "hi": dhi, "p": pval}
        print(f"  S1 rtok AUROC2={fmt(a_rt)}; rtok-minus-conf {fmt(d)} "
              f"CI[{fmt(dlo)},{fmt(dhi)}] p={fmt(pval,4)}")

    # bets (descriptive)
    bets = [r for r in scored if r["bet"] is not None]
    if bets:
        keep = [r for r in bets if r["bet"] == "KEEP"]
        acc_keep = mean(1.0 if r["correct"] else 0.0
                        for r in keep) if keep else None
        pass_r = [r for r in bets if r["bet"] == "PASS"]
        acc_pass = mean(1.0 if r["correct"] else 0.0
                        for r in pass_r) if pass_r else None
        bet_pairs = [(1.0 if r["bet"] == "KEEP" else 0.0, r["correct"])
                     for r in bets]
        out["bets"] = {"n": len(bets), "keep_frac": len(keep) / len(bets),
                       "acc_keep": acc_keep, "acc_pass": acc_pass,
                       "auroc_bet": bs.auroc(bet_pairs)}
        print(f"  bets: keep={len(keep)}/{len(bets)} acc_keep="
              f"{fmt(acc_keep,2)} acc_pass={fmt(acc_pass,2)} "
              f"AUROC={fmt(out['bets']['auroc_bet'])}")

    # consistency (descriptive)
    cons = [r for r in scored if r.get("consistency")]
    if len(cons) >= 3:
        cscore = [mean(1.0 if c == r["pred"] else 0.0
                       for c in r["consistency"]) for r in cons]
        cpairs = list(zip(cscore, [r["correct"] for r in cons]))
        out["consistency"] = {"n": len(cons),
                              "auroc": bs.auroc(cpairs),
                              "mean_stability": mean(cscore)}
        print(f"  consistency(n={len(cons)}): stability="
              f"{mean(cscore):.2f} AUROC={fmt(out['consistency']['auroc'])}")
    return out

def analyze_run(out, seed=0):
    print(f"\n===== ANALYSIS {out['arm']} ({out['mode']}) =====")
    fam_outs = {}
    prim_ps, prim_names = [], []
    all_scored = []
    for family, fam in out["families"].items():
        rows = fam.get("rows") or []
        if fam.get("dissolution_flag"):
            print(f"\n[{family}] FLAGGED: {fam['dissolution_flag']}")
            continue
        res = analyze_family_rows(family, rows, seed=seed)
        fam_outs[family] = res
        all_scored += [r for r in rows if r["confidence"] is not None
                       and r["pred"] is not None]
        if res.get("primary"):
            # undefined p enters Holm as 1.0 so the correction's family
            # size m never silently shrinks (gate-3 finding)
            prim_ps.append(res["primary"]["p"]
                           if res["primary"]["p"] is not None else 1.0)
            prim_names.append(family)
    confirmed = []
    if prim_ps:
        rejected = bs.holm(prim_ps)
        for fam_name, rej in zip(prim_names, rejected):
            res = fam_outs[fam_name]
            # prereg criterion: Holm rejection AND paired-diff CI lower
            # bound > 0 AND no drop flag (gate-3 critical fix: Holm alone
            # was being reported as confirmation)
            if rej and res["primary"].get("beats") \
                    and not res.get("drop_flag"):
                confirmed.append(fam_name)
        bounded = [n for n in prim_names if fam_outs[n].get("drop_flag")]
        print(f"\nPRIMARY across families (Holm AND CI>0 AND no drop "
              f"flag): confirmed in {confirmed if confirmed else 'NONE'}"
              + (f"; BOUNDED by drops: {bounded}" if bounded else ""))
    # S3 preregistered readout: pooled calibration gap with item-bootstrap CI
    s3 = None
    if len(all_scored) >= 20:
        gaps = [r["confidence"] - (100.0 if r["correct"] else 0.0)
                for r in all_scored]
        gap = mean(gaps)
        rng = random.Random(seed)
        boots = []
        for _ in range(2000):
            samp = [gaps[rng.randrange(len(gaps))] for _ in range(len(gaps))]
            boots.append(mean(samp))
        boots.sort()
        s3 = {"gap": gap, "ci": [boots[int(0.025 * len(boots))],
                                 boots[min(len(boots) - 1,
                                           int(0.975 * len(boots)))]],
              "n": len(all_scored)}
        direction = ("UNDERconfident" if s3["ci"][1] < 0 else
                     "OVERconfident" if s3["ci"][0] > 0 else
                     "CI includes 0")
        print(f"S3 pooled calibration gap: {gap:+.1f} "
              f"CI[{s3['ci'][0]:+.1f},{s3['ci'][1]:+.1f}] "
              f"(n={len(all_scored)}) -> {direction}")
    # pooled cross-family SECONDARY: the battery gains family-identity
    # indicator cues here (per the mission's cue list; constant within a
    # single family, so only meaningful pooled)
    pooled = None
    if len(all_scored) >= 40 and len({r["family"] for r in all_scored}) > 1:
        labels = [r["correct"] for r in all_scored]
        conf_pairs = [(r["confidence"], c)
                      for r, c in zip(all_scored, labels)]
        a_conf = bs.auroc(conf_pairs)
        best_cue = best_a = best_scores = None
        for cue, scores in cue_battery(all_scored, pooled=True).items():
            a = bs.auroc(list(zip(scores, labels)))
            if best_a is None or a > best_a:
                best_cue, best_a, best_scores = cue, a, scores
        d, dlo, dhi, pval = bs.paired_auroc_diff_p(
            conf_pairs, list(zip(best_scores, labels)), seed=seed)
        pooled = {"auroc_conf": a_conf, "best_cue": best_cue,
                  "best_cue_auroc": best_a, "diff": d, "lo": dlo,
                  "hi": dhi, "p": pval,
                  "beats": (dlo is not None and dlo > 0)}
        print(f"POOLED secondary: conf AUROC={fmt(a_conf)} vs best cue "
              f"{best_cue}={fmt(best_a)}; diff {fmt(d)} "
              f"CI[{fmt(dlo)},{fmt(dhi)}] p={fmt(pval, 4)}")
    return {"families": fam_outs, "primary_holm_confirmed": confirmed,
            "s3_pooled_gap": s3, "pooled_secondary": pooled}

def s2_effort_contrast(rows_low, rows_high, iters=2000, seed=0):
    """sol-low vs sol-high on the SAME items: joint item bootstrap of
    AUROC2(conf) difference (high - low) + one-sided p for a rise, plus a
    McNemar accuracy contrast. Items aligned by (family, idx)."""
    kl = {(r["family"], r["idx"]): r for r in rows_low
          if r["confidence"] is not None}
    kh = {(r["family"], r["idx"]): r for r in rows_high
          if r["confidence"] is not None}
    keys = sorted(set(kl) & set(kh))
    # gate-3 fix: (family, idx) alone does not identify an item - level and
    # seed do too. Enforce identity via level and body; ANY mismatch means
    # the two arms did not run the same problems and S2 is invalid.
    mismatched = [k for k in keys
                  if kl[k].get("level") != kh[k].get("level")
                  or (kl[k].get("body") is not None
                      and kh[k].get("body") is not None
                      and kl[k]["body"] != kh[k]["body"])]
    if mismatched:
        return {"n_items": len(keys), "n_mismatched": len(mismatched),
                "note": "ITEM MISMATCH between arms - S2 INVALID "
                        "(different levels/seeds ran different problems)"}
    if len(keys) < 20:
        return {"n_items": len(keys), "note": "too few shared items"}
    lo_rows = [kl[k] for k in keys]
    hi_rows = [kh[k] for k in keys]
    a_lo = bs.auroc([(r["confidence"], r["correct"]) for r in lo_rows])
    a_hi = bs.auroc([(r["confidence"], r["correct"]) for r in hi_rows])
    rng = random.Random(seed)
    diffs = []
    skipped = 0
    for _ in range(iters):
        idx = [rng.randrange(len(keys)) for _ in range(len(keys))]
        al = bs.auroc([(lo_rows[i]["confidence"], lo_rows[i]["correct"])
                       for i in idx])
        ah = bs.auroc([(hi_rows[i]["confidence"], hi_rows[i]["correct"])
                       for i in idx])
        if al is None or ah is None:
            skipped += 1
            continue
        diffs.append(ah - al)
    diffs.sort()
    if not diffs or skipped > 0.10 * iters:
        return {"n_items": len(keys), "bootstrap_skipped": skipped,
                "note": "bootstrap degenerate (class loss in >10% of "
                        "resamples) - S2 readout UNSTABLE"}
    p_rise = (1 + sum(1 for d in diffs if d <= 0)) / (len(diffs) + 1)
    b = sum(1 for l, h in zip(lo_rows, hi_rows)
            if l["correct"] and not h["correct"])
    c = sum(1 for l, h in zip(lo_rows, hi_rows)
            if not l["correct"] and h["correct"])
    return {"n_items": len(keys), "auroc_low": a_lo, "auroc_high": a_hi,
            "diff": (a_hi - a_lo) if (a_hi is not None and a_lo is not None)
            else None,
            "ci": [diffs[int(0.025 * len(diffs))],
                   diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]],
            "p_rise": p_rise, "bootstrap_skipped": skipped,
            "acc_low": mean(1.0 if r["correct"] else 0.0 for r in lo_rows),
            "acc_high": mean(1.0 if r["correct"] else 0.0 for r in hi_rows),
            "mcnemar_p": bs.mcnemar_exact(b, c)}

# ----------------------------- SELFTEST --------------------------------------

ORACLE = {}          # body -> (true, p_err); populated by selftest makers

def _synth_item(family, level, rng, i):
    """Synthetic item for bot-ranking selftests: body length scales with
    level (so the length cue carries level information), params carry the
    level, p_err = level trend + independent item noise (the calibrated
    bot's private edge over every cue)."""
    body = f"[{family} synthetic level {level} item {i}] " + \
           "x" * (120 + 90 * level + rng.randrange(0, 40))
    true = rng.randrange(2, 9000)
    p_err = min(0.95, max(0.02,
                          0.10 + 0.085 * level + rng.uniform(-0.30, 0.30)))
    ORACLE[body] = (true, p_err)
    return {"family": family, "level": level, "idx": i, "body": body,
            "true": true, "params": {f"{family[:2]}_size": level}}

def _stable_hash(s):
    """Process-stable string hash (built-in hash() is randomized per process
    - the m4_families item_rng lesson, relearned here in the selftest)."""
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")

def _mk_bot(name):
    """Returns a fake transport fn(model, messages, cap, effort) with the
    bot's answer/conf/bet/rtok policy. Bots see only what a model would."""
    brng = random.Random(_stable_hash(name))

    def bot(model, messages, cap, effort):
        last = messages[-1]["content"]
        if last.startswith("How confident"):
            body = messages[0]["content"]
            true, p_err = ORACLE[_body_key(body)]
            if name == "calibrated":
                conf = round(min(100, max(0, (1 - p_err) * 100
                                          + brng.gauss(0, 4))))
            elif name == "cue_reader":
                conf = round(min(100, max(0, 100 - len(body) / 14
                                          + brng.gauss(0, 2))))
            elif name == "token_honest":
                conf = 90
            else:
                conf = 90
            return (str(conf), "stop", 50, 5, _bot_rtok(name, p_err, brng),
                    {"bot": name})
        if last.startswith("Now a bet"):
            return ("KEEP", "stop", 50, 2, 0, {"bot": name})
        body = _body_key(last)
        true, p_err = ORACLE[body]
        if name == "perfect":
            pred = true
        else:
            pred = true if brng.random() > p_err else true + \
                brng.randrange(1, 9)
        return (str(pred), "stop", 200, 8, _bot_rtok(name, p_err, brng),
                {"bot": name})
    return bot

def _bot_rtok(name, p_err, brng):
    if name == "token_honest":
        return int(300 + 6000 * p_err + brng.gauss(0, 150))
    return int(400 + brng.gauss(0, 60))

def _body_key(prompt_content):
    for body in ORACLE:
        if body in prompt_content:
            return body
    raise KeyError("selftest bot could not find item body in prompt")

def selftest():
    print("m4_study selftest: OFFLINE (transports swapped, clients raise)\n")
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    # parser fuzz
    check(parse_int_strict(" **-42** ") == -42
          and parse_int_strict("42 is the answer") is None
          and parse_int_strict("") is None,
          "int parser: bare only, no prose")
    check(parse_conf_strict("85%") == 85
          and parse_conf_strict("about 85") is None
          and parse_conf_strict("300") is None,
          "conf parser: bare 0-100 only")
    check(parse_bet_strict(" KEEP.") == "KEEP"
          and parse_bet_strict("I would KEEP") is None
          and parse_bet_strict("KEEP or PASS") is None,
          "bet parser: bare KEEP/PASS only")

    # no-network guard (covers m4's own client factory too)
    global _m4_client
    real_transport = dict(TRANSPORT)
    real_clients = (P._openai_client, P._anthropic_client, _m4_client)
    def _no_net(*a, **k):
        raise RuntimeError("OFFLINE selftest attempted network access")
    P._openai_client = _no_net
    P._anthropic_client = _no_net
    _m4_client = _no_net

    try:
        # ---- integration smoke on REAL generated items ----
        USAGE.clear(); REGIME.clear(); ORACLE.clear()
        for fam in F.FAMILIES:
            for lvl in (1, 3):
                for i in (0, 1):
                    it = F.make_item(fam, lvl, 11, i)
                    ORACLE[it["body"]] = (it["true"], 0.3)
        TRANSPORT["openai"] = _mk_bot("calibrated")
        rng = random.Random(0)
        attempts = []
        smoke_rows = []
        for fam in F.FAMILIES:
            it = F.make_item(fam, 1, 11, 0)
            smoke_rows.append(run_trial("sol-high", it, rng, attempts,
                                        do_consistency=True))
        check(all(r["pred"] is not None and r["confidence"] is not None
                  and r["bet"] is not None and len(r["consistency"]) == 3
                  for r in smoke_rows),
              "integration smoke: all channels populated on real items")
        check(all(a["kind"] in PARSERS for a in attempts)
              and len(attempts) >= 4 * (3 + CONSISTENCY_K),
              "integration smoke: attempts logged for every call")

        # ---- bot-ranking on synthetic items ----
        def run_bot_arm(bot_name, n=170, seed=5):
            ORACLE.clear()
            TRANSPORT["openai"] = _mk_bot(bot_name)
            rng = random.Random(seed)
            rows = []
            att = []
            for fam in F.FAMILIES:
                srng = random.Random(seed * 31 + _stable_hash(fam) % 1000)
                for i in range(n):
                    lvl = srng.randrange(1, 8)
                    it = _synth_item(fam, lvl, srng, i)
                    rows.append(run_trial("sol-high", it, rng, att,
                                          do_consistency=False))
            return rows

        def primary_confirmed_families(rows, seed=5):
            """Routes through analyze_run so the REAL confirm criterion
            (Holm AND CI>0 AND no drop flag) is what gets exercised."""
            byfam = {}
            for r in rows:
                byfam.setdefault(r["family"], []).append(r)
            out = {"arm": "sol-high", "mode": "selftest",
                   "families": {f: {"rows": rr} for f, rr in byfam.items()}}
            res = analyze_run(out, seed=seed)
            return res["primary_holm_confirmed"], res["families"]

        print("\n  --- bot: calibrated (conf SHOULD beat cues) ---")
        rows_cal = run_bot_arm("calibrated")
        conf_cal, res_cal = primary_confirmed_families(rows_cal)
        check(len(conf_cal) >= 3,
              f"calibrated bot: primary confirmed in >=3 families "
              f"(got {len(conf_cal)}: {conf_cal})")

        print("\n  --- bot: cue_reader (conf must NOT beat cues) ---")
        rows_cue = run_bot_arm("cue_reader")
        conf_cue, res_cue = primary_confirmed_families(rows_cue)
        check(len(conf_cue) == 0,
              f"cue_reader bot: primary confirmed NOWHERE (got {conf_cue})")

        print("\n  --- bot: token_honest (S1: rtok must beat conf) ---")
        rows_tok = run_bot_arm("token_honest")
        _, res_tok = primary_confirmed_families(rows_tok)
        s1_wins = sum(1 for fam, r in res_tok.items()
                      if r.get("s1") and r["s1"]["p"] is not None
                      and r["s1"]["p"] < 0.05
                      and r["s1"]["diff_rtok_minus_conf"] > 0)
        check(s1_wins >= 3,
              f"token_honest bot: S1 rtok>conf in >=3 families "
              f"(got {s1_wins})")
        s1_cal_wins = sum(1 for fam, r in res_cal.items()
                          if r.get("s1") and r["s1"]["p"] is not None
                          and r["s1"]["p"] < 0.05
                          and r["s1"]["diff_rtok_minus_conf"] > 0)
        check(s1_cal_wins == 0,
              f"calibrated bot: S1 does NOT credit noise tokens "
              f"(got {s1_cal_wins})")

        # ---- S2 machinery: detects a rise only when one exists ----
        def synth_arm_rows(conf_mode, err_scale, seed=9, level_shift=0):
            rng = random.Random(seed)
            rows = []
            for fam in F.FAMILIES:
                for i in range(80):
                    lvl = rng.randrange(1, 8)
                    p_err = min(0.9, max(0.05,
                                         (0.1 + 0.08 * lvl) * err_scale))
                    correct = rng.random() > p_err
                    conf = (round((1 - p_err) * 100) if conf_mode == "cal"
                            else 90)
                    rows.append({"family": fam, "idx": i,
                                 "level": lvl + level_shift,
                                 "confidence": conf, "correct": correct})
            return rows
        r_pin_lo = synth_arm_rows("pin", 1.0, seed=9)
        r_pin_hi = synth_arm_rows("pin", 0.6, seed=9)
        r_cal_hi = synth_arm_rows("cal", 0.6, seed=9)
        s2_null = s2_effort_contrast(r_pin_lo, r_pin_hi, seed=1)
        s2_rise = s2_effort_contrast(r_pin_lo, r_cal_hi, seed=1)
        check(s2_null.get("p_rise", 0) > 0.05,
              f"S2 null: pinned->pinned no rise (p={s2_null.get('p_rise')})")
        check(s2_rise.get("p_rise", 1) < 0.05,
              f"S2 rise: pinned->calibrated detected "
              f"(p={s2_rise.get('p_rise')})")
        # item-identity guard: shifted levels must invalidate S2, not
        # silently compare different problems (gate-3 fix)
        r_shift = synth_arm_rows("cal", 0.6, seed=9, level_shift=1)
        s2_bad = s2_effort_contrast(r_pin_lo, r_shift, seed=1)
        check("INVALID" in s2_bad.get("note", ""),
              f"S2 mismatch guard: shifted levels -> INVALID "
              f"(note={s2_bad.get('note', 'MISSING')!r})")

        # ---- drop rule: >10% lost confidences bound the family ----
        rows_drop = [dict(r) for r in rows_cal
                     if r["family"] == "arith_long"]
        for r in rows_drop[:len(rows_drop) // 6]:      # ~17% drops
            r["confidence"] = None
        res_drop = analyze_run({"arm": "x", "mode": "selftest",
                                "families": {"arith_long":
                                             {"rows": rows_drop}}}, seed=5)
        check(res_drop["families"]["arith_long"].get("drop_flag") is True
              and "arith_long" not in res_drop["primary_holm_confirmed"],
              "drop rule: >10% lost confidences flag the family and block "
              "confirmation")

        # ---- dissolution gate: perfect bot flags the family ----
        ORACLE.clear()
        TRANSPORT["openai"] = _mk_bot("perfect")
        # register probe items the calibrator will generate
        for lvl in range(F.N_LEVELS):
            for i in range(60):
                pass  # probe items are real; register lazily below
        real_make = F.make_item
        def make_and_register(fam, lvl, seed, i):
            it = real_make(fam, lvl, seed, i)
            ORACLE[it["body"]] = (it["true"], 0.0)
            return it
        F.make_item = make_and_register
        try:
            att = []
            lvl, trace, flag = calibrate("sol-high", "arith_long", 13, att)
            check(flag is not None and flag.startswith("dissolved"),
                  f"dissolution gate fires on a perfect bot (flag={flag})")
        finally:
            F.make_item = real_make

        # ---- drop accounting: a noncompliant answer counts as dropped ----
        USAGE.clear()
        def noisy_bot(model, messages, cap, effort):
            return ("The answer is 7, I believe.", "stop", 10, 10, 0, {})
        TRANSPORT["openai"] = noisy_bot
        ORACLE.clear()
        it = _synth_item("arith_long", 2, random.Random(3), 0)
        row = run_trial("sol-high", it, random.Random(3), [], False)
        check(row["pred"] is None
              and USAGE.get("sol-high_answer_dropped") == 1
              and USAGE.get("sol-high_answer_parse") == PARSE_RETRIES,
              "noncompliant answers: parse-counted then dropped, no guess")

        # ---- cost accumulator ----
        USAGE.clear()
        USAGE["sol-high_in"] = 1_000_000
        USAGE["sol-high_out"] = 1_000_000
        c = spent_usd()
        check(abs(c - 35.0) < 1e-6,
              f"cost accumulator: 1M in + 1M out on sol = $35 (got {c})")
        USAGE.clear()

        # ---- unique_path never overwrites ----
        check(unique_path("m4_study.py") != "m4_study.py",
              "unique_path suffixes existing files")
    finally:
        TRANSPORT.update(real_transport)
        P._openai_client, P._anthropic_client, _m4_client = real_clients
        USAGE.clear()
        REGIME.clear()
        ORACLE.clear()

    print("\nselftest", "PASSED" if ok else "FAILED")
    return ok

# ----------------------------- CLI -------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--main", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--levels", help="JSON file {family: level} for --main")
    ap.add_argument("--approved-by-reid", action="store_true")
    ap.add_argument("--analyze", nargs="+",
                    help="offline re-analysis of saved run JSONs")
    ap.add_argument("--budget", type=float, default=None,
                    help="hard USD ceiling for this process (abort+save); "
                         "default 14 for --pilot, REQUIRED for --main")
    ap.add_argument("--consistency-items", type=int, default=None,
                    help="override pilot consistency-item count")
    ap.add_argument("--s2", nargs=2, metavar=("LOW_JSON", "HIGH_JSON"),
                    help="offline S2 effort contrast from two saved runs")
    ap.add_argument("--start-idx", type=int, default=0,
                    help="resume a budget-aborted main block at this item "
                         "index (continuation runs merge at analysis)")
    args = ap.parse_args()
    if args.main and args.budget is None:
        sys.exit("--main requires an explicit --budget sized to the "
                 "approved projection (gate-3 fix: the $14 pilot default "
                 "would abort a $40-90 main run mid-arm)")
    BUDGET["usd"] = args.budget if args.budget is not None else 14.0
    if args.consistency_items is not None:
        PILOT_CONSISTENCY_ITEMS = args.consistency_items
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.s2:
        runs = []
        for path in args.s2:
            with open(path) as f:
                d = json.load(f)
            runs.append([r for fam in d["families"].values()
                         for r in (fam.get("rows") or [])])
        res = s2_effort_contrast(runs[0], runs[1])
        print(json.dumps(res, indent=1))
        sys.exit(0)
    if args.analyze:
        for path in args.analyze:
            with open(path) as f:
                d = json.load(f)
            # re-analysis must reproduce the run's own numbers: use the
            # run's stored seed, not the CLI default (gate-3 finding)
            analyze_run(d, seed=d.get("seed", args.seed))
        sys.exit(0)
    if args.ping:
        if not args.arm:
            sys.exit("--ping needs --arm")
        sys.exit(0 if ping(args.arm) else 1)
    if args.pilot:
        if not args.arm:
            sys.exit("--pilot needs --arm")
        run_pilot(args.arm, args.seed)
        sys.exit(0)
    if args.main:
        if not (args.arm and args.levels):
            sys.exit("--main needs --arm and --levels")
        run_main(args.arm, args.seed, args.levels, args.approved_by_reid,
                 start_idx=args.start_idx)
        sys.exit(0)
    ap.print_help()
