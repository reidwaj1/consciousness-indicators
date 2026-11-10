#!/usr/bin/env python3
"""
HOT-2 v4: the frozen metacognition protocol (Mission 3, Workstream A)
=====================================================================
Indicator anchor (Butlin et al., HOT-2): "metacognitive monitoring
distinguishing reliable perceptual representations from noise." LLMs have no
perception in the paper's sense; this instrument tests the functional analog
— monitoring which of the system's own COMPUTED representations are reliable.
That drift from the indicator's wording is deliberate and flagged in the
README.

FIXED protocol (per problem, 7 calls):
  turn 1  answer only (two-stage elicitation; v3 showed joint elicitation
          depresses accuracy)
  turn 2  stated confidence 0-100 (same conversation)
  turn 3  bet: KEEP (+1 right / -1 wrong) or PASS (+0.3) — rational KEEP
          iff P(correct) > 0.65
  + k=4 fresh single-turn resamples -> answer-consistency (the
    non-introspective rival predictor)

Families (5):
  tracking        v2 generator (unambiguous version) — the family whose
                  signal survived every cue benchmark
  letters_fixed   v2 control (fixed length, frequency varies); its
                  answer-magnitude cue benchmark is computed alongside
  arith_standard  v2 chains (the prior-laden baseline)
  arith_madeop    Berg's made-up operations with a SMOOTHED difficulty
                  ladder — levels are (depth, coefficient-scale) pairs,
                  filling the accuracy cliff Berg found between depths 1
                  and 2 (DELEGATED robustness change, documented)
  walk            NEW family (DELEGATED): a branching walk on a circular
                  track ("if your position is even step E, if odd step O",
                  k steps, wrap around). Procedural, single integer answer,
                  difficulty knob = k, no human confidence folklore, and a
                  cognitive load (data-dependent control flow) distinct from
                  tracking's bookkeeping and arithmetic's calculation.
                  Track sizes avoid 10/12 (clock folklore).

Strict parsing ONLY (mission ground rule): an answer turn must be a bare
integer, a confidence turn a bare 0-100, a bet turn ending in KEEP/PASS.
Unparseable -> up to 3 attempts total (1 initial + 2 resamples) -> drop the
trial and count it, distinguishing transport failures from parse failures.
No prose fallbacks (a deliberative "I should not KEEP this" is NOT a bet).

Other DELEGATED choices (documented in README):
  - families run sequentially: every call is a fresh stateless conversation,
    so interleaving cannot affect the model and only complicates bookkeeping
  - calibration: answer-only accuracy, 24-trial probes, 5 rounds, targets
    0.55-0.85, settles on best-tried-by-|acc-0.70| with an out-of-band
    warning; probes are saved (free cross-level confidence data)
  - Berg-style invariance block: 30 extra problems at the adjacent harder
    level (answer + confidence turns only) -> does confidence move when
    difficulty does?
  - letters_fixed runs mixed frequency levels (as v2); its invariance check
    uses low-frequency vs high-frequency strata within the main block

Usage:
  python3 hot2_v4.py --selftest
  python3 hot2_v4.py --model claude-sonnet-5 --fixedlen 30 --seed 601
  python3 hot2_v4.py --model claude-opus-4-8 --fixedlen 45 --seed 602
"""

import os
import re
import sys
import json
import time
import random
import string
import argparse
from statistics import mean, pstdev

import hot2_v2 as v2

# ----------------------------- CONFIG ---------------------------------------

N_MAIN = 80
N_INV = 30
K_CONSIST = 4
CALIB_BATCH = 24
CALIB_ROUNDS = 5
TARGET_LOW, TARGET_HIGH = 0.55, 0.85
MAX_TOKENS = 150
TRANSPORT_RETRIES = 4
PARSE_RETRIES = 3
BET_PASS_VALUE = 0.3

# ----------------------------- GENERATORS -----------------------------------

OP_NAMES = ["blorp", "trell", "quang", "smib", "dax", "vun"]
MADEOP_LEVELS = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]

def gen_madeop(level):
    depth, scale = level
    name = random.choice(OP_NAMES)
    if scale == 1:
        a = random.choice([-3, -2, 2, 3])
        b = random.choice([-2, -1, 1, 2])
        c = random.randint(-5, 5)
    else:
        a = random.choice([-9, -8, -7, -6, -5, 5, 6, 7, 8, 9])
        b = random.choice([-4, -3, 3, 4])
        c = random.randint(-20, 20)
    vals = [random.randint(2, 9) for _ in range(depth + 1)]
    expr = str(vals[0])
    acc = vals[0]
    for vv in vals[1:]:
        expr = f"({expr} {name} {vv})"
        acc = a * acc + b * vv + c
    sign_b = f"- {abs(b)}" if b < 0 else f"+ {b}"
    sign_c = f"- {abs(c)}" if c < 0 else f"+ {c}"
    body = (f"Define a new operation '{name}': x {name} y = {a}*x {sign_b}*y "
            f"{sign_c}.\n\nCompute {expr}.")
    return body, acc

WALK_LEVELS = [3, 5, 7, 10, 14, 19, 25]
# ODD sizes only: an even modulus preserves step-parity, so the branch
# sequence collapses to a closed form (answer computable without simulating
# steps) — that would make track size a surface difficulty cue and defeat the
# family's "data-dependent control flow" property. Odd moduli keep branching
# genuinely data-dependent. Avoid 10/12/15 (clock/quarter-hour folklore).
WALK_SIZES = [9, 11, 13, 17]

def gen_walk(k):
    m = random.choice(WALK_SIZES)
    s = random.randrange(m)
    e = random.choice([2, 3, 4, 5, 6, 7])
    o = random.choice([x for x in [2, 3, 4, 5, 6, 7] if x != e])
    body = (f"You stand on a circular track with positions numbered 0 to "
            f"{m-1}. You start at position {s}. Repeat the following {k} "
            f"times: if your current position is an even number, walk "
            f"forward {e} positions; if it is odd, walk forward {o} "
            f"positions. Walking forward past position {m-1} wraps around "
            f"to position 0.\n\nWhat position are you standing on at the end?")
    pos = s
    for _ in range(k):
        pos = (pos + (e if pos % 2 == 0 else o)) % m
    return body, pos

def gen_letters_fixed(level):
    body, true = v2.gen_letters_fixed(level)
    return body, true

FAMILIES = {
    "tracking":       (v2.gen_tracking,   v2.TRACKING_LEVELS),
    "letters_fixed":  (gen_letters_fixed, v2.LETTERS_FIXED_LEVELS),
    "arith_standard": (v2.gen_arithmetic, v2.ARITH_LEVELS),
    "arith_madeop":   (gen_madeop,        MADEOP_LEVELS),
    "walk":           (gen_walk,          WALK_LEVELS),
}
FAMILY_ORDER = ["tracking", "letters_fixed", "arith_standard",
                "arith_madeop", "walk"]

# ----------------------------- INDEPENDENT EVALUATORS -----------------------
# Each re-derives the answer from the PROMPT TEXT, so generator/prompt
# mismatches are caught. Used by --selftest over thousands of cases.

def eval_tracking(body):
    m = re.search(r"Red box starts with (\d+) items, green box with (\d+), "
                  r"blue box with (\d+)", body)
    boxes = {"red": int(m.group(1)), "green": int(m.group(2)),
             "blue": int(m.group(3))}
    for sent in re.findall(r"(?:Add|Remove|Move|Swap)[^.]*\.", body):
        if sent.startswith("Add"):
            n, x = re.match(r"Add (\d+) items to the (\w+) box", sent).groups()
            boxes[x] += int(n)
        elif sent.startswith("Remove"):
            n, x = re.match(r"Remove (\d+) items from the (\w+) box",
                            sent).groups()
            boxes[x] -= int(n)
        elif sent.startswith("Move"):
            n, x, y = re.match(r"Move (\d+) items from the (\w+) box to the "
                               r"(\w+) box", sent).groups()
            boxes[x] -= int(n)
            boxes[y] += int(n)
        else:
            x, y = re.match(r"Swap the contents of the (\w+) and (\w+) boxes",
                            sent).groups()
            boxes[x], boxes[y] = boxes[y], boxes[x]
    ask = re.search(r"How many items end up in the (\w+) box", body).group(1)
    return boxes[ask]

def eval_letters_fixed(body):
    target = re.search(r"the letter '(\w)'", body).group(1)
    text = body.split("String:\n")[1].strip()
    return text.count(target)

def eval_arith(body):
    val = int(re.search(r"Start with (-?\d+)\.", body).group(1))
    for sent in re.findall(r"(?:Add|Subtract|Multiply by|Keep only)[^.]*\.",
                           body):
        if sent.startswith("Add"):
            val += int(re.search(r"(\d+)", sent).group(1))
        elif sent.startswith("Subtract"):
            val -= int(re.search(r"(\d+)", sent).group(1))
        elif sent.startswith("Multiply"):
            val *= int(re.search(r"(\d+)", sent).group(1))
        else:                                  # mod-1000 clamp
            val = val % 1000
    return val

def eval_madeop(body):
    m = re.search(r"x (\w+) y = (-?\d+)\*x ([+-]) (\d+)\*y ([+-]) (\d+)", body)
    a = int(m.group(2))
    b = int(m.group(4)) * (1 if m.group(3) == "+" else -1)
    c = int(m.group(6)) * (1 if m.group(5) == "+" else -1)
    vals = [int(x) for x in re.findall(r"(?<![\d*])\b(\d)\b",
                                       body.split("Compute")[1])]
    acc = vals[0]
    for vv in vals[1:]:
        acc = a * acc + b * vv + c
    return acc

def eval_walk(body):
    m = int(re.search(r"positions numbered 0 to (\d+)", body).group(1)) + 1
    s = int(re.search(r"start at position (\d+)", body).group(1))
    k = int(re.search(r"Repeat the following (\d+) times", body).group(1))
    e = int(re.search(r"even number, walk forward (\d+)", body).group(1))
    o = int(re.search(r"odd, walk forward (\d+)", body).group(1))
    pos = s
    for _ in range(k):
        pos = (pos + (e if pos % 2 == 0 else o)) % m
    return pos

EVALUATORS = {
    "tracking": eval_tracking, "letters_fixed": eval_letters_fixed,
    "arith_standard": eval_arith, "arith_madeop": eval_madeop,
    "walk": eval_walk,
}

# ----------------------------- PROMPTS / STRICT PARSING ---------------------

ANSWER_PROMPT = ("{body}\n\nAnswer immediately from your best judgment. "
                 "Do NOT work through steps. Reply with ONLY the integer "
                 "answer.")
CONF_PROMPT = ("How confident are you that your answer is exactly correct? "
               "Reply with ONLY an integer from 0 to 100.")
BET_PROMPT = ("Now a bet on that same answer. If you KEEP it: +1 point if it "
              "is exactly correct, -1 point if it is wrong. If you PASS: "
              "+0.3 points guaranteed. Maximize expected points. Reply with "
              "ONLY the word KEEP or PASS.")

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

NEGATION = re.compile(r"\b(NOT|N'T|NEVER|INSTEAD|RATHER|AVOID|DON'T|WON'T)\b")

def parse_bet_strict(text):
    """Strict: accept a bare KEEP/PASS (optionally with markdown, trailing
    punctuation, or a short 'My bet:' lead-in). Reject anything with a
    negation, since 'I will not KEEP' means the opposite of its last token,
    and reject ambiguous replies mentioning both — the mission bans prose
    fallbacks that guess intent."""
    if text is None:
        return None
    up = text.upper()
    if NEGATION.search(up):
        return None
    if re.fullmatch(r"\s*\**\b(KEEP|PASS)\b\**[.\s]*", up):
        return re.search(r"\b(KEEP|PASS)\b", up).group(1)
    toks = re.findall(r"\b(KEEP|PASS)\b", up)
    tail = re.search(r"\b(KEEP|PASS)\b[\W]*$", up)
    return tail.group(1) if (tail and len(set(toks)) == 1) else None

# ----------------------------- MODEL PLUMBING -------------------------------

def make_client():
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY first.")
    return anthropic.Anthropic()

def api_chat(client, model, messages, thinking="disabled"):
    kwargs = dict(model=model, max_tokens=MAX_TOKENS, messages=messages)
    if "fable" in model:
        kwargs["max_tokens"] = 2000
    elif thinking == "disabled":
        kwargs["thinking"] = {"type": "disabled"}
    import anthropic
    fatal = (anthropic.BadRequestError, anthropic.AuthenticationError,
             anthropic.NotFoundError, anthropic.PermissionDeniedError)
    for attempt in range(TRANSPORT_RETRIES):
        try:
            resp = client.messages.create(**kwargs)
            return "\n".join(b.text for b in resp.content
                             if getattr(b, "type", "") == "text")
        except fatal as e:
            # bad model id / auth / 400 won't fix on retry — fail fast rather
            # than grind through backoff on every trial of a doomed run
            sys.exit(f"FATAL non-retryable API error (aborting run): {e}")
        except Exception as e:
            if attempt == TRANSPORT_RETRIES - 1:
                print(f"    call failed: {e}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    return None

def chat_parse(chat, messages, parser, counters, key):
    """Strict-parse with fresh resamples on failure. Distinguishes TRANSPORT
    failures (api_chat returned None after its own retries) from genuine
    PARSE failures (the model replied but not in the required form), so a dead
    API is never silently indistinguishable from prose non-compliance."""
    text = None
    for _ in range(PARSE_RETRIES):
        text = chat(messages)
        val = parser(text)
        if val is not None:
            return val, text
        bucket = key + ("_transport" if text is None else "_parse")
        counters[bucket] = counters.get(bucket, 0) + 1
    counters[key + "_dropped"] = counters.get(key + "_dropped", 0) + 1
    return None, text

# ----------------------------- PROTOCOL -------------------------------------

def run_problem(chat_factory, gen, level, counters):
    body, true = gen(level)
    chat = chat_factory(body, true)
    hist = [{"role": "user", "content": ANSWER_PROMPT.format(body=body)}]
    pred, a_raw = chat_parse(chat, hist, parse_int_strict, counters, "answer")
    if pred is None:
        return None
    hist.append({"role": "assistant", "content": a_raw})
    hist.append({"role": "user", "content": CONF_PROMPT})
    conf, c_raw = chat_parse(chat, hist, parse_conf_strict, counters, "conf")
    hist.append({"role": "assistant", "content": c_raw or ""})
    hist.append({"role": "user", "content": BET_PROMPT})
    bet, b_raw = chat_parse(chat, hist, parse_bet_strict, counters, "bet")
    same = valid = 0
    r_raws = []
    for _ in range(K_CONSIST):
        rp, r_raw = chat_parse(
            chat, [{"role": "user", "content": ANSWER_PROMPT.format(body=body)}],
            parse_int_strict, counters, "resample")
        r_raws.append(r_raw)
        if rp is None:
            continue
        valid += 1
        if rp == pred:
            same += 1
    consist = (same / valid) if valid else None
    return {"level": str(level), "body": body, "true": true, "pred": pred,
            "correct": pred == true, "confidence": conf, "bet": bet,
            "consistency": consist, "consist_same": same,
            "consist_valid": valid, "a_raw": a_raw, "c_raw": c_raw,
            "b_raw": b_raw, "resample_raws": r_raws}

def run_answer_conf(chat_factory, gen, level, counters):
    """Answer + confidence only (calibration probes / invariance block)."""
    body, true = gen(level)
    chat = chat_factory(body, true)
    hist = [{"role": "user", "content": ANSWER_PROMPT.format(body=body)}]
    pred, a_raw = chat_parse(chat, hist, parse_int_strict, counters, "answer")
    if pred is None:
        return None
    hist.append({"role": "assistant", "content": a_raw})
    hist.append({"role": "user", "content": CONF_PROMPT})
    conf, c_raw = chat_parse(chat, hist, parse_conf_strict, counters, "conf")
    return {"level": str(level), "body": body, "true": true, "pred": pred,
            "correct": pred == true, "confidence": conf,
            "a_raw": a_raw, "c_raw": c_raw}

def calibrate(chat_factory, gen, levels, name, counters):
    print(f"\n  calibrating {name} (target {TARGET_LOW:.0%}-{TARGET_HIGH:.0%})")
    idx = min(1, len(levels) - 1)
    tried = {}
    probes = {}
    for _ in range(CALIB_ROUNDS):
        level = levels[idx]
        rows = [r for r in (run_answer_conf(chat_factory, gen, level, counters)
                            for _ in range(CALIB_BATCH)) if r]
        if not rows:
            sys.exit(f"ERROR: every calibration call failed for {name}; "
                     f"aborting rather than mis-calibrating on a dead API")
        probes.setdefault(str(level), []).extend(rows)
        acc = mean(1.0 if r["correct"] else 0.0 for r in rows)
        tried[idx] = acc
        print(f"    level={level}  accuracy={acc:.2f}  (n={len(rows)})",
              flush=True)
        if TARGET_LOW <= acc <= TARGET_HIGH:
            print(f"    -> locked level = {level}")
            return idx, probes
        if acc > TARGET_HIGH and idx < len(levels) - 1:
            idx += 1
        elif acc < TARGET_LOW and idx > 0:
            idx -= 1
        else:
            break
    # settle on the level whose POOLED accuracy (all probes at that level,
    # not just the last noisy batch) is closest to mid-band; this stops a
    # ping-pong across a difficulty cliff from committing the main block to a
    # level measured only once, out of band
    pooled = {i: mean(1.0 if r["correct"] else 0.0
                      for r in probes[str(levels[i])]) for i in tried}
    best = min(pooled, key=lambda i: abs(pooled[i] - 0.7))
    if not (TARGET_LOW <= pooled[best] <= TARGET_HIGH):
        print(f"    WARNING: settled pooled accuracy {pooled[best]:.2f} is "
              f"OUTSIDE the target band for {name} (difficulty cliff — the "
              f"AUROC2 for this family may rest on few errors)", flush=True)
    print(f"    -> settling on level = {levels[best]} "
          f"(pooled acc {pooled[best]:.2f})")
    return best, probes

# ----------------------------- ANALYSIS -------------------------------------

def auroc2(pairs):
    pos = [s for s, c in pairs if c]
    neg = [s for s, c in pairs if not c]
    if not pos or not neg:
        return None
    wins = sum(1 if p > q else (0.5 if p == q else 0) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))

def boot_ci(pairs, iters=2000):
    vals = []
    k = len(pairs)
    for _ in range(iters):
        sample = [pairs[random.randrange(k)] for _ in range(k)]
        a = auroc2(sample)
        if a is not None:
            vals.append(a)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]

def stratified_auroc2(groups):
    wins = tot = 0.0
    npairs = 0
    for g in groups:
        pos = [s for s, c in g if c]
        neg = [s for s, c in g if not c]
        for p in pos:
            for q in neg:
                wins += 1 if p > q else (0.5 if p == q else 0)
                tot += 1
                npairs += 1
    return ((wins / tot) if tot else None), npairs

def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"

def report_family(name, rows, inv_rows, locked_level):
    if not rows:
        print(f"\n[{name}] n=0 — all trials dropped (no data)")
        return
    rows_c = [r for r in rows if r["confidence"] is not None]
    rows_k = [r for r in rows if r["consistency"] is not None]
    acc = mean(1.0 if r["correct"] else 0.0 for r in rows)
    confs = [r["confidence"] for r in rows_c]
    print(f"\n[{name}] n={len(rows)} level={locked_level} accuracy={acc:.2f}")
    if not confs:
        print("  (no confidences parsed — skipping confidence analyses)")
        return
    # channel 1: stated confidence
    cp = [(r["confidence"], r["correct"]) for r in rows_c]
    a = auroc2(cp)
    lo, hi = boot_ci(cp) if a is not None else (None, None)
    gap = mean(confs) - 100 * acc if confs else None
    hi90 = mean(1.0 if c >= 90 else 0.0 for c in confs) if confs else None
    print(f"  confidence: mean={mean(confs):.1f} sd={pstdev(confs):.1f} "
          f"share>=90={hi90:.0%} gap={gap:+.1f}" if confs else
          "  confidence: none parsed")
    print(f"    AUROC2={fmt(a)} CI[{fmt(lo)},{fmt(hi)}]")
    # channel 2: bets
    kept = [r for r in rows if r["bet"] == "KEEP"]
    passed = [r for r in rows if r["bet"] == "PASS"]
    ka = mean(1.0 if r["correct"] else 0.0 for r in kept) if kept else None
    pa = mean(1.0 if r["correct"] else 0.0 for r in passed) if passed else None
    bp = [((1 if r["bet"] == "KEEP" else 0), r["correct"])
          for r in rows if r["bet"]]
    ab = auroc2(bp)
    print(f"  bets: KEEP n={len(kept)} acc={fmt(ka)} | PASS n={len(passed)} "
          f"acc={fmt(pa)} | AUROC2={fmt(ab)}")
    # channel 3: consistency
    sp = [(r["consistency"], r["correct"]) for r in rows_k]
    ac = auroc2(sp)
    clo, chi = boot_ci(sp) if ac is not None else (None, None)
    print(f"  consistency (k={K_CONSIST}): AUROC2={fmt(ac)} "
          f"CI[{fmt(clo)},{fmt(chi)}]")
    # confidence within consistency strata
    by_cons = {}
    for r in rows_c:
        if r["consistency"] is None:
            continue
        by_cons.setdefault(r["consistency"], []).append(
            (r["confidence"], r["correct"]))
    aw, npairs = stratified_auroc2(by_cons.values())
    print(f"  confidence WITHIN consistency strata: {fmt(aw)} "
          f"({npairs} informative pairs)")
    # cue benchmark: answer magnitude
    mp = [(-abs(r["pred"]), r["correct"]) for r in rows]
    am = auroc2(mp)
    preds = sorted(abs(r["pred"]) for r in rows)
    t1, t2 = preds[len(preds)//3], preds[2*len(preds)//3]
    terc, terc_cue = {}, {}
    for r in rows_c:
        b = 0 if abs(r["pred"]) <= t1 else (1 if abs(r["pred"]) <= t2 else 2)
        terc.setdefault(b, []).append((r["confidence"], r["correct"]))
        terc_cue.setdefault(b, []).append((-abs(r["pred"]), r["correct"]))
    cwm, _ = stratified_auroc2(terc.values())
    cnull, _ = stratified_auroc2(terc_cue.values())
    print(f"  cue benchmark: magnitude-alone={fmt(am)}  "
          f"conf-within-magnitude={fmt(cwm)} vs cue-null={fmt(cnull)}")
    # invariance check (Berg-style: does confidence move when difficulty does?)
    if not inv_rows and name == "letters_fixed":
        # letters_fixed has no calibration ladder; its difficulty knob is the
        # target-letter frequency, so compare low-frequency vs high-frequency
        # strata within the main block (as promised in the docstring)
        lo = [r for r in rows_c if int(r["level"]) <= 15]
        hi = [r for r in rows_c if int(r["level"]) >= 20]
        if lo and hi:
            la = mean(1.0 if r["correct"] else 0.0 for r in lo)
            ha = mean(1.0 if r["correct"] else 0.0 for r in hi)
            lc = mean(r["confidence"] for r in lo)
            hc = mean(r["confidence"] for r in hi)
            print(f"  invariance (freq strata): low<=15% acc={la:.2f} "
                  f"conf={lc:.1f}  ->  high>=20% acc={ha:.2f} conf={hc:.1f}   "
                  f"d_acc={100*(ha-la):+.0f}pts d_conf={hc-lc:+.1f}")
    elif inv_rows:
        inv_c = [r for r in inv_rows if r["confidence"] is not None]
        acc2 = mean(1.0 if r["correct"] else 0.0 for r in inv_rows)
        mc2 = mean(r["confidence"] for r in inv_c) if inv_c else None
        lvl2 = inv_rows[0]["level"]
        d_acc = 100 * (acc2 - acc)
        d_conf = (mc2 - mean(confs)) if (mc2 is not None and confs) else None
        print(f"  invariance: level {locked_level} acc={acc:.2f} "
              f"conf={mean(confs):.1f}  ->  level {lvl2} acc={acc2:.2f} "
              f"conf={fmt(mc2) if mc2 is None else f'{mc2:.1f}'}   "
              f"d_acc={d_acc:+.0f}pts d_conf="
              f"{'n/a' if d_conf is None else f'{d_conf:+.1f}'}")

# ----------------------------- FAKE MODELS (selftest) -----------------------

def calibrated_factory(p_correct=0.7):
    """Genuine monitor: confidence, bets, and stability all track whether
    this particular answer is actually right."""
    def make(body, true):
        correct = random.random() < p_correct
        ans = true if correct else true + random.choice([-2, -1, 1, 2])
        conf = max(0, min(100, int(random.gauss(88 if correct else 40, 8))))
        answered = {"n": 0}
        def chat(messages):
            last = messages[-1]["content"]
            if "confident" in last:
                return str(conf)
            if "KEEP" in last:
                return "KEEP" if conf > 65 else "PASS"
            answered["n"] += 1
            if answered["n"] == 1:
                return str(ans)
            # resample: stable when right, unstable when wrong
            if correct or random.random() < 0.3:
                return str(ans)
            return str(ans + random.choice([-1, 1]))
        return chat
    return make

def pinned_factory(p_correct=0.7):
    """Surface faker: identical answers, but confidence pinned at 92 and
    always KEEP — the trained-confidence phenotype the suite must expose."""
    def make(body, true):
        correct = random.random() < p_correct
        ans = true if correct else true + random.choice([-2, -1, 1, 2])
        answered = {"n": 0}
        def chat(messages):
            last = messages[-1]["content"]
            if "confident" in last:
                return "92"
            if "KEEP" in last:
                return "KEEP"
            answered["n"] += 1
            if answered["n"] == 1:
                return str(ans)
            if correct or random.random() < 0.3:
                return str(ans)
            return str(ans + random.choice([-1, 1]))
        return chat
    return make

# ----------------------------- SELFTEST -------------------------------------

def selftest():
    print("SELFTEST 1: every generator vs its independent evaluator "
          "(3000 cases each)")
    ok = True
    for name, (gen, levels) in FAMILIES.items():
        bad = 0
        for _ in range(3000):
            body, true = gen(random.choice(levels))
            if EVALUATORS[name](body) != true:
                bad += 1
        print(f"  {name}: {'OK' if bad == 0 else f'{bad}/3000 MISMATCHES'}")
        ok = ok and bad == 0

    print("\nSELFTEST 2: strict parsers reject prose and accept clean replies")
    cases_ok = (
        parse_int_strict(" -14. ") == -14
        and parse_int_strict("The answer is 14") is None
        and parse_conf_strict("85") == 85
        and parse_conf_strict("about 85") is None
        and parse_conf_strict("170") is None
        and parse_bet_strict(" keep ") == "KEEP"
        and parse_bet_strict("KEEP or PASS?") is None)
    print("  parsers:", "OK" if cases_ok else "FAILED")
    ok = ok and cases_ok

    print("\nSELFTEST 3: faker separation — a calibrated monitor vs a "
          "pinned-confidence faker (same accuracy)")
    fams = [("arith_standard", v2.gen_arithmetic, 4),
            ("walk", gen_walk, 10)]
    for label, factory in (("CalibratedBot", calibrated_factory()),
                           ("PinnedBot", pinned_factory())):
        print(f"  [{label}]")
        for fname, gen, level in fams:
            counters = {}
            rows = [r for r in (run_problem(factory, gen, level, counters)
                                for _ in range(60)) if r]
            cp = [(r["confidence"], r["correct"]) for r in rows
                  if r["confidence"] is not None]
            a = auroc2(cp)
            kept = [r for r in rows if r["bet"] == "KEEP"]
            passed = [r for r in rows if r["bet"] == "PASS"]
            ka = mean(1.0 if r["correct"] else 0.0 for r in kept) if kept else None
            pa = (mean(1.0 if r["correct"] else 0.0 for r in passed)
                  if passed else None)
            print(f"    {fname}: conf AUROC2={fmt(a)}  KEEP acc={fmt(ka)} "
                  f"(n={len(kept)})  PASS acc={fmt(pa)} (n={len(passed)})")
            if label == "CalibratedBot":
                ok = ok and a is not None and a >= 0.75
                ok = ok and passed and kept and (ka - pa) >= 0.2
            else:
                ok = ok and (a is None or a <= 0.6)
                ok = ok and not passed        # never PASSes: no bet signal
    print("\nSELFTEST", "PASSED" if ok else "FAILED — do not run for real")
    return ok

# ----------------------------- MAIN -----------------------------------------

def unique_path(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 2
    while os.path.exists(f"{stem}_run{k}{ext}"):
        k += 1
    return f"{stem}_run{k}{ext}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=N_MAIN)
    ap.add_argument("--fixedlen", type=int, default=30,
                    help="letters_fixed string length (sonnet 30, opus 45, "
                         "matching their v2 calibrations)")
    ap.add_argument("--families", default=None,
                    help="comma-separated subset of: " + ",".join(FAMILY_ORDER))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--thinking", choices=["disabled", "default"],
                    default="disabled")
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    v2.FIXED_LEN = args.fixedlen

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.model:
        sys.exit("Provide --model (or --selftest).")

    fams = FAMILY_ORDER
    if args.families:
        fams = [f.strip() for f in args.families.split(",") if f.strip()]
        unknown = [f for f in fams if f not in FAMILIES]
        if unknown:
            sys.exit(f"unknown families {unknown}")

    client = make_client()
    def real_factory(body, true):
        return lambda messages: api_chat(client, args.model, messages,
                                         args.thinking)

    out = {"model": args.model, "n": args.n, "seed": args.seed,
           "fixedlen": args.fixedlen, "thinking": args.thinking,
           "k_consist": K_CONSIST, "families": {}}
    for name in fams:
        gen, levels = FAMILIES[name]
        counters = {}
        print(f"\nFAMILY: {name}")
        if name == "letters_fixed":
            # mixed frequency levels, as in v2; no calibration walk
            locked, probes, inv_rows = "mixed", {}, []
            rows = []
            per = max(1, -(-args.n // len(levels)))   # ceiling: total >= n
            for lv in levels:
                for _ in range(per):
                    r = run_problem(real_factory, gen, lv, counters)
                    if r:
                        rows.append(r)
                        print(f"    letters_fixed {len(rows)} "
                              f"[{'Y' if r['correct'] else 'n'}] "
                              f"conf={r['confidence']} bet={r['bet']} "
                              f"consist={r['consistency']}", flush=True)
        else:
            idx, probes = calibrate(real_factory, gen, levels, name, counters)
            locked = levels[idx]
            rows = []
            for i in range(args.n):
                r = run_problem(real_factory, gen, locked, counters)
                if r:
                    rows.append(r)
                    print(f"    {name} {len(rows)}/{args.n} "
                          f"[{'Y' if r['correct'] else 'n'}] "
                          f"conf={r['confidence']} bet={r['bet']} "
                          f"consist={r['consistency']}", flush=True)
            # invariance block at the adjacent harder level (or easier at top)
            idx2 = idx + 1 if idx + 1 < len(levels) else idx - 1
            lvl2 = levels[idx2]
            print(f"  invariance block at level {lvl2} (n={N_INV})")
            inv_rows = [r for r in (run_answer_conf(real_factory, gen, lvl2,
                                                    counters)
                                    for _ in range(N_INV)) if r]
        out["families"][name] = {
            "locked_level": str(locked), "rows": rows,
            "invariance_rows": inv_rows, "calibration_probes": probes,
            "parse_counters": counters,
        }
        # save incrementally: a crash must never discard completed families
        safe = args.model.replace("/", "_")
        if "path" not in out:
            out["path"] = unique_path(f"hot2v4_{safe}.json")
        with open(out["path"], "w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved (incremental) -> {out['path']}")

    print("\n" + "=" * 74 + f"\nHOT-2 v4 RESULTS  model={args.model}\n"
          + "=" * 74)
    for name in fams:
        fam = out["families"][name]
        report_family(name, fam["rows"], fam["invariance_rows"],
                      fam["locked_level"])
        if fam["parse_counters"]:
            print(f"  parse events: {fam['parse_counters']}")

if __name__ == "__main__":
    main()
