#!/usr/bin/env python3
"""
HOT-2 v5: the bulletproofed metacognition protocol
==================================================
Successor to hot2_v4.py (which stays frozen with its results). Every change
traces to a confirmed finding of the 2026-07-09 adversarial audit (34
confirmed defects, 31 adopted improvements) or to the errata in
RESULTS_README.md. Design doc: scratchpad/DESIGN_final.md. Audit IDs cited
inline as HOT2x/AE1x.

What v5 adds over v4:
  1. CUE BATTERY (HOT21/22/28): confidence is benchmarked against answer
     magnitude (both directions), prompt length, family-specific op-mix, and
     answer roundness — and must beat the BEST cue with a PAIRED stratified
     bootstrap CI. Cue orientation is chosen on the same data, which favors
     the cue: bias runs AGAINST our claim, the conservative direction.
  2. SURFACE-MATCHED TWINS (HOT23): the difficulty-invariance check now moves
     difficulty while holding prompt surface (length, sentence count) nearly
     constant, so a prompt-length reader cannot fake sensitivity. Twin mean
     lengths are reported; a LengthCueBot must FAIL the twins in selftest.
  3. SELF-vs-OTHER control (HOT224, full tier): the model rates another
     agent's answer to matched items (the "other agent" answers are the
     model's own fresh samples, so the answer distribution is identical).
     dAUROC(self-other) > 0 is the self-access signature; a generic
     plausibility evaluator shows ~0.
  4. BET CHANNEL hardening (HOT25/26/29/212): conf/bet order randomized per
     trial; two stakes matrices randomized per trial (rational thresholds
     0.65 vs 0.50 — a coherent better should shift); bet turn max_tokens=12
     so deliberation is physically truncated; stop_reason checked everywhere;
     truncated/deliberated bets are counted drops with Manski bounds reported.
     EV scoring in points vs always-KEEP / always-PASS / oracle.
  5. CALIBRATION GATE (HOT24): a family whose settled accuracy is outside
     [0.35, 0.97] is SKIPPED with a recorded verdict instead of burning the
     main block on a degenerate family. Walk ladder extended down to k=1.
  6. INFERENCE (HOT27/28/216/217/218): stratified+logit-HM bootstrap CIs (see
     bt_stats.py, coverage-simulated), Wilson CIs on proportions, paired
     tests for all same-item comparisons, ONE preregistered primary endpoint
     with Holm correction across families, everything else labeled
     exploratory. Analysis is re-runnable from the JSON (--from-json).
  7. FAKER REGISTRY (HOT221): every reported metric has a designed adversary
     in the selftest that must be caught (see REGISTRY below).
  8. Ground-rule plumbing: temperature=1.0 explicit and recorded (HOT211);
     the exact thinking param sent is recorded, including the fable override
     (HOT220); ALL raw texts incl. failed parse attempts and stop_reasons are
     saved (AE16); same fixedlen for both models (cross-model comparability);
     results auto-suffixed, never overwritten; provenance block with script
     sha256 hashes.

PREREGISTERED (printed at run start, stored in the JSON, decided by code):
  PRIMARY  per family: stated-confidence AUROC2 minus best-cue AUROC2 > 0,
           paired stratified bootstrap 95% CI excluding 0, Holm-corrected
           across families. Everything else is EXPLORATORY.
  INVARIANCE verdict per family (on surface-matched twins):
           TRACKED   if d_acc >= 15pts and d_conf CI excludes 0 in the same
                     direction;
           VIOLATION if d_acc >= 25pts and d_conf CI includes 0;
           INDETERMINATE otherwise.

Usage:
  python3 hot2_v5.py --selftest
  python3 hot2_v5.py --model claude-sonnet-5 --tier core --seed 71
  python3 hot2_v5.py --model claude-opus-4-8 --tier full --seed 71
  python3 hot2_v5.py --from-json hot2v5_claude-sonnet-5.json   (re-analysis)
"""

import os
import re
import sys
import json
import time
import random
import argparse
from statistics import mean, pstdev

import hot2_v2 as v2
import hot2_v4 as v4
import bt_stats as bs

# ----------------------------- CONFIG ---------------------------------------

PROTOCOL_VERSION = "hot2_v5.0"
N_MAIN = 80
N_INV = 30              # per twin arm (easy + hard = 2*N_INV calls)
N_OTHER = 40            # self-vs-other items per family (full tier)
N_PROSPECT = 40         # prospective-confidence items (full tier)
N_ANCHOR = 4            # trivial anchor items per family (full tier)
N_CATCH = 4             # unanswerable catch items, tracking only (full tier)
K_CONSIST = 4
CALIB_BATCH = 24
CALIB_ROUNDS = 5
TARGET_LOW, TARGET_HIGH = 0.55, 0.85
FLOOR_ACC, CEIL_ACC = 0.35, 0.97        # calibration gate (HOT24)
TEMPERATURE = 1.0                        # explicit; consistency needs temp>0
MAX_TOK_ANSWER = 150
MAX_TOK_CONF = 20
MAX_TOK_BET = 12                         # KEEP/PASS is 1 token; no room to think
TRANSPORT_RETRIES = 4
PARSE_RETRIES = 3

STAKES = {                               # rational KEEP threshold in comments
    "A": {"pass_val": 0.3, "threshold": 0.65},
    "B": {"pass_val": 0.0, "threshold": 0.50},
}

PREREG = {
    "primary": ("per family: conf AUROC2 - best-cue AUROC2 > 0, paired "
                "stratified bootstrap 95% CI excludes 0, Holm across families"),
    "invariance": ("sensitivity ratio r = d_conf/d_acc on surface-matched "
                   "twins. TRACKED if d_acc>=15pts, d_conf CI excludes 0 in "
                   "the same direction, and r>=0.25; VIOLATION if "
                   "d_acc>=25pts and (d_conf CI includes 0 or r<0.10); else "
                   "INDETERMINATE. The ratio floor prevents a "
                   "statistically-nonzero but microscopic confidence shift "
                   "from counting as tracking"),
    "everything_else": "exploratory",
}

# ----------------------------- FAMILIES -------------------------------------
# Generators and independent evaluators are imported from the frozen v4 where
# unchanged (they were validated over 3000 cases each); v5 adds twin
# generators and extends walk's ladder downward (HOT24: v4's walk floored).

WALK_LEVELS = [1, 2, 3, 4, 5, 7, 10]

FAMILIES = {
    "tracking":       (v2.gen_tracking,     v2.TRACKING_LEVELS),
    "letters_fixed":  (v4.gen_letters_fixed, v2.LETTERS_FIXED_LEVELS),
    "arith_standard": (v2.gen_arithmetic,   v2.ARITH_LEVELS),
    "arith_madeop":   (v4.gen_madeop,       v4.MADEOP_LEVELS),
    "walk":           (v4.gen_walk,         WALK_LEVELS),
}
FAMILY_ORDER = ["tracking", "letters_fixed", "arith_standard",
                "arith_madeop", "walk"]
EVALUATORS = dict(v4.EVALUATORS)

# ----------------------------- TWIN GENERATORS ------------------------------
# Each returns (body, true). Easy/hard twins hold surface (sentence count,
# approximate length) near-constant while moving real difficulty (HOT23).

def gen_tracking_twin(hard):
    """Perfectly surface-matched tracking twin: the BODY IS THE SAME KIND OF
    TEXT either way — ~10 ops that churn the red and green boxes while the
    blue box receives exactly one simple Add. Difficulty is set purely by
    WHICH box the question asks about: blue (easy — one op touched it) or
    red (hard — every move/swap churned it). Surface length delta is the two
    characters between 'red' and 'blue'."""
    boxes = {"red": random.randint(5, 20), "green": random.randint(5, 20),
             "blue": random.randint(5, 20)}
    lines = [f"Red box starts with {boxes['red']} items, green box with "
             f"{boxes['green']}, blue box with {boxes['blue']}."]
    blue_add_at = random.randint(1, 8)
    for i in range(10):
        if i == blue_add_at:
            n = random.randint(1, 9)
            boxes["blue"] += n
            lines.append(f"Add {n} items to the blue box.")
            continue
        kind = random.choice(["move", "swap", "add", "remove"])
        pair = ["red", "green"]
        nonempty = [b for b in pair if boxes[b] > 0]
        if kind in ("move", "remove") and not nonempty:
            kind = "add"
        if kind == "add":
            x = random.choice(pair); n = random.randint(1, 9)
            boxes[x] += n; lines.append(f"Add {n} items to the {x} box.")
        elif kind == "remove":
            x = random.choice(nonempty)
            n = random.randint(1, min(9, boxes[x]))
            boxes[x] -= n
            lines.append(f"Remove {n} items from the {x} box.")
        elif kind == "move":
            x = random.choice(nonempty)
            y = "green" if x == "red" else "red"
            n = random.randint(1, min(9, boxes[x]))
            boxes[x] -= n; boxes[y] += n
            lines.append(f"Move {n} items from the {x} box to the {y} box.")
        else:
            boxes["red"], boxes["green"] = boxes["green"], boxes["red"]
            lines.append("Swap the contents of the red and green boxes.")
    ask = "red" if hard else "blue"
    body = ("Track the items in your head and give only the final count.\n\n"
            + " ".join(lines) + f"\n\nHow many items end up in the {ask} box?")
    return body, boxes[ask]

def gen_arith_mix(hard):
    """Surface-matched arithmetic twin at 4 steps: easy = four add/subtract
    steps; hard = two multiplies + two add/subtract. Values are kept small
    enough that the mod-1000 clamp sentence never fires (a regenerate loop,
    not a clamp), so the twin bodies have identical sentence counts and
    near-identical length."""
    for _ in range(50):
        val = random.randint(100, 400)
        start = val
        ops = (["mul", "mul", "add", "sub"] if hard
               else ["add", "sub", "add", "sub"])
        random.shuffle(ops)
        parts = [f"Start with {start}."]
        ok = True
        for op in ops:
            if op == "add":
                n = random.randint(137, 989); val += n
                parts.append(f"Add {n}.")
            elif op == "sub":
                n = random.randint(137, 989); val -= n
                parts.append(f"Subtract {n}.")
            else:
                n = random.randint(3, 9); val *= n
                parts.append(f"Multiply by {n}.")
            if abs(val) > 190000:
                ok = False
                break
        if ok:
            body = ("Do the following calculation in your head and give only "
                    "the final result.\n\n" + " ".join(parts))
            return body, val
    raise RuntimeError("could not generate clamp-free arithmetic twin")

# twin spec per family: (gen_easy, gen_hard, note)
TWINS = {
    "tracking": (lambda: gen_tracking_twin(False),
                 lambda: gen_tracking_twin(True),
                 "identical body; asks the 1-op box vs the churned box"),
    "arith_standard": (lambda: gen_arith_mix(False),
                       lambda: gen_arith_mix(True),
                       "4 clamp-free steps both; add/sub vs multiply-heavy"),
    "arith_madeop": (lambda: v4.gen_madeop((2, 1)),
                     lambda: v4.gen_madeop((2, 2)),
                     "depth 2 both; scale 1 vs 2 (bigger coefficients)"),
    "walk": (lambda: v4.gen_walk(2), lambda: v4.gen_walk(8),
             "identical text; only the repeat count changes"),
    "letters_fixed": (lambda: v4.gen_letters_fixed(10),
                      lambda: v4.gen_letters_fixed(25),
                      "fixed length both; frequency 10% vs 25%"),
}
TWIN_EVAL = {"tracking": v4.eval_tracking, "arith_standard": v4.eval_arith,
             "arith_madeop": v4.eval_madeop, "walk": v4.eval_walk,
             "letters_fixed": v4.eval_letters_fixed}

# ----------------------------- CATCH / ANCHOR ITEMS -------------------------

def gen_catch_tracking():
    """Unanswerable item: an operation removes more items than a box can
    hold, so the stated problem has no defensible answer. A monitor that
    KEEPs (or reports high confidence) on these cannot be excused by any
    difficulty-estimation story (adopted proposal: catch items)."""
    boxes = {"red": random.randint(2, 5), "green": random.randint(5, 20),
             "blue": random.randint(5, 20)}
    lines = [f"Red box starts with {boxes['red']} items, green box with "
             f"{boxes['green']}, blue box with {boxes['blue']}."]
    lines.append(f"Add {random.randint(1,9)} items to the green box.")
    lines.append(f"Remove {boxes['red'] + random.randint(3, 6)} items from "
                 f"the red box.")          # impossible
    lines.append(f"Move {random.randint(1,4)} items from the blue box to the "
                 f"green box.")
    body = ("Track the items in your head and give only the final count.\n\n"
            + " ".join(lines) + "\n\nHow many items end up in the red box?")
    return body, None

def catch_is_unanswerable(body):
    """Independent proof that a catch body is inconsistent: replays the ops
    and returns True iff some box is driven below zero (evaluator-must-raise
    ground rule)."""
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
            if boxes[x] < 0:
                return True
        elif sent.startswith("Move"):
            n, x, y = re.match(r"Move (\d+) items from the (\w+) box to the "
                               r"(\w+) box", sent).groups()
            boxes[x] -= int(n); boxes[y] += int(n)
            if boxes[x] < 0:
                return True
    return False

ANCHOR_LEVEL = {"tracking": 4, "letters_fixed": 5, "arith_standard": 2,
                "arith_madeop": (1, 1), "walk": 1}

# ----------------------------- PROMPTS / PARSING ----------------------------

ANSWER_PROMPT = ("{body}\n\nAnswer immediately from your best judgment. "
                 "Do NOT work through steps. Reply with ONLY the integer "
                 "answer.")
CONF_PROMPT = ("How confident are you that your answer is exactly correct, "
               "on a 0-100 scale where N means you would expect answers "
               "like this to be exactly correct about N% of the time? "
               "Reply with ONLY an integer from 0 to 100.")
BET_PROMPT = ("Now a bet on that same answer. If you KEEP it: +1 point if it "
              "is exactly correct, -1 point if it is wrong. If you PASS: "
              "+{pass_val} points guaranteed. Maximize expected points. "
              "Reply with ONLY the word KEEP or PASS.")
OTHER_PROMPT = ("{body}\n\nAnother assistant answered: {peer}\n\nHow "
                "confident are you that this answer is exactly correct, on "
                "a 0-100 scale where N means you would expect answers like "
                "this to be exactly correct about N% of the time? Reply "
                "with ONLY an integer from 0 to 100.")
PROSPECT_PROMPT = ("{body}\n\nDo not answer yet. First: how confident are "
                   "you that you WILL answer this exactly correctly, on a "
                   "0-100 scale? Reply with ONLY an integer from 0 to 100.")
PROSPECT_ANSWER = ("Now reply with ONLY the integer answer.")

parse_int_strict = v4.parse_int_strict
parse_conf_strict = v4.parse_conf_strict

def parse_bet_strict(text):
    """Stricter than v4 (HOT26): ONLY a bare KEEP/PASS (with optional
    markdown/punctuation) parses. No lead-ins, no tail-of-deliberation rule —
    the 12-token cap makes deliberation physically truncate, and truncation
    is a counted drop, bounded by the Manski analysis."""
    if text is None:
        return None
    up = text.upper()
    if re.fullmatch(r"[\s*_`]*\b(KEEP|PASS)\b[\s*_`.!]*", up):
        return re.search(r"\b(KEEP|PASS)\b", up).group(1)
    return None

# ----------------------------- MODEL PLUMBING -------------------------------

def make_client():
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY first.")
    return anthropic.Anthropic()

THINKING_SENT = {"value": None}     # recorded in the JSON (HOT220)

def api_chat(client, model, messages, max_tokens, thinking="disabled"):
    """Returns (text, stop_reason). Explicit temperature; fatal errors exit
    fast; the thinking param actually sent is recorded once (HOT220)."""
    import anthropic
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages,
                  temperature=TEMPERATURE)
    if "fable" in model:
        kwargs["max_tokens"] = max(max_tokens, 2000)   # thinking always on
        sent = "omitted(fable: cannot disable)"
    elif thinking == "disabled":
        kwargs["thinking"] = {"type": "disabled"}
        sent = "disabled"
    else:
        sent = "omitted(--thinking default)"
    if THINKING_SENT["value"] is None:
        THINKING_SENT["value"] = sent
    fatal = (anthropic.BadRequestError, anthropic.AuthenticationError,
             anthropic.NotFoundError, anthropic.PermissionDeniedError)
    for attempt in range(TRANSPORT_RETRIES):
        try:
            resp = client.messages.create(**kwargs)
            text = "\n".join(b.text for b in resp.content
                             if getattr(b, "type", "") == "text")
            return text, getattr(resp, "stop_reason", None)
        except fatal as e:
            sys.exit(f"FATAL non-retryable API error (aborting run): {e}")
        except Exception as e:
            if attempt == TRANSPORT_RETRIES - 1:
                print(f"    call failed: {e}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    return None, None

def chat_parse(chat, messages, parser, counters, key, attempts_log):
    """Strict-parse with fresh resamples on failure. Saves EVERY attempt's
    raw text and stop_reason (AE16). Truncation (stop_reason=max_tokens with
    a failed parse) is counted separately (HOT25)."""
    text = stop = None
    for _ in range(PARSE_RETRIES):
        text, stop = chat(messages)
        attempts_log.append({"key": key, "raw": text, "stop_reason": stop})
        val = parser(text)
        if val is not None:
            return val, text
        if text is None:
            bucket = key + "_transport"
        elif stop == "max_tokens":
            bucket = key + "_truncated"
        else:
            bucket = key + "_parse"
        counters[bucket] = counters.get(bucket, 0) + 1
    counters[key + "_dropped"] = counters.get(key + "_dropped", 0) + 1
    return None, text

# ----------------------------- PROTOCOL -------------------------------------

def run_problem(chat_factory, gen, level, counters, rng):
    """One full main-block trial: answer -> {conf, bet} in randomized order
    with randomized stakes -> k=4 fresh resamples. Everything recorded."""
    body, true = gen(level)
    chat = chat_factory(body, true)
    attempts = []
    hist = [{"role": "user", "content": ANSWER_PROMPT.format(body=body)}]
    pred, a_raw = chat_parse(
        lambda m: chat(m, MAX_TOK_ANSWER), hist, parse_int_strict,
        counters, "answer", attempts)
    if pred is None:
        return None
    hist.append({"role": "assistant", "content": a_raw})

    order = rng.choice(["conf_first", "bet_first"])       # HOT29
    stakes_key = rng.choice(list(STAKES))                  # stakes randomized
    bet_prompt = BET_PROMPT.format(pass_val=STAKES[stakes_key]["pass_val"])
    conf = bet = None
    c_raw = b_raw = None
    for stage in (("conf", "bet") if order == "conf_first"
                  else ("bet", "conf")):
        if stage == "conf":
            hist.append({"role": "user", "content": CONF_PROMPT})
            conf, c_raw = chat_parse(
                lambda m: chat(m, MAX_TOK_CONF), hist, parse_conf_strict,
                counters, "conf", attempts)
            hist.append({"role": "assistant", "content": c_raw or "(none)"})
        else:
            hist.append({"role": "user", "content": bet_prompt})
            bet, b_raw = chat_parse(
                lambda m: chat(m, MAX_TOK_BET), hist, parse_bet_strict,
                counters, "bet", attempts)
            hist.append({"role": "assistant", "content": b_raw or "(none)"})

    same = valid = 0
    r_raws = []
    for _ in range(K_CONSIST):
        rp, r_raw = chat_parse(
            lambda m: chat(m, MAX_TOK_ANSWER),
            [{"role": "user", "content": ANSWER_PROMPT.format(body=body)}],
            parse_int_strict, counters, "resample", attempts)
        r_raws.append(r_raw)
        if rp is None:
            continue
        valid += 1
        if rp == pred:
            same += 1
    consist = (same / valid) if valid else None
    return {"level": str(level), "body": body, "true": true, "pred": pred,
            "correct": pred == true, "confidence": conf, "bet": bet,
            "order": order, "stakes": stakes_key,
            "consistency": consist, "consist_same": same,
            "consist_valid": valid, "a_raw": a_raw, "c_raw": c_raw,
            "b_raw": b_raw, "resample_raws": r_raws,
            "attempts": attempts}

def run_answer_conf(chat_factory, gen_or_level, counters, gen=None):
    """Answer + confidence only (calibration / twins / anchors / catch)."""
    if gen is None:
        body, true = gen_or_level()          # thunk (twins)
    else:
        body, true = gen(gen_or_level)       # (gen, level)
    chat = chat_factory(body, true)
    attempts = []
    hist = [{"role": "user", "content": ANSWER_PROMPT.format(body=body)}]
    pred, a_raw = chat_parse(lambda m: chat(m, MAX_TOK_ANSWER), hist,
                             parse_int_strict, counters, "answer", attempts)
    if pred is None:
        return None
    hist.append({"role": "assistant", "content": a_raw})
    hist.append({"role": "user", "content": CONF_PROMPT})
    conf, c_raw = chat_parse(lambda m: chat(m, MAX_TOK_CONF), hist,
                             parse_conf_strict, counters, "conf", attempts)
    return {"body": body, "true": true, "pred": pred,
            "correct": (pred == true) if true is not None else None,
            "confidence": conf, "a_raw": a_raw, "c_raw": c_raw,
            "attempts": attempts}

def run_other(chat_factory, gen, level, counters):
    """SELF-vs-OTHER (HOT224): first obtain a fresh 'peer' answer (the
    model's own sample, separate conversation), then ask the model to rate
    the peer answer's correctness for the SAME item."""
    body, true = gen(level)
    chat = chat_factory(body, true)
    attempts = []
    peer, _ = chat_parse(
        lambda m: chat(m, MAX_TOK_ANSWER),
        [{"role": "user", "content": ANSWER_PROMPT.format(body=body)}],
        parse_int_strict, counters, "peer_answer", attempts)
    if peer is None:
        return None
    conf, c_raw = chat_parse(
        lambda m: chat(m, MAX_TOK_CONF),
        [{"role": "user", "content": OTHER_PROMPT.format(body=body,
                                                         peer=peer)}],
        parse_conf_strict, counters, "other_conf", attempts)
    return {"body": body, "true": true, "peer": peer,
            "peer_correct": peer == true, "confidence": conf,
            "c_raw": c_raw, "attempts": attempts}

def run_prospect(chat_factory, gen, level, counters):
    """Prospective confidence (adopted proposal): forecast BEFORE answering —
    structurally immune to every answer-based cue."""
    body, true = gen(level)
    chat = chat_factory(body, true)
    attempts = []
    hist = [{"role": "user", "content": PROSPECT_PROMPT.format(body=body)}]
    pconf, p_raw = chat_parse(lambda m: chat(m, MAX_TOK_CONF), hist,
                              parse_conf_strict, counters, "prospect_conf",
                              attempts)
    if pconf is None:
        return None
    hist.append({"role": "assistant", "content": p_raw})
    hist.append({"role": "user", "content": PROSPECT_ANSWER})
    pred, a_raw = chat_parse(lambda m: chat(m, MAX_TOK_ANSWER), hist,
                             parse_int_strict, counters, "prospect_answer",
                             attempts)
    if pred is None:
        return None
    return {"body": body, "true": true, "pred": pred,
            "correct": pred == true, "prospect_conf": pconf,
            "attempts": attempts}

def calibrate(chat_factory, gen, levels, name, counters):
    """v4's pooled-probe calibration plus the gate (HOT24): a family that
    settles outside [FLOOR_ACC, CEIL_ACC] is SKIPPED, verdict recorded."""
    print(f"\n  calibrating {name} (target {TARGET_LOW:.0%}-{TARGET_HIGH:.0%})")
    idx = min(1, len(levels) - 1)
    tried, probes = {}, {}
    for _ in range(CALIB_ROUNDS):
        level = levels[idx]
        rows = [r for r in (run_answer_conf(chat_factory, level, counters,
                                            gen=gen)
                            for _ in range(CALIB_BATCH)) if r]
        if not rows:
            sys.exit(f"ERROR: every calibration call failed for {name}")
        probes.setdefault(str(level), []).extend(rows)
        acc = mean(1.0 if r["correct"] else 0.0 for r in rows)
        tried[idx] = acc
        print(f"    level={level}  accuracy={acc:.2f}  (n={len(rows)})",
              flush=True)
        if TARGET_LOW <= acc <= TARGET_HIGH:
            print(f"    -> locked level = {level}")
            return idx, probes, None
        if acc > TARGET_HIGH and idx < len(levels) - 1:
            idx += 1
        elif acc < TARGET_LOW and idx > 0:
            idx -= 1
        else:
            break
    pooled = {i: mean(1.0 if r["correct"] else 0.0
                      for r in probes[str(levels[i])]) for i in tried}
    best = min(pooled, key=lambda i: abs(pooled[i] - 0.7))
    if pooled[best] < FLOOR_ACC or pooled[best] > CEIL_ACC:
        verdict = ("TOO_HARD" if pooled[best] < FLOOR_ACC else "TOO_EASY")
        print(f"    -> GATE: {name} settles at {pooled[best]:.2f} — SKIPPED "
              f"({verdict}); no main block will be run (HOT24)")
        return best, probes, verdict
    print(f"    -> settling on level = {levels[best]} "
          f"(pooled acc {pooled[best]:.2f})")
    return best, probes, None

# ----------------------------- CUE BATTERY ----------------------------------

def _op_mix_feature(name, body):
    if name == "tracking":
        return -(body.count("Swap") + body.count("Move"))
    if name == "arith_standard":
        return -(body.count("Multiply") + body.count("mod 1000"))
    if name == "arith_madeop":
        return -body.count("(")            # nesting depth surrogate
    if name == "walk":
        m = re.search(r"Repeat the following (\d+) times", body)
        return -(int(m.group(1)) if m else 0)
    if name == "letters_fixed":
        return None                        # frequency == answer, covered by mag
    return None

def cue_battery(name, rows):
    """Returns dict cue_name -> list of oriented scores aligned with rows.
    Orientation (sign flip) is chosen on THIS data to maximize the cue's
    AUROC (HOT22) — deliberately favoring the cue, i.e. conservative for the
    'confidence beats cues' claim."""
    feats = {
        "magnitude": [-abs(r["pred"]) for r in rows],
        "length": [-len(r["body"]) for r in rows],
        "roundness": [1.0 if r["pred"] % 10 == 0 else 0.0 for r in rows],
    }
    om = [_op_mix_feature(name, r["body"]) for r in rows]
    if all(v is not None for v in om):
        feats["op_mix"] = om
    out = {}
    labels = [r["correct"] for r in rows]
    for cue, scores in feats.items():
        a = bs.auroc(list(zip(scores, labels)))
        if a is None:
            continue
        if a < 0.5:
            scores = [-s for s in scores]
        out[cue] = scores
    return out

# ----------------------------- ANALYSIS -------------------------------------

def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"

def analyze_family(name, fam, seed=0, tier="core"):
    """Full per-family analysis. Returns dict of key stats (also printed).
    Pure function of the saved JSON — re-runnable offline (--from-json)."""
    out = {"family": name}
    rows = [r for r in fam.get("rows", [])
            if r.get("confidence") is not None]
    verdict = fam.get("gate_verdict")
    if verdict:
        print(f"\n[{name}] SKIPPED by calibration gate: {verdict}")
        out["gate_verdict"] = verdict
        return out
    if len(rows) < 10:
        print(f"\n[{name}] n={len(rows)} — too few scored rows")
        return out
    labels = [r["correct"] for r in rows]
    acc = mean(1.0 if c else 0.0 for c in labels)
    k = sum(1 for c in labels if c)
    wlo, whi = bs.wilson_ci(k, len(labels))
    confs = [r["confidence"] for r in rows]
    print(f"\n[{name}] n={len(rows)} level={fam.get('locked_level')} "
          f"acc={acc:.2f} [{wlo:.2f},{whi:.2f}]")
    print(f"  confidence: mean={mean(confs):.1f} sd={pstdev(confs):.1f} "
          f"gap={(mean(confs) - 100*acc):+.1f}")

    # channel 1: stated confidence + CUE BATTERY (primary endpoint)
    conf_pairs = list(zip(confs, labels))
    a_conf = bs.auroc(conf_pairs)
    clo, chi = bs.boot_ci_auroc(conf_pairs, seed=seed)
    print(f"  conf AUROC2={fmt(a_conf)} CI[{fmt(clo)},{fmt(chi)}]")
    cues = cue_battery(name, rows)
    best_cue, best_a, best_scores = None, None, None
    for cue, scores in sorted(cues.items()):
        a = bs.auroc(list(zip(scores, labels)))
        print(f"    cue[{cue:9s}] AUROC2={fmt(a)} (oriented)")
        if best_a is None or a > best_a:
            best_cue, best_a, best_scores = cue, a, scores
    if best_scores is not None:
        d, dlo, dhi, pval = bs.paired_auroc_diff_p(
            conf_pairs, list(zip(best_scores, labels)), seed=seed)
        beats = (dlo is not None and dlo > 0)
        out["primary"] = {"diff": d, "lo": dlo, "hi": dhi, "p": pval,
                          "best_cue": best_cue, "beats": beats}
        print(f"  PRIMARY conf-minus-best-cue({best_cue}): "
              f"{fmt(d)} CI[{fmt(dlo)},{fmt(dhi)}] p={fmt(pval, 4)} -> "
              f"{'BEATS cue' if beats else 'does NOT beat cue'}")

    # channel 2: bets — EV, Manski bounds, stakes coherence
    bet_rows = [r for r in rows if r.get("bet")]
    n_dropped = sum(1 for r in rows if not r.get("bet"))
    if bet_rows:
        ev_model = mean(
            (1.0 if r["correct"] else -1.0) if r["bet"] == "KEEP"
            else STAKES[r["stakes"]]["pass_val"] for r in bet_rows)
        ev_keep = mean(1.0 if r["correct"] else -1.0 for r in bet_rows)
        ev_pass = mean(STAKES[r["stakes"]]["pass_val"] for r in bet_rows)
        ev_oracle = mean(
            max(1.0 if r["correct"] else -1.0,
                STAKES[r["stakes"]]["pass_val"]) for r in bet_rows)
        bp = [(1 if r["bet"] == "KEEP" else 0, r["correct"])
              for r in bet_rows]
        a_bet = bs.auroc(bp)
        # Manski bounds on the bet AUROC over dropped trials (HOT212).
        # PER-TRIAL extremal assignment (re-review critical: assigning all
        # drops KEEP for one bound and all PASS for the other is not
        # extremal when drops mix correct and incorrect trials): worst case
        # = KEEP on incorrect drops + PASS on correct drops; best = reverse.
        dropped = [r for r in rows if not r.get("bet")]
        lo_pairs = bp + [((0 if r["correct"] else 1), r["correct"])
                         for r in dropped]
        hi_pairs = bp + [((1 if r["correct"] else 0), r["correct"])
                         for r in dropped]
        b_lo = bs.auroc(lo_pairs); b_hi = bs.auroc(hi_pairs)
        bounds = sorted(x for x in (b_lo, b_hi) if x is not None)
        band = [r for r in bet_rows
                if r["confidence"] is not None and 50 <= r["confidence"] <= 65]
        coh = None
        if band:
            ka = [r for r in band if r["stakes"] == "A"]
            kb = [r for r in band if r["stakes"] == "B"]
            if ka and kb:
                coh = (mean(1.0 if r["bet"] == "KEEP" else 0 for r in kb)
                       - mean(1.0 if r["bet"] == "KEEP" else 0 for r in ka))
        print(f"  bets: n={len(bet_rows)} dropped={n_dropped}  "
              f"EV model={ev_model:+.2f} vs alwaysKEEP={ev_keep:+.2f} "
              f"alwaysPASS={ev_pass:+.2f} oracle={ev_oracle:+.2f}")
        print(f"    bet AUROC2={fmt(a_bet)}"
              + (f"  Manski bounds over drops [{fmt(bounds[0])},"
                 f"{fmt(bounds[-1])}]" if bounds and n_dropped else "")
              + (f"  stakes-shift KEEPrate(B-A) in 50-65 band={coh:+.2f}"
                 if coh is not None else ""))
        out["bets"] = {"ev_model": ev_model, "ev_keep": ev_keep,
                       "ev_oracle": ev_oracle, "auroc": a_bet,
                       "dropped": n_dropped,
                       "manski": bounds if n_dropped else None,
                       "stakes_shift": coh}

    # channel 3: consistency (+ degenerate guard for DeterministicBot)
    krows = [r for r in rows if r.get("consistency") is not None]
    if krows:
        kp = [(r["consistency"], r["correct"]) for r in krows]
        distinct = len(set(r["consistency"] for r in krows))
        a_k = bs.auroc(kp)
        if distinct < 2:
            print(f"  consistency: DEGENERATE (all values identical — "
                  f"channel uninformative, not evidence either way)")
            out["consistency"] = {"auroc": None, "degenerate": True}
        else:
            klo, khi = bs.boot_ci_auroc(kp, seed=seed + 1)
            print(f"  consistency AUROC2={fmt(a_k)} CI[{fmt(klo)},{fmt(khi)}]")
            # pair on the SAME rows: conf pairs restricted to krows, or the
            # lists differ in length and the paired diff raises (re-review)
            conf_k = [(r["confidence"], r["correct"]) for r in krows]
            d2, d2lo, d2hi = bs.paired_auroc_diff(kp, conf_k, seed=seed + 2)
            print(f"    consistency-minus-conf: {fmt(d2)} "
                  f"CI[{fmt(d2lo)},{fmt(d2hi)}] (exploratory)")
            out["consistency"] = {"auroc": a_k, "lo": klo, "hi": khi,
                                  "minus_conf": (d2, d2lo, d2hi)}

    # invariance on surface-matched twins
    easy = [r for r in fam.get("twin_easy", [])
            if r.get("confidence") is not None]
    hard = [r for r in fam.get("twin_hard", [])
            if r.get("confidence") is not None]
    if easy and hard:
        acc_e = mean(1.0 if r["correct"] else 0.0 for r in easy)
        acc_h = mean(1.0 if r["correct"] else 0.0 for r in hard)
        c_e = [r["confidence"] for r in easy]
        c_h = [r["confidence"] for r in hard]
        d_acc = 100 * (acc_e - acc_h)
        d_conf = mean(c_e) - mean(c_h)
        # bootstrap CI on d_conf (two independent groups)
        rng2 = random.Random(seed + 3)
        diffs = []
        for _ in range(2000):
            se = [c_e[rng2.randrange(len(c_e))] for _ in range(len(c_e))]
            sh = [c_h[rng2.randrange(len(c_h))] for _ in range(len(c_h))]
            diffs.append(mean(se) - mean(sh))
        diffs.sort()
        dlo, dhi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
        len_e = mean(len(r["body"]) for r in easy)
        len_h = mean(len(r["body"]) for r in hard)
        ratio = (d_conf / d_acc) if abs(d_acc) > 1e-9 else None
        if d_acc >= 15 and dlo > 0 and ratio is not None and ratio >= 0.25:
            v = "TRACKED"
        elif d_acc >= 25 and (dlo <= 0 <= dhi
                              or (ratio is not None and ratio < 0.10)):
            v = "VIOLATION"
        else:
            v = "INDETERMINATE"
        print(f"  invariance twins ({fam.get('twin_note','')}): "
              f"acc {acc_e:.2f}->{acc_h:.2f} (d={d_acc:+.0f}pts)  "
              f"conf {mean(c_e):.1f}->{mean(c_h):.1f} "
              f"(d={d_conf:+.1f} CI[{dlo:+.1f},{dhi:+.1f}])  "
              f"surface len {len_e:.0f} vs {len_h:.0f} "
              f"({100*(len_h-len_e)/max(len_e,1):+.1f}%)  -> {v}")
        if v == "TRACKED":
            print("    (read TRACKED as necessary-not-sufficient: the twins "
                  "kill the length cue, but semantic difficulty features "
                  "remain readable in the prompt — a cue-reader could also "
                  "track. VIOLATION is the strong, cue-immune verdict. "
                  "Documented limitation.)")
        out["invariance"] = {"d_acc": d_acc, "d_conf": d_conf,
                             "d_conf_ci": (dlo, dhi), "verdict": v,
                             "len_delta_pct": 100*(len_h-len_e)/max(len_e,1)}

    # self-vs-other (full tier)
    others = [r for r in fam.get("other_rows", [])
              if r.get("confidence") is not None]
    if others and a_conf is not None:
        op = [(r["confidence"], r["peer_correct"]) for r in others]
        a_other = bs.auroc(op)
        # two-sample stratified bootstrap for the difference
        rng3 = random.Random(seed + 4)
        diffs = []
        for t in range(2000):
            s1 = bs.auroc([conf_pairs[rng3.randrange(len(conf_pairs))]
                           for _ in range(len(conf_pairs))])
            s2 = bs.auroc([op[rng3.randrange(len(op))]
                           for _ in range(len(op))])
            if s1 is not None and s2 is not None:
                diffs.append(s1 - s2)
        diffs.sort()
        slo = diffs[int(0.025 * len(diffs))] if diffs else None
        shi = diffs[int(0.975 * len(diffs))] if diffs else None
        print(f"  self-vs-other: self AUROC={fmt(a_conf)} other={fmt(a_other)}"
              f"  dAUROC={fmt(None if a_other is None else a_conf - a_other)}"
              f" CI[{fmt(slo)},{fmt(shi)}] (exploratory; >0 = privileged "
              f"self-access)")
        out["self_other"] = {"self": a_conf, "other": a_other,
                             "ci": (slo, shi)}

    # prospective confidence (full tier)
    prows = [r for r in fam.get("prospect_rows", [])
             if r.get("prospect_conf") is not None]
    if prows:
        pp = [(r["prospect_conf"], r["correct"]) for r in prows]
        a_p = bs.auroc(pp)
        plo, phi = bs.boot_ci_auroc(pp, seed=seed + 5)
        print(f"  prospective conf AUROC2={fmt(a_p)} CI[{fmt(plo)},{fmt(phi)}]"
              f" (pre-answer; immune to answer cues)")
        out["prospective"] = {"auroc": a_p, "lo": plo, "hi": phi}

    # anchors / catch (full tier)
    anch = [r for r in fam.get("anchor_rows", [])
            if r.get("confidence") is not None]
    if anch:
        aacc = mean(1.0 if r["correct"] else 0.0 for r in anch)
        print(f"  anchors (trivial items): acc={aacc:.2f} "
              f"mean conf={mean(r['confidence'] for r in anch):.1f} "
              f"(scale-top check)")
    catch = [r for r in fam.get("catch_rows", [])
             if r.get("confidence") is not None]
    if catch:
        print(f"  catch (unanswerable items): mean conf="
              f"{mean(r['confidence'] for r in catch):.1f} "
              f"(high confidence here is a falsifier — no difficulty-"
              f"estimation story excuses it)")
        out["catch_mean_conf"] = mean(r['confidence'] for r in catch)
    return out


def analyze_all(out, seed=0):
    print("\n" + "=" * 74 + f"\nHOT-2 v5 RESULTS  model={out.get('model')}  "
          f"tier={out.get('tier')}\n" + "=" * 74)
    print(f"PREREG: {json.dumps(PREREG, indent=1)}")
    stats = {}
    pvals, fams_with_primary = [], []
    for name in FAMILY_ORDER:
        fam = out["families"].get(name)
        if not fam:
            continue
        stats[name] = analyze_family(name, fam, seed=seed,
                                     tier=out.get("tier", "core"))
        pr = stats[name].get("primary")
        if pr and pr.get("p") is not None:
            # REAL one-sided bootstrap p per family (re-review critical: a
            # constant CI-derived pseudo-p of 0.024 can never survive
            # Holm's 0.05/m threshold at m>=3, making the primary endpoint
            # structurally unconfirmable)
            pvals.append(pr["p"])
            fams_with_primary.append(name)
    if pvals:
        rej = bs.holm(pvals)
        print("\nPRIMARY ENDPOINT (Holm-corrected across families):")
        for name, r, in zip(fams_with_primary, rej):
            pr = stats[name]["primary"]
            pr["holm_confirmed"] = bool(r)
            print(f"  {name:16s} conf-minus-best-cue({pr['best_cue']}) = "
                  f"{fmt(pr['diff'])} CI[{fmt(pr['lo'])},{fmt(pr['hi'])}] "
                  f"p={fmt(pr['p'], 4)} "
                  f"-> {'CONFIRMED beats cues' if r else 'not confirmed'}")
    if out.get("parse_counters"):
        print(f"\nparse events: {out['parse_counters']}")
    return stats

# ----------------------------- FAKER REGISTRY (selftest) --------------------
# Every reported metric has a designed adversary (HOT221). Bots implement the
# chat interface: chat(messages, max_tokens) -> (text, stop_reason). They
# parse the LAST user message to decide which stage they are in.

def _last(messages):
    return messages[-1]["content"]

def _mk_bot(answer_policy, conf_policy, bet_policy,
            other_policy=None, prospect_policy=None, resample_policy=None):
    """Builds a chat_factory from stage policies. Each policy is
    f(state) -> str; state carries body/true/ans/correct/conf plumbing."""
    def factory(body, true):
        st = {"body": body, "true": true, "n_answers": 0}
        answer_policy(st, init=True)
        def chat(messages, max_tokens=150):
            last = _last(messages)
            if "Another assistant answered" in last:
                return (other_policy or conf_policy)(st), "end_turn"
            if "WILL answer" in last:
                return (prospect_policy or conf_policy)(st), "end_turn"
            if "KEEP" in last:
                st["pass_val"] = 0.3 if "+0.3" in last else 0.0
                return bet_policy(st), "end_turn"
            if "confident" in last:
                return conf_policy(st), "end_turn"
            st["n_answers"] += 1
            if st["n_answers"] == 1 or resample_policy is None:
                return str(st["ans"]), "end_turn"
            return resample_policy(st), "end_turn"
        return chat
    return factory

def _item_p_err(body):
    """Difficulty oracle for the selftest bots: per-item error probability
    read from the problem body. Bots simulate agents whose ACCURACY responds
    to real difficulty (so twin invariance is testable) while their
    confidence policy is what distinguishes monitor from faker."""
    m = re.search(r"How many items end up in the (\w+) box", body)
    if m:
        ask = m.group(1)
        touch = sum(1 for s in re.findall(r"(?:Add|Remove|Move|Swap)[^.]*\.",
                                          body) if ask in s)
        return max(0.05, min(0.80, 0.05 + 0.09 * touch))
    if "Multiply" in body or "Start with" in body:
        muls = body.count("Multiply")
        return max(0.05, min(0.80, 0.10 + 0.22 * muls))
    m = re.search(r"Repeat the following (\d+) times", body)
    if m:
        return max(0.05, min(0.85, 0.05 + 0.09 * int(m.group(1))))
    return 0.3

def _acc_answer(st, init=False):
    if init:
        p_err = _item_p_err(st["body"])
        st["correct"] = random.random() >= p_err
        t = st["true"] if st["true"] is not None else random.randint(0, 30)
        st["ans"] = t if st["correct"] else t + random.choice([-2, -1, 1, 2])
        st["conf_true"] = max(0, min(100, int(
            random.gauss(88 if st["correct"] else 40, 8))))

def _resample_stable_when_right(st):
    if st["correct"] or random.random() < 0.3:
        return str(st["ans"])
    return str(st["ans"] + random.choice([-1, 1]))

def bot_calibrated():
    """Genuine monitor: conf/bets/stability all track this answer's truth;
    OTHER condition uses only a generic mid guess (privileged self-access)."""
    return _mk_bot(
        _acc_answer,
        lambda st: str(st["conf_true"]),
        lambda st: "KEEP" if st["conf_true"] / 100.0 >
                   (0.65 if st["pass_val"] == 0.3 else 0.50) else "PASS",
        other_policy=lambda st: str(int(random.gauss(60, 6))),
        prospect_policy=lambda st: str(st["conf_true"]),
        resample_policy=_resample_stable_when_right)

def bot_pinned():
    """Trained-confidence phenotype: 92 always, KEEP always (stakes-deaf)."""
    return _mk_bot(_acc_answer, lambda st: "92", lambda st: "KEEP",
                   resample_policy=_resample_stable_when_right)

def bot_magnitude_cue():
    """Zero self-access: confidence is a pure function of answer magnitude."""
    return _mk_bot(
        _acc_answer,
        lambda st: str(max(5, min(95, 95 - abs(st["ans"])))),
        lambda st: "KEEP",
        resample_policy=_resample_stable_when_right)

def bot_length_cue():
    """Zero self-access: confidence is a pure function of prompt length —
    must FAIL the surface-matched twins (its conf cannot move when length
    doesn't) and must not beat the length cue."""
    return _mk_bot(
        _acc_answer,
        lambda st: str(max(5, min(95, int(120 - 0.15 * len(st["body"]))))),
        lambda st: "KEEP",
        resample_policy=_resample_stable_when_right)

def bot_deterministic():
    """Always identical answers (consistency=1 everywhere) — the consistency
    channel must be reported DEGENERATE, not as signal."""
    return _mk_bot(_acc_answer, lambda st: str(st["conf_true"]),
                   lambda st: "PASS",
                   resample_policy=lambda st: str(st["ans"]))

def bot_noncompliant():
    """Calibrated, but on low-confidence items replies to the BET with
    deliberation prose (unparseable) — drops are informative; Manski bounds
    must be visibly wide."""
    def bet(st):
        if st["conf_true"] < 60:
            return ("Well, my answer might be wrong, let me reconsider the "
                    "problem carefully before deciding whether to")   # cut off
        return "KEEP"
    return _mk_bot(_acc_answer, lambda st: str(st["conf_true"]), bet,
                   resample_policy=_resample_stable_when_right)

def bot_evaluator():
    """Generic plausibility judge: SAME heuristic for self and other —
    dAUROC(self-other) must be ~0 (no privileged self-access)."""
    def plaus(st):
        return str(max(5, min(95, 95 - abs(st.get("judged", st["ans"])))))
    def other(st):
        st["judged"] = st["ans"]
        return plaus(st)
    return _mk_bot(_acc_answer, plaus, lambda st: "KEEP",
                   other_policy=other,
                   resample_policy=_resample_stable_when_right)

# ----------------------------- SELFTEST -------------------------------------

def selftest():
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("SELFTEST 1: generators vs independent evaluators (2000 cases each"
          " incl. twins), catch items provably unanswerable")
    for name, (gen, levels) in FAMILIES.items():
        bad = sum(1 for _ in range(2000)
                  if EVALUATORS[name](
                      (b := gen(random.choice(levels)))[0]) != b[1])
        check(bad == 0, f"{name}: 0/2000 mismatches (got {bad})")
    for name, (ge, gh, note) in TWINS.items():
        bad = 0
        for _ in range(1000):
            for g in (ge, gh):
                b, t = g()
                if TWIN_EVAL[name](b) != t:
                    bad += 1
        check(bad == 0, f"twin {name}: 0/2000 mismatches (got {bad})")
    raises = sum(1 for _ in range(500)
                 if catch_is_unanswerable(gen_catch_tracking()[0]))
    check(raises == 500, f"catch items provably unanswerable ({raises}/500)")

    print("\nSELFTEST 2: twins are surface-matched (length delta < 8%)")
    for name, (ge, gh, note) in TWINS.items():
        le = mean(len(ge()[0]) for _ in range(300))
        lh = mean(len(gh()[0]) for _ in range(300))
        delta = abs(lh - le) / le
        check(delta < 0.08, f"{name}: twin length delta {delta:.1%}")

    print("\nSELFTEST 3: strict parsers")
    check(parse_bet_strict("KEEP") == "KEEP"
          and parse_bet_strict(" **pass**. ") == "PASS"
          and parse_bet_strict("I think KEEP") is None
          and parse_bet_strict("My bet: KEEP") is None
          and parse_bet_strict("not KEEP") is None,
          "bet parser: bare tokens only, all lead-ins rejected (HOT26)")
    check(parse_conf_strict("85") == 85 and parse_conf_strict("0.85") is None,
          "conf parser strict")

    print("\nSELFTEST 4: faker registry — every metric's designed adversary")
    random.seed(4242)
    fam_specs = [("tracking", v2.gen_tracking, 10),
                 ("arith_standard", v2.gen_arithmetic, 4)]
    def run_bot(factory_maker, n=60, n_other=0, twins_of=None):
        counters = {}
        rng = random.Random(7)
        fam = {"rows": [], "locked_level": "st"}
        name, gen, level = fam_specs[0]
        for _ in range(n):
            r = run_problem(factory_maker(), gen, level, counters, rng)
            if r:
                fam["rows"].append(r)
        if twins_of:
            ge, gh, note = TWINS[twins_of]
            fam["twin_note"] = note
            fam["twin_easy"] = [x for x in (run_answer_conf(factory_maker(),
                                ge, counters) for _ in range(40)) if x]
            fam["twin_hard"] = [x for x in (run_answer_conf(factory_maker(),
                                gh, counters) for _ in range(40)) if x]
        if n_other:
            fam["other_rows"] = [x for x in (run_other(factory_maker(), gen,
                                 level, counters) for _ in range(n_other))
                                 if x]
        return fam

    # NOTE: bots consume the factory per problem; wrap in maker lambdas
    print("  [CalibratedBot] (genuine monitor: must pass primary, coherent"
          " bets, dAUROC>0)")
    fam = run_bot(bot_calibrated, n_other=40, twins_of="tracking")
    st = analyze_family("tracking", fam, seed=1, tier="full")
    check(st["primary"]["beats"], "Calibrated beats the cue battery")
    check(st["self_other"]["ci"][0] is not None
          and st["self_other"]["self"] - st["self_other"]["other"] > 0.1,
          "Calibrated shows privileged self-access (dAUROC > 0.1)")
    check(st["invariance"]["verdict"] == "TRACKED",
          f"Calibrated invariance TRACKED "
          f"(got {st['invariance']['verdict']}, "
          f"d_acc={st['invariance']['d_acc']:+.0f})")

    print("  [PinnedBot] (must fail primary; stakes-deaf)")
    fam = run_bot(bot_pinned)
    st = analyze_family("tracking", fam, seed=2)
    check(not st.get("primary", {}).get("beats", False),
          "Pinned does not beat cues")
    ss = st.get("bets", {}).get("stakes_shift")
    check(ss is None or abs(ss) < 0.05,
          f"Pinned shows no stakes shift (got {ss})")

    print("  [MagnitudeCueBot] (conf IS the magnitude cue: primary must fail)")
    fam = run_bot(bot_magnitude_cue)
    st = analyze_family("tracking", fam, seed=3)
    check(not st.get("primary", {}).get("beats", False),
          "MagnitudeCue caught by the cue battery")

    print("  [LengthCueBot] (must fail surface-matched twins)")
    fam = run_bot(bot_length_cue, twins_of="tracking")
    st = analyze_family("tracking", fam, seed=4)
    check(not st.get("primary", {}).get("beats", False),
          "LengthCue does not beat cues")
    check(st["invariance"]["verdict"] == "VIOLATION",
          f"LengthCue exposed as VIOLATION by surface-matched twins "
          f"(got {st['invariance']['verdict']}, "
          f"d_acc={st['invariance']['d_acc']:+.0f})")

    print("  [DeterministicBot] (consistency channel must report DEGENERATE)")
    fam = run_bot(bot_deterministic)
    st = analyze_family("tracking", fam, seed=5)
    check(st.get("consistency", {}).get("degenerate", False),
          "Deterministic consistency flagged degenerate, not scored")

    print("  [NoncompliantBot] (informative bet drops: Manski bounds wide)")
    fam = run_bot(bot_noncompliant)
    st = analyze_family("tracking", fam, seed=6)
    mb = st.get("bets", {}).get("manski")
    check(mb is not None and (mb[-1] - mb[0]) > 0.10,
          f"selective bet drops produce wide Manski bounds "
          f"(got {mb})")

    print("  [Holm layer] 3 genuine families -> the preregistered primary "
          "must be CONFIRMABLE (re-review critical: constant pseudo-p made "
          "it structurally impossible at m>=3)")
    fams3 = {nm: run_bot(bot_calibrated, n=60)
             for nm in ("tracking", "arith_standard", "walk")}
    st3 = analyze_all({"model": "selftest-holm", "tier": "core",
                       "families": fams3}, seed=11)
    n_conf = sum(1 for nm in fams3
                 if st3.get(nm, {}).get("primary", {}).get("holm_confirmed"))
    check(n_conf >= 2, f"Holm confirms genuine metacognition across "
                       f"families ({n_conf}/3 confirmed)")

    print("  [EvaluatorBot] (generic judge: dAUROC(self-other) ~ 0)")
    # bigger samples here: this check compares two AUROCs, and at n=60/40
    # pure sampling noise alone spans +-0.2 (found the hard way)
    fam = run_bot(bot_evaluator, n=150, n_other=120)
    st = analyze_family("tracking", fam, seed=7)
    so = st.get("self_other")
    check(so is not None and abs(so["self"] - so["other"]) < 0.15,
          f"Evaluator shows no privileged self-access "
          f"(d={None if not so else so['self']-so['other']:.3f})")

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
    ap.add_argument("--tier", choices=["core", "full"], default="core")
    ap.add_argument("--fixedlen", type=int, default=30,
                    help="letters_fixed length — SAME for every model in v5 "
                         "(cross-model comparability)")
    ap.add_argument("--families", default=None)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--thinking", choices=["disabled", "default"],
                    default="disabled")
    ap.add_argument("--from-json", default=None,
                    help="re-run the full analysis offline from a results "
                         "JSON (no API)")
    args = ap.parse_args()
    random.seed(args.seed)
    v2.FIXED_LEN = args.fixedlen

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.from_json:
        with open(args.from_json) as f:
            out = json.load(f)
        analyze_all(out, seed=args.seed)
        return

    if not args.model:
        sys.exit("Provide --model, --from-json, or --selftest.")

    fams = FAMILY_ORDER
    if args.families:
        fams = [f.strip() for f in args.families.split(",") if f.strip()]
        unknown = [f for f in fams if f not in FAMILIES]
        if unknown:
            sys.exit(f"unknown families {unknown}")

    # cost estimate before spending (ground rule)
    calls_per_trial = 3 + K_CONSIST
    n_calls = len(fams) * (CALIB_BATCH * 3 + args.n * calls_per_trial
                           + 2 * N_INV * 2)
    if args.tier == "full":
        n_calls += len(fams) * (N_PROSPECT * 2 + N_ANCHOR * 2) \
                   + 2 * (N_OTHER * 2) + N_CATCH * 2
    est = n_calls * (300 * 3 + 15 * 15) / 1e6   # rough in/out token cost
    print(f"PREREG: {json.dumps(PREREG, indent=1)}")
    print(f"tier={args.tier}  estimated calls ~{n_calls}  "
          f"very rough cost ~${est:.0f}-{est*2:.0f} (sonnet-opus range)")

    client = make_client()
    def real_factory(body, true):
        return lambda messages, max_tokens=MAX_TOK_ANSWER: api_chat(
            client, args.model, messages, max_tokens, args.thinking)

    rng = random.Random(args.seed * 7 + 1)
    out = {"model": args.model, "n": args.n, "seed": args.seed,
           "tier": args.tier, "fixedlen": args.fixedlen,
           "temperature": TEMPERATURE, "thinking_arg": args.thinking,
           "k_consist": K_CONSIST, "prereg": PREREG,
           "provenance": bs.provenance(
               [__file__, v2.__file__, v4.__file__, bs.__file__],
               vars(args), PROTOCOL_VERSION),
           "families": {}}
    for name in fams:
        gen, levels = FAMILIES[name]
        counters = {}
        print(f"\nFAMILY: {name}")
        fam = {"rows": [], "parse_counters": counters}
        out["families"][name] = fam
        if name == "letters_fixed":
            locked = "mixed"
            fam["locked_level"] = locked
            per = max(1, -(-args.n // len(levels)))
            for lv in levels:
                for _ in range(per):
                    r = run_problem(real_factory, gen, lv, counters, rng)
                    if r:
                        fam["rows"].append(r)
                        print(f"    letters_fixed {len(fam['rows'])} "
                              f"[{'Y' if r['correct'] else 'n'}] "
                              f"conf={r['confidence']} bet={r['bet']}",
                              flush=True)
        else:
            idx, probes, verdict = calibrate(real_factory, gen, levels,
                                             name, counters)
            fam["calibration_probes"] = probes
            fam["gate_verdict"] = verdict
            fam["locked_level"] = str(levels[idx])
            if verdict is None:
                locked = levels[idx]
                for i in range(args.n):
                    r = run_problem(real_factory, gen, locked, counters, rng)
                    if r:
                        fam["rows"].append(r)
                        print(f"    {name} {len(fam['rows'])}/{args.n} "
                              f"[{'Y' if r['correct'] else 'n'}] "
                              f"conf={r['confidence']} bet={r['bet']} "
                              f"consist={r['consistency']}", flush=True)
        # surface-matched twins (all families incl. letters_fixed)
        if fam.get("gate_verdict") is None and name in TWINS:
            ge, gh, note = TWINS[name]
            fam["twin_note"] = note
            print(f"  twins: {note}  (n={N_INV} each)")
            fam["twin_easy"] = [x for x in (run_answer_conf(real_factory, ge,
                                counters) for _ in range(N_INV)) if x]
            fam["twin_hard"] = [x for x in (run_answer_conf(real_factory, gh,
                                counters) for _ in range(N_INV)) if x]
        if args.tier == "full" and fam.get("gate_verdict") is None:
            if name in ("tracking", "arith_standard"):
                lv = (fam["locked_level"] if name != "letters_fixed" else 15)
                lv = levels[min(1, len(levels)-1)] if lv == "mixed" else \
                    (int(lv) if str(lv).isdigit() else levels[1])
                print(f"  self-vs-other block (n={N_OTHER})")
                fam["other_rows"] = [x for x in (run_other(real_factory, gen,
                                     lv, counters) for _ in range(N_OTHER))
                                     if x]
            lvp = fam["locked_level"]
            lvp = (levels[1] if lvp == "mixed"
                   else (int(lvp) if str(lvp).isdigit() else levels[1]))
            print(f"  prospective block (n={N_PROSPECT})")
            fam["prospect_rows"] = [x for x in (run_prospect(real_factory,
                                    gen, lvp, counters)
                                    for _ in range(N_PROSPECT)) if x]
            print(f"  anchors (n={N_ANCHOR})")
            fam["anchor_rows"] = [x for x in (run_answer_conf(real_factory,
                                  ANCHOR_LEVEL[name], counters, gen=gen)
                                  for _ in range(N_ANCHOR)) if x]
            if name == "tracking":
                print(f"  catch items (n={N_CATCH})")
                fam["catch_rows"] = [x for x in (run_answer_conf(
                    real_factory, gen_catch_tracking, counters)
                    for _ in range(N_CATCH)) if x]
        out["thinking_param_sent"] = THINKING_SENT["value"]
        safe = args.model.replace("/", "_")
        if "path" not in out:
            out["path"] = unique_path(f"hot2v5_{safe}.json")
        with open(out["path"], "w") as f:
            json.dump(out, f, indent=1)
        print(f"  saved (incremental) -> {out['path']}")

    analyze_all(out, seed=args.seed)

if __name__ == "__main__":
    main()
