#!/usr/bin/env python3
"""
Berg test: does arithmetic overconfidence follow SURFACE FORM or DIFFICULTY?
============================================================================
Hypothesis (C. Berg): RLHF trains models to sound confident on math-shaped
problems, masking genuine self-monitoring there. Random-string counting has
no such prior, so the real signal shows through.

Discriminating design: three problem types with the same "math" surface form,
same elicitation format as the v2 arithmetic baseline (answer + confidence in
one JSON), each auto-calibrated to the same accuracy band:

  arith_standard   ordinary add/subtract/multiply chains (the v2 baseline)
  arith_mod        chains in modular arithmetic with unusual moduli (17, 19..)
  arith_madeop     a made-up operation DEFINED FRESH in each problem
                   (e.g. "x blorp y = 2x - y + 3"), nested

Readout, at matched accuracy:
  - training-artifact branch: novel math stays pinned at high confidence with
    low AUROC2, just like standard arithmetic (overconfidence follows the
    surface form)
  - masked-signal branch: novel math shows a wider confidence spread, a
    smaller calibration gap, and higher AUROC2 (confidence follows actual
    difficulty once the trained prior is gone)

Usage:
  python3 berg_test.py --selftest
  export ANTHROPIC_API_KEY=...
  python3 berg_test.py --model claude-sonnet-5
  python3 berg_test.py --model claude-opus-4-8
"""

import os
import re
import sys
import json
import time
import random
import argparse
from statistics import mean

N_MAIN = 60
TARGET_LOW, TARGET_HIGH = 0.55, 0.85
CALIB_BATCH = 24     # 12 -> 24: halves probe noise; 'matched accuracy' is the
CALIB_ROUNDS = 5     # design's core premise and 12-trial probes can't deliver it
MAX_TOKENS = 120
RETRIES = 4

# ----------------------------- GENERATORS -----------------------------------
# Each: gen(level) -> (body, true_answer)

def gen_standard(level):
    steps = level
    val = random.randint(100, 999)
    parts = [f"Start with {val}."]
    for _ in range(steps):
        op = random.choice(["add", "sub", "mul"])
        if op == "add":
            n = random.randint(37, 989); val += n; parts.append(f"Add {n}.")
        elif op == "sub":
            n = random.randint(37, 989); val -= n; parts.append(f"Subtract {n}.")
        else:
            n = random.randint(3, 9); val *= n; parts.append(f"Multiply by {n}.")
        if abs(val) > 200000:
            val = val % 1000
            parts.append("Keep only the last three digits (result mod 1000).")
    body = ("Do the following calculation in your head and give only the "
            "final result.\n\n" + " ".join(parts))
    return body, val

STANDARD_LEVELS = [2, 3, 4, 5, 6, 8]

# genuinely unusual moduli only (the docstring promises "17, 19..": 7/11/13
# are the most textbook-standard moduli there are, which would let familiarity
# leak into the "novel math" arm; larger moduli also shrink lucky-guess rates)
UNUSUAL_MODULI = [17, 19, 23, 29, 31]

def gen_mod(level):
    steps = level
    m = random.choice(UNUSUAL_MODULI)
    val = random.randint(100, 999)
    parts = [f"Start with {val}."]
    for _ in range(steps):
        op = random.choice(["add", "sub", "mul"])
        if op == "add":
            n = random.randint(13, 97); val += n; parts.append(f"Add {n},")
        elif op == "sub":
            n = random.randint(13, 97); val -= n; parts.append(f"Subtract {n},")
        else:
            n = random.randint(3, 9); val *= n; parts.append(f"Multiply by {n},")
        val = val % m
        parts.append(f"then take the result mod {m}.")
    body = ("Do the following calculation in your head and give only the "
            "final result.\n\n" + " ".join(parts))
    return body, val

MOD_LEVELS = [1, 2, 3, 4, 5, 6]

OP_NAMES = ["blorp", "trell", "quang", "smib", "dax", "vun"]

def gen_madeop(level):
    depth = level
    name = random.choice(OP_NAMES)
    a = random.choice([-3, -2, 2, 3])
    b = random.choice([-2, -1, 1, 2])
    c = random.randint(-5, 5)
    def op(x, y):
        return a * x + b * y + c
    vals = [random.randint(2, 9) for _ in range(depth + 1)]
    expr = str(vals[0])
    acc = vals[0]
    for v in vals[1:]:
        expr = f"({expr} {name} {v})"
        acc = op(acc, v)
    sign_b = f"- {abs(b)}" if b < 0 else f"+ {b}"
    sign_c = f"- {abs(c)}" if c < 0 else f"+ {c}"
    body = (f"Define a new operation '{name}': x {name} y = {a}*x {sign_b}*y "
            f"{sign_c}.\n\nDo the following calculation in your head and give "
            f"only the final result.\n\nCompute {expr}.")
    return body, acc

MADEOP_LEVELS = [1, 2, 3, 4, 5]

TYPES = {
    "arith_standard": (gen_standard, STANDARD_LEVELS),
    "arith_mod":      (gen_mod,      MOD_LEVELS),
    "arith_madeop":   (gen_madeop,   MADEOP_LEVELS),
}

PROMPT = ("{body}\n\nAnswer immediately from your best judgment. Do NOT work "
          "through steps out loud. Respond with ONLY a JSON object on one "
          'line in exactly this form: {{"answer": <integer>, "confidence": '
          "<integer 0-100>}}\n"
          "confidence is how sure you are that your answer is exactly correct.")

# ----------------------------- PLUMBING -------------------------------------

def parse_reply(text):
    """Returns (answer, confidence, parse_mode) or None.
    Prefers the LAST parseable JSON object (replies sometimes echo the
    prompt's template braces before the real answer). The numeric fallback
    is strict: exactly two integers with the second in 0-100 — otherwise
    None, so the call is retried instead of fabricating a row (a loose
    fallback demonstrably injects fake low-confidence rows, concentrated
    on mod problems whose replies contain stray moduli)."""
    if text is None:
        return None
    best = None
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            best = (int(obj["answer"]),
                    max(0.0, min(100.0, float(obj["confidence"]))), "json")
        except Exception:
            continue
    if best is not None:
        return best
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    if len(nums) == 2:
        try:
            conf = float(nums[1])
            if 0 <= conf <= 100:
                return int(nums[0]), conf, "fallback"
        except Exception:
            pass
    return None

def make_client():
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY first.")
    return anthropic.Anthropic()

def ask(client, model, body):
    prompt = PROMPT.format(body=body)
    for attempt in range(RETRIES):
        try:
            try:
                resp = client.messages.create(
                    model=model, max_tokens=MAX_TOKENS,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}])
            except Exception as e:
                if "thinking" in str(e).lower():
                    resp = client.messages.create(
                        model=model, max_tokens=MAX_TOKENS,
                        messages=[{"role": "user", "content": prompt}])
                else:
                    raise
            text = "\n".join(b.text for b in resp.content
                             if getattr(b, "type", "") == "text")
            parsed = parse_reply(text)
            if parsed is not None:
                pred, conf, mode = parsed
                return (pred, conf, mode), text
        except Exception as e:
            if attempt == RETRIES - 1:
                print(f"    call failed: {e}")
            time.sleep(1.5 * (attempt + 1))
    return None, None

# ----------------------------- RUN / ANALYZE --------------------------------

def run_batch(answer_fn, gen, level, n, label=""):
    rows = []
    for i in range(n):
        body, true = gen(level)
        out, raw = answer_fn(body)
        if out is None:
            continue
        pred, conf, mode = out
        rows.append({"level": level, "true": true, "pred": pred,
                     "correct": pred == true, "confidence": conf,
                     "parse_mode": mode, "body": body, "raw": raw})
        if label:
            print(f"    {label} {i+1}/{n} [{'Y' if pred == true else 'n'}] "
                  f"conf={conf:.0f}", flush=True)
    if len(rows) < n:
        print(f"    WARNING: {n - len(rows)}/{n} calls failed or were "
              f"unparseable and were dropped", flush=True)
    return rows

def accuracy(rows):
    return mean(1.0 if r["correct"] else 0.0 for r in rows) if rows else 0.0

def calibrate(answer_fn, gen, levels, name):
    print(f"\n  calibrating {name} (target {TARGET_LOW:.0%}-{TARGET_HIGH:.0%})")
    idx = min(1, len(levels) - 1)
    tried = {}
    probes = {}      # saved with results: free cross-level confidence data
    for _ in range(CALIB_ROUNDS):
        level = levels[idx]
        rows = run_batch(answer_fn, gen, level, CALIB_BATCH)
        if not rows:
            sys.exit(f"ERROR: every calibration call failed for {name} at "
                     f"level {level}; aborting rather than mis-calibrating "
                     f"on a dead API")
        probes.setdefault(str(level), []).extend(rows)
        acc = accuracy(rows)
        tried[level] = acc
        print(f"    level={level}  accuracy={acc:.2f}  (n={len(rows)})", flush=True)
        if TARGET_LOW <= acc <= TARGET_HIGH:
            print(f"    -> locked level = {level}")
            return level, probes
        if acc > TARGET_HIGH and idx < len(levels) - 1:
            idx += 1
        elif acc < TARGET_LOW and idx > 0:
            idx -= 1
        else:
            break
    best = min(tried, key=lambda L: abs(tried[L] - 0.7))
    if not (TARGET_LOW <= tried[best] <= TARGET_HIGH):
        print(f"    WARNING: settled accuracy {tried[best]:.2f} is OUTSIDE "
              f"the target band; 'matched accuracy' may not hold for {name}",
              flush=True)
    print(f"    -> settling on level = {best}")
    return best, probes

def auroc2(rows):
    pos = [r["confidence"] for r in rows if r["correct"]]
    neg = [r["confidence"] for r in rows if not r["correct"]]
    if not pos or not neg:
        return None, len(pos), len(neg)
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1 if p > q else (0.5 if p == q else 0)
    return wins / (len(pos) * len(neg)), len(pos), len(neg)

def bootstrap_ci(rows, iters=2000):
    vals = []
    k = len(rows)
    for _ in range(iters):
        sample = [rows[random.randrange(k)] for _ in range(k)]
        a, _, _ = auroc2(sample)
        if a is not None:
            vals.append(a)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]

def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"

def unique_path(path):
    """Never overwrite a previous results file: add _run2, _run3, ..."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 2
    while os.path.exists(f"{stem}_run{k}{ext}"):
        k += 1
    return f"{stem}_run{k}{ext}"

def report(all_rows, model):
    print("\n" + "=" * 74)
    print(f"BERG TEST RESULTS  model={model}")
    print("=" * 74)
    stats = {}
    for name, rows in all_rows.items():
        if not rows:
            continue
        acc = accuracy(rows)
        confs = [r["confidence"] for r in rows]
        hi = mean(1.0 if c >= 90 else 0.0 for c in confs)
        gap = mean(confs) - 100 * acc
        a, npos, nneg = auroc2(rows)
        lo, hi_ci = bootstrap_ci(rows) if a is not None else (None, None)
        cc = [r["confidence"] for r in rows if r["correct"]]
        cw = [r["confidence"] for r in rows if not r["correct"]]
        stats[name] = dict(acc=acc, meanconf=mean(confs), hi=hi, gap=gap, a=a)
        print(f"\n[{name}] n={len(rows)}  level={rows[0]['level']}  "
              f"accuracy={acc:.2f}")
        print(f"  confidence: mean={mean(confs):.1f}  share>=90: {hi:.0%}  "
              f"range {min(confs):.0f}-{max(confs):.0f}")
        print(f"  calibration gap (mean conf - accuracy%): {gap:+.1f}")
        if cc and cw:
            print(f"  conf when right={mean(cc):.1f}  when wrong={mean(cw):.1f}")
        print(f"  AUROC2={fmt(a)}  CI[{fmt(lo)},{fmt(hi_ci)}]")
    if "arith_standard" in stats:
        s = stats["arith_standard"]
        print("\n--- verdict guide ---")
        print(f"standard arithmetic: gap {s['gap']:+.1f}, AUROC2 {fmt(s['a'])}, "
              f"{s['hi']:.0%} of answers at conf>=90")
        for name in ("arith_mod", "arith_madeop"):
            if name not in stats:
                continue
            t = stats[name]
            print(f"{name}: gap {t['gap']:+.1f}, AUROC2 {fmt(t['a'])}, "
                  f"{t['hi']:.0%} at conf>=90")
        print("If the novel-math rows look like the standard row (pinned-high "
              "confidence, low AUROC2) at matched accuracy, overconfidence "
              "follows SURFACE FORM: training-artifact branch. If they show "
              "lower gaps, wider spread, and higher AUROC2, confidence follows "
              "actual difficulty: masked-signal branch.")

# ----------------------------- SELFTEST -------------------------------------

def selftest():
    print("SELFTEST 1: generator ground truths vs an independent evaluator")
    ok = True
    for _ in range(4000):
        body, true = gen_mod(random.choice(MOD_LEVELS))
        val = int(re.search(r"Start with (\d+)", body).group(1))
        for mm in re.finditer(r"(Add|Subtract|Multiply by) (\d+), then take "
                              r"the result mod (\d+)\.", body):
            n, m = int(mm.group(2)), int(mm.group(3))
            if mm.group(1) == "Add":
                val += n
            elif mm.group(1) == "Subtract":
                val -= n
            else:
                val *= n
            val %= m
        if val != true:
            ok = False
            break
    print("  arith_mod:", "OK" if ok else "MISMATCH")
    ok2 = True
    for _ in range(4000):
        body, true = gen_madeop(random.choice(MADEOP_LEVELS))
        m = re.search(r"x (\w+) y = (-?\d+)\*x ([+-]) (\d+)\*y ([+-]) (\d+)", body)
        a = int(m.group(2))
        b = int(m.group(4)) * (1 if m.group(3) == "+" else -1)
        c = int(m.group(6)) * (1 if m.group(5) == "+" else -1)
        vals = [int(v) for v in re.findall(r"(?<![\d*])\b(\d)\b",
                body.split("Compute")[1])]
        acc = vals[0]
        for v in vals[1:]:
            acc = a * acc + b * v + c
        if acc != true:
            ok2 = False
            break
    print("  arith_madeop:", "OK" if ok2 else "MISMATCH")

    print("\nSELFTEST 2: report rendering on a fake training-artifact model")
    fake = {}
    for name in TYPES:
        rows = []
        for _ in range(60):
            correct = random.random() < 0.65
            conf = random.gauss(95, 3)
            rows.append({"level": 3, "true": 0, "pred": 0, "correct": correct,
                         "confidence": max(0, min(100, conf)), "raw": ""})
        fake[name] = rows
    report(fake, "selftest")
    print("\nSELFTEST done." if (ok and ok2) else
          "\nSELFTEST FAILED: do not run for real.")

# ----------------------------- MAIN -----------------------------------------

def main():
    global N_MAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=N_MAIN)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the problem generators for reproducibility")
    args = ap.parse_args()
    N_MAIN = args.n
    if args.seed is not None:
        random.seed(args.seed)
    if args.selftest:
        selftest()
        return
    if not args.model:
        sys.exit("Provide --model (or --selftest).")
    client = make_client()
    answer_fn = lambda body: ask(client, args.model, body)
    all_rows = {}
    all_probes = {}
    for name, (gen, levels) in TYPES.items():
        print(f"\nTYPE: {name}")
        level, probes = calibrate(answer_fn, gen, levels, name)
        all_probes[name] = probes
        all_rows[name] = run_batch(answer_fn, gen, level, N_MAIN, label=name)
    # save BEFORE reporting: analysis crashes must never discard paid data
    safe = args.model.replace("/", "_")
    path = unique_path(f"bergtest_{safe}.json")
    with open(path, "w") as f:
        json.dump({"model": args.model, "n": N_MAIN, "seed": args.seed,
                   "calib_batch": CALIB_BATCH, "moduli": UNUSUAL_MODULI,
                   "rows": all_rows, "calibration": all_probes}, f, indent=2)
    print(f"\nsaved -> {path}")
    report(all_rows, args.model)

if __name__ == "__main__":
    main()
