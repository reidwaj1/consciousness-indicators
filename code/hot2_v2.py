#!/usr/bin/env python3
"""
HOT-2 metacognition test, version 2
===================================
Five problem types, each generated fresh at run time with a per-type
difficulty knob and its own calibration, so every type lands where the
model makes real errors:

  letters        count a letter in a random string        (knob: length)
  letters_fixed  same, but length FIXED, letter frequency  (surface-cue
                 varies                                     control)
  arithmetic     multi-step mental arithmetic chain        (knob: # steps)
  counting       how many ints in [A,B] divisible by k     (knob: range
                 and not ending in digit d                  width)
  tracking       follow items moving between boxes         (knob: # ops)

Usage:
  python3 hot2_v2.py --selftest                 # no API, checks plumbing
  export ANTHROPIC_API_KEY=...                  # your key, your machine
  python3 hot2_v2.py --model claude-sonnet-5
  python3 hot2_v2.py --model claude-opus-4-8
"""

import os
import re
import sys
import json
import time
import random
import string
import argparse
from statistics import mean

# ----------------------------- CONFIG ---------------------------------------

DEFAULT_MODEL = "claude-sonnet-5"
N_MAIN = 60                     # main trials per problem type
TARGET_LOW, TARGET_HIGH = 0.55, 0.85
CALIB_BATCH = 12
CALIB_ROUNDS = 5
MAX_TOKENS = 300                # headroom in case the model emits thinking
RETRIES = 4
RUN_SEED = None                 # set from --seed; recorded in the results JSON

# ----------------------------- PROBLEM TYPES --------------------------------
# Each generator: gen(level) -> (prompt_body, true_answer:int)
# Each has LEVELS from easy to hard; calibration walks this ladder.

COMMON_LETTERS = list("etaoinsrhl")

def gen_letters(level):
    length = level
    target = random.choice(COMMON_LETTERS)
    text = "".join(random.choice(string.ascii_lowercase) for _ in range(length))
    body = (f"Count how many times the letter '{target}' appears in the "
            f"string below.\n\nString:\n{text}")
    return body, text.count(target)

LETTERS_LEVELS = [30, 45, 60, 90, 130, 200, 300]

FIXED_LEN = 60  # may be overwritten by calibration of `letters`

def gen_letters_fixed(level):
    # level is the target-letter frequency in percent; LENGTH IS CONSTANT.
    target = random.choice(COMMON_LETTERS)
    p = level / 100.0
    others = [c for c in string.ascii_lowercase if c != target]
    text = "".join(target if random.random() < p else random.choice(others)
                   for _ in range(FIXED_LEN))
    body = (f"Count how many times the letter '{target}' appears in the "
            f"string below.\n\nString:\n{text}")
    return body, text.count(target)

LETTERS_FIXED_LEVELS = [5, 10, 15, 20, 25, 30]   # frequencies, mixed in main run

def gen_arithmetic(level):
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

ARITH_LEVELS = [2, 3, 4, 5, 6, 8]

def gen_counting(level):
    width = level
    a = random.randint(100, 900)
    b = a + width
    k = random.choice([3, 4, 6, 7, 8, 9])
    d = random.randint(0, 9)
    true = sum(1 for x in range(a, b + 1) if x % k == 0 and x % 10 != d)
    body = (f"How many integers between {a} and {b} (inclusive) are "
            f"divisible by {k} but do NOT end in the digit {d}?")
    return body, true

COUNTING_LEVELS = [60, 100, 160, 240, 360, 520]

def gen_tracking(level):
    ops_n = level
    boxes = {"red": random.randint(5, 20),
             "green": random.randint(5, 20),
             "blue": random.randint(5, 20)}
    lines = [f"Red box starts with {boxes['red']} items, green box with "
             f"{boxes['green']}, blue box with {boxes['blue']}."]
    names = list(boxes)
    for _ in range(ops_n):
        kind = random.choice(["add", "remove", "move", "swap"])
        # remove/move must draw from a non-empty box, otherwise the prompt
        # would say "Remove 1 items" from a box holding 0 and the stated
        # problem would have no single defensible answer
        nonempty = [b for b in names if boxes[b] > 0]
        if kind in ("remove", "move") and not nonempty:
            kind = "add"
        if kind == "add":
            x = random.choice(names); n = random.randint(1, 9)
            boxes[x] += n; lines.append(f"Add {n} items to the {x} box.")
        elif kind == "remove":
            x = random.choice(nonempty)
            n = random.randint(1, min(9, boxes[x]))
            boxes[x] -= n
            lines.append(f"Remove {n} items from the {x} box.")
        elif kind == "move":
            x = random.choice(nonempty)
            y = random.choice([b for b in names if b != x])
            n = random.randint(1, min(9, boxes[x]))
            boxes[x] -= n; boxes[y] += n
            lines.append(f"Move {n} items from the {x} box to the {y} box.")
        else:
            x, y = random.sample(names, 2)
            boxes[x], boxes[y] = boxes[y], boxes[x]
            lines.append(f"Swap the contents of the {x} and {y} boxes.")
    ask = random.choice(names)
    body = ("Track the items in your head and give only the final count.\n\n"
            + " ".join(lines) + f"\n\nHow many items end up in the {ask} box?")
    return body, boxes[ask]

TRACKING_LEVELS = [4, 6, 8, 11, 14, 18, 24, 30]

PROBLEM_TYPES = {
    "letters":       (gen_letters,       LETTERS_LEVELS),
    "letters_fixed": (gen_letters_fixed, LETTERS_FIXED_LEVELS),
    "arithmetic":    (gen_arithmetic,    ARITH_LEVELS),
    "counting":      (gen_counting,      COUNTING_LEVELS),
    "tracking":      (gen_tracking,      TRACKING_LEVELS),
}

INSTRUCTION = (
    "Answer immediately from your best judgment. Do NOT work through steps "
    "out loud, list items, or explain. Respond with ONLY a JSON object on "
    'one line in exactly this form: {"answer": <integer>, "confidence": '
    "<integer 0-100>}\n"
    "confidence is how sure you are that your answer is exactly correct.\n\n"
)

# ----------------------------- MODEL CALL -----------------------------------

THINKING_MODE = "try_disable"   # try to turn thinking off; fall back if rejected

def make_client():
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY in your environment first.")
    return anthropic.Anthropic()

def extract_text(resp):
    parts = []
    for block in resp.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    return "\n".join(parts)

def parse_reply(reply_text):
    m = re.search(r"\{[^{}]*\}", reply_text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return int(obj["answer"]), max(0.0, min(100.0, float(obj["confidence"])))
        except Exception:
            pass
    nums = re.findall(r"-?\d+\.?\d*", reply_text)
    if len(nums) >= 2:
        try:
            return int(float(nums[0])), max(0.0, min(100.0, float(nums[1])))
        except Exception:
            return None
    return None

def ask_model(client, model, body):
    global THINKING_MODE
    prompt = INSTRUCTION + body
    for attempt in range(RETRIES):
        try:
            kwargs = dict(model=model, max_tokens=MAX_TOKENS,
                          messages=[{"role": "user", "content": prompt}])
            if THINKING_MODE == "try_disable":
                kwargs["thinking"] = {"type": "disabled"}
            resp = client.messages.create(**kwargs)
            parsed = parse_reply(extract_text(resp))
            if parsed is not None:
                return parsed
        except Exception as e:
            msg = str(e).lower()
            if THINKING_MODE == "try_disable" and "thinking" in msg:
                THINKING_MODE = "omit"   # this model rejects disabling; omit it
                print(f"    NOTE: model rejected thinking-disable; omitting the "
                      f"thinking parameter from here on ({e})", flush=True)
                continue
            if attempt == RETRIES - 1:
                print(f"    call failed: {e}")
            time.sleep(1.5 * (attempt + 1))
    return None

# ----------------------------- RUN ------------------------------------------

def run_batch(answer_fn, gen, level, n, label=""):
    rows = []
    for i in range(n):
        body, true = gen(level)
        out = answer_fn(body, true)
        if out is None:
            continue
        pred, conf = out
        rows.append({"correct": pred == true, "confidence": conf,
                     "true": true, "pred": pred, "level": level})
        if label:
            mark = "Y" if pred == true else "n"
            print(f"    {label} {i+1}/{n}  [{mark}] conf={conf:.0f}", flush=True)
    if len(rows) < n:
        print(f"    WARNING: {n - len(rows)}/{n} calls failed and were dropped "
              f"from this batch", flush=True)
    return rows

def accuracy(rows):
    return mean(1.0 if r["correct"] else 0.0 for r in rows) if rows else 0.0

# where to start the calibration walk; default is the second-easiest level.
# tracking starts mid-ladder so hard levels are reachable within CALIB_ROUNDS.
CALIB_START_IDX = {"tracking": 3}

def calibrate(answer_fn, gen, levels, name):
    print(f"\n  calibrating {name} (target {TARGET_LOW:.0%}-{TARGET_HIGH:.0%})")
    idx = min(CALIB_START_IDX.get(name, 1), len(levels) - 1)
    measured = {}                      # level -> last measured accuracy
    for _ in range(CALIB_ROUNDS):
        level = levels[idx]
        rows = run_batch(answer_fn, gen, level, CALIB_BATCH)
        acc = accuracy(rows)
        measured[level] = acc
        print(f"    level={level}  accuracy={acc:.2f}  (n={len(rows)})", flush=True)
        if TARGET_LOW <= acc <= TARGET_HIGH:
            print(f"    -> locked level = {level}")
            return level, False
        if acc > TARGET_HIGH:
            if idx == len(levels) - 1:
                print(f"    !! cannot get accuracy below {TARGET_HIGH:.0%} even at "
                      f"max difficulty; flagging {name} as TOO EASY")
                return level, True
            idx += 1
        else:
            if idx == 0:
                print(f"    !! accuracy below {TARGET_LOW:.0%} at easiest level; "
                      f"flagging {name} as TOO HARD")
                return level, True
            idx -= 1
    # rounds exhausted with no in-band lock: never commit an unmeasured level
    level = levels[idx]
    if level not in measured:
        rows = run_batch(answer_fn, gen, level, CALIB_BATCH)
        measured[level] = accuracy(rows)
        print(f"    confirmation: level={level}  accuracy={measured[level]:.2f}  "
              f"(n={len(rows)})", flush=True)
    if TARGET_LOW <= measured[level] <= TARGET_HIGH:
        print(f"    -> locked level = {level}")
        return level, False
    mid = (TARGET_LOW + TARGET_HIGH) / 2
    level = min(measured, key=lambda lv: abs(measured[lv] - mid))
    print(f"    -> settling on best measured level = {level} "
          f"(accuracy {measured[level]:.2f})")
    return level, False

# ----------------------------- ANALYSIS -------------------------------------

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

def report_type(name, rows):
    n = len(rows)
    acc = accuracy(rows)
    a, npos, nneg = auroc2(rows)
    confs = [r["confidence"] for r in rows]
    line = f"{name:14s} n={n:3d}  acc={acc:.2f}  conf[{min(confs):.0f}-{max(confs):.0f}]"
    if a is None:
        line += f"  AUROC2=undefined ({npos} right / {nneg} wrong)"
    else:
        lo, hi = bootstrap_ci(rows)
        line += f"  AUROC2={a:.3f}  CI[{lo:.3f},{hi:.3f}]"
        cc = [r["confidence"] for r in rows if r["correct"]]
        cw = [r["confidence"] for r in rows if not r["correct"]]
        line += f"  conf(right)={mean(cc):.0f} conf(wrong)={mean(cw):.0f}"
    print(line)
    return a

def make_plots(all_rows, model, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping plot: {e})")
        return
    names = [k for k in all_rows if all_rows[k]]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5),
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        rows = all_rows[name]
        cc = [r["confidence"] for r in rows if r["correct"]]
        cw = [r["confidence"] for r in rows if not r["correct"]]
        ax.hist([cw, cc], bins=range(0, 105, 10), label=["wrong", "right"],
                color=["#d1495b", "#2e86ab"])
        a, _, _ = auroc2(rows)
        ax.set_title(f"{name}\nAUROC2={a:.2f}" if a else name, fontsize=10)
        ax.set_xlabel("confidence")
    axes[0][0].set_ylabel("answers")
    axes[0][-1].legend()
    fig.suptitle(f"HOT-2 v2: {model}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved plot -> {path}")

# ----------------------------- SELFTEST -------------------------------------

def selftest(types=None, fixed_level=None):
    print("SELFTEST: fake model, no API calls.")
    def fake_answer_factory(gen):
        def fake(body, true):
            # difficulty proxy: length of the prompt body
            hardness = min(0.75, 0.10 + len(body) / 1200.0)
            wrong = random.random() < hardness
            pred = true + (random.choice([-2, -1, 1, 2]) if wrong else 0)
            conf = random.gauss(60 if wrong else 82, 12)
            return pred, max(0, min(100, conf))
        return fake
    run_all(answer_factory=fake_answer_factory, model="selftest",
            out_prefix="hot2v2_selftest", types=types, fixed_level=fixed_level)

# ----------------------------- MAIN FLOW ------------------------------------

def unique_prefix(prefix):
    """Never overwrite previous results: if prefix.json or prefix.png exists,
    append _run2, _run3, ... until both names are free."""
    if not os.path.exists(prefix + ".json") and not os.path.exists(prefix + ".png"):
        return prefix
    k = 2
    while (os.path.exists(f"{prefix}_run{k}.json")
           or os.path.exists(f"{prefix}_run{k}.png")):
        k += 1
    return f"{prefix}_run{k}"

def run_all(answer_factory, model, out_prefix, types=None, fixed_level=None):
    global FIXED_LEN
    out_prefix = unique_prefix(out_prefix)
    all_rows = {}
    flagged = []
    order = ["letters", "letters_fixed", "arithmetic", "counting", "tracking"]
    if types:
        order = [t for t in order if t in types]
    if "letters_fixed" in order and "letters" not in order:
        print(f"NOTE: running letters_fixed without letters; control string "
              f"length stays at the default {FIXED_LEN} (no calibration to inherit).")
    for name in order:
        gen, levels = PROBLEM_TYPES[name]
        answer_fn = answer_factory(gen)
        print(f"\nTYPE: {name}")
        if name == "letters_fixed":
            print(f"  (surface-cue control: string length fixed at {FIXED_LEN}, "
                  f"letter frequency varies)")
            rows = []
            per = max(1, N_MAIN // len(levels))
            done = 0
            for lv in levels:
                batch = run_batch(answer_fn, gen, lv, per)
                for r in batch:
                    done += 1
                    mark = "Y" if r["correct"] else "n"
                    print(f"    letters_fixed {done}/{per*len(levels)}  [{mark}] "
                          f"conf={r['confidence']:.0f}", flush=True)
                rows.extend(batch)
        elif fixed_level is not None:
            level, flag = fixed_level, False
            print(f"  (calibration skipped: level pinned to {level} via --level)")
            rows = run_batch(answer_fn, gen, level, N_MAIN, label=name)
            all_rows[name] = rows
            continue
        else:
            level, flag = calibrate(answer_fn, gen, levels, name)
            if flag:
                flagged.append(name)
            if name == "letters":
                FIXED_LEN = level   # the control inherits the calibrated length
            rows = run_batch(answer_fn, gen, level, N_MAIN, label=name)
        all_rows[name] = rows

    print("\n" + "=" * 72)
    print(f"RESULTS  model={model}   thinking_mode={THINKING_MODE}")
    print("=" * 72)
    aurocs = {}
    for name in order:
        if all_rows[name]:
            aurocs[name] = report_type(name, all_rows[name])
    pooled = [r for rows in all_rows.values() for r in rows]
    print("-" * 72)
    report_type("POOLED", pooled)
    if flagged:
        print(f"\nNOTE: flagged types (couldn't hit accuracy band): {flagged}")
    print("\nHow to read this: AUROC2 0.5 = confidence carries no information; "
          "above 0.5 = more confident when right. The letters_fixed row is the "
          "surface-cue control: length is constant there, so a healthy AUROC2 "
          "on that row cannot come from reading string length.")

    make_plots(all_rows, model, f"{out_prefix}.png")
    with open(f"{out_prefix}.json", "w") as f:
        json.dump({"model": model, "thinking_mode": THINKING_MODE,
                   "seed": RUN_SEED, "rows": all_rows}, f, indent=2)
    print(f"saved raw data -> {out_prefix}.json")

def main():
    global N_MAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=N_MAIN, help="trials per type")
    ap.add_argument("--types", default=None,
                    help="comma-separated problem types to run (default: all); "
                         "choices: " + ",".join(PROBLEM_TYPES))
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the problem generators for reproducibility")
    ap.add_argument("--level", type=int, default=None,
                    help="pin all non-letters_fixed types to this difficulty "
                         "level and skip calibration")
    args = ap.parse_args()
    N_MAIN = args.n
    if args.seed is not None:
        random.seed(args.seed)
    global RUN_SEED
    RUN_SEED = args.seed
    types = None
    if args.types:
        types = [t.strip() for t in args.types.split(",") if t.strip()]
        unknown = [t for t in types if t not in PROBLEM_TYPES]
        if unknown:
            sys.exit(f"ERROR: unknown problem type(s) {unknown}; "
                     f"choices: {list(PROBLEM_TYPES)}")
    if args.selftest:
        selftest(types, args.level)
        return
    client = make_client()
    def factory(gen):
        return lambda body, true: ask_model(client, args.model, body)
    safe = args.model.replace("/", "_")
    suffix = ("_" + "-".join(types)) if types else ""
    run_all(answer_factory=factory, model=args.model,
            out_prefix=f"hot2v2_{safe}{suffix}", types=types,
            fixed_level=args.level)

if __name__ == "__main__":
    main()
