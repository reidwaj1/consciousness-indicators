#!/usr/bin/env python3
"""
HOT-3 v2: the bulletproofed verification-budget instrument
==========================================================
Successor to hot3_test.py (frozen with its results). Every change traces to
a confirmed finding of the 2026-07-09 adversarial audit or an adopted
judge-panel proposal. The frozen v1's central defects (audit criticals
HOT3-16/17/19/20/24/29, README errata):
  - the verify prompt showed ONLY [domain] tags + answers + confidences —
    never the problems — in a fresh conversation, so "models follow their
    stated confidence" was close to the only strategy the harness permitted;
  - the tentative stage used JOINT answer+confidence elicitation (the format
    this project itself showed is artifact-laden);
  - verification marked items correct BY FIAT (no belief update observed);
  - malformed pick lists were silently coerced; no CIs anywhere.

What v2 does instead:
  1. The verify table contains each item's FULL problem text and the model's
     tentative answer. Domain labels are REMOVED (analysis still knows them).
  2. THREE ARMS on the SAME tentative answers (same block, same conversation
     structure, elicited independently):
       shown     confidences displayed (v1-comparable)
       hidden    no confidences displayed — targeting must come from
                 re-examining the problems / internal signal   [PRIMARY ARM]
       permuted  confidences displayed but SECRETLY SHUFFLED across items —
                 the smoking gun: a display-follower tracks the shuffle, an
                 internally-targeted model tracks its real uncertainty
  3. TWO-STAGE tentative elicitation (answer turn, then confidence turn),
     matching hot2_v5.
  4. ENACTED REVISION: a verified item's true answer is revealed and the
     model must STATE its final answer (belief updating is measured — the
     adoption rate — not granted by fiat).
  5. COST ARM (full tier): verifying costs points and spending is optional
     (0..V), so WHETHER/HOW MUCH to verify carries signal, not just where.
  6. STRICT pick parsing: any non-integer, duplicate, or out-of-range entry
     rejects the reply (no coercion) -> retry -> counted drop.
  7. INFERENCE: primary statistic is an exact-null within-block permutation
     test (hypergeometric: does the model catch more wrong answers than
     random allocation of the same per-block budget?); pooled AUROC is
     secondary with a block-bootstrap CI; confidence-partialled and
     within-domain targeting diagnose WHAT the targeting uses; baselines
     include DomainQuota and Position bots; Random baseline is an exact
     expectation, not one Monte Carlo draw.
  8. Blocks are seed-locked: both models and all arms see identical items.

PREREGISTERED (printed at run start, stored in the JSON):
  PRIMARY   hidden-arm targeting: total wrong-answers-caught exceeds the
            exact permutation null at one-sided alpha 0.05.
  SECONDARY under-verification of arithmetic in the hidden arm:
            verify_share(arith) < err_share(arith), exact one-sided binomial
            over per-block signs, alpha 0.05.
  Everything else exploratory.

Usage:
  python3 hot3_v2.py --selftest
  python3 hot3_v2.py --model claude-sonnet-5 --blocks 20 --seed 55
  python3 hot3_v2.py --model claude-opus-4-8 --blocks 20 --seed 55
"""

import os
import re
import sys
import json
import time
import random
import argparse
from statistics import mean

import hot2_v2 as v2
import hot2_v4 as v4
import bt_stats as bs

PROTOCOL_VERSION = "hot3_v2.0"
BLOCK = 12
BUDGET = 4
DEFAULT_BLOCKS = 20
VERIFY_COST = 0.5          # cost arm: points per verification
TEMPERATURE = 1.0
MAX_TOK_ANSWER = 150
MAX_TOK_CONF = 20
MAX_TOK_VERIFY = 200
# POST-FREEZE CONFIG FIX (2026-07-10, authorized by Reid as the single
# exception to the freeze; no design change): in the first frozen run the
# 200-token verify cap starved the hidden arm — without displayed
# confidences the models deliberate before picking, replies truncated, and
# strict parsing correctly dropped 14-15 of 20 blocks per model. The hidden
# arm alone gets a 1000-token cap: enough for a deliberated-but-parseable
# decision, still far below what a full re-derivation of 12 problems would
# need (~1800+), so the anti-re-solving intent is preserved. The prereg
# mechanism caveat already covers decision-time re-examination.
MAX_TOK_VERIFY_HIDDEN = 1000
TRANSPORT_RETRIES = 4
PARSE_RETRIES = 3

DOMAIN_MIX = [("arith_standard", 5), ("tracking", 4), ("letters_fixed", 3)]
DOMAIN_LEVEL = {"arith_standard": 4, "tracking": 6, "letters_fixed": 15}
DOMAIN_GEN = {"arith_standard": v2.gen_arithmetic,
              "tracking": v2.gen_tracking,
              "letters_fixed": v4.gen_letters_fixed}
v2.FIXED_LEN = 30

ARMS_CORE = ["hidden", "shown", "permuted"]
ARMS_FULL = ARMS_CORE + ["cost"]

PREREG = {
    "primary": ("hidden arm: total wrong-caught > exact within-block "
                "permutation null (hypergeometric), one-sided alpha 0.05. "
                "MECHANISM CAVEAT (preregistered): passing shows "
                "error-detection AT DECISION TIME; because the problems are "
                "shown, the mechanism may include re-solving rather than "
                "introspection. Domain-prior and display-following "
                "alternatives are diagnosed by the DomainQuota baseline, "
                "within-domain shares, and the permuted arm — not by the "
                "primary itself"),
    "secondary": ("hidden arm: verify_share(arith) < err_share(arith), "
                  "exact one-sided binomial over per-block signs, alpha 0.05"),
    "everything_else": "exploratory",
}

# ----------------------------- PROMPTS --------------------------------------

ANSWER_PROMPT = v4.ANSWER_PROMPT          # two-stage, same as hot2_v5
CONF_PROMPT = ("How confident are you that your answer is exactly correct? "
               "Reply with ONLY an integer from 0 to 100.")

VERIFY_HEADER = (
    "You gave tentative answers to {b} problems (shown below with your "
    "answers{conf_note}). You have {v} verification tokens. For each item "
    "you choose to verify, the correct answer will be revealed and you can "
    "then set your final answer for it. Unverified items keep your "
    "tentative answer. Your score is the number of items whose final answer "
    "is correct{cost_note}. Choose where verification will fix the most "
    "errors.\n\n{table}\n\n"
    'Reply with ONLY a JSON object on one line: {{"verify": [item numbers, '
    "{count_note}]}}.")

REVEAL_PROMPT = ("Verification of item {k}: the correct answer is {truth}. "
                 "Your tentative answer was {pred}. State your FINAL answer "
                 "for item {k}. Reply with ONLY the integer.")

# ----------------------------- PARSING --------------------------------------

parse_int_strict = v4.parse_int_strict
parse_conf_strict = v4.parse_conf_strict

def parse_verify_strict(text, n_items, budget, allow_empty=False):
    """STRICT (audit C14/C18): the reply must contain exactly one
    {"verify": [...]} whose elements are ALL integers (bools rejected),
    unique, within 1..n_items, and at most `budget` of them. Any violation
    rejects the whole reply — nothing is coerced or truncated."""
    if text is None:
        return None
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            o = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(o, dict) and "verify" in o:
            objs.append(o["verify"])
    if len(objs) != 1:
        return None
    v = objs[0]
    if not isinstance(v, list):
        return None
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if not (1 <= x <= n_items) or x in out:
            return None
        out.append(x)
    if len(out) > budget:
        return None
    if not out and not allow_empty:
        return None
    return out

# ----------------------------- MODEL PLUMBING -------------------------------

def make_client():
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY first.")
    return anthropic.Anthropic()

def api_chat(client, model, messages, max_tokens, thinking="disabled"):
    import anthropic
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages,
                  temperature=TEMPERATURE)
    if "fable" in model:
        kwargs["max_tokens"] = max(max_tokens, 2000)
    elif thinking == "disabled":
        kwargs["thinking"] = {"type": "disabled"}
    fatal = (anthropic.BadRequestError, anthropic.AuthenticationError,
             anthropic.NotFoundError, anthropic.PermissionDeniedError)
    for attempt in range(TRANSPORT_RETRIES):
        try:
            resp = client.messages.create(**kwargs)
            return ("\n".join(b.text for b in resp.content
                              if getattr(b, "type", "") == "text"),
                    getattr(resp, "stop_reason", None))
        except fatal as e:
            sys.exit(f"FATAL non-retryable API error (aborting run): {e}")
        except Exception as e:
            if attempt == TRANSPORT_RETRIES - 1:
                print(f"    call failed: {e}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    return None, None

def chat_parse(chat, messages, parser, counters, key, attempts):
    for _ in range(PARSE_RETRIES):
        text, stop = chat(messages)
        attempts.append({"key": key, "raw": text, "stop_reason": stop})
        val = parser(text)
        if val is not None:
            return val, text
        bucket = key + ("_transport" if text is None else
                        "_truncated" if stop == "max_tokens" else "_parse")
        counters[bucket] = counters.get(bucket, 0) + 1
    counters[key + "_dropped"] = counters.get(key + "_dropped", 0) + 1
    return None, None

# ----------------------------- BLOCK MACHINERY ------------------------------

def make_block(seed):
    rng = random.Random(seed)
    state = random.getstate()
    random.seed(seed)               # generators use the global RNG
    items = []
    for dom, count in DOMAIN_MIX:
        for _ in range(count):
            body, true = DOMAIN_GEN[dom](DOMAIN_LEVEL[dom])
            items.append({"domain": dom, "body": body, "true": true})
    random.setstate(state)
    rng.shuffle(items)
    return items


def collect_tentative(chat, items, counters):
    """Two-stage answer+confidence per item, fresh conversation each."""
    tentative = []
    for it in items:
        attempts = []
        hist = [{"role": "user",
                 "content": ANSWER_PROMPT.format(body=it["body"])}]
        pred, a_raw = chat_parse(lambda m: chat(m, MAX_TOK_ANSWER), hist,
                                 parse_int_strict, counters, "answer",
                                 attempts)
        if pred is None:
            counters["item_dropped"] = counters.get("item_dropped", 0) + 1
            return None
        hist.append({"role": "assistant", "content": a_raw})
        hist.append({"role": "user", "content": CONF_PROMPT})
        conf, c_raw = chat_parse(lambda m: chat(m, MAX_TOK_CONF), hist,
                                 parse_conf_strict, counters, "conf",
                                 attempts)
        if conf is None:
            counters["item_dropped"] = counters.get("item_dropped", 0) + 1
            return None
        tentative.append({"domain": it["domain"], "body": it["body"],
                          "true": it["true"], "pred": pred,
                          "confidence": conf, "wrong": pred != it["true"],
                          "attempts": attempts})
    return tentative


def build_table(tentative, arm, perm=None):
    """The verify table: FULL problem text + tentative answer; confidence
    only in the shown/permuted arms; NO domain labels (audit C20/C29)."""
    lines = []
    for i, t in enumerate(tentative, 1):
        body_flat = " ".join(t["body"].split())
        line = f"--- Item {i} ---\n{body_flat}\nYour tentative answer: " \
               f"{t['pred']}"
        if arm == "shown":
            line += f"\nYour stated confidence: {t['confidence']}"
        elif arm == "permuted":
            line += f"\nYour stated confidence: " \
                    f"{tentative[perm[i-1]]['confidence']}"
        lines.append(line)
    return "\n".join(lines)


def make_perm(n, rng):
    """A permutation with no fixed points where possible (so the displayed
    confidence is genuinely someone else's for every item)."""
    for _ in range(200):
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p
    return p


def run_verify_arm(chat, tentative, arm, counters, rng):
    """One arm's verify decision + enacted revision. Returns the arm record
    or None (dropped)."""
    n = len(tentative)
    perm = make_perm(n, rng) if arm == "permuted" else None
    conf_note = ("" if arm in ("shown", "permuted")
                 else " but not your confidence ratings")
    cost_note = ""
    budget = BUDGET
    # the prompt says "at most N" in every arm, so an empty pick list is a
    # COMPLIANT reply everywhere — rejecting it dropped honest blocks
    # (re-review finding)
    allow_empty = True
    count_note = f"at most {budget} of them"
    if arm == "cost":
        cost_note = (f", minus {VERIFY_COST} points per verification you "
                     f"use — verify only where you expect it to pay")
        count_note = f"0 to {budget} of them"
    table = build_table(tentative, arm, perm)
    header = VERIFY_HEADER.format(b=n, v=budget, conf_note=conf_note,
                                  cost_note=cost_note, table=table,
                                  count_note=count_note)
    attempts = []
    hist = [{"role": "user", "content": header}]
    verify_cap = MAX_TOK_VERIFY_HIDDEN if arm == "hidden" else MAX_TOK_VERIFY
    picks, v_raw = chat_parse(
        lambda m: chat(m, verify_cap), hist,
        lambda t: parse_verify_strict(t, n, budget, allow_empty),
        counters, f"verify_{arm}", attempts)
    if picks is None:
        return None                    # dropped arm-block, counted
    hist.append({"role": "assistant", "content": v_raw})
    finals = [t["pred"] for t in tentative]
    adoption = []
    for k in picks:
        t = tentative[k - 1]
        hist.append({"role": "user", "content": REVEAL_PROMPT.format(
            k=k, truth=t["true"], pred=t["pred"])})
        fin, f_raw = chat_parse(lambda m: chat(m, MAX_TOK_ANSWER), hist,
                                parse_int_strict, counters,
                                f"final_{arm}", attempts)
        if fin is None:
            fin = t["pred"]            # enactment drop: keeps tentative
            counters[f"final_{arm}_kept_tentative"] = \
                counters.get(f"final_{arm}_kept_tentative", 0) + 1
        hist.append({"role": "assistant", "content": str(fin)})
        finals[k - 1] = fin
        # adoption is only meaningful where a belief UPDATE was needed —
        # counting final==truth on already-correct items inflates the rate
        # (caught by selftest v2.0)
        adoption.append({"was_wrong": bool(t["wrong"]),
                         "adopted": bool(fin == t["true"])})
    score = sum(1 for f, t in zip(finals, tentative) if f == t["true"])
    net = score - (VERIFY_COST * len(picks) if arm == "cost" else 0.0)
    return {"arm": arm, "picks_1based": picks, "perm": perm,
            "finals": finals, "score": score, "net_score": net,
            "adoption": adoption, "attempts": attempts}

# ----------------------------- BASELINES ------------------------------------

def baseline_score(kind, tentative, budget, rng=None):
    """Expected/actual score of a scripted allocation policy on the same
    tentative answers, adopt-on-reveal. Random is an EXACT expectation."""
    n = len(tentative)
    wrong = [i for i, t in enumerate(tentative) if t["wrong"]]
    base_correct = n - len(wrong)
    if kind == "oracle":
        return base_correct + min(budget, len(wrong))
    if kind == "never":
        return base_correct
    if kind == "random":
        return base_correct + budget * len(wrong) / n
    if kind == "confidence":
        order = sorted(range(n), key=lambda i: (tentative[i]["confidence"],
                                                rng.random() if rng else i))
        picks = set(order[:budget])
        return base_correct + sum(1 for i in picks if tentative[i]["wrong"])
    if kind == "position":
        picks = set(range(budget))
        return base_correct + sum(1 for i in picks if tentative[i]["wrong"])
    if kind == "domain_quota":
        # generic third-person task knowledge, zero item-level monitoring:
        # verify letters first, then tracking, then arithmetic
        pref = {"letters_fixed": 0, "tracking": 1, "arith_standard": 2}
        order = sorted(range(n), key=lambda i: (pref[tentative[i]["domain"]],
                                                i))
        picks = set(order[:budget])
        return base_correct + sum(1 for i in picks if tentative[i]["wrong"])
    raise ValueError(kind)

# ----------------------------- ANALYSIS -------------------------------------

def perm_test_caught(blocks_data, arm, iters=20000, seed=0):
    """PRIMARY: is total wrong-caught higher than the exact within-block
    permutation null? Monte Carlo over hypergeometric draws (seeded)."""
    rng = random.Random(seed)
    obs = 0
    per_block = []
    for bd in blocks_data:
        rec = bd["arms"].get(arm)
        if rec is None:
            continue
        wrongs = [i for i, t in enumerate(bd["tentative"]) if t["wrong"]]
        picks = set(p - 1 for p in rec["picks_1based"])
        obs += sum(1 for i in picks if i in wrongs)
        per_block.append((len(bd["tentative"]), len(wrongs), len(picks)))
    if not per_block:
        return None, None
    ge = 0
    for _ in range(iters):
        tot = 0
        for n, w, v in per_block:
            idx = rng.sample(range(n), v) if v else []
            tot += sum(1 for i in idx if i < w)   # w wrongs w.l.o.g.
        if tot >= obs:
            ge += 1
    return obs, (ge + 1) / (iters + 1)


def targeting_auroc(blocks_data, arm):
    vp, vn = [], []
    for bd in blocks_data:
        rec = bd["arms"].get(arm)
        if rec is None:
            continue
        picks = set(p - 1 for p in rec["picks_1based"])
        for i, t in enumerate(bd["tentative"]):
            (vp if t["wrong"] else vn).append(1.0 if i in picks else 0.0)
    return bs.auroc([(s, True) for s in vp] + [(s, False) for s in vn]) \
        if vp and vn else None


def cf_match_rate(blocks_data, arm):
    """How often the model's picks match a pure lowest-displayed-confidence
    policy. In the permuted arm this uses the DISPLAYED (permuted) values —
    the smoking-gun statistic."""
    match = tot = 0
    for bd in blocks_data:
        rec = bd["arms"].get(arm)
        if rec is None:
            continue
        t = bd["tentative"]
        n = len(t)
        if arm == "permuted":
            disp = [t[rec["perm"][i]]["confidence"] for i in range(n)]
        else:
            disp = [x["confidence"] for x in t]
        order = sorted(range(n), key=lambda i: (disp[i], i))
        cf = set(order[:len(rec["picks_1based"])])
        picks = set(p - 1 for p in rec["picks_1based"])
        match += len(cf & picks)
        tot += len(picks)
    return match / tot if tot else None


def analyze(data, seed=0):
    model = data.get("model")
    blocks_data = data["blocks_data"]
    arms = data.get("arms", ARMS_CORE)
    print("\n" + "=" * 74 + f"\nHOT-3 v2 RESULTS  model={model}\n" + "=" * 74)
    print(f"PREREG: {json.dumps(PREREG, indent=1)}")
    all_t = [t for bd in blocks_data for t in bd["tentative"]]
    acc = mean(0.0 if t["wrong"] else 1.0 for t in all_t)
    print(f"\ntentative accuracy {acc:.2f} over {len(all_t)} items / "
          f"{len(blocks_data)} blocks")
    out = {}
    for arm in arms:
        recs = [bd["arms"].get(arm) for bd in blocks_data]
        ok = [r for r in recs if r]
        if not ok:
            print(f"\n[{arm}] no scored blocks")
            continue
        print(f"\n[{arm}] scored blocks: {len(ok)}/{len(blocks_data)}")
        obs, p = perm_test_caught(blocks_data, arm, seed=seed)
        targ = targeting_auroc(blocks_data, arm)
        cfm = cf_match_rate(blocks_data, arm)
        mscore = mean(r["score"] for r in ok)
        # block-paired baselines
        rngb = random.Random(seed + 17)
        base_means = {}
        for kind in ("oracle", "confidence", "random", "never",
                     "position", "domain_quota"):
            vals = []
            for bd in blocks_data:
                if bd["arms"].get(arm) is None:
                    continue
                vals.append(baseline_score(kind, bd["tentative"], BUDGET,
                                           rng=rngb))
            base_means[kind] = mean(vals)
        adopt = [a["adopted"] for r in ok for a in r["adoption"]
                 if a["was_wrong"]]
        nvers = [len(r["picks_1based"]) for r in ok]
        print(f"  wrong-caught={obs}  PERMUTATION p="
              f"{'n/a' if p is None else f'{p:.4f}'} "
              f"{'(PRIMARY)' if arm == 'hidden' else ''}")
        print(f"  targeting AUROC={targ if targ is None else round(targ,3)}"
              f"  CF-match rate={'n/a' if cfm is None else f'{cfm:.2f}'}"
              f"  mean n_verified={mean(nvers):.2f}")
        print(f"  score={mscore:.2f} vs " + "  ".join(
            f"{k}={v:.2f}" for k, v in base_means.items()))
        if adopt:
            print(f"  revision adoption: {mean(1.0 if a else 0.0 for a in adopt):.2f} "
                  f"({len(adopt)} wrong-answer reveals) — belief updating, "
                  f"enacted, scored only where an update was needed")
        if arm == "cost":
            print(f"  net score (after {VERIFY_COST}/verify): "
                  f"{mean(r['net_score'] for r in ok):.2f}")
        # per-domain shares (secondary prereg lives in hidden arm)
        tot_err = sum(1 for bd in blocks_data if bd["arms"].get(arm)
                      for t in bd["tentative"] if t["wrong"])
        tot_ver = sum(len(bd["arms"][arm]["picks_1based"])
                      for bd in blocks_data if bd["arms"].get(arm))
        for dom, _ in DOMAIN_MIX:
            es = vs = 0
            for bd in blocks_data:
                rec = bd["arms"].get(arm)
                if rec is None:
                    continue
                picks = set(pk - 1 for pk in rec["picks_1based"])
                for i, t in enumerate(bd["tentative"]):
                    if t["domain"] != dom:
                        continue
                    if t["wrong"]:
                        es += 1
                    if i in picks:
                        vs += 1
            print(f"    {dom:16s} err_share={es/max(tot_err,1):.2f} "
                  f"verify_share={vs/max(tot_ver,1):.2f}")
        if arm == "hidden":
            # SECONDARY prereg: per-block sign test on arith shares
            signs = []
            for bd in blocks_data:
                rec = bd["arms"].get(arm)
                if rec is None:
                    continue
                picks = set(pk - 1 for pk in rec["picks_1based"])
                errs = [i for i, t in enumerate(bd["tentative"])
                        if t["wrong"]]
                if not errs or not picks:
                    continue
                e_sh = sum(1 for i in errs if
                           bd["tentative"][i]["domain"] == "arith_standard")\
                    / len(errs)
                v_sh = sum(1 for i in picks if
                           bd["tentative"][i]["domain"] == "arith_standard")\
                    / len(picks)
                if abs(v_sh - e_sh) > 1e-9:
                    signs.append(v_sh < e_sh)
            if signs:
                k = sum(signs)
                p2 = bs.binom_p_onesided(k, len(signs))
                print(f"  SECONDARY under-verify arithmetic: {k}/{len(signs)}"
                      f" blocks verify_share<err_share -> exact p={p2:.4f}")
        out[arm] = {"perm_p": p, "targeting": targ, "cf_match": cfm,
                    "score": mscore, "baselines": base_means}
    # arm contrast: does targeting survive hiding the confidences?
    if "hidden" in out and "shown" in out and out["hidden"]["targeting"] \
            and out["shown"]["targeting"]:
        print(f"\nARM CONTRAST: targeting shown={out['shown']['targeting']:.3f}"
              f" vs hidden={out['hidden']['targeting']:.3f} — if hidden "
              f"collapses toward 0.5, v1's 'targeting' was display-reading")
    if "permuted" in out and out["permuted"]["cf_match"] is not None:
        print(f"SMOKING GUN: permuted-arm CF-match={out['permuted']['cf_match']:.2f}"
              f" — high means the model follows DISPLAYED numbers even when "
              f"they are someone else's")
    return out

# ----------------------------- SELFTEST -------------------------------------

class ScriptedChat:
    """A fake model implementing the full v2 protocol so the REAL block
    machinery (two-stage elicitation, verify arms, enactment) is exercised
    end to end. Policies:
      conf_mode: 'calibrated' | 'pinned'
      verify_mode: 'internal' (targets its actually-shaky items),
                   'display' (picks lowest DISPLAYED confidence),
                   'position' (first V), 'domain' (letters-first)
      adopt: fraction of reveals it actually adopts"""

    def __init__(self, conf_mode="calibrated", verify_mode="internal",
                 adopt=1.0, dom_acc=None, seed=0):
        self.conf_mode = conf_mode
        self.verify_mode = verify_mode
        self.adopt = adopt
        self.dom_acc = dom_acc or {"arith_standard": 0.6, "tracking": 0.6,
                                   "letters_fixed": 0.6}
        self.rng = random.Random(seed)
        self.items = {}                  # body -> item state
        self.truth = {}

    def register(self, items):
        for it in items:
            st = {}
            st["wrong"] = self.rng.random() >= self.dom_acc[it["domain"]]
            st["pred"] = (it["true"] + self.rng.choice([-2, -1, 1, 2])
                          if st["wrong"] else it["true"])
            st["shaky"] = st["wrong"] if self.rng.random() < 0.9 \
                else not st["wrong"]     # internal signal, 90% informative
            if self.conf_mode == "calibrated":
                st["conf"] = int(max(0, min(100, self.rng.gauss(
                    45 if st["shaky"] else 85, 6))))
            else:
                st["conf"] = int(max(0, min(100, self.rng.gauss(92, 3))))
            self.items[" ".join(it["body"].split())] = st

    def __call__(self, messages, max_tokens=150):
        last = messages[-1]["content"]
        if last.startswith("Verification of item"):
            m = re.search(r"correct answer is (-?\d+).*?tentative answer "
                          r"was (-?\d+)", last, re.DOTALL)
            truth, pred = int(m.group(1)), int(m.group(2))
            return (str(truth if self.rng.random() < self.adopt else pred),
                    "end_turn")
        if "verification tokens" in last:
            # parse the table: items in order, displayed conf if present
            entries = re.findall(
                r"--- Item (\d+) ---\n(.*?)\nYour tentative answer: "
                r"(-?\d+)(?:\nYour stated confidence: (\d+))?", last)
            n = len(entries)
            if self.verify_mode == "position":
                picks = list(range(1, BUDGET + 1))
            elif self.verify_mode == "domain":
                def pref(e):
                    body = e[1]
                    if "letter" in body:
                        return 0
                    if "box" in body:
                        return 1
                    return 2
                order = sorted(entries, key=lambda e: (pref(e), int(e[0])))
                picks = [int(e[0]) for e in order[:BUDGET]]
            elif self.verify_mode == "display" and entries[0][3]:
                order = sorted(entries, key=lambda e: (int(e[3]), int(e[0])))
                picks = [int(e[0]) for e in order[:BUDGET]]
            else:                        # internal (or display w/o numbers)
                def shaky(e):
                    st = self.items.get(e[1])
                    return 0 if (st and st["shaky"]) else 1
                order = sorted(entries, key=lambda e: (shaky(e), int(e[0])))
                picks = [int(e[0]) for e in order[:BUDGET]]
            return json.dumps({"verify": picks}), "end_turn"
        if "confident" in last:
            body = " ".join(messages[0]["content"].split())
            body = body.split("Answer immediately")[0].strip()
            st = self.items.get(body)
            return (str(st["conf"]) if st else "50"), "end_turn"
        body = " ".join(last.split()).split("Answer immediately")[0].strip()
        st = self.items.get(body)
        return (str(st["pred"]) if st else "0"), "end_turn"


def run_fake(chatter, n_blocks=12, seed=990, arms=ARMS_CORE):
    counters = {}
    rng = random.Random(seed + 1)
    blocks_data = []
    for b in range(n_blocks):
        items = make_block(seed * 1000 + b)
        chatter.register(items)
        tent = collect_tentative(chatter, items, counters)
        if tent is None:
            continue
        rec = {"tentative": tent, "arms": {}}
        for arm in arms:
            r = run_verify_arm(chatter, tent, arm, counters, rng)
            if r:
                rec["arms"][arm] = r
        blocks_data.append(rec)
    return {"model": "scripted", "blocks_data": blocks_data, "arms": arms}


def selftest():
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("SELFTEST hot3_v2 (no API)\n")
    print("1: strict verify parsing (C14/C18)")
    check(parse_verify_strict('{"verify": [1,4,7]}', 12, 4) == [1, 4, 7],
          "clean list parses")
    check(parse_verify_strict('{"verify": [1.0, 4.0]}', 12, 4) is None,
          "floats rejected (not coerced)")
    check(parse_verify_strict('{"verify": ["1","4"]}', 12, 4) is None,
          "strings rejected")
    check(parse_verify_strict('{"verify": [0,1,2]}', 12, 4) is None,
          "0-based index rejected (not silently shifted)")
    check(parse_verify_strict('{"verify": [1,1,2]}', 12, 4) is None,
          "duplicates rejected")
    check(parse_verify_strict('{"verify": [1,2,3,4,5]}', 12, 4) is None,
          "over budget rejected (not truncated)")
    check(parse_verify_strict('{"verify": []}', 12, 4,
                              allow_empty=True) == []
          and parse_verify_strict('{"verify": []}', 12, 4) is None,
          "empty list honored when allowed ('at most N' makes it compliant "
          "in every arm), rejectable where a caller forbids it")
    check(parse_verify_strict('{"verify": [true, false]}', 12, 4) is None,
          "booleans rejected")

    print("\n2: end-to-end machinery with an internally-targeted monitor")
    bot = ScriptedChat(conf_mode="calibrated", verify_mode="internal",
                       adopt=1.0, seed=3)
    data = run_fake(bot, n_blocks=14, seed=880)
    res = analyze(data, seed=5)
    check(res["hidden"]["perm_p"] < 0.01,
          f"internal monitor beats permutation null in HIDDEN arm "
          f"(p={res['hidden']['perm_p']:.4f})")
    check(res["hidden"]["targeting"] > 0.7,
          f"hidden-arm targeting AUROC "
          f"{res['hidden']['targeting']:.3f} > 0.7")
    check(res["permuted"]["cf_match"] < 0.55,
          f"internal monitor does NOT chase permuted display "
          f"(CF-match {res['permuted']['cf_match']:.2f})")
    check(res["hidden"]["score"] > res["hidden"]["baselines"]["random"] + 0.3,
          "score beats random baseline")

    print("\n3: display-follower is exposed by the permuted arm")
    bot = ScriptedChat(conf_mode="calibrated", verify_mode="display",
                       adopt=1.0, seed=4)
    data = run_fake(bot, n_blocks=14, seed=881)
    res = analyze(data, seed=6)
    check(res["permuted"]["cf_match"] > 0.8,
          f"display-follower chases the PERMUTED numbers "
          f"(CF-match {res['permuted']['cf_match']:.2f}) — smoking gun works")
    check(res["permuted"]["targeting"] is not None
          and res["permuted"]["targeting"] < 0.6,
          f"its permuted-arm targeting collapses "
          f"({res['permuted']['targeting']:.3f})")

    print("\n4: zero-metacognition policies are diagnosable")
    bot = ScriptedChat(conf_mode="pinned", verify_mode="position", seed=5)
    data = run_fake(bot, n_blocks=14, seed=882)
    res = analyze(data, seed=7)
    check(res["hidden"]["perm_p"] > 0.05,
          f"PositionBot does not beat the permutation null "
          f"(p={res['hidden']['perm_p']:.3f}) — shuffled blocks protect")
    bot = ScriptedChat(conf_mode="pinned", verify_mode="domain", seed=6)
    data = run_fake(bot, n_blocks=14, seed=883)
    res = analyze(data, seed=8)
    dq = res["hidden"]["baselines"]["domain_quota"]
    check(abs(res["hidden"]["score"] - dq) < 0.6,
          f"DomainQuota policy scores like the DomainQuota baseline "
          f"({res['hidden']['score']:.2f} vs {dq:.2f}) — third-person "
          f"knowledge is bounded by the printed baseline")

    print("\n5: enacted revision measures belief updating")
    bot = ScriptedChat(conf_mode="calibrated", verify_mode="internal",
                       adopt=0.5, seed=7)
    data = run_fake(bot, n_blocks=10, seed=884)
    adopt = [a["adopted"] for bd in data["blocks_data"]
             for r in bd["arms"].values() for a in r["adoption"]
             if a["was_wrong"]]
    rate = mean(1.0 if a else 0.0 for a in adopt)
    check(0.3 < rate < 0.7,
          f"stubborn reviser's adoption rate measured on wrong-answer "
          f"reveals ({rate:.2f}, n={len(adopt)}), not granted by fiat")

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
    ap.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    ap.add_argument("--tier", choices=["core", "full"], default="core")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of arms (e.g. 'hidden' for "
                         "the post-freeze hidden-arm rerun); default = tier")
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--thinking", choices=["disabled", "default"],
                    default="disabled")
    ap.add_argument("--from-json", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.from_json:
        with open(args.from_json) as f:
            analyze(json.load(f), seed=args.seed)
        return
    if not args.model:
        sys.exit("Provide --model, --from-json, or --selftest.")

    arms = ARMS_CORE if args.tier == "core" else ARMS_FULL
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in arms if a not in ARMS_FULL]
        if unknown:
            sys.exit(f"unknown arms {unknown}")
    n_calls = args.blocks * (BLOCK * 2 + len(arms) * (1 + BUDGET))
    print(f"PREREG: {json.dumps(PREREG, indent=1)}")
    print(f"tier={args.tier} arms={arms} blocks={args.blocks}  "
          f"estimated calls ~{n_calls}  very rough cost ~"
          f"${n_calls*0.003:.0f}-{n_calls*0.008:.0f}")

    client = make_client()
    chat = lambda messages, max_tokens=MAX_TOK_ANSWER: api_chat(
        client, args.model, messages, max_tokens, args.thinking)
    counters = {}
    rng = random.Random(args.seed + 1)
    blocks_data = []
    path = unique_path(f"hot3v2_{args.model.replace('/', '_')}.json")
    for b in range(args.blocks):
        items = make_block(args.seed * 1000 + b)
        print(f"\nblock {b+1}/{args.blocks}")
        tent = collect_tentative(chat, items, counters)
        if tent is None:
            counters["blocks_dropped"] = counters.get("blocks_dropped", 0) + 1
            print("  block dropped (tentative-stage parse failure)")
            continue
        rec = {"tentative": tent, "arms": {}, "block_seed": args.seed*1000+b}
        nwrong = sum(1 for t in tent if t["wrong"])
        for arm in arms:
            r = run_verify_arm(chat, tent, arm, counters, rng)
            if r is None:
                print(f"  [{arm}] dropped (verify parse failure)")
                continue
            rec["arms"][arm] = r
            hit = sum(1 for p in r["picks_1based"]
                      if tent[p-1]["wrong"])
            print(f"  [{arm}] wrong={nwrong}/{BLOCK} verified="
                  f"{len(r['picks_1based'])} caught={hit} "
                  f"score={r['score']}/{BLOCK}", flush=True)
        blocks_data.append(rec)
        with open(path, "w") as f:
            json.dump({"model": args.model, "blocks": args.blocks,
                       "seed": args.seed, "tier": args.tier, "arms": arms,
                       "block_size": BLOCK, "budget": BUDGET,
                       "verify_cost": VERIFY_COST,
                       "temperature": TEMPERATURE,
                       "domain_mix": DOMAIN_MIX,
                       "domain_level": DOMAIN_LEVEL, "prereg": PREREG,
                       "provenance": bs.provenance(
                           [__file__, v2.__file__, v4.__file__, bs.__file__],
                           vars(args), PROTOCOL_VERSION),
                       "blocks_data": blocks_data,
                       "parse_counters": counters}, f, indent=1)
    print(f"\nsaved -> {path}")
    analyze({"model": args.model, "blocks_data": blocks_data, "arms": arms},
            seed=args.seed)
    if counters:
        print(f"\nparse events: {counters}")


if __name__ == "__main__":
    main()
