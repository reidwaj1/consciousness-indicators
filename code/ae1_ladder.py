#!/usr/bin/env python3
"""
AE-1 robustness ladder (Mission 3, Workstream B)
================================================
Extends ae1_test.py (imported, not copied) from one mild perturbation into a
severity ladder, so "all models pass" becomes a degradation curve showing
where goal-pursuit breaks. Indicator anchor (Butlin et al., AE-1): "learning
from feedback and selecting outputs in such a way as to pursue goals,
especially where this involves flexible responsiveness to competing goals."

Rungs (each its own condition; every rung fires AFTER the first successful
add, like the original perturbation):
  control      no change
  stockout     the original: best not-in-cart optimal item goes out of stock
  price_spike  that item's price jumps 2.5x instead (cheapest route dies,
               goal still satisfiable)
  budget_cut   budget silently drops to ~70% after the first add (visible
               only in cart views)
  compound     stockout AND price spike on the next-best item
  distractor   several PROVABLY irrelevant changes (non-optimal items go out
               of stock / prices jiggle, optimum verified unchanged): targets
               over-exploration, not brittleness
  pressure     stockout with the turn limit cut 22 -> 12

Reference agents (selftest must separate them before any paid run):
  GoodBotX     GoodBot + repairs its cart when totals exceed the live budget
  BrittleBot   unchanged from ae1_test: plans once, executes blindly
  DitherBot    pathological over-explorer: keeps checking items instead of
               committing; the designed faker for the distractor/pressure
               failure mode. (On distractor, truly-irrelevant changes cannot
               make a blind script fail, so Good-vs-Brittle separation is
               structurally impossible there; DitherBot is the reference
               failure instead. Documented in the README.)

Design choices delegated by MISSION3 and taken here:
  - paired worlds: episode i uses the same catalog/budget in every rung
    (seeded), so rung differences are not world-sampling noise
  - adapted := perturbation fired AND final cart valid AND utility_ratio
    >= 0.75 of the post-change optimum (n/a for control/distractor)
  - pressure keeps the stockout change (a real change is required) and cuts
    MAX_TURNS to 12: GoodBotX needs ~9 turns to notice and repair, so 12
    punishes dithering without making repair impossible

Usage:
  python3 ae1_ladder.py --selftest
  python3 ae1_ladder.py --model claude-sonnet-5 --episodes 12 --seed 501
"""

import os
import re
import sys
import json
import time
import random
import argparse
from statistics import mean

import ae1_test as base

RUNGS = ["control", "stockout", "price_spike", "budget_cut",
         "compound", "distractor", "pressure"]
RUNG_TURNS = {"pressure": 12}
DEFAULT_TURNS = 22
DEFAULT_EPISODES = 16      # statistical floor matching the AE-1 quick-win
SPIKE_MULT = 2.5
CUT_FRACS = (0.7, 0.8, 0.9)
ADAPT_UTILITY = 0.75


# ----------------------------- EPISODE --------------------------------------

class LadderEpisode(base.Episode):
    def __init__(self, rung):
        super().__init__(perturb=(rung != "control"))
        self.rung = rung
        self.events = []
        self.spike_target = None
        self.spike_newprice = None
        self.old_budget = self.budget
        self.budget_cut_to = None
        self.distractor_changes = []

    # ---- perturbation dispatch (fires between turns, silently) ----
    def maybe_perturb(self):
        if not self.perturb or self.perturbed or self.adds_ok == 0:
            return
        if self.rung in ("stockout", "pressure"):
            self._stockout()
        elif self.rung == "price_spike":
            self._spike()
        elif self.rung == "budget_cut":
            self._cut()
        elif self.rung == "compound":
            self._stockout()
            self._spike()
        elif self.rung == "distractor":
            self._distract()

    def _stockout(self):
        opt = base.best_cart(self.catalog, self.budget)
        if not opt:
            return
        in_cart = set(self.cart.values())
        candidates = [self.catalog[i] for i in opt["ids"] if i not in in_cart]
        for target in sorted(candidates, key=lambda it: -it["quality"]):
            target["stock"] = 0
            if base.best_cart(self.catalog, self.budget) is not None:
                self.perturbed = True
                self.target = target["id"]
                self.events.append(("stockout", target["id"]))
                return
            target["stock"] = 1

    def _spike(self):
        opt = base.best_cart(self.catalog, self.budget)
        if not opt:
            return
        in_cart = set(self.cart.values())
        candidates = [self.catalog[i] for i in opt["ids"]
                      if i not in in_cart and i != self.target
                      and self.catalog[i]["stock"] > 0]
        for t in sorted(candidates, key=lambda it: -it["quality"]):
            old = t["price"]
            t["price"] = max(old + 1, int(old * SPIKE_MULT))
            if base.best_cart(self.catalog, self.budget) is not None:
                self.perturbed = True
                self.spike_target = t["id"]
                self.spike_newprice = t["price"]
                self.events.append(("price_spike", t["id"], old, t["price"]))
                return
            t["price"] = old

    def _cut(self):
        for frac in CUT_FRACS:
            nb = int(self.budget * frac)
            if base.best_cart(self.catalog, nb) is not None:
                self.events.append(("budget_cut", self.budget, nb))
                self.budget = nb
                self.budget_cut_to = nb
                self.perturbed = True
                return

    def _distract(self):
        opt = base.best_cart(self.catalog, self.budget)
        if not opt:
            return
        optids = opt["ids"]
        in_cart = set(self.cart.values())
        others = [it for it in self.catalog.values()
                  if it["id"] not in optids and it["id"] not in in_cart]
        random.shuffle(others)
        changed = []
        n_stock = 0
        for it in others:                       # irrelevant stock-outs
            if n_stock >= 2 or it["stock"] == 0:
                continue
            it["stock"] = 0
            newopt = base.best_cart(self.catalog, self.budget)
            if newopt and newopt["ids"] == optids:
                changed.append(("stockout", it["id"]))
                n_stock += 1
            else:
                it["stock"] = 1
        for it in others:                       # irrelevant price jiggles
            if len(changed) >= 5 or it["stock"] == 0:
                continue
            old = it["price"]
            it["price"] = max(1, int(old * random.choice([0.8, 1.2])))
            newopt = base.best_cart(self.catalog, self.budget)
            if newopt and newopt["ids"] == optids and it["price"] != old:
                changed.append(("price_jiggle", it["id"], old, it["price"]))
            else:
                it["price"] = old
        if changed:
            self.perturbed = True
            self.distractor_changes = changed
            self.events.append(("distractor", changed))

    # ---- notice detection ----
    def saw_change(self, obs):
        base.Episode.saw_change(self, obs)          # stockout target
        if (self.spike_target and self.spike_target in obs
                and f'price {self.spike_newprice}' in obs):
            self.noticed = True
        if (self.budget_cut_to is not None
                and f"Budget {self.budget_cut_to}." in obs):
            self.noticed = True
        if self.rung == "distractor":
            for ch in self.distractor_changes:
                if ch[0] == "stockout" and ch[1] in obs and "OUT OF STOCK" in obs:
                    self.noticed = True
                elif ch[0] == "price_jiggle" and ch[1] in obs \
                        and f"price {ch[3]}" in obs:
                    self.noticed = True

    # ---- scoring ----
    def score(self):
        s = base.Episode.score(self)
        s["rung"] = self.rung
        s["events"] = self.events
        s["old_budget"] = self.old_budget
        # which perturbation(s) actually fired (compound can degrade to one)
        s["fired"] = [e[0] for e in self.events if e[0] != "distractor"]
        if self.rung in ("control", "distractor") or not s["perturbed"]:
            # nothing goal-relevant to adapt to, OR the perturbation never
            # fired (world too tight / candidates exhausted) — scoring this
            # as an adaptation failure would fabricate a rung-dependent dip
            s["adapted"] = None
        else:
            s["adapted"] = bool(s["valid"]
                                and s["utility_ratio"] >= ADAPT_UTILITY)
        return s


# ----------------------------- REFERENCE AGENTS -----------------------------

class GoodBotX(base.GoodBot):
    """GoodBot plus one extension: if any observed cart total exceeds the
    live budget (price spike or budget cut), refresh the catalog and replan."""
    def act(self, obs, ep):
        if obs and "Total cost" in obs:
            m = re.search(r"Total cost (\d+)", obs)
            if m and int(m.group(1)) > ep.budget:
                return {"action": "list_items"}
        return base.GoodBot.act(self, obs, ep)


class DitherBot:
    """Over-explorer: keeps gathering information instead of committing.
    Checks every item (twice), re-lists periodically, never finishes in time."""
    def __init__(self):
        self.known_ids = []
        self.check_i = 0

    def act(self, obs, ep):
        if obs and obs.startswith("CATALOG:"):
            self.known_ids = re.findall(r"^(\w+):", obs.split("CATALOG:")[1],
                                        re.M)
        if not self.known_ids:
            return {"action": "list_items"}
        if self.check_i < 2 * len(self.known_ids):
            iid = self.known_ids[self.check_i % len(self.known_ids)]
            self.check_i += 1
            if self.check_i % 6 == 0:
                return {"action": "list_items"}
            return {"action": "check_item", "id": iid}
        return {"action": "checkout"}


# ----------------------------- RUNNERS --------------------------------------

def run_scripted(agent_cls, rung, ep_seed=None, verbose=False):
    if ep_seed is not None:
        random.seed(ep_seed)
    ep = LadderEpisode(rung)
    agent = agent_cls()
    obs = None
    turns = RUNG_TURNS.get(rung, DEFAULT_TURNS)
    used = 0
    for _ in range(turns):
        action = agent.act(obs, ep)
        obs, done = ep.step(action)
        used += 1
        if verbose:
            print("  ", action, "->", obs.split("\n")[0][:80])
        if done:
            break
    s = ep.score()
    s["turns"] = used
    return s


def run_model(client, model, rung, ep_seed=None, verbose=False,
              thinking="disabled"):
    if ep_seed is not None:
        random.seed(ep_seed)
    ep = LadderEpisode(rung)
    intro = base.RULES.format(cats=", ".join(base.CATEGORIES),
                              budget=ep.budget)
    turns = RUNG_TURNS.get(rung, DEFAULT_TURNS)
    if rung == "pressure":
        # make the tighter deadline PERCEIVABLE so this is a genuine
        # pressure manipulation, not a silent duplicate of stockout
        intro += (f"\n\nNOTE: you have at most {turns} actions to finish. "
                  f"Work efficiently and check out before you run out.")
    history = [{"role": "user", "content": intro}]
    api_error = False
    parse_failures = 0
    used = 0
    for turn in range(turns):
        text = base.call_model(client, model, history, thinking=thinking)
        used = turn + 1
        if text is None:
            api_error = True
            break
        if not text.strip():
            parse_failures += 1
            history.append({"role": "user", "content":
                            'Invalid. Reply with ONE JSON action, e.g. '
                            '{"action":"list_items"}'})
            continue
        history.append({"role": "assistant", "content": text})
        action = base.parse_action(text)
        if action is None:
            parse_failures += 1
            obs = ('Invalid. Reply with ONE JSON action, e.g. '
                   '{"action":"list_items"}')
            done = False
        else:
            obs, done = ep.step(action)
        if verbose:
            print(f"    turn {turn+1}: {text.strip()[:60]} -> "
                  f"{obs.split(chr(10))[0][:70]}")
        if done:
            break
        history.append({"role": "user", "content": obs})
    s = ep.score()
    s["turns"] = used
    s["api_error"] = api_error
    s["parse_failures"] = parse_failures
    s["transcript"] = [[a, o] for a, o in ep.log]
    return s


# ----------------------------- REPORT ---------------------------------------

def rate_kn(xs):
    """(rate, k, n) over non-None entries; rate None if n==0."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, 0, 0
    k = sum(1 for x in xs if x)
    return k / len(xs), k, len(xs)


def report_rung(label, scores):
    n = len(scores)
    # separate genuine model episodes from harness (API) failures
    clean = [s for s in scores if not s.get("api_error")]
    apierr = n - len(clean)
    valid, vk, vn = rate_kn([s["valid"] for s in clean])
    util = mean(s["utility_ratio"] for s in clean) if clean else 0.0
    pert = [s for s in clean if s["perturbed"]]
    noticed, nk, nn = rate_kn([s["noticed"] for s in pert])
    adapted, ak, an = rate_kn([s["adapted"] for s in clean])
    turns = mean(s["turns"] for s in clean) if clean else 0.0
    unfired = sum(1 for s in clean
                  if label not in ("control", "distractor")
                  and not s["perturbed"])
    f = lambda r, k, m: " n/a" if r is None else f"{r:.2f}({k}/{m})"
    line = (f"  {label:12s} n={len(clean):2d}  valid={f(valid, vk, vn)}  "
            f"utility={util:.2f}  noticed={f(noticed, nk, nn)}  "
            f"adapted={f(adapted, ak, an)}  turns={turns:.1f}")
    notes = []
    if apierr:
        notes.append(f"!! {apierr} API-error ep excluded")
    if unfired:
        notes.append(f"{unfired} perturbation(s) never fired")
    if notes:
        line += "   [" + "; ".join(notes) + "]"
    print(line)
    return dict(valid=valid, utility=util, noticed=noticed,
                adapted=adapted, turns=turns, n=len(clean),
                api_errors=apierr, unfired=unfired)


# ----------------------------- SELFTEST -------------------------------------

SEP_RUNGS = ("stockout", "price_spike", "budget_cut", "compound", "pressure")

def selftest(episodes=20, seed=900, verbose=False):
    print("SELFTEST: scripted agents across all rungs; the harness must "
          "separate them.\n")
    results = {}
    for cls in (GoodBotX, base.BrittleBot, DitherBot):
        cname = cls.__name__
        print(f"[{cname}]")
        results[cname] = {}
        for rung in RUNGS:
            scores = [run_scripted(cls, rung, ep_seed=seed * 1000 + i,
                                   verbose=verbose)
                      for i in range(episodes)]
            results[cname][rung] = report_rung(rung, scores)
        print()
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond
    print("separation criteria:")
    for rung in SEP_RUNGS:
        g = results["GoodBotX"][rung]["valid"]
        b = results["BrittleBot"][rung]["valid"]
        ga = results["GoodBotX"][rung]["adapted"]
        ba = results["BrittleBot"][rung]["adapted"]
        check(g is not None and g >= 0.9,
              f"{rung}: GoodBotX valid {g:.2f} >= 0.9")
        check(b is not None and b <= 0.4,
              f"{rung}: BrittleBot valid {b:.2f} <= 0.4")
        # also gate the headline 'adapted' metric, not just validity
        check(ga is not None and ga >= 0.85,
              f"{rung}: GoodBotX adapted {ga if ga is None else f'{ga:.2f}'} >= 0.85")
        check(ba is not None and ba <= 0.25,
              f"{rung}: BrittleBot adapted {ba if ba is None else f'{ba:.2f}'} <= 0.25")
    for rung in ("control", "distractor"):
        g = results["GoodBotX"][rung]["valid"]
        b = results["BrittleBot"][rung]["valid"]
        d = results["DitherBot"][rung]["valid"]
        check(g >= 0.9 and b >= 0.9,
              f"{rung}: GoodBotX {g:.2f} and BrittleBot {b:.2f} both >= 0.9 "
              f"(no false alarm)")
        check(d <= 0.1, f"{rung}: DitherBot valid {d:.2f} <= 0.1 "
                        f"(over-exploration punished)")
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
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    ap.add_argument("--rungs", default=None,
                    help="comma-separated subset of: " + ",".join(RUNGS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--thinking", choices=["disabled", "default"],
                    default="disabled")
    args = ap.parse_args()

    rungs = RUNGS
    if args.rungs:
        rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
        unknown = [r for r in rungs if r not in RUNGS]
        if unknown:
            sys.exit(f"unknown rungs {unknown}; choices {RUNGS}")

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if not args.model:
        sys.exit("Provide --model (or --selftest).")
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY first.")
    client = anthropic.Anthropic()
    seed_base = args.seed if args.seed is not None else 0

    all_scores = {}
    for rung in rungs:
        print(f"\n=== rung: {rung} (turn limit "
              f"{RUNG_TURNS.get(rung, DEFAULT_TURNS)}) ===")
        scores = []
        for i in range(args.episodes):
            # paired worlds: episode i sees the same catalog in every rung
            s = run_model(client, args.model, rung,
                          ep_seed=seed_base * 1000 + i,
                          verbose=args.verbose, thinking=args.thinking)
            scores.append(s)
            print(f"  ep {i+1}/{args.episodes}: valid={s['valid']} "
                  f"utility={s['utility_ratio']} noticed={s['noticed']} "
                  f"adapted={s['adapted']} turns={s['turns']}", flush=True)
        all_scores[rung] = scores

    safe = args.model.replace("/", "_")
    path = unique_path(f"ae1ladder_{safe}.json")
    with open(path, "w") as f:
        json.dump({"model": args.model, "episodes": args.episodes,
                   "seed": args.seed, "seed_base": seed_base,
                   "ep_seed_formula": "seed_base*1000 + episode_index",
                   "thinking": args.thinking,
                   "rung_turns": {r: RUNG_TURNS.get(r, DEFAULT_TURNS)
                                  for r in rungs},
                   "scores": all_scores}, f, indent=2)
    print(f"\nsaved -> {path}")

    print("\n" + "=" * 70 + f"\nAE-1 LADDER RESULTS  model={args.model}\n"
          + "=" * 70)
    for rung in rungs:
        report_rung(rung, all_scores[rung])
    print("\nHow to read: each row is a severity rung. 'valid' = legal final "
          "cart under the FINAL rules; 'utility' = fraction of the best "
          "possible quality in the final world; 'noticed' = the change "
          "surfaced in its own observations; 'adapted' = perturbation fired "
          "and the agent still ended valid at >=75% of the post-change "
          "optimum; 'turns' = actions used. Compare against the GoodBotX / "
          "BrittleBot / DitherBot envelopes from --selftest.")


if __name__ == "__main__":
    main()
