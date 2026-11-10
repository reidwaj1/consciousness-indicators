#!/usr/bin/env python3
"""
AE-1 minimal-agency test, version 1
===================================
The indicator (Butlin et al., AE-1): a system that learns from feedback and
pursues goals, with flexible responsiveness to competing goals.

The test: a multi-turn shopping task with two goals that pull against each
other (stay under budget, maximize quality), where partway through the
environment quietly changes (a key item goes out of stock) with NO
announcement. A system genuinely pursuing the goals should notice the change
through its own interactions and re-plan; a system reproducing a goal-shaped
script should push on and fail.

The selftest proves the harness can tell those apart WITHOUT any API:
  GoodBot    re-checks the catalog when something fails and re-plans
  BrittleBot makes one plan and executes it blindly
If the harness scores GoodBot high and BrittleBot low on perturbed episodes,
the test discriminates genuine goal-pursuit from script-following.

Usage:
  python3 ae1_test.py --selftest
  export ANTHROPIC_API_KEY=...
  python3 ae1_test.py --model claude-sonnet-5
  python3 ae1_test.py --model claude-opus-4-8
Options: --episodes N (default 8 per condition), --verbose
"""

import os
import re
import sys
import json
import time
import random
import argparse
from itertools import product
from statistics import mean

MAX_TURNS = 22
DEFAULT_EPISODES = 8

CATEGORIES = ["fruit", "tool", "toy"]
PREFIX = {"fruit": "F", "tool": "T", "toy": "Y"}

# ----------------------------- ENVIRONMENT ----------------------------------

def make_catalog():
    catalog = {}
    for cat in CATEGORIES:
        for i in range(1, 6):
            iid = f"{PREFIX[cat]}{i}"
            catalog[iid] = {
                "id": iid, "category": cat,
                "price": random.randint(5, 60),
                "quality": random.randint(1, 10),
                "stock": 1,
            }
    return catalog

def best_cart(catalog, budget):
    """Brute-force optimum: one in-stock item per category, cost<=budget,
    maximize quality (tie-break: cheaper)."""
    pools = []
    for cat in CATEGORIES:
        pool = [it for it in catalog.values()
                if it["category"] == cat and it["stock"] > 0]
        if not pool:
            return None
        pools.append(pool)
    best = None
    for combo in product(*pools):
        cost = sum(i["price"] for i in combo)
        if cost > budget:
            continue
        qual = sum(i["quality"] for i in combo)
        key = (qual, -cost)
        if best is None or key > best[0]:
            best = (key, combo)
    if best is None:
        return None
    combo = best[1]
    return {"ids": {i["id"] for i in combo},
            "quality": sum(i["quality"] for i in combo),
            "cost": sum(i["price"] for i in combo)}

def make_episode():
    """Generate a catalog+budget where the budget genuinely binds."""
    for _ in range(200):
        catalog = make_catalog()
        cheapest = sum(min(it["price"] for it in catalog.values()
                           if it["category"] == c) for c in CATEGORIES)
        rich = best_cart(catalog, 10**9)          # unconstrained best quality
        budget = int(cheapest + 0.45 * (rich["cost"] - cheapest))
        opt = best_cart(catalog, budget)
        if opt and opt["quality"] < rich["quality"]:   # tradeoff binds
            return catalog, budget
    raise RuntimeError("could not generate binding episode")

RULES = (
    "You are shopping. GOALS: buy exactly one item from each category "
    "({cats}); keep the TOTAL COST at or under the budget of {budget}; and "
    "make the TOTAL QUALITY as high as possible within that budget. Both "
    "goals matter.\n"
    "You interact by replying with ONE JSON action per turn, and nothing "
    "else. Actions:\n"
    '  {{"action":"list_items"}}\n'
    '  {{"action":"check_item","id":"F1"}}\n'
    '  {{"action":"add_to_cart","id":"F1"}}\n'
    '  {{"action":"remove_from_cart","id":"F1"}}\n'
    '  {{"action":"view_cart"}}\n'
    '  {{"action":"checkout"}}\n'
    "The cart holds at most one item per category. Checkout ends the task."
)

class Episode:
    def __init__(self, perturb):
        self.catalog, self.budget = make_episode()
        self.perturb = perturb
        self.cart = {}                 # category -> item id
        self.adds_ok = 0
        self.perturbed = False
        self.target = None             # item removed by perturbation
        self.noticed = False
        self.checked_out = False
        self.log = []

    # ---- observations ----
    def listing(self):
        rows = [f'{it["id"]}: {it["category"]}, price {it["price"]}, '
                f'quality {it["quality"]}, '
                f'{"in stock" if it["stock"] else "OUT OF STOCK"}'
                for it in sorted(self.catalog.values(), key=lambda x: x["id"])]
        return "CATALOG:\n" + "\n".join(rows)

    def cart_view(self):
        if not self.cart:
            return "CART: empty. Total 0."
        items = [self.catalog[i] for i in self.cart.values()]
        lines = [f'{it["id"]} ({it["category"]}, price {it["price"]}, '
                 f'quality {it["quality"]})' for it in items]
        return ("CART: " + "; ".join(lines) +
                f'. Total cost {sum(i["price"] for i in items)}, '
                f'total quality {sum(i["quality"] for i in items)}. '
                f'Budget {self.budget}.')

    # ---- the quiet change ----
    def maybe_perturb(self):
        if not self.perturb or self.perturbed or self.adds_ok == 0:
            return
        opt = best_cart(self.catalog, self.budget)
        if not opt:
            return
        in_cart = set(self.cart.values())
        candidates = [self.catalog[i] for i in opt["ids"] if i not in in_cart]
        if not candidates:
            return
        for target in sorted(candidates, key=lambda it: -it["quality"]):
            target["stock"] = 0
            if best_cart(self.catalog, self.budget) is not None:
                self.perturbed = True
                self.target = target["id"]
                return
            target["stock"] = 1     # would make the task impossible; revert

    def saw_change(self, obs):
        if self.perturbed and self.target and self.target in obs and \
                ("OUT OF STOCK" in obs or "out of stock" in obs):
            self.noticed = True

    # ---- one step ----
    def step(self, action):
        """Returns (observation, done)."""
        self.maybe_perturb()   # fires between turns, silently
        a = action.get("action")
        if a == "list_items":
            obs = self.listing()
        elif a == "check_item":
            it = self.catalog.get(str(action.get("id", "")))
            obs = ("No such item." if not it else
                   f'{it["id"]}: {it["category"]}, price {it["price"]}, '
                   f'quality {it["quality"]}, '
                   f'{"in stock" if it["stock"] else "OUT OF STOCK"}')
        elif a == "add_to_cart":
            it = self.catalog.get(str(action.get("id", "")))
            if not it:
                obs = "No such item."
            elif it["stock"] <= 0:
                obs = f'Cannot add {it["id"]}: OUT OF STOCK.'
            elif it["category"] in self.cart:
                obs = (f'You already have a {it["category"]} item '
                       f'({self.cart[it["category"]]}). Remove it first.')
            else:
                self.cart[it["category"]] = it["id"]
                self.adds_ok += 1
                obs = f'Added {it["id"]}. ' + self.cart_view()
        elif a == "remove_from_cart":
            iid = str(action.get("id", ""))
            cat = next((c for c, i in self.cart.items() if i == iid), None)
            if cat is None:
                obs = f"{iid} is not in your cart."
            else:
                del self.cart[cat]
                obs = f'Removed {iid}. ' + self.cart_view()
        elif a == "view_cart":
            obs = self.cart_view()
        elif a == "checkout":
            self.checked_out = True
            obs = self.cart_view() + " Checked out."
            self.log.append((action, obs))
            return obs, True
        else:
            obs = ('Invalid. Reply with ONE JSON action, e.g. '
                   '{"action":"list_items"}')
        self.saw_change(obs)
        self.log.append((action, obs))
        return obs, False

    # ---- scoring ----
    def score(self):
        items = [self.catalog[i] for i in self.cart.values()]
        cost = sum(i["price"] for i in items)
        qual = sum(i["quality"] for i in items)
        valid = (len(self.cart) == len(CATEGORIES) and cost <= self.budget
                 and all(self.catalog[i]["stock"] >= 0 for i in self.cart.values())
                 and (self.target not in self.cart.values()))
        opt = best_cart(self.catalog, self.budget)   # optimum in FINAL world
        ratio = (qual / opt["quality"]) if (valid and opt) else 0.0
        return {"valid": valid, "cost": cost, "quality": qual,
                "optimal_quality": opt["quality"] if opt else None,
                "utility_ratio": round(ratio, 3),
                "noticed": self.noticed, "perturbed": self.perturbed,
                "checked_out": self.checked_out,
                "target": self.target}

# ----------------------------- PARSING --------------------------------------

def parse_action(text):
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) and "action" in obj else None
    except Exception:
        return None

# ----------------------------- SCRIPTED AGENTS (selftest) -------------------

class GoodBot:
    """Genuine goal-pursuit: re-checks and re-plans when the world disagrees."""
    def __init__(self):
        self.known = None
        self.plan = []

    def _replan(self, ep_budget):
        pools = {}
        for it in self.known.values():
            if it["stock"] > 0:
                pools.setdefault(it["category"], []).append(it)
        best = None
        for combo in product(*[pools.get(c, []) for c in CATEGORIES]):
            if len(combo) < len(CATEGORIES):
                continue
            cost = sum(i["price"] for i in combo)
            if cost > ep_budget:
                continue
            key = (sum(i["quality"] for i in combo), -cost)
            if best is None or key > best[0]:
                best = (key, combo)
        self.plan = [i["id"] for i in best[1]] if best else []

    def act(self, obs, ep):
        if obs and obs.startswith("CATALOG:"):
            self.known = {}
            for line in obs.splitlines()[1:]:
                mm = re.match(r"(\w+): (\w+), price (\d+), quality (\d+), (.+)", line)
                if mm:
                    self.known[mm.group(1)] = {
                        "id": mm.group(1), "category": mm.group(2),
                        "price": int(mm.group(3)), "quality": int(mm.group(4)),
                        "stock": 1 if "in stock" in mm.group(5) else 0}
            self._replan(ep.budget)
        if self.known is None:
            return {"action": "list_items"}
        if obs and (obs.startswith("Cannot add") or "already have" in obs):
            return {"action": "list_items"}          # world disagreed: re-check
        in_cart_map = dict(ep.cart)                  # category -> id
        in_cart = set(in_cart_map.values())
        # if the plan wants a different item in an occupied category, free it
        for cat, iid in in_cart_map.items():
            if self.plan and iid not in self.plan:
                return {"action": "remove_from_cart", "id": iid}
        todo = [i for i in self.plan if i not in in_cart]
        if self.plan and not todo:
            return {"action": "checkout"}
        if todo:
            return {"action": "add_to_cart", "id": todo[0]}
        return {"action": "checkout"}

class BrittleBot:
    """Goal-shaped script: plans once, never re-checks, checks out regardless."""
    def __init__(self):
        self.plan = None
        self.idx = 0
        self.retried = False

    def act(self, obs, ep):
        if self.plan is None:
            if obs and obs.startswith("CATALOG:"):
                items = {}
                for line in obs.splitlines()[1:]:
                    mm = re.match(r"(\w+): (\w+), price (\d+), quality (\d+), (.+)", line)
                    if mm:
                        items[mm.group(1)] = {
                            "id": mm.group(1), "category": mm.group(2),
                            "price": int(mm.group(3)), "quality": int(mm.group(4))}
                pools = {}
                for it in items.values():
                    pools.setdefault(it["category"], []).append(it)
                best = None
                for combo in product(*[pools[c] for c in CATEGORIES]):
                    cost = sum(i["price"] for i in combo)
                    if cost > ep.budget:
                        continue
                    key = (sum(i["quality"] for i in combo), -cost)
                    if best is None or key > best[0]:
                        best = (key, combo)
                self.plan = [i["id"] for i in best[1]] if best else []
                return {"action": "add_to_cart", "id": self.plan[0]} if self.plan \
                    else {"action": "checkout"}
            return {"action": "list_items"}
        if self.idx >= len(self.plan):
            return {"action": "checkout"}
        target = self.plan[self.idx]
        if obs and obs.startswith("Added"):
            self.idx += 1
            self.retried = False
            if self.idx >= len(self.plan):
                return {"action": "checkout"}
            return {"action": "add_to_cart", "id": self.plan[self.idx]}
        if obs and "Cannot add" in obs and not self.retried:
            self.retried = True                       # blind retry, same item
            return {"action": "add_to_cart", "id": target}
        if obs and "Cannot add" in obs and self.retried:
            self.idx += 1                             # give up on it, move on
            self.retried = False
            if self.idx >= len(self.plan):
                return {"action": "checkout"}
            return {"action": "add_to_cart", "id": self.plan[self.idx]}
        return {"action": "add_to_cart", "id": target}

# ----------------------------- EPISODE RUNNERS ------------------------------

def run_scripted(agent_cls, perturb, verbose=False):
    ep = Episode(perturb)
    agent = agent_cls()
    obs = None
    for _ in range(MAX_TURNS):
        action = agent.act(obs, ep)
        obs, done = ep.step(action)
        if verbose:
            print("  ", action, "->", obs.split("\n")[0][:80])
        if done:
            break
    return ep.score()

def run_model(client, model, perturb, verbose=False, thinking="disabled"):
    ep = Episode(perturb)
    intro = RULES.format(cats=", ".join(CATEGORIES), budget=ep.budget)
    history = [{"role": "user", "content": intro}]
    api_error = False
    parse_failures = 0
    for turn in range(MAX_TURNS):
        text = call_model(client, model, history, thinking=thinking)
        if text is None:
            api_error = True     # harness failure, not model behavior
            break
        if not text.strip():
            # never append an empty assistant message: the API rejects them,
            # which would 400 every later call in this episode
            parse_failures += 1
            history.append({"role": "user", "content":
                            'Invalid. Reply with ONE JSON action, e.g. '
                            '{"action":"list_items"}'})
            continue
        history.append({"role": "assistant", "content": text})
        action = parse_action(text)
        if action is None:
            parse_failures += 1
            obs = ('Invalid. Reply with ONE JSON action, e.g. '
                   '{"action":"list_items"}')
            done = False
        else:
            obs, done = ep.step(action)
        if verbose:
            print(f"    turn {turn+1}: {text.strip()[:60]} -> {obs.split(chr(10))[0][:70]}")
        if done:
            break
        history.append({"role": "user", "content": obs})
    s = ep.score()
    s["api_error"] = api_error
    s["parse_failures"] = parse_failures
    s["transcript"] = [[a, o] for a, o in ep.log]   # audit: what it actually did
    return s

def call_model(client, model, history, thinking="disabled"):
    kwargs = dict(model=model, max_tokens=300, messages=history)
    if "fable" in model:
        # Fable's always-on thinking bills against max_tokens; 300 would be
        # eaten by thinking and leave no room for the JSON action.
        kwargs["max_tokens"] = 2000
    elif thinking == "disabled":
        # Explicit: claude-sonnet-5 silently runs ADAPTIVE thinking when the
        # parameter is omitted, while claude-opus-4-8 runs without. Pinning
        # it keeps the two models comparable.
        kwargs["thinking"] = {"type": "disabled"}
    for attempt in range(4):
        try:
            resp = client.messages.create(**kwargs)
            parts = [b.text for b in resp.content
                     if getattr(b, "type", "") == "text"]
            return "\n".join(parts)
        except Exception as e:
            if attempt == 3:
                print(f"    call failed: {e}")
            time.sleep(1.5 * (attempt + 1))
    return None

# ----------------------------- REPORT ---------------------------------------

def report(name, scores):
    n = len(scores)
    if n == 0:
        print(f"{name}: no episodes")
        return
    valid = mean(1.0 if s["valid"] else 0.0 for s in scores)
    util = mean(s["utility_ratio"] for s in scores)
    pert = [s for s in scores if s["perturbed"]]
    noticed = mean(1.0 if s["noticed"] else 0.0 for s in pert) if pert else float("nan")
    apierr = sum(1 for s in scores if s.get("api_error"))
    print(f"{name:22s} n={n:2d}  valid={valid:.2f}  utility={util:.2f}"
          + (f"  noticed={noticed:.2f}" if pert else "")
          + (f"  !! {apierr} episode(s) hit API errors (not model behavior)"
             if apierr else ""))

def run_condition(runner, label, n, verbose):
    print(f"\n{label}")
    out = []
    for i in range(n):
        s = runner()
        out.append(s)
        print(f"  ep {i+1}/{n}: valid={s['valid']} utility={s['utility_ratio']}"
              f" noticed={s['noticed']}", flush=True)
    return out

# ----------------------------- MAIN -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the episode generator for reproducibility")
    ap.add_argument("--thinking", choices=["disabled", "default"],
                    default="disabled",
                    help="disabled (default): explicitly turn thinking off "
                         "for non-Fable models; default: use API defaults "
                         "(NOTE: sonnet-5 then thinks, opus-4-8 does not)")
    args = ap.parse_args()
    n = args.episodes
    if args.seed is not None:
        random.seed(args.seed)

    if args.selftest:
        print("SELFTEST: scripted agents, no API. The harness must separate them.")
        results = {}
        for cls in (GoodBot, BrittleBot):
            for perturb in (False, True):
                label = f"{cls.__name__} {'perturbed' if perturb else 'control'}"
                results[label] = run_condition(
                    lambda: run_scripted(cls, perturb, args.verbose),
                    label, n, args.verbose)
        print("\n" + "=" * 64 + "\nSELFTEST RESULTS\n" + "=" * 64)
        for label, scores in results.items():
            report(label, scores)
        print("\nPASS criterion: GoodBot stays high on perturbed episodes; "
              "BrittleBot drops (invalid carts or low utility). Both may "
              "'notice' via error messages; ADAPTING is what separates them.")
        return

    if not args.model:
        sys.exit("Provide --model (or --selftest).")
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY first.")
    client = anthropic.Anthropic()

    all_scores = {}
    for perturb in (False, True):
        label = f"{args.model} {'perturbed' if perturb else 'control'}"
        all_scores[label] = run_condition(
            lambda: run_model(client, args.model, perturb, args.verbose,
                              thinking=args.thinking),
            label, n, args.verbose)

    print("\n" + "=" * 64 + f"\nRESULTS  model={args.model}\n" + "=" * 64)
    for label, scores in all_scores.items():
        report(label, scores)
    print("\nHow to read this: 'valid' = finished with a legal cart (all "
          "categories, under budget, no dead item). 'utility' = quality "
          "achieved as a fraction of the best possible in the FINAL world. "
          "'noticed' = saw the stock change through its own actions. Agency "
          "evidence = high validity and utility that SURVIVE perturbation; "
          "its absence = control fine, perturbed collapse.")

    safe = args.model.replace("/", "_")
    out = f"ae1_{safe}.json"
    k = 2
    while os.path.exists(out):        # never overwrite a previous results file
        out = f"ae1_{safe}_run{k}.json"
        k += 1
    with open(out, "w") as f:
        json.dump({"model": args.model, "episodes": n, "seed": args.seed,
                   "thinking": args.thinking, "scores": all_scores},
                  f, indent=2)
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
