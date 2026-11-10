#!/usr/bin/env python3
"""
M10 Stage -1, item 6: the NUMERIC-FAMILY WHITELIST
==================================================
Mission: M10_MISSION.md Stage -1 item 6. Enumerate the m4/tracking task
families whose verified answer is a single integer and, for each
whitelisted family, freeze NOW (pod-free, model-free): the generation
adapter with its independent verifier called on every item, the F-/F+
foil rule, the foil audit, and the analytically derived
infeasible-true-value set. Non-whitelisted families are ineligible,
period; they are listed in NONWHITELISTED with reasons. This module
feeds m10_prereg_template.md (Stage -1 item 7) and the Stage-1
calibration walk ("if saturated, walk ONLY whitelisted numeric m4
families").

Imports cleanly with stdlib + numpy only (m4_families / m8_force are
imported for their generators and independent verifiers; neither loads
a model at import time). The tokenizer is NEVER loaded here: every foil
decision takes an injected tokcount_fn(str) -> int. On the pod, inject
a cached wrapper (functools.lru_cache) around the real tokenizer.

WHITELIST DECISION TABLE
------------------------
  family            single-int answer?     foil rule?    verdict
  tracking (m8)     yes ([3,14] at L0)     yes           WHITELISTED
  program_exec      yes ([0, 9999])        yes           WHITELISTED
  constraint_count  yes ([5, width])       yes (caveat)  WHITELISTED
  arith_long        yes (signed, |T|<1e7)  yes           WHITELISTED
  logic_grid        yes ([1, n], n <= 7)   NO            NONWHITELISTED

logic_grid passes the single-integer test but fails the foil-rule test;
the full argument is in NONWHITELISTED.

FOIL RULE (shared by all whitelisted families; the rule depends only on
the true value, so the per-family content is the answer range, the
infeasible set, and the leading-digit documentation in WHITELIST notes)
-----------------------------------------------------------------------
pick_foils(true, err_offsets, tokcount_fn, rng) -> (Fminus, Fplus) with
  Fminus < true < Fplus, and both foils matching true on:
    sign               (arith_long extension; trivial elsewhere)
    digit count        of |value|
    roundness class    (|value| % 10 == 0)
    tokenizer tokens   (tokcount_fn(str(value)))
    leading digit      of |value| -- TIER 1, applied when feasible
Preference order within a side, ported from m8_force.pick_foil:
candidates at offsets seen in the model's own stage-1 error-offset
distribution, sampled with the observed frequencies; else the nearest
matching value (unique per side, so no tie-break is needed). rng is
consumed only by the weighted branch.

TIER SYSTEM (mission audit list: digit count, leading digit, roundness)
Tier 1 adds same-leading-digit to the match. It is applied only when
BOTH sides admit a tier-1 candidate under the full constraint set
(including tokcount_fn); otherwise BOTH foils fall back to tier 2
(digit count + roundness) so the pair is never mixed-stringency.
FoilError is raised when tier 2 is empty on either side.

FOIL FEASIBILITY, DERIVED ANALYTICALLY (generalizing m8_force's level-0
derivation of {9, 10, 11})
-----------------------------------------------------------------------
The tier-2 cell of T (same sign, digit count k of |T|, roundness class)
is a contiguous run of magnitudes:
  non-round, k = 1 : |c| in [1, 9]
  non-round, k >= 2: |c| in [10^(k-1) + 1, 10^k - 1]
  round,     k >= 2: |c| in [10^(k-1), 10^k - 10], step 10
  T = 0            : cell {0} (0 is the only 1-digit round value)
Fminus < T < Fplus needs a strictly smaller AND a strictly larger cell
member, so T is infeasible exactly at the cell edges:
  T == 0;  |T| in {1, 9} (k = 1);
  |T| in {10^(k-1), 10^k - 10} (round);
  |T| in {10^(k-1) + 1, 10^k - 1} (non-round, k >= 2).
For T < 0 the empty side flips (the mirror of +10's empty minus side is
-10's empty plus side) but the pair is broken either way, so the
predicate depends only on |T|. Intersecting with each family's answer
range gives the per-family sets; tracking level 0 ([3, 14]) reproduces
m8's {9, 10, 11} exactly (selftested against m8_force.INFEASIBLE_T).

Tier-1 structural feasibility (leading digit d, magnitude a = |T|):
  k = 1            : never (the leading digit IS the value)
  round,     k = 2 : never (d0 is the only round 2-digit lead-d value)
  round,     k >= 3: d*10^(k-1) < a < (d+1)*10^(k-1) - 10
  non-round, k >= 2: d*10^(k-1) + 1 < a < (d+1)*10^(k-1) - 1
A fragmenting tokenizer can only SHRINK cells (tokcount is an equality
constraint), which surfaces at runtime as extra tier-2 fallbacks or
FoilErrors -- never as a silently weaker match (selftested with a
deliberately fragmenting fake tokenizer).

PER-FAMILY ANSWER RANGES AND INFEASIBLE SETS (details in WHITELIST)
  tracking L0  : [3, 14] -> {9, 10, 11};  L >= 1: [0, 12 + 6L]
  program_exec : [0, 9999] -> {0, 1, 9, 10, 11, 90, 99, 100, 101, 990,
                 999, 1000, 1001, 9990, 9999} (15 values)
  constraint_count: [5, width(level)], width <= 200000
  arith_long   : signed, |T| < 10^7 (= m4.AL_CAP; the multiply guard
                 forbids mul/div once |v| > AL_CAP // 12, so one final
                 multiply by <= 9 tops out below 7.5e6 and <= 300 adds
                 of < 1000 add < 3e5)

python3 m10_families.py --selftest          full offline selftest
python3 m10_families.py --show LEVEL        one item + foils per family
python3 m10_families.py --infeasible        per-family infeasible sets
"""

import argparse
import random
import sys
import time

import m4_families as m4
import m8_force as m8

# ----------------------------- SURFACE HELPERS --------------------------------
# Sign-extended versions of m8_force's helpers (m8's operate on the
# positive level-0 range only; these agree with them there -- selftested
# -- and additionally handle arith_long's signed answers).


def n_digits(v):
    """Digit count of |v| (sign-blind; matches m8_force.n_digits on
    positive values)."""
    return len(str(abs(v)))


def is_round(v):
    """Roundness class of |v| (|v| % 10 == 0; matches m8_force.is_round
    on positive values)."""
    return abs(v) % 10 == 0


def leading_digit(v):
    return int(str(abs(v))[0])


# ----------------------------- FOIL RULE --------------------------------------


class FoilError(Exception):
    pass


def _matches(true, c, tokcount_fn, require_lead):
    """Full per-candidate constraint set (order is checked by the
    caller's scan direction)."""
    if c == true:
        return False
    if (c < 0) != (true < 0):
        return False
    if n_digits(c) != n_digits(true):
        return False
    if is_round(c) != is_round(true):
        return False
    if require_lead and leading_digit(c) != leading_digit(true):
        return False
    if tokcount_fn(str(c)) != tokcount_fn(str(true)):
        return False
    return True


def _nearest(true, side, tokcount_fn, require_lead):
    """Nearest matching value strictly on the requested side of true,
    scanning outward inside the (lead cell if require_lead else digit
    cell); None if the cell side is empty. Doubles as the per-side
    feasibility test."""
    if true == 0:
        return None
    a = abs(true)
    k = n_digits(true)
    if require_lead:
        if k == 1:
            return None            # leading digit == value: no peers
        d = leading_digit(true)
        lo_a = d * 10 ** (k - 1)
        hi_a = (d + 1) * 10 ** (k - 1) - 1
    else:
        lo_a, hi_a = (1, 9) if k == 1 else (10 ** (k - 1), 10 ** k - 1)
    lo, hi = (lo_a, hi_a) if true > 0 else (-hi_a, -lo_a)
    step = -1 if side == "minus" else 1
    c = true + step
    while lo <= c <= hi:
        if _matches(true, c, tokcount_fn, require_lead):
            return c
        c += step
    return None


def pick_foil(true, side, err_offsets, tokcount_fn, rng, require_lead):
    """One side. m8_force.pick_foil's preregistered preference order,
    generalized: candidates at offsets present in the model's own
    error-offset distribution, sampled with the observed frequencies;
    else the nearest matching value (unique per side). Raises FoilError
    if the cell side is empty."""
    step = -1 if side == "minus" else 1
    cands, weights = [], []
    for off, w in sorted(err_offsets.items()):
        if off <= 0 or w <= 0:
            continue
        c = true + step * off
        if _matches(true, c, tokcount_fn, require_lead):
            cands.append(c)
            weights.append(w)
    if cands:
        return rng.choices(cands, weights=weights, k=1)[0]
    c = _nearest(true, side, tokcount_fn, require_lead)
    if c is None:
        raise FoilError(f"no matched {side} foil for T={true} "
                        f"(require_lead={require_lead})")
    return c


def pick_foils(true, err_offsets, tokcount_fn, rng):
    """The M10 foil rule (shared across whitelisted families; see module
    docstring). Tier 1 (adds same leading digit) when both sides admit a
    tier-1 candidate under the full constraint set including
    tokcount_fn; else tier 2 for BOTH foils; FoilError when tier 2 is
    empty on either side. Returns (Fminus, Fplus) with
    Fminus < true < Fplus."""
    for require_lead in (True, False):
        if (_nearest(true, "minus", tokcount_fn, require_lead) is not None
                and _nearest(true, "plus", tokcount_fn,
                             require_lead) is not None):
            fm = pick_foil(true, "minus", err_offsets, tokcount_fn, rng,
                           require_lead)
            fp = pick_foil(true, "plus", err_offsets, tokcount_fn, rng,
                           require_lead)
            return fm, fp
    raise FoilError(
        f"no matched foil pair for T={true} (structurally infeasible: "
        f"{structurally_infeasible(true)})")


def pair_tier(true, fm, fp):
    """1 if the saved pair matches T's leading digit (tier 1), else 2.
    Logging helper; the audit decides required tier from feasibility,
    not from this."""
    return 1 if (leading_digit(fm) == leading_digit(true)
                 == leading_digit(fp)) else 2


# ----------------------------- FEASIBILITY ------------------------------------


def structurally_infeasible(true):
    """True iff no (F-, F+) pair exists under the tier-2 constraints
    (sign, digit count, roundness, order) alone -- the derivation is in
    the module docstring. Tokenizer effects can only add runtime
    FoilErrors on top, never remove one of these."""
    if true == 0:
        return True
    a, k = abs(true), n_digits(true)
    if is_round(true):             # k >= 2 here: positive round => a >= 10
        return a == 10 ** (k - 1) or a == 10 ** k - 10
    if k == 1:
        return a in (1, 9)
    return a == 10 ** (k - 1) + 1 or a == 10 ** k - 1


def lead_feasible(true):
    """Tier-1 structural feasibility (both sides, ignoring tokcount);
    derivation in the module docstring."""
    if true == 0:
        return False
    a, k = abs(true), n_digits(true)
    if k == 1:
        return False
    d = leading_digit(true)
    if is_round(true):
        return d * 10 ** (k - 1) < a < (d + 1) * 10 ** (k - 1) - 10
    return d * 10 ** (k - 1) + 1 < a < (d + 1) * 10 ** (k - 1) - 1


def infeasible_in(lo, hi):
    """All structurally infeasible T in [lo, hi], in closed form
    (O(log hi) construction; selftested against the predicate)."""
    out = set()
    if lo <= 0 <= hi:
        out.add(0)

    def edges(bound):
        vals = {1, 9}
        for k in range(2, len(str(bound)) + 1):
            vals.update({10 ** (k - 1), 10 ** (k - 1) + 1,
                         10 ** k - 10, 10 ** k - 1})
        return {v for v in vals if v <= bound}

    if hi >= 1:
        out.update(v for v in edges(hi) if v >= max(1, lo))
    if lo <= -1:
        out.update(-v for v in edges(-lo) if -v <= min(-1, hi))
    return sorted(out)


def answer_range(family, level):
    """Analytic answer range of a whitelisted family at a level
    (derivations in the WHITELIST notes)."""
    if family == "tracking":
        return (3, 14) if level == 0 else (0, 12 + 6 * level)
    if family == "program_exec":
        return (0, m4.PE_MOD - 1)
    if family == "constraint_count":
        return (5, m4.CC_LEVELS[level][0])
    if family == "arith_long":
        return (-m4.AL_CAP, m4.AL_CAP)
    raise KeyError(family)


def infeasible_true_values(family, level):
    """The family's infeasible-true-value set at a level: the structural
    edge set intersected with the answer range (m8's {9, 10, 11} is the
    tracking level-0 instance)."""
    return infeasible_in(*answer_range(family, level))


# ----------------------------- GENERATE ADAPTERS ------------------------------
# generate(level, rng) -> (body, true, meta). The family's INDEPENDENT
# verifier (m8_force.independent_solve for tracking; m4_families'
# text-level re-parse + re-solve verifiers otherwise) is called on every
# single item; a mismatch raises VerifyError -- no unverified item can
# leave this module.


class VerifyError(Exception):
    pass


def verify_true(family, body, true):
    """True iff the family's independent verifier reproduces `true`
    from the rendered body alone."""
    if family == "tracking":
        return m8.independent_solve(body) == true
    ok, _detail = m4.verify_item({"family": family, "body": body,
                                  "true": true})
    return ok


def _require_verified(family, body, true, level):
    if not verify_true(family, body, true):
        raise VerifyError(f"{family} L{level}: independent verifier "
                          f"disagrees with generator (claimed {true})")


def gen_tracking_item(level, rng):
    """tracking: m8_force.gen_tracking (the verbatim M7 port) with
    n_ops = level; verified by m8_force.independent_solve (the M8 G2
    verifier) on the rendered body."""
    body, true, ask = m8.gen_tracking(level, rng)
    _require_verified("tracking", body, true, level)
    return body, true, {"family": "tracking", "level": level, "ask": ask,
                        "params": {"tr_ops": level}}


def _gen_m4_item(family, level, rng):
    item = m4.FAMILIES[family](level, rng)
    _require_verified(family, item["body"], item["true"], level)
    return item["body"], item["true"], {"family": family, "level": level,
                                        "params": item["params"]}


def gen_program_exec_item(level, rng):
    return _gen_m4_item("program_exec", level, rng)


def gen_constraint_count_item(level, rng):
    return _gen_m4_item("constraint_count", level, rng)


def gen_arith_long_item(level, rng):
    return _gen_m4_item("arith_long", level, rng)


def gen_feasible_item(family, level, rng, counter=None, tokcount_fn=None):
    """m8_force.gen_item's rejection-sampling pattern, family-general:
    draw verified items until the true value admits a matched foil pair.
    Structural feasibility always; if tokcount_fn is given, tier-2
    tokenizer feasibility of both sides is required too (the pipeline
    passes the real tokenizer here so tokenizer-fragmented cells are
    rejected at build time, with the reject count recorded)."""
    gen = WHITELIST[family]["generate"]
    while True:
        body, true, meta = gen(level, rng)
        feasible = not structurally_infeasible(true)
        if feasible and tokcount_fn is not None:
            feasible = (
                _nearest(true, "minus", tokcount_fn, False) is not None
                and _nearest(true, "plus", tokcount_fn, False) is not None)
        if feasible:
            return body, true, meta
        if counter is not None:
            counter["rejected"] = counter.get("rejected", 0) + 1


# ----------------------------- REGISTRY ---------------------------------------


def _entry(name, gen, notes):
    return {"generate": gen,
            "pick_foils": pick_foils,   # shared rule: depends only on T
            "answer_range": (lambda level, _n=name: answer_range(_n, level)),
            "infeasible": (lambda level, _n=name:
                           infeasible_true_values(_n, level)),
            "notes": notes}


WHITELIST = {
    "tracking": _entry(
        "tracking", gen_tracking_item,
        "Answers: level 0 in [3, 14] (m8 derivation; infeasible "
        "{9, 10, 11} == m8_force.INFEASIBLE_T, selftested); level "
        "L >= 1 in [0, 12 + 6L] (boxes start <= 12, each op adds <= 6; "
        "removes floor at 0). LEADING-DIGIT TIER: 1-digit T (most of "
        "the L0 range) always falls back to digit-count + roundness "
        "because a 1-digit leading digit IS the value; at L0 tier 1 "
        "binds only for T in {12, 13, 14}. This is the one deliberate "
        "tightening over m8's rule (leading digit added per the M10 "
        "mission audit list)."),
    "program_exec": _entry(
        "program_exec", gen_program_exec_item,
        "Answers: r0 at halt, in [0, 9999] (mod-10000 registers, stated "
        "in the problem header). Infeasible set (15 values): {0, 1, 9, "
        "10, 11, 90, 99, 100, 101, 990, 999, 1000, 1001, 9990, 9999}. "
        "Foils share T's digit cell, so every foil is a representable "
        "register value -- in-domain by construction. LEADING-DIGIT "
        "TIER: infeasible for 1-digit T, for all round 2-digit T (d0 "
        "is the only round 2-digit lead-d value), and at per-lead cell "
        "edges; tier-2 fallback documented in the module docstring."),
    "constraint_count": _entry(
        "constraint_count", gen_constraint_count_item,
        "Answers: counts in [5, width] (generator floor 5 after the m4 "
        "gate-3 review fix; width = CC_LEVELS[level][0] <= 200000). "
        "LEADING-DIGIT TIER: as derived generically (never for 1-digit "
        "or round 2-digit counts). CAVEAT for the prereg template: "
        "pick_foils cannot see the item's width, so when T sits within "
        "its digit cell of width, F+ can exceed the largest possible "
        "count b - a + 1 and become surface-detectable; the pipeline "
        "must reject such items at build time using "
        "meta['params']['cc_width'] (require Fplus <= cc_width)."),
    "arith_long": _entry(
        "arith_long", gen_arith_long_item,
        "Answers: signed; |T| < 10^7 = m4.AL_CAP analytically (multiply "
        "guard forbids mul/div once |v| > AL_CAP // 12, so one multiply "
        "by <= 9 tops out below 7.5e6, and <= 300 adds of < 1000 add "
        "< 3e5). M10 EXTENSION over m8's positive-only rule: foils "
        "additionally match T's sign; digit count, leading digit and "
        "roundness are taken on |T|; for negative T the empty-side "
        "geometry mirrors (the predicate depends only on |T|). T = 0 "
        "is structurally infeasible and rejection-sampled like any "
        "other infeasible value. LEADING-DIGIT TIER: as derived "
        "generically, on |T|."),
}

NONWHITELISTED = {
    "logic_grid": (
        "Single-integer test: PASSES (the verified answer is a house "
        "index in [1, n], n <= 7). Foil-rule test: FAILS, three ways. "
        "(1) The valid-answer domain [1, n] is item-level metadata "
        "(LG_LEVELS n varies 3..7 by level) that the frozen "
        "pick_foils(true, err_offsets, tokcount_fn, rng) contract "
        "cannot see, and any foil > n is surface-detectably wrong from "
        "the question line itself ('in which house (1-n) ...'), which "
        "destroys the beyond-surface-features control the foil design "
        "exists for. (2) Restricting to in-domain foils makes the "
        "endpoint answers T = 1 and T = n structurally infeasible "
        "(F- < T < F+ needs room on both sides) -- at n = 3 that "
        "excludes 2 of the 3 possible answers, and endpoint rejection "
        "would bias the item distribution over answer positions at "
        "every level. (3) All answers are 1-digit, so the mission "
        "audit list degenerates: the leading-digit constraint is never "
        "applicable (leading digit == value) and the roundness class "
        "is constant (no value in 1..7 is round) -- only digit count "
        "survives, which every candidate ties on. Ineligible, period "
        "(M10_MISSION.md Stage -1 item 6)."),
}


# ----------------------------- FOIL AUDIT -------------------------------------


def foil_audit(rows, tokcount_fn=None):
    """Re-verify every saved foil pair against the mission audit list
    (digit count, leading digit, roundness class) plus order, sign, and
    token count -- m8_force's posttest foil-audit bot, family-general.

    rows: dicts with at least {"true", "Fminus", "Fplus"}; optional
    "tok3" = [tokcount(str(T)), tokcount(str(F-)), tokcount(str(F+))]
    recorded at extraction time. When tokcount_fn is given, tier-1
    feasibility is recomputed under it (exact); otherwise the
    structural tier-1 predicate is used -- correct whenever the real
    tokenizer does not fragment same-cell numerals differently (the
    pipeline audits with the real tokcount_fn, like m8's posttest ran
    with the model loaded). Leading-digit equality is REQUIRED exactly
    when tier 1 is feasible for that T; differing leads on a
    tier-1-infeasible T are the documented fallback, not a failure.
    Returns {"n", "n_bad", "n_tier1", "n_tier2", "ok", "failures"}."""
    failures = []
    n_t1 = n_t2 = 0
    for i, r in enumerate(rows):
        T, fm, fp = r["true"], r["Fminus"], r["Fplus"]
        reasons = []
        if not fm < T < fp:
            reasons.append("order")
        if not (n_digits(fm) == n_digits(T) == n_digits(fp)):
            reasons.append("digit_count")
        if not (is_round(fm) == is_round(T) == is_round(fp)):
            reasons.append("roundness")
        if not ((fm < 0) == (T < 0) == (fp < 0)):
            reasons.append("sign")
        if tokcount_fn is not None:
            tk = tokcount_fn(str(T))
            if not (tokcount_fn(str(fm)) == tk == tokcount_fn(str(fp))):
                reasons.append("tokcount")
            t1 = (_nearest(T, "minus", tokcount_fn, True) is not None
                  and _nearest(T, "plus", tokcount_fn, True) is not None)
        else:
            t1 = lead_feasible(T)
        if t1 and not (leading_digit(fm) == leading_digit(T)
                       == leading_digit(fp)):
            reasons.append("leading_digit")
        if "tok3" in r and not (r["tok3"][0] == r["tok3"][1]
                                == r["tok3"][2]):
            reasons.append("tok3")
        if t1:
            n_t1 += 1
        else:
            n_t2 += 1
        if reasons:
            failures.append((i, reasons))
    return {"n": len(rows), "n_bad": len(failures), "n_tier1": n_t1,
            "n_tier2": n_t2, "ok": not failures, "failures": failures}


# ----------------------------- SELFTEST ---------------------------------------

# 200 verified items per family; constraint_count's allocation leans on
# the cheap levels (a level-9 gen+verify walks a 200,000-wide range) but
# still covers every level.
_SELFTEST_ALLOC = {
    "tracking": [20] * 10,
    "program_exec": [20] * 10,
    "constraint_count": [35, 30, 28, 25, 22, 18, 15, 12, 9, 6],
    "arith_long": [20] * 10,
}


def _boundaries(kmax, hi):
    """Digit-cell edge neighborhoods up to kmax digits, clipped to
    [1, hi] -- infeasibility is a cell-edge property (derivation above),
    so exhaustively testing edges plus random interior samples covers a
    large range."""
    out = set()
    for k in range(2, kmax + 1):
        for d in range(-2, 13):
            out.add(10 ** (k - 1) + d)
        for d in range(-13, 3):
            out.add(10 ** k + d)
    return sorted(v for v in out if 1 <= v <= hi)


def selftest():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("m10_families selftest (M10 Stage -1 item 6)\n")

    # ---- registry coverage ---------------------------------------------------
    check(set(WHITELIST) | set(NONWHITELISTED)
          == set(m4.FAMILIES) | {"tracking"},
          "every m4 family plus tracking is classified")
    check(not set(WHITELIST) & set(NONWHITELISTED),
          "whitelist and non-whitelist are disjoint")
    check("logic_grid" in NONWHITELISTED
          and len(NONWHITELISTED["logic_grid"]) > 200,
          "logic_grid excluded with a substantive documented reason")

    # ---- surface helpers agree with m8 on its (positive) domain --------------
    check(all(n_digits(v) == m8.n_digits(v) and is_round(v) == m8.is_round(v)
              for v in range(1, 1000)),
          "n_digits/is_round match m8_force on all positive 1-3 digit values")

    # ---- infeasibility derivations -------------------------------------------
    got = infeasible_true_values("tracking", 0)
    check(got == sorted(m8.INFEASIBLE_T),
          f"tracking L0 infeasible set == m8's {{9,10,11}} (got {got})")
    brute = [t for t in range(-1500, 2501) if structurally_infeasible(t)]
    check(brute == infeasible_in(-1500, 2500),
          "closed-form infeasible_in matches the predicate on [-1500, 2500]")
    check(infeasible_true_values("program_exec", 0)
          == [0, 1, 9, 10, 11, 90, 99, 100, 101, 990, 999, 1000, 1001,
              9990, 9999],
          "program_exec infeasible set matches the 15-value derivation")
    check(not any(lead_feasible(t) for t in range(0, 10))
          and all(lead_feasible(t) for t in (12, 13, 14))
          and not lead_feasible(11) and not lead_feasible(19)
          and not lead_feasible(30) and lead_feasible(340)
          and not lead_feasible(300) and not lead_feasible(390),
          "leading-digit tier feasibility matches the derivation")

    # ---- generator vs verifier: 200 verified items per family ----------------
    for fam, alloc in _SELFTEST_ALLOC.items():
        t0 = time.time()
        ints_ok = metas_ok = True
        try:
            for level, cnt in enumerate(alloc):
                for i in range(cnt):
                    rng = m4.item_rng(7, "m10:" + fam, level, i)
                    body, true, meta = WHITELIST[fam]["generate"](level, rng)
                    ints_ok &= (isinstance(true, int)
                                and not isinstance(true, bool))
                    metas_ok &= (meta.get("family") == fam
                                 and meta.get("level") == level
                                 and "params" in meta)
            check(True, f"{fam}: generator-vs-verifier clean on "
                        f"{sum(alloc)} items [{time.time() - t0:.1f}s]")
        except VerifyError as e:
            check(False, f"{fam}: verifier mismatch: {e}")
        check(ints_ok, f"{fam}: every verified answer is a single int")
        check(metas_ok, f"{fam}: meta carries family/level/params")

    # ---- adapter determinism -------------------------------------------------
    det = True
    for fam in WHITELIST:
        a = WHITELIST[fam]["generate"](1, random.Random(11))
        b = WHITELIST[fam]["generate"](1, random.Random(11))
        det &= a[0] == b[0] and a[1] == b[1]
    check(det, "adapters deterministic given the rng state")

    # ---- the verifier-raise path is real -------------------------------------
    body, true, _ = gen_tracking_item(0, random.Random(3))
    check(verify_true("tracking", body, true),
          "tracking verifier confirms a fresh item")
    raised = 0
    for fam, bt in [("tracking", (body, true)),
                    ("constraint_count",
                     gen_constraint_count_item(2, random.Random(3))[:2]),
                    ("arith_long",
                     gen_arith_long_item(3, random.Random(3))[:2])]:
        try:
            _require_verified(fam, bt[0], bt[1] + 1, 0)
        except VerifyError:
            raised += 1
    check(raised == 3,
          "corrupted true values raise VerifyError (3/3 families probed)")

    # ---- foil properties over the full feasible range ------------------------
    tok_len = len          # stand-in tokenizer: len(str(v)); constant
    #                        within a (sign, digit-count) cell, so it is
    #                        non-binding and structural == actual feasibility

    def sweep(values, msg):
        rngf = random.Random(101)
        offs_cycle = [{}, {1: 3, 2: 2, 5: 1}]
        n_feas = n_inf = 0
        viol = []
        for j, t in enumerate(values):
            expect_raise = structurally_infeasible(t)
            try:
                fm, fp = pick_foils(t, offs_cycle[j % 2], tok_len, rngf)
            except FoilError:
                if expect_raise:
                    n_inf += 1
                else:
                    viol.append((t, "raised though structurally feasible"))
                continue
            if expect_raise:
                viol.append((t, "no raise though infeasible"))
                continue
            n_feas += 1
            if not fm < t < fp:
                viol.append((t, f"order {fm},{fp}"))
            if not (n_digits(fm) == n_digits(t) == n_digits(fp)):
                viol.append((t, "digit count"))
            if not (is_round(fm) == is_round(t) == is_round(fp)):
                viol.append((t, "roundness"))
            if not ((fm < 0) == (t < 0) == (fp < 0)):
                viol.append((t, "sign"))
            if lead_feasible(t) and not (
                    leading_digit(fm) == leading_digit(t)
                    == leading_digit(fp)):
                viol.append((t, "leading digit"))
        check(not viol, f"{msg} ({n_feas} feasible ok, {n_inf} infeasible "
                        f"raised{'' if not viol else '; bad: '}"
                        f"{viol[:3] if viol else ''})")

    sweep(range(0, 67),
          "tracking: exhaustive over [0, 66] (= L9 range, includes all "
          "lower levels)")
    sweep(range(0, 10000),
          "program_exec: exhaustive over the full [0, 9999] range")
    rs = random.Random(77)
    sweep(list(range(5, 2501)) + _boundaries(6, 200000)
          + [rs.randrange(5, 200001) for _ in range(300)],
          "constraint_count: exhaustive [5, 2500] + all cell edges to "
          "200000 + 300 interior samples")
    rs = random.Random(78)
    arith_vals = ([0] + [s * t for t in range(1, 2501) for s in (1, -1)]
                  + [s * b for b in _boundaries(8, 10 ** 7)
                     for s in (1, -1)]
                  + [rs.choice((1, -1)) * rs.randrange(1, 10 ** 7)
                     for _ in range(300)])
    sweep(arith_vals,
          "arith_long: signed exhaustive [-2500, 2500] + all cell edges "
          "to +/-1e7 + 300 interior samples")

    # ---- per-family infeasible sets all raise --------------------------------
    for fam, level in [("tracking", 0), ("tracking", 9),
                       ("program_exec", 0), ("constraint_count", 9),
                       ("arith_long", 0)]:
        vals = infeasible_true_values(fam, level)
        raised = 0
        for t in vals:
            try:
                pick_foils(t, {}, tok_len, random.Random(1))
            except FoilError:
                raised += 1
        check(raised == len(vals),
              f"{fam} L{level}: all {len(vals)} infeasible values raise "
              f"FoilError")

    # ---- tier / tokenizer / error-offset behavior ----------------------------
    r = random.Random(1)
    check(pick_foils(8, {2: 100}, tok_len, r) == (6, 9),
          "err-offset preference honored; nearest fallback on the "
          "offset-empty side (T=8 -> (6, 9), the m8 semantics)")
    check(pick_foils(-45, {}, tok_len, r) == (-46, -44),
          "negative T: sign-matched tier-1 pair (T=-45 -> (-46, -44))")
    fm, fp = pick_foils(12, {}, tok_len, r)
    check((fm, fp) == (11, 13) and pair_tier(12, fm, fp) == 1,
          "tier 1 binds where feasible (T=12 -> (11, 13), leads match)")
    fm, fp = pick_foils(5, {}, tok_len, r)
    check((fm, fp) == (4, 6) and pair_tier(5, fm, fp) == 2,
          "1-digit T falls back to tier 2 (T=5 -> (4, 6))")

    def tok4(s):
        return 2 if "4" in s else 1        # deliberately fragmenting fake

    check(pick_foils(5, {}, tok4, r) == (3, 6),
          "tokcount constraint binds (fragmented '4' skipped: "
          "T=5 -> (3, 6))")
    try:
        pick_foils(40, {}, tok4, r)
        check(False, "tokenizer-fragmented cell must raise")
    except FoilError:
        check(True, "tokenizer-induced empty cell raises FoilError "
                    "(T=40 under the fragmenting fake)")

    # ---- feasible-item rejection sampling ------------------------------------
    ctr = {}
    rngg = random.Random(5)
    vals = [gen_feasible_item("tracking", 0, rngg, ctr)[1]
            for _ in range(100)]
    check(all(v not in m8.INFEASIBLE_T for v in vals)
          and ctr.get("rejected", 0) >= 1,
          f"gen_feasible_item rejects infeasible T "
          f"({ctr.get('rejected', 0)} rejects across 100 draws)")

    # ---- foil audit ----------------------------------------------------------
    rows = []
    rnga = random.Random(31)
    for t in [3, 5, 12, 14, 25, 47, 60, 137, 250, 941, 4321, -45, -137,
              199998]:
        fm, fp = pick_foils(t, {1: 2, 3: 1}, tok_len, rnga)
        rows.append({"true": t, "Fminus": fm, "Fplus": fp,
                     "tok3": [tok_len(str(t)), tok_len(str(fm)),
                              tok_len(str(fp))]})
    rep = foil_audit(rows, tokcount_fn=tok_len)
    check(rep["ok"] and rep["n_bad"] == 0,
          f"foil audit clean on {rep['n']} generated rows "
          f"(tier1={rep['n_tier1']}, tier2={rep['n_tier2']})")
    check(foil_audit(rows)["ok"],
          "foil audit clean in structural mode (no tokcount_fn)")
    bad_rows = [
        {"true": 45, "Fminus": 47, "Fplus": 46},            # order
        {"true": 45, "Fminus": 9, "Fplus": 47},             # digit count
        {"true": 45, "Fminus": 40, "Fplus": 47},            # roundness
        {"true": 45, "Fminus": 32, "Fplus": 47},            # leading digit
        {"true": -45, "Fminus": -47, "Fplus": 44},          # sign
        {"true": 45, "Fminus": 44, "Fplus": 46,
         "tok3": [1, 2, 1]},                                # token count
    ]
    repb = foil_audit(bad_rows)
    check(repb["n_bad"] == len(bad_rows),
          f"foil audit flags all {len(bad_rows)} corrupted rows: "
          f"{[fl[1] for fl in repb['failures']]}")
    check(foil_audit([{"true": 19, "Fminus": 13, "Fplus": 27}])["ok"],
          "audit permits differing leads where tier 1 is infeasible "
          "(T=19, documented fallback)")

    print("\nselftest", "PASSED" if ok else "FAILED")
    return ok


# ----------------------------- MAIN -------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", type=int, default=-1,
                    help="print one item + foils per whitelisted family "
                         "at LEVEL")
    ap.add_argument("--infeasible", action="store_true",
                    help="print per-family answer ranges and infeasible "
                         "sets")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.infeasible:
        for fam in WHITELIST:
            for lvl in (0, 9):
                lo, hi = answer_range(fam, lvl)
                vals = infeasible_true_values(fam, lvl)
                print(f"{fam:17s} L{lvl}: answers [{lo}, {hi}], "
                      f"{len(vals)} infeasible: {vals}")
        sys.exit(0)
    if args.show >= 0:
        rng = random.Random(args.seed)
        for fam in WHITELIST:
            body, true, meta = gen_feasible_item(fam, args.show, rng)
            fm, fp = pick_foils(true, {}, len, random.Random(args.seed))
            print(f"===== {fam} L{args.show} (T={true}, F-={fm}, F+={fp}, "
                  f"tier={pair_tier(true, fm, fp)}) =====")
            print(body[:1200])
            print()
        sys.exit(0)
    ap.print_help()
