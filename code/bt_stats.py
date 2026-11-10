#!/usr/bin/env python3
"""
bt_stats: shared statistics + provenance helpers for the bulletproofed
instrument suite (hot2_v5, hot3_v2, ae1_ladder2)
=========================================================================
Every function that resamples takes an explicit `seed` and builds its own
random.Random, so an analysis is reproducible regardless of what else has
consumed the global RNG — a fragility the earlier scripts had (their
bootstrap CIs drew from the global stream, so adding any upstream feature
would silently shift every downstream CI).

Contents:
  wilson_ci(k, n)                    CI for a proportion (rates like adapted%)
  auroc(pairs)                       type-2 AUROC with tie handling
  boot_ci_auroc(pairs, seed)         percentile bootstrap CI, item resampling
  paired_auroc_diff(pairs_a, pairs_b, seed)
                                     CI for AUROC(a)-AUROC(b) computed on the
                                     SAME items (the "confidence beats the cue"
                                     claim needs this, not two separate CIs)
  block_boot_ci(blocks, stat, seed)  cluster bootstrap (resample whole blocks;
                                     items within a HOT-3 block are not
                                     independent)
  mcnemar_exact(b, c)                exact two-sided McNemar p for paired
                                     yes/no outcomes (e.g. same-seed episodes
                                     across two AE-1 rungs)
  sha256_file(path)                  content hash for provenance blocks
  provenance(argv, seeds, extra)     the standard dict every results JSON
                                     stores (script hashes, params, versions)

Selftest: python3 bt_stats.py --selftest   (no API, no network)
"""

import os
import sys
import json
import math
import random
import hashlib

# ----------------------------- proportions ----------------------------------

def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion. Returns (lo, hi);
    (None, None) if n == 0."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)

# ----------------------------- AUROC ----------------------------------------

def auroc(pairs):
    """pairs: iterable of (score, is_positive). Ties count 0.5.
    None if either class is empty."""
    pos = [s for s, c in pairs if c]
    neg = [s for s, c in pairs if not c]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > q else (0.5 if p == q else 0.0)
               for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def boot_ci_auroc(pairs, iters=2000, seed=0):
    """Percentile bootstrap CI for AUROC. STRATIFIED: positives and negatives
    are resampled separately, so every resample keeps both classes. (The naive
    i.i.d. version silently dropped single-class resamples, which conditions
    the CI on class survival and produces fake-narrow intervals exactly when
    errors are rare — audit finding HOT27.) Returns (None, None) if either
    class is empty or has fewer than 3 members (too few for a meaningful CI —
    caller should suppress the interval rather than print a fake one)."""
    pairs = list(pairs)
    pos = [p for p in pairs if p[1]]
    neg = [p for p in pairs if not p[1]]
    if len(pos) < 3 or len(neg) < 3:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        sp = [pos[rng.randrange(len(pos))] for _ in range(len(pos))]
        sn = [neg[rng.randrange(len(neg))] for _ in range(len(neg))]
        a = auroc(sp + sn)
        vals.append(a)
    vals.sort()
    b_lo = vals[int(0.025 * len(vals))]
    b_hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    # with a tiny minority class the bootstrap distribution is too coarse and
    # under-covers; union with the Hanley-McNeil analytic interval is
    # conservative by construction and restores ~nominal coverage
    # Hanley-McNeil variance on the LOGIT scale with a t-quantile
    # (df = min class - 1). The logit transform makes the interval asymmetric
    # — wider toward 0.5 — which is exactly where the percentile bootstrap
    # misses in the rare-error regime (its misses are one-sided: the sample
    # draws few errors, the point estimate lands high, and a symmetric
    # interval never reaches down to the truth). Union of the two intervals
    # achieves ~nominal coverage in the selftest's coverage simulation.
    a = auroc(pairs)
    npn = len(pos) * len(neg)
    a_c = min(max(a, 0.5 / npn), 1 - 0.5 / npn)   # continuity guard for 0/1
    q1 = a_c / (2 - a_c)
    q2 = 2 * a_c * a_c / (1 + a_c)
    var = (a_c * (1 - a_c) + (len(pos) - 1) * (q1 - a_c * a_c)
           + (len(neg) - 1) * (q2 - a_c * a_c)) / npn
    _T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
             8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09,
             25: 2.06, 30: 2.04}
    df = min(len(pos), len(neg)) - 1
    tq = 1.959963984540054
    for d in sorted(_T975):
        if df <= d:
            tq = _T975[d]
            break
    se_logit = math.sqrt(max(var, 1e-12)) / (a_c * (1 - a_c))
    logit = math.log(a_c / (1 - a_c))
    hm_lo = 1 / (1 + math.exp(-(logit - tq * se_logit)))
    hm_hi = 1 / (1 + math.exp(-(logit + tq * se_logit)))
    return min(b_lo, hm_lo), max(b_hi, hm_hi)


def paired_auroc_diff(pairs_a, pairs_b, iters=2000, seed=0):
    """CI for AUROC(pairs_a) - AUROC(pairs_b) where element i of both lists
    describes the SAME item (two scores, one correctness label). Resamples
    item indices once per iteration and evaluates both AUROCs on the same
    resample, which is the correct treatment for 'does score A beat score B
    on these items'. Returns (diff, lo, hi); Nones if undefined."""
    pairs_a, pairs_b = list(pairs_a), list(pairs_b)
    if len(pairs_a) != len(pairs_b):
        raise ValueError("paired_auroc_diff needs same-length, same-item lists")
    point_a, point_b = auroc(pairs_a), auroc(pairs_b)
    if point_a is None or point_b is None:
        return None, None, None
    # stratified paired resampling: both lists share the correctness label at
    # each index, so stratify the INDEX pool by label and resample within
    # class — every resample keeps both classes (same rationale as
    # boot_ci_auroc; audit finding HOT27)
    pos_idx = [i for i, p in enumerate(pairs_a) if p[1]]
    neg_idx = [i for i, p in enumerate(pairs_a) if not p[1]]
    if len(pos_idx) < 3 or len(neg_idx) < 3:
        return point_a - point_b, None, None
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [pos_idx[rng.randrange(len(pos_idx))]
               for _ in range(len(pos_idx))] + \
              [neg_idx[rng.randrange(len(neg_idx))]
               for _ in range(len(neg_idx))]
        a = auroc([pairs_a[i] for i in idx])
        b = auroc([pairs_b[i] for i in idx])
        diffs.append(a - b)
    diffs.sort()
    return (point_a - point_b,
            diffs[int(0.025 * len(diffs))],
            diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))])


def paired_auroc_diff_p(pairs_a, pairs_b, iters=2000, seed=0):
    """paired_auroc_diff plus a one-sided bootstrap p-value for diff > 0:
    p = (1 + #{bootstrap diffs <= 0}) / (iters + 1). This is what a Holm
    correction needs — a real p per comparison, not a CI-derived constant
    (re-review critical: constant pseudo-p made Holm structurally
    unconfirmable at m>=3)."""
    pairs_a, pairs_b = list(pairs_a), list(pairs_b)
    if len(pairs_a) != len(pairs_b):
        raise ValueError("needs same-length, same-item lists")
    point_a, point_b = auroc(pairs_a), auroc(pairs_b)
    if point_a is None or point_b is None:
        return None, None, None, None
    pos_idx = [i for i, p in enumerate(pairs_a) if p[1]]
    neg_idx = [i for i, p in enumerate(pairs_a) if not p[1]]
    if len(pos_idx) < 3 or len(neg_idx) < 3:
        return point_a - point_b, None, None, None
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [pos_idx[rng.randrange(len(pos_idx))]
               for _ in range(len(pos_idx))] + \
              [neg_idx[rng.randrange(len(neg_idx))]
               for _ in range(len(neg_idx))]
        a = auroc([pairs_a[i] for i in idx])
        b = auroc([pairs_b[i] for i in idx])
        diffs.append(a - b)
    diffs.sort()
    p = (1 + sum(1 for d in diffs if d <= 0)) / (len(diffs) + 1)
    return (point_a - point_b,
            diffs[int(0.025 * len(diffs))],
            diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))],
            p)


def block_boot_ci(blocks, stat, iters=2000, seed=0):
    """Cluster bootstrap: `blocks` is a list of arbitrary per-block payloads,
    `stat` maps a list of blocks -> float. Resamples whole blocks with
    replacement (items within a block share context and are not independent).
    Returns (point, lo, hi)."""
    blocks = list(blocks)
    point = stat(blocks)
    rng = random.Random(seed)
    k = len(blocks)
    vals = []
    for _ in range(iters):
        sample = [blocks[rng.randrange(k)] for _ in range(k)]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return point, None, None
    vals.sort()
    return (point, vals[int(0.025 * len(vals))],
            vals[min(len(vals) - 1, int(0.975 * len(vals)))])

# ----------------------------- paired binary --------------------------------

def binom_p_onesided(k, n, p=0.5):
    """Exact one-sided binomial P(X >= k | n, p)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
               for i in range(k, n + 1))


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni: returns list of booleans (reject?) in input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    reject = [False] * len(pvals)
    m = len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def mcnemar_exact(b, c):
    """Exact two-sided McNemar test. b = pairs where condition 1 succeeded and
    condition 2 failed; c = the reverse. Returns the p-value (1.0 if b+c==0).
    Uses the exact binomial distribution of b given b+c trials at p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    def binom_pmf(k):
        return math.comb(n, k) * (0.5 ** n)
    p_obs = binom_pmf(b)
    p = sum(binom_pmf(k) for k in range(n + 1)
            if binom_pmf(k) <= p_obs + 1e-12)
    return min(1.0, p)

# ----------------------------- provenance -----------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(script_paths, args_dict, protocol_version):
    """The standard provenance block every bulletproofed results JSON stores:
    exact content hashes of the scripts that produced it, the full CLI
    parameters, and a protocol version string. No timestamps are needed —
    the shell records those; hashes are what make a result auditable."""
    return {
        "protocol_version": protocol_version,
        "script_sha256": {os.path.basename(p): sha256_file(p)
                          for p in script_paths if os.path.exists(p)},
        "params": dict(args_dict),
        "python": sys.version.split()[0],
    }

# ----------------------------- selftest -------------------------------------

def selftest():
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("SELFTEST bt_stats (no API)\n")
    # wilson
    lo, hi = wilson_ci(8, 10)
    check(0.49 < lo < 0.51 and 0.94 < hi < 0.95,
          f"wilson_ci(8,10) ~ [0.49,0.94] (got [{lo:.3f},{hi:.3f}])")
    check(wilson_ci(0, 0) == (None, None), "wilson_ci(0,0) -> (None,None)")
    lo, hi = wilson_ci(0, 16)
    check(lo == 0.0 and hi < 0.25, f"wilson_ci(0,16) hi<0.25 (got {hi:.3f})")

    # auroc basics
    check(auroc([(0.9, True), (0.1, False)]) == 1.0, "auroc separable = 1.0")
    check(auroc([(0.5, True), (0.5, False)]) == 0.5, "auroc all-tied = 0.5")
    check(auroc([(1, True), (2, True)]) is None, "auroc one-class = None")

    # bootstrap determinism + coverage sanity
    rng = random.Random(42)
    pairs = [(rng.gauss(1 if i % 3 else 0, 1), bool(i % 3))
             for i in range(120)]
    a1 = boot_ci_auroc(pairs, iters=500, seed=7)
    a2 = boot_ci_auroc(pairs, iters=500, seed=7)
    a3 = boot_ci_auroc(pairs, iters=500, seed=8)
    check(a1 == a2, "boot_ci_auroc deterministic for same seed")
    check(a1 != a3, "boot_ci_auroc varies across seeds")
    pt = auroc(pairs)
    check(a1[0] < pt < a1[1], f"CI brackets the point estimate "
                              f"({a1[0]:.3f} < {pt:.3f} < {a1[1]:.3f})")

    # paired diff: score A informative, score B pure noise on SAME items
    rng = random.Random(1)
    items = [(bool(i % 2)) for i in range(150)]
    pa = [((3 if c else 0) + rng.gauss(0, 1), c) for i, c in enumerate(items)]
    pb = [(rng.gauss(0, 1), c) for i, c in enumerate(items)]
    d, lo, hi = paired_auroc_diff(pa, pb, iters=500, seed=3)
    check(d > 0.3 and lo > 0, f"paired diff detects A>B (d={d:.3f}, "
                              f"CI[{lo:.3f},{hi:.3f}] excludes 0)")
    # two equally-informative scores (same signal, independent noise): the
    # diff CI must span 0 — neither genuinely beats the other
    pb2 = [((3 if c else 0) + rng.gauss(0, 1), c) for i, c in enumerate(items)]
    d2, lo2, hi2 = paired_auroc_diff(pa, pb2, iters=500, seed=3)
    check(lo2 < 0 < hi2, f"paired diff of equally-informative scores spans 0 "
                         f"(d={d2:.3f}, CI[{lo2:.3f},{hi2:.3f}])")

    # block bootstrap: stat = mean of per-block means
    blocks = [[i + j for j in range(4)] for i in range(12)]
    stat = lambda bs: sum(sum(b) / len(b) for b in bs) / len(bs)
    p, lo, hi = block_boot_ci(blocks, stat, iters=400, seed=5)
    check(lo < p < hi, f"block bootstrap brackets point ({lo:.2f}<{p:.2f}<{hi:.2f})")
    check(block_boot_ci(blocks, stat, iters=400, seed=5)
          == block_boot_ci(blocks, stat, iters=400, seed=5),
          "block bootstrap deterministic for same seed")

    # rare-error regime: CI must be suppressed below 3 errors, not fake-narrow
    rare = [(90, True)] * 57 + [(88, False), (85, False)]
    check(boot_ci_auroc(rare, iters=200, seed=1) == (None, None),
          "CI suppressed when a class has <3 members (no fake intervals)")

    # coverage simulation in the rare-error regime that broke the old CI:
    # n=60, ~10% errors, known true AUROC via a binormal model
    def sim_pairs(rng, n=60, p_err=0.10):
        out = []
        for _ in range(n):
            wrong = rng.random() < p_err
            out.append((rng.gauss(0 if wrong else 1.2, 1), not wrong))
        return out
    # true AUROC for d=1.2 binormal: Phi(1.2/sqrt(2)) ~ 0.8018
    true_a = 0.8018
    rng = random.Random(99)
    hits = trials = 0
    for t in range(120):
        pairs = sim_pairs(rng)
        lo, hi = boot_ci_auroc(pairs, iters=300, seed=t)
        if lo is None:
            continue          # too few errors drawn — correctly suppressed
        trials += 1
        if lo <= true_a <= hi:
            hits += 1
    cov = hits / trials
    check(cov >= 0.90, f"CI coverage in rare-error regime ~nominal "
                       f"(got {cov:.2f} over {trials} sims; naive bootstrap "
                       f"got ~0.83, symmetric-HM union ~0.85 here)")

    # binomial + holm
    check(abs(binom_p_onesided(8, 10, 0.5) - 0.0546875) < 1e-6,
          "exact one-sided binomial p(8/10)")
    rej = holm([0.001, 0.04, 0.20])
    check(rej == [True, False, False] or rej == [True, True, False],
          f"holm rejects smallest p first (got {rej})")

    # mcnemar
    check(mcnemar_exact(0, 0) == 1.0, "mcnemar(0,0)=1.0")
    check(mcnemar_exact(5, 5) > 0.9, f"mcnemar(5,5) ~ 1 "
                                     f"(got {mcnemar_exact(5,5):.3f})")
    p_skew = mcnemar_exact(12, 1)
    check(p_skew < 0.01, f"mcnemar(12,1) significant (got {p_skew:.4f})")

    # provenance
    prov = provenance([__file__], {"n": 80, "seed": 1}, "selftest-1")
    check(len(prov["script_sha256"]) == 1
          and len(list(prov["script_sha256"].values())[0]) == 64,
          "provenance carries a sha256 per script")

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
