#!/usr/bin/env python3
"""
M10 Stage 4: the frozen-prereg analysis, executed locally on the
pulled pod artifacts. Statistics reuse m8_force's battle-tested
machinery (auroc, OOF grouped CV, pair slices, item-clustered
bootstrap) verbatim. Implements m10_prereg.md (frozen, read-only):
  ARM B P1: (a) within-item AND (c) beats-concat-baseline AND
            (b) matched-answer (>=30 pairs; est 11,706) at L39
            pos-ANS; L35/L41 secondaries; cross-sign descriptive.
  ARM C P2 (PINNED branch): differential margin S at alpha=+4 vs 5
            matched-norm random nulls, item-clustered bootstrap,
            -4 sign flip, both doses in-regime; crack rate counts
            ANY sampled report != 100 (unparseable included), with
            unparsed rates reported separately.
  LENS: pod-computed curves (m10_lens_curves.json), descriptive.
"""

import json
import math
import os
import random
import sys
import tarfile
import time
from collections import Counter

import numpy as np

import m8_force as m8

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "m10_data")
PRIMARY_L = 39
SECONDARY_LS = [35, 41]
N_BOOT = 1000
ALPHAS = [-8, -4, -1, 0, 1, 4, 8]
N_RAND = 5
PARSE_GATE = 0.90


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(DIR, "m10_run.log"), "a",
              encoding="utf-8") as f:
        f.write(line + "\n")


def extract_pull():
    tgz = os.path.join(DIR, "m10_pod_pull.tar.gz")
    os.makedirs(DATA, exist_ok=True)
    with tarfile.open(tgz) as t:
        t.extractall(DATA)
    n = sum(len(fs) for _, _, fs in os.walk(DATA))
    log(f"extracted {n} files to m10_data/")


def jsonl(path):
    return [json.loads(x) for x in
            open(path, encoding="utf-8") if x.strip()]


def v4_tokcount():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V4-Flash",
        revision="60d8d70770c6776ff598c94bb586a859a38244f1")
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


def leading_digit(v):
    return int(str(abs(int(v)))[0])


def analyze():
    res = {"mission": "M10", "model": "DeepSeek-V4-Flash",
           "revision": "60d8d70", "prereg": "m10_prereg.md (frozen)",
           "n_items": 400}
    ck = os.path.join(DATA, "checkpoints")
    tokcount = v4_tokcount()

    # ---------------- ARM A (descriptive) ----------------
    arm_a = json.load(open(os.path.join(DATA, "m10_arm_a.json"),
                           encoding="utf-8"))
    res["arm_a"] = arm_a["summary"]
    res["arm_a"]["diag_battery"] = json.load(
        open(os.path.join(DATA, "m10_diag_battery.json"),
             encoding="utf-8"))
    own_by_item = {r["item"]: r for r in arm_a["rows"]}
    log(f"ARM A: {json.dumps(res['arm_a'] | {'diag_battery': '...'})}")

    # ---------------- ARM B (P1) ----------------
    rows_b = jsonl(os.path.join(ck, "arm_b.jsonl"))
    for r in rows_b:
        r["forced_str"] = str(r["forced"])
        r["correct"] = bool(r["correct"])
    y = np.array([1 if r["correct"] else 0 for r in rows_b])
    groups = np.array([r["item"] for r in rows_b])
    item_ids = sorted(set(groups))

    def X_of(layer, pos):
        return np.stack([
            np.load(os.path.join(ck, f"b_{r['item']}_{r['cond']}.npz")
                    )[f"L{layer}_{pos}"].astype(np.float64)
            for r in rows_b])

    surf = np.array([[float(r["forced"]), abs(float(r["forced"])),
                      float(len(str(abs(int(r["forced"]))))),
                      1.0 if int(r["forced"]) % 10 == 0 else 0.0,
                      float(tokcount(str(r["forced"])))]
                     for r in rows_b])
    logp = np.array([[r["forced_logp"]] for r in rows_b])

    res["arm_b"] = {}
    for L in [PRIMARY_L] + SECONDARY_LS:
        X = X_of(L, "ans")
        s, folds = m8.oof_scores(X, y, groups, seed=1)
        assert m8.leakage_check(groups, folds)
        out = {"overall_auc": m8.auroc(s, y)}
        stat_a = m8.slice_within_item(s, rows_b)
        pa = stat_a(item_ids)
        lo, hi, _ = m8.cluster_boot(stat_a, item_ids, seed=2)
        out["a_within_item"] = {"auc": pa, "ci": [lo, hi],
                                "pass": lo is not None and lo > 0.5}
        stat_b = m8.slice_matched_answer(s, rows_b)
        pb = stat_b(item_ids)
        lob, hib, _ = m8.cluster_boot(stat_b, item_ids, seed=3)
        out["b_matched_answer"] = {"auc": pb, "ci": [lob, hib],
                                   "pass": lob is not None
                                   and lob > 0.5}
        Xbase = np.column_stack([X_of(L, "pre"), surf, logp])
        sb, _ = m8.oof_scores(Xbase, y, groups, seed=1)
        base_auc = m8.auroc(sb, y)
        idx_of = {}
        for i, r in enumerate(rows_b):
            idx_of.setdefault(r["item"], []).append(i)

        def diff_stat(ims, s=s, sb=sb, idx_of=idx_of):
            take = np.array([i for it in ims for i in idx_of[it]])
            a1, a2 = m8.auroc(s[take], y[take]), m8.auroc(sb[take],
                                                          y[take])
            return None if a1 is None or a2 is None else a1 - a2
        _, _, diffs = m8.cluster_boot(diff_stat, item_ids, seed=4)
        p_c = (sum(1 for d in diffs if d <= 0) + 1) / (len(diffs) + 1)
        out["c_vs_concat_baseline"] = {
            "probe_auc": out["overall_auc"], "baseline_auc": base_auc,
            "p_one_sided": p_c, "pass": p_c < 0.05}
        d1 = m8.cross_sign_transfer(X, rows_b, groups, "Fminus",
                                    "Fplus", seed=5)
        d2 = m8.cross_sign_transfer(X, rows_b, groups, "Fplus",
                                    "Fminus", seed=6)
        out["cross_sign_descriptive"] = {
            "train_Fminus_test_Fplus": d1, "train_Fplus_test_Fminus":
            d2}
        res["arm_b"][f"L{L}"] = out
        log(f"ARM B L{L}: a={pa:.3f} b={pb:.3f} "
            f"c: {out['overall_auc']:.3f} vs {base_auc:.3f} "
            f"(p={p_c:.4f})")

    prim = res["arm_b"][f"L{PRIMARY_L}"]
    n_pairs = sum(1 for _ in ())  # matched pairs measured inside slice
    p1 = (prim["a_within_item"]["pass"]
          and prim["c_vs_concat_baseline"]["pass"]
          and prim["b_matched_answer"]["pass"])
    res["arm_b"]["P1_pass"] = p1
    log(f"ARM B P1 (a AND c AND b at L{PRIMARY_L} pos-ANS): "
        f"{'PASS' if p1 else 'FAIL'}")

    # secondaries: pos-PRE null check + T-rows own-greedy control
    Xp = X_of(PRIMARY_L, "pre")
    sp, _ = m8.oof_scores(Xp, y, groups, seed=1)
    res["arm_b"]["pos_pre_auc"] = m8.auroc(sp, y)
    s39, _ = m8.oof_scores(X_of(PRIMARY_L, "ans"), y, groups, seed=1)
    tmask = np.array([r["cond"] == "T"
                      and own_by_item.get(r["item"], {}).get(
                          "own_correct") is not None
                      for r in rows_b])
    t_scores = s39[tmask]
    t_own = np.array([1 if own_by_item[r["item"]]["own_correct"]
                      else 0 for r in rows_b
                      if r["cond"] == "T"
                      and own_by_item.get(r["item"], {}).get(
                          "own_correct") is not None])
    res["arm_b"]["T_rows_own_greedy_auc"] = m8.auroc(t_scores, t_own)
    # rep-position descriptive
    sr, _ = m8.oof_scores(X_of(PRIMARY_L, "rep"), y, groups, seed=1)
    res["arm_b"]["rep_position_auc"] = m8.auroc(sr, y)
    log(f"ARM B secondaries: pos-PRE {res['arm_b']['pos_pre_auc']:.3f}"
        f", T-own {res['arm_b']['T_rows_own_greedy_auc']:.3f}, "
        f"rep-pos {res['arm_b']['rep_position_auc']:.3f}")

    # ---------------- ARM C (P2, PINNED branch) ----------------
    rows_c = jsonl(os.path.join(ck, "arm_c.jsonl"))
    regime = {}
    for r in rows_c:
        regime.setdefault(f"{r['dir']}_a{r['alpha']}", []).append(
            r["sampled"] is not None)
    res["arm_c"] = {"parse_rates": {k: float(np.mean(v))
                                    for k, v in sorted(regime.items())}}
    res["arm_c"]["in_regime"] = {k: v >= PARSE_GATE for k, v in
                                 res["arm_c"]["parse_rates"].items()}

    marg = {}
    for r in rows_c:
        marg.setdefault(r["dir"], {}).setdefault(
            r["alpha"], {}).setdefault(r["cond"], {})[r["item"]] = \
            r["margin"]
    items_c = sorted({r["item"] for r in rows_c})

    def S(dname, alpha, ims):
        b = marg["probe"][0]
        s = marg[dname][alpha]
        dF = [s["F"][i] - b["F"][i] for i in ims]
        dT = [s["T"][i] - b["T"][i] for i in ims]
        return float(np.mean(dF) - np.mean(dT))

    def boot(fn, seed):
        rs = np.random.RandomState(seed)
        arr = np.array(items_c)
        return [fn(list(arr[rs.randint(0, len(arr), len(arr))]))
                for _ in range(N_BOOT)]

    S_probe = S("probe", 4, items_c)
    S_rand = {f"rand{i+1}": S(f"rand{i+1}", 4, items_c)
              for i in range(N_RAND)}
    boot_c = boot(lambda ims: S("probe", 4, ims)
                  - np.mean([S(f"rand{i+1}", 4, ims)
                             for i in range(N_RAND)]), 21)
    p_primary = (sum(1 for v in boot_c if v >= 0) + 1) / (N_BOOT + 1)
    rank_ok = all(S_probe < v for v in S_rand.values())
    S_neg = S("probe", -4, items_c)
    boot_n = boot(lambda ims: S("probe", -4, ims), 22)
    flip_p = (sum(1 for v in boot_n if v <= 0) + 1) / (N_BOOT + 1)
    S_rand_neg = {f"rand{i+1}": S(f"rand{i+1}", -4, items_c)
                  for i in range(N_RAND)}
    ok4 = res["arm_c"]["in_regime"].get("probe_a4", False)
    okn4 = res["arm_c"]["in_regime"].get("probe_a-4", False)
    p2 = (ok4 and okn4 and S_probe < 0 and p_primary < 0.05
          and rank_ok and S_neg > 0 and flip_p < 0.05)
    res["arm_c"]["P2_pinned"] = {
        "S_probe_alpha4": S_probe, "S_random": S_rand,
        "p_vs_random": p_primary, "below_every_random": rank_ok,
        "S_probe_alpha_minus4": S_neg, "flip_p": flip_p,
        "S_random_alpha_minus4": S_rand_neg,
        "in_regime_a4": ok4, "in_regime_a_minus4": okn4,
        "pass": p2}
    log(f"ARM C P2: S(+4)={S_probe:+.4f} vs rand "
        f"{np.mean(list(S_rand.values())):+.4f} (p={p_primary:.4f}, "
        f"below-all={rank_ok}); S(-4)={S_neg:+.4f} "
        f"(flip p={flip_p:.4f}) -> {'PASS' if p2 else 'FAIL'}")

    res["arm_c"]["dose_curve"] = {str(a): S("probe", a, items_c)
                                  for a in ALPHAS}
    crack, unparsed, crack_rows = {}, {}, []
    for r in rows_c:
        key = f"{r['dir']}_a{r['alpha']}_{r['cond']}"
        is_crack = r["sampled"] != 100
        crack.setdefault(key, []).append(is_crack)
        unparsed.setdefault(key, []).append(r["sampled"] is None)
        if is_crack:
            crack_rows.append({k: r[k] for k in
                               ["item", "cond", "dir", "alpha",
                                "sampled_raw", "sampled", "margin"]})
    res["arm_c"]["crack_rates"] = {k: float(np.mean(v))
                                   for k, v in sorted(crack.items())}
    res["arm_c"]["unparsed_rates"] = {
        k: float(np.mean(v)) for k, v in sorted(unparsed.items())
        if np.mean(v) > 0}
    res["arm_c"]["crack_rows"] = crack_rows
    res["arm_c"]["n_cracks"] = len(crack_rows)
    log(f"ARM C cracks: {len(crack_rows)} of {len(rows_c)} rows")

    def abs_shift(cond, alpha, ims):
        b, s = marg["probe"][0][cond], marg["probe"][alpha][cond]
        return float(np.mean([s[i] - b[i] for i in ims]))
    res["arm_c"]["absolute_shifts"] = {}
    for cond in ["T", "F"]:
        for a in [4, -4]:
            bv = boot(lambda ims, c=cond, aa=a: abs_shift(c, aa, ims),
                      30 + a + (0 if cond == "T" else 1))
            res["arm_c"]["absolute_shifts"][f"{cond}_a{a}"] = {
                "mean": abs_shift(cond, a, items_c),
                "ci": [float(np.percentile(bv, 2.5)),
                       float(np.percentile(bv, 97.5))]}

    # ---------------- LENS (descriptive, pod-computed) -------------
    lens_path = os.path.join(DATA, "m10_lens_curves.json")
    if os.path.exists(lens_path):
        res["lens"] = json.load(open(lens_path, encoding="utf-8"))
    else:
        res["lens"] = {"status": "pod file missing"}

    p = m8.fresh_path(os.path.join(DIR, "m10_results.json"))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    log(f"saved {p}")
    return True


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract_pull()
    sys.exit(0 if analyze() else 1)
