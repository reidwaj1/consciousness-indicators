#!/usr/bin/env python3
"""
M10B: belief or truth? Local analysis over M10's frozen captures.
Prereg: m10b_prereg.md (frozen, hashed). $0, no pod, no model runs.

Design: fit the probe ONLY on model-RIGHT items (belief == truth
there), freeze it, apply to model-WRONG items (belief != truth) and
ask which one the within-item discrimination follows.
"""

import json
import os
import sys
import time
from collections import Counter

import numpy as np

import m8_force as m8

DIR = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(DIR, "m10_data", "checkpoints")
PRIMARY_L = 39
ALL_LS = [7, 13, 20, 26, 35, 39, 41]
N_BOOT = 1000


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(DIR, "m10b_run.log"), "a",
              encoding="utf-8") as f:
        f.write(line + "\n")


def jsonl(path):
    return [json.loads(x) for x in open(path, encoding="utf-8")
            if x.strip()]


def fit_frozen_probe(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000, C=0.1).fit(
        sc.transform(X), y)
    return lambda Xn: lr.decision_function(sc.transform(Xn))


def within_item_stat(scores, rows):
    return m8.slice_within_item(scores, rows)


def main():
    res = {"mission": "M10B", "prereg": "m10b_prereg.md (frozen)"}
    rows = jsonl(os.path.join(CK, "arm_b.jsonl"))
    for r in rows:
        r["forced_str"] = str(r["forced"])
        r["correct"] = bool(r["correct"])
    arm_a = json.load(open(os.path.join(DIR, "m10_data",
                                        "m10_arm_a.json"),
                           encoding="utf-8"))
    own = {r["item"]: r for r in arm_a["rows"]}

    right_items = sorted(i for i, r in own.items()
                         if r["own_correct"] is True)
    wrong_items = sorted(i for i, r in own.items()
                         if r["own_correct"] is False)
    res["n_model_right"] = len(right_items)
    res["n_model_wrong"] = len(wrong_items)
    log(f"split: {len(right_items)} model-right / "
        f"{len(wrong_items)} model-wrong items")

    def load_X(rws, layer, pos):
        return np.stack([
            np.load(os.path.join(CK, f"b_{r['item']}_{r['cond']}.npz")
                    )[f"L{layer}_{pos}"].astype(np.float64)
            for r in rws])

    # ---------- selftests ----------
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    # 1. reload/alignment spot-check on 10 rows
    import random as _rnd
    spot = _rnd.Random(3).sample(rows, 10)
    good = 0
    for r in spot:
        p = os.path.join(CK, f"b_{r['item']}_{r['cond']}.npz")
        z = np.load(p)
        v = z[f"L{PRIMARY_L}_ans"]
        good += int(v.shape == (16384,) and np.isfinite(
            v.astype(np.float64)).all()
            and r["pos_rep"] == r["seq_len"] - 1
            and r["pos_ans"] == r["ans_span"][1] - 1)
    check(good == 10, f"reload/alignment spot-check ({good}/10)")

    # 2+3. regression gate: reproduce M10's within-item on ALL items
    y_all = np.array([1 if r["correct"] else 0 for r in rows])
    groups = np.array([r["item"] for r in rows])
    X_all = load_X(rows, PRIMARY_L, "ans")
    s_all, folds = m8.oof_scores(X_all, y_all, groups, seed=1)
    check(m8.leakage_check(groups, folds), "grouped CV leakage check")
    stat_all = within_item_stat(s_all, rows)
    wi_all = stat_all(sorted(set(groups)))
    frozen_ref = json.load(open(os.path.join(
        DIR, "m10_results_run2.json"), encoding="utf-8"))[
        "arm_b"][f"L{PRIMARY_L}"]["a_within_item"]["auc"]
    check(abs(wi_all - frozen_ref) < 0.01,
          f"M10 within-item reproduces ({wi_all:.4f} vs "
          f"{frozen_ref:.4f})")
    y_shuf = y_all.copy()
    np.random.RandomState(9).shuffle(y_shuf)
    s_shuf, _ = m8.oof_scores(X_all, y_shuf, groups, seed=1)
    a_shuf = m8.auroc(s_shuf, y_shuf)
    check(abs(a_shuf - 0.5) < 0.06,
          f"shuffled-label null ({a_shuf:.3f})")
    if not ok:
        log("SELFTESTS RED — stopping")
        return False
    res["selftests"] = "green"

    # ---------- primary + secondaries ----------
    rows_right = [r for r in rows if r["item"] in set(right_items)]
    rows_wrong = [r for r in rows if r["item"] in set(wrong_items)]
    y_right = np.array([1 if r["correct"] else 0 for r in rows_right])

    def primary_at(layer, pos):
        Xr = load_X(rows_right, layer, pos)
        Xw = load_X(rows_wrong, layer, pos)
        probe = fit_frozen_probe(Xr, y_right)
        sw = probe(Xw)
        stat_w = within_item_stat(sw, rows_wrong)
        point = stat_w(wrong_items)
        lo, hi, _ = m8.cluster_boot(stat_w, wrong_items, seed=11)
        # held-out model-right comparison via grouped OOF inside right
        gr = np.array([r["item"] for r in rows_right])
        sr, _ = m8.oof_scores(Xr, y_right, gr, seed=1)
        stat_r = within_item_stat(sr, rows_right)
        point_r = stat_r(right_items)
        lor, hir, _ = m8.cluster_boot(stat_r, right_items, seed=12)
        return {"wrong_auc": point, "wrong_ci": [lo, hi],
                "right_auc": point_r, "right_ci": [lor, hir],
                "scores_wrong": sw}

    prim = primary_at(PRIMARY_L, "ans")
    sw = prim.pop("scores_wrong")
    res["primary_L39_ans"] = prim

    w, lo = prim["wrong_auc"], prim["wrong_ci"][0]
    if w >= 0.85 and lo > 0.5:
        verdict = "TRUTH-TRACKING"
    elif w <= 0.65 or lo <= 0.5:
        verdict = ("BELIEF-TRACKING (inverted)" if w < 0.5
                   else "BELIEF-TRACKING (collapse)")
    else:
        verdict = "MIXED"
    res["verdict"] = verdict
    log(f"PRIMARY: wrong-items within-item "
        f"{prim['wrong_auc']:.4f} {prim['wrong_ci']} vs right-items "
        f"{prim['right_auc']:.4f} {prim['right_ci']} -> {verdict}")

    # secondary 1: truth-AUC vs belief-AUC on model-wrong rows
    y_truth = np.array([1 if r["correct"] else 0 for r in rows_wrong])
    y_belief = np.array([1 if r["forced"] == own[r["item"]]["own_pred"]
                         else 0 for r in rows_wrong])
    res["secondary_belief_vs_truth"] = {
        "truth_auc": m8.auroc(sw, y_truth),
        "belief_auc": m8.auroc(sw, y_belief) if 0 < y_belief.sum()
        < len(y_belief) else None,
        "n_belief_positive_rows": int(y_belief.sum())}
    log(f"S1: truth-AUC "
        f"{res['secondary_belief_vs_truth']['truth_auc']:.4f} vs "
        f"belief-AUC {res['secondary_belief_vs_truth']['belief_auc']}"
        f" (belief-positive rows: {int(y_belief.sum())})")

    # secondary 2: foil == belief purest cells
    pure = [i for i, r in enumerate(rows_wrong)
            if r["cond"] != "T"
            and r["forced"] == own[r["item"]]["own_pred"]]
    res["secondary_pure_cells"] = {"n": len(pure)}
    if len(pure) >= 1:
        pure_items = {rows_wrong[i]["item"] for i in pure}
        t_idx = [i for i, r in enumerate(rows_wrong)
                 if r["cond"] == "T" and r["item"] in pure_items]
        res["secondary_pure_cells"].update({
            "mean_score_belief_foils": float(np.mean(sw[pure])),
            "mean_score_matched_T": float(np.mean(sw[t_idx])),
            "frac_T_above_belief_foil": float(np.mean([
                sw[t] > sw[p] for p in pure for t in t_idx
                if rows_wrong[t]["item"] == rows_wrong[p]["item"]]))})
    log(f"S2 pure cells: {res['secondary_pure_cells']}")

    # secondary 3: per-layer sweep
    res["secondary_layer_sweep"] = {}
    for L in ALL_LS:
        p = primary_at(L, "ans")
        p.pop("scores_wrong")
        res["secondary_layer_sweep"][f"L{L}"] = p
        log(f"S3 L{L}: wrong {p['wrong_auc']:.4f} "
            f"right {p['right_auc']:.4f}")

    # secondary 4: pos-REP repeat
    p = primary_at(PRIMARY_L, "rep")
    p.pop("scores_wrong")
    res["secondary_pos_rep"] = p
    log(f"S4 pos-REP: wrong {p['wrong_auc']:.4f} "
        f"right {p['right_auc']:.4f}")

    res["confound_note"] = ("model-wrong items are harder; the "
                            "primary is within-item (same item, T vs "
                            "foil) which difficulty cannot explain; "
                            "cross-item statistics carry this caveat")

    p = m8.fresh_path(os.path.join(DIR, "m10b_results.json"))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    log(f"saved {p}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
