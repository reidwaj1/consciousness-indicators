"""Incremental-information analysis (Section 3.1 of the paper).

Question: does a model's stated confidence, or its KEEP/PASS bet, help to predict
its errors once the surface cues are already known?

For each pairing of model and problem type, two logistic regressions predicting
error are compared by five-fold stratified cross-validated AUROC: one on the
surface cues alone, one on the cues plus the channel. The gain is tested by
shuffling the channel across problems.

    python3 code/incremental_information.py              # 300 permutations per cell
    python3 code/incremental_information.py --perms 3000 # slower, steadier p-values
    python3 code/incremental_information.py --perms 0    # AUROC values only

Needs numpy and scikit-learn.

The script that first ran this analysis was not kept. This re-implementation
reproduces every saved AUROC value in data/incremental_information/ to three
decimals. Permutation p-values depend on the random shuffles and so differ a
little from the saved ones and from run to run. The follow-up with three further
cues for the tracking problems uses this script's own definitions of those cues,
which may differ from the original ones, so its figures are close to but not
identical with the saved ones (see the README).
"""
import argparse, glob, json, os, re, sys, warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code")); sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import hot2_v5 as H


def find(name):
    hits = glob.glob(os.path.join(ROOT, "data", "**", name), recursive=True)
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: found {len(hits)} copies under data/")
    return hits[0]


def cv_auc(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def gain(X, channel, y, perms, rng):
    """AUROC of cues, AUROC of cues + channel, and the permutation p-value of the gain."""
    a0 = cv_auc(X, y)
    a1 = cv_auc(np.hstack([X, channel]), y)
    if not perms:
        return a0, a1, None
    ge = sum(cv_auc(np.hstack([X, rng.permutation(channel)]), y) - a0 >= a1 - a0 - 1e-12
             for _ in range(perms))
    return a0, a1, (ge + 1) / (perms + 1)


def main_protocol_cell(model, family, channel):
    """Rows, error labels, surface cues and the channel for one pairing of the main protocol."""
    rows = json.load(open(find(f"hot2v5_claude-{model}.json")))["families"][family]["rows"]
    rows = [r for r in rows if (r.get("confidence") is not None if channel == "confidence" else r.get("bet"))]
    y = np.array([0 if r["correct"] else 1 for r in rows])
    cues = [[abs(r["pred"]) for r in rows],                       # magnitude of the answer
            [len(r["body"]) for r in rows],                       # length of the problem text
            [1.0 if r["pred"] % 10 == 0 else 0.0 for r in rows]]  # roundness of the answer
    ops = [H._op_mix_feature(family, r["body"]) for r in rows]    # operation count, where defined
    if all(v is not None for v in ops):
        cues.append([-v for v in ops])
    ch = [[r["confidence"]] if channel == "confidence" else [1.0 if r["bet"] == "KEEP" else 0.0] for r in rows]
    return rows, y, np.array(cues, float).T, np.array(ch, float)


def reasoning_round_cell(arm, family):
    f = {"grok": "m4main_grok.json", "sol-low": "m4main_sol-low.json"}[arm]
    rows = json.load(open(find(f)))["families"][family]["rows"]
    rows = [r for r in rows if r["confidence"] is not None and r["pred"] is not None]
    y = np.array([0 if r["correct"] else 1 for r in rows])
    cues = [[r["body_len"] for r in rows], [abs(r["pred"]) for r in rows],
            [len(str(abs(r["pred"]))) for r in rows], [1.0 if r["pred"] % 10 == 0 else 0.0 for r in rows]]
    for k in sorted({k for r in rows for k in r["params"]}):      # generator parameters that vary
        vals = [float(r["params"].get(k, 0)) for r in rows]
        if len(set(vals)) > 1:
            cues.append(vals)
    return rows, y, np.array(cues, float).T, np.array([[r["confidence"]] for r in rows], float)


def box_cues(body):
    """How often the box asked about is mentioned, swapped and moved in a tracking problem."""
    box = re.search(r"end up in the (\w+) box", body).group(1)
    ops = [s.strip() for s in re.split(r"(?<=[.?])\s+", body) if s.strip().startswith(("Move", "Swap", "Add", "Remove"))]
    named = lambda s: re.search(r"\b" + box + r"\b", s.lower()) is not None
    return [sum(len(re.findall(r"\b" + box + r"\b", s.lower())) for s in ops),
            sum(1 for s in ops if s.startswith("Swap") and named(s)),
            sum(1 for s in ops if s.startswith("Move") and named(s))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=300, help="permutations per comparison (0 = none)")
    ap.add_argument("--seed", type=int, default=7, help="seed for the permutations")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    saved = {}
    for f in ("posthoc_incremental_v5.json", "posthoc_incremental_m4.json"):
        saved.update(json.load(open(find(f))))

    print(f"{'comparison':40s} {'n':>3s}  cues -> cues+channel   gain      p      saved AUROCs       saved p")
    mismatches = 0
    for key, rec in saved.items():
        parts = key.split("/")
        if parts[0] == "v5":
            rows, y, X, ch = main_protocol_cell(parts[1], parts[2], parts[3])
        else:
            rows, y, X, ch = reasoning_round_cell(parts[1], parts[2])
        a0, a1, p = gain(X, ch, y, args.perms, rng)
        s1 = rec.get("auc_cues_plus", rec.get("auc_plus"))
        same = round(a0, 3) == rec["auc_cues"] and round(a1, 3) == s1
        mismatches += not same
        ptxt = "   -  " if p is None else f"{p:6.3f}"
        print(f"{key:40s} {len(rows):3d}  {a0:.3f} -> {a1:.3f}        {a1 - a0:+.3f}  {ptxt}   "
              f"{rec['auc_cues']:.3f} -> {s1:.3f}   {rec['p_perm']:.3f}  {'' if same else 'AUROC DIFFERS FROM SAVED'}")
    print(f"\n{len(saved)} comparisons; saved AUROC values reproduced in {len(saved) - mismatches}.")

    print("\nFollow-up: claude-sonnet-5 on tracking, with three further cues for the box asked about")
    for channel in ("confidence", "bet"):
        rows, y, X, ch = main_protocol_cell("sonnet-5", "tracking", channel)
        X2 = np.hstack([X, np.array([box_cues(r["body"]) for r in rows], float)])
        a0, a1, p = gain(X2, ch, y, args.perms, rng)
        ptxt = "" if p is None else f", p = {p:.3f}"
        print(f"  {channel:10s} cues {a0:.3f} -> with channel {a1:.3f} (gain {a1 - a0:+.3f}{ptxt})")
    sv = json.load(open(find("posthoc_churn_resolution.json")))["churn_cue_reanalysis"]
    print(f"  saved result for confidence: {sv['cues_auc_with_churn']:.3f} -> {sv['cues_plus_conf']:.3f} "
          f"(gain {sv['delta']:+.3f}, p = {sv['p_perm']:.3f})")


if __name__ == "__main__":
    main()
