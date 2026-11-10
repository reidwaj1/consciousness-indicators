"""Recompute every cell of Tables 5 to 13 of the paper from the result files in data/
and compare it with the table as printed (tables/paper_tables.md).

Run from anywhere:   python3 code/check_tables.py

Needs only the Python standard library. Bootstrap intervals use the seeds that the
analysis functions were given when the paper's tables were made, so they reproduce
exactly. The permutation p-values of Table 13 are Monte Carlo estimates; they are
printed in the paper as "< 0.001" when below that value.
"""
import sys, os, io, json, re, contextlib, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code")); sys.dont_write_bytecode = True
from statistics import mean, pstdev
import bt_stats as bs
import hot2_v5 as H
import m4_study as M4
import hot3_v2 as H3


def find(name):
    """Locate a result file anywhere under data/."""
    hits = glob.glob(os.path.join(ROOT, "data", "**", name), recursive=True)
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: found {len(hits)} copies under data/")
    return hits[0]


md = open(os.path.join(ROOT, "tables", "paper_tables.md"), encoding="utf-8").read()

def table_after(caption_start):
    i = md.index(caption_start)
    lines = md[i:].split("\n")
    rows = []; started = False
    for ln in lines[1:]:
        if ln.startswith("|"):
            started = True
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "): continue
            rows.append(cells)
        elif started: break
    return rows[0], rows[1:]

errors = []
def check(label, got, want):
    if got != want:
        errors.append(f"{label}: paper {got!r} != recomputed {want!r}")

def sgn(x, nd=3): return (f"+{x:.{nd}f}" if x >= 0 else f"−{abs(x):.{nd}f}")
def neg(x, nd=1): return (f"{x:.{nd}f}" if x >= 0 else f"−{abs(x):.{nd}f}")
NAME = {"tracking":"tracking","letters_fixed":"letters","arith_standard":"arithmetic","arith_madeop":"invented-operation","walk":"walk"}
CUE = {"length":"length","magnitude":"magnitude","op_mix":"operation count","roundness":"roundness"}
V5 = {"claude-sonnet-5":"hot2v5_claude-sonnet-5.json","claude-opus-4-8":"hot2v5_claude-opus-4-8.json","gpt-5.6-luna":"hot2v5_gpt-5.6-luna.json","grok-4.20-non-reasoning":"hot2v5_grok-4.20-0309-non-reasoning.json","gemini-3.5-flash":"hot2v5_gemini-3.5-flash.json","grok-4.3":"hot2v5_grok-4.3.json"}
FAMS = list(NAME)
def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): r = fn(*a, **k)
    return r
data = {m: json.load(open(find(f))) for m, f in V5.items()}
stats = {m: {f: quiet(H.analyze_family, f, data[m]["families"][f], seed=71) for f in FAMS if data[m]["families"].get(f)} for m in data}

def conf_rows(m, f): return [r for r in data[m]["families"][f]["rows"] if r.get("confidence") is not None]
def cue_auc(m, f, cue):
    rows = conf_rows(m, f); sc = H.cue_battery(f, rows)[cue]
    return bs.auroc(list(zip(sc, [r["correct"] for r in rows])))

# ---- Table 5
hdr, rows = table_after("*Table 5.")
for cells in rows:
    m, typ = cells[0], cells[1]; f = [k for k, v in NAME.items() if v == typ][0]
    rc = conf_rows(m, f); pr = stats[m][f]["primary"]
    acc = mean(1.0 if r["correct"] else 0.0 for r in rc)
    a_conf = bs.auroc([(r["confidence"], r["correct"]) for r in rc])
    check(f"T5 {m} {typ} acc", cells[2], f"{acc:.2f}")
    check(f"T5 {m} {typ} conf", cells[3], f"{a_conf:.3f}")
    check(f"T5 {m} {typ} cue", cells[4], f"{CUE[pr['best_cue']]} ({cue_auc(m, f, pr['best_cue']):.3f})")
    check(f"T5 {m} {typ} diff", cells[5], f"{sgn(pr['diff'])} [{sgn(pr['lo'])}, {sgn(pr['hi'])}]")
print("Table 5 rows:", len(rows))
# ---- Table 6
hdr, rows = table_after("*Table 6.")
for cells in rows:
    m, typ = cells[0], cells[1]; f = [k for k, v in NAME.items() if v == typ][0]
    rc = conf_rows(m, f)
    cc = [r["confidence"] for r in rc if r["correct"]]; ci = [r["confidence"] for r in rc if not r["correct"]]
    check(f"T6 {m} {typ} n", cells[2], str(len(rc)))
    check(f"T6 {m} {typ} correct", cells[3], f"{mean(cc):.1f} ± {pstdev(cc):.1f}")
    check(f"T6 {m} {typ} incorrect", cells[4], f"{mean(ci):.1f} ± {pstdev(ci):.1f}")
print("Table 6 rows:", len(rows))
# ---- Table 7
hdr, rows = table_after("*Table 7.")
for cells in rows:
    m, typ = cells[0], cells[1]; f = [k for k, v in NAME.items() if v == typ][0]
    allr = data[m]["families"][f]["rows"]; rc = conf_rows(m, f); pr = stats[m][f]["primary"]
    check(f"T7 {m} {typ} usable", cells[2], f"{len(rc)} of {len(allr)}")
    check(f"T7 {m} {typ} acc", cells[3], f"{mean(1.0 if r['correct'] else 0.0 for r in rc):.2f}")
    check(f"T7 {m} {typ} meanconf", cells[4], f"{mean(r['confidence'] for r in rc):.1f}")
    check(f"T7 {m} {typ} conf", cells[5], f"{bs.auroc([(r['confidence'], r['correct']) for r in rc]):.3f}")
    check(f"T7 {m} {typ} cue", cells[6], f"{CUE[pr['best_cue']]} ({cue_auc(m, f, pr['best_cue']):.3f})")
    want = f"{sgn(pr['diff'])} [{sgn(pr['lo'])}, {sgn(pr['hi'])}]" if pr["lo"] is not None else f"{sgn(pr['diff'])} (two errors, no interval)"
    check(f"T7 {m} {typ} diff", cells[7], want)
print("Table 7 rows:", len(rows))
# ---- Table 8
hdr, rows = table_after("*Table 8.")
for cells in rows:
    m, typ = cells[0], cells[1]; f = [k for k, v in NAME.items() if v == typ][0]
    fam = data[m]["families"][f]
    e = [r for r in fam["twin_easy"] if r.get("confidence") is not None]; h = [r for r in fam["twin_hard"] if r.get("confidence") is not None]
    ae = mean(1.0 if r["correct"] else 0.0 for r in e); ah = mean(1.0 if r["correct"] else 0.0 for r in h)
    ce = mean(r["confidence"] for r in e); ch = mean(r["confidence"] for r in h)
    inv = stats[m][f]["invariance"]; lo, hi = inv["d_conf_ci"]
    check(f"T8 {m} {typ} acc", cells[2], f"{ae:.2f} to {ah:.2f}")
    check(f"T8 {m} {typ} conf", cells[3], f"{ce:.1f} to {ch:.1f}")
    check(f"T8 {m} {typ} fall", cells[4], f"{neg(inv['d_conf'])} [{neg(lo)}, {neg(hi)}]")
    if len(e) != 30 or len(h) != 30: print(f"   note: {m} {typ}: {len(e)} and {len(h)} usable ratings in the two sets of 30")
print("Table 8 rows:", len(rows))
# ---- Table 9 (joint)
J = {"claude-sonnet-5":"bergtest_claude-sonnet-5.json","claude-opus-4-8":"bergtest_claude-opus-4-8.json","gemini-3.5-flash":"bergtest_gemini-3.5-flash.json","gpt-5.6-luna":"bergtest_gpt-5.6-luna.json","gpt-5.6-sol":"bergtest_gpt-5.6-sol.json","grok-4.20-non-reasoning":"bergtest_grok-4.20-0309-non-reasoning.json","grok-4.3":"bergtest_grok-4.3.json"}
hdr, rows = table_after("*Table 9.")
for cells in rows:
    d = json.load(open(find(J[cells[0]])))
    allr = []
    for j, k in enumerate(("arith_standard", "arith_mod", "arith_madeop")):
        rs = d["rows"][k]; allr += rs
        assert len(rs) == 100
        check(f"T9 {cells[0]} {k}", cells[1 + j], f"{mean(1.0 if r['correct'] else 0.0 for r in rs):.2f} / {mean(r['confidence'] for r in rs):.1f}")
    sh = 100 * mean(1.0 if r["confidence"] >= 90 else 0.0 for r in allr)
    check(f"T9 {cells[0]} share", cells[4], "100%" if sh == 100 else f"{sh:.1f}%")
print("Table 9 rows:", len(rows))
# ---- Table 10 (reasoning round)
ARMS = {"grok-4.3":"m4main_grok.json","gpt-5.6-sol, low effort":"m4main_sol-low.json","gpt-5.6-sol, high effort":"m4main_sol-high.json"}
TN = {"logic grid":"logic_grid","program":"program_exec","constraint count":"constraint_count","long arithmetic":"arith_long"}
CUE4 = {"magnitude":"magnitude","param:cc_digitish":"digit conditions","param:cc_square":"square condition","param:al_muls":"multiplications","length":"length"}
hdr, rows = table_after("*Table 10.")
m4 = {a: json.load(open(find(f))) for a, f in ARMS.items()}
for cells in rows:
    arm, typ = cells[0], cells[1]; out = m4[arm]
    fam = out["families"][TN[typ]]
    frows = fam.get("rows") or fam.get("main")
    o = quiet(M4.analyze_family_rows, TN[typ], frows, seed=out.get("seed", 0))
    pr = o["primary"]
    check(f"T10 {arm} {typ} n", cells[2], str(len([r for r in frows if r['confidence'] is not None and r['pred'] is not None])))
    check(f"T10 {arm} {typ} acc", cells[3], f"{o['acc']:.2f}")
    check(f"T10 {arm} {typ} meanconf", cells[4], f"{o['mean_conf']:.1f}")
    check(f"T10 {arm} {typ} conf", cells[5], f"{o['auroc_conf']:.3f}")
    check(f"T10 {arm} {typ} cue", cells[6], f"{CUE4[pr['best_cue']]} ({pr['best_cue_auroc']:.3f})")
    want = f"{sgn(pr['diff'])} [{sgn(pr['lo'])}, {sgn(pr['hi'])}]" if pr["lo"] is not None else f"{sgn(pr['diff'])}"
    check(f"T10 {arm} {typ} diff", cells[7], want)
    check(f"T10 {arm} {typ} cons", cells[8], f"{o['consistency']['auroc']:.3f}")
    check(f"T10 {arm} {typ} bet", cells[9], f"{o['bets']['auroc_bet']:.3f}")
print("Table 10 rows:", len(rows), "| seeds used:", {a: m4[a].get("seed") for a in m4})
# ---- Table 11
hdr, rows = table_after("*Table 11.")
for cells in rows:
    m, typ = cells[0], cells[1]; f = [k for k, v in NAME.items() if v == typ][0]
    allr = data[m]["families"][f]["rows"]; rc = conf_rows(m, f); rb = [r for r in allr if r.get("bet")]
    rk = [r for r in allr if r.get("consistency") is not None]
    keep = [r for r in rb if r["bet"] == "KEEP"]; pas = [r for r in rb if r["bet"] == "PASS"]
    acc = lambda rs: f"{sum(r['correct'] for r in rs)/len(rs):.2f}" if rs else "–"
    check(f"T11 {m} {typ} conf", cells[2], f"{bs.auroc([(r['confidence'], r['correct']) for r in rc]):.3f}")
    check(f"T11 {m} {typ} cons", cells[3], f"{bs.auroc([(r['consistency'], r['correct']) for r in rk]):.3f}")
    check(f"T11 {m} {typ} bet", cells[4], f"{bs.auroc([(1 if r['bet']=='KEEP' else 0, r['correct']) for r in rb]):.3f}")
    check(f"T11 {m} {typ} keep", cells[5], acc(keep)); check(f"T11 {m} {typ} pass", cells[6], acc(pas))
print("Table 11 rows:", len(rows), "| bet n range:", min(len([r for r in data[m]['families'][f]['rows'] if r.get('bet')]) for m in ('claude-sonnet-5','claude-opus-4-8') for f in FAMS), max(len([r for r in data[m]['families'][f]['rows'] if r.get('bet')]) for m in ('claude-sonnet-5','claude-opus-4-8') for f in FAMS))
# ---- Table 12 (AE-1)
AE = {"claude-sonnet-5":"ae1ladder2_claude-sonnet-5.json","claude-opus-4-8":"ae1ladder2_claude-opus-4-8.json","gpt-5.6-sol":"ae1ladder2_gpt-5.6-sol.json","gpt-5.6-luna":"ae1ladder2_gpt-5.6-luna.json","grok-4.20-non-reasoning":"ae1ladder2_grok-4.20-0309-non-reasoning.json"}
ae = {m: json.load(open(find(f))) for m, f in AE.items()}
RN = {"control":"control","static low budget":"static_hard","stockout":"stockout","silent cut":"budget_cut","announced cut":"budget_cut_announced","enforced cut":"budget_cut_enforced"}
hdr, rows = table_after("*Table 12.")
for cells in rows:
    cond = cells[0]
    for j, m in enumerate(hdr[1:]):
        if cond.startswith("Reported a change"):
            r = RN[cond.split(", ")[1]]
            sc = ae[m]["scores"][r]
            want = f"{sum(1 for s in sc if s['probe'] and s['probe']['changed'])}/{len(sc)}"
        else:
            r = RN[cond]; sc = ae[m]["scores"][r]
            if r in ("control", "static_hard"):
                want = f"{sum(1 for s in sc if s['valid'] and s['utility_ratio'] >= 0.75)}/{len(sc)}"
            else:
                den = [s for s in sc if s["adapted"] is not None]
                want = f"{sum(1 for s in den if s['adapted'])}/{len(den)}"
        check(f"T12 {cond} {m}", cells[1 + j], want)
print("Table 12 rows:", len(rows))
# ---- Table 13 (HOT-3)
hdr, rows = table_after("*Table 13.")
ARM = {"ratings shown":("", "shown"), "ratings shuffled":("", "permuted"), "ratings withheld":("_run2", "hidden")}
for cells in rows:
    run, arm = ARM[cells[0]]; m = cells[1]
    d = json.load(open(find(f"hot3v2_{m}{run}.json"))); bd = d["blocks_data"]
    blocks = [b for b in bd if b["arms"].get(arm)]
    obs, p = H3.perm_test_caught(bd, arm, seed=d.get("seed", 55))
    obs0, p0 = H3.perm_test_caught(bd, arm, seed=0)
    targ = H3.targeting_auroc(bd, arm); cfm = H3.cf_match_rate(bd, arm)
    nw = sum(1 for b in blocks for t in b["tentative"] if t["wrong"])
    import random
    rngb = random.Random(d.get("seed", 55) + 17); base = {}
    for kind in ("oracle", "confidence", "random", "never", "position", "domain_quota"):
        base[kind] = mean(H3.baseline_score(kind, b["tentative"], H3.BUDGET, rng=rngb) for b in blocks)
    score = mean(b["arms"][arm]["score"] for b in blocks)
    check(f"T13 {cells[0]} {m} blocks", cells[2], str(len(blocks)))
    check(f"T13 {cells[0]} {m} wrong", cells[3], f"{obs} of {nw}")
    check(f"T13 {cells[0]} {m} auroc", cells[4], f"{targ:.3f}")
    pfmt = lambda p: ("< 0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}"))
    if cells[5] not in (pfmt(p), pfmt(p0)):
        errors.append(f"T13 {cells[0]} {m} p: paper {cells[5]!r}; recomputed {p:.5f} (seed {d.get('seed', 55)}), {p0:.5f} (seed 0)")
    check(f"T13 {cells[0]} {m} match", cells[6], "–" if arm == "hidden" else f"{100*cfm:.0f}%")
    want = f"{score:.2f} / {base['random']:.2f} / {base['confidence']:.2f} / {base['oracle']:.2f}"
    if cells[7] != want:
        errors.append(f"T13 {cells[0]} {m} scores: paper {cells[7]!r} != recomputed {want!r}")
print("Table 13 rows:", len(rows))
print("\nMISMATCHES:" if errors else "\nALL TABLE CELLS MATCH THE RECOMPUTED VALUES")
for e in errors: print("  ", e)
