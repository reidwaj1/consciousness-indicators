#!/usr/bin/env python3
"""
M10 pod-side stages: G0 environment gates, Stage-1 calibration, Arms
A/B/C, and on-pod analysis. Runs ON THE POD (Linux) inside tmux with a
per-stage `timeout` wrapper; every stage writes per-item JSONL
checkpoints under checkpoints/ and supports --resume-from.

Usage (pod):
  python m10_stages.py --g0
  python m10_stages.py --calibrate
  python m10_stages.py --arm-a [--resume-from checkpoints/arm_a.jsonl]
  python m10_stages.py --arm-b [--resume-from ...]
  python m10_stages.py --arm-c [--resume-from ...]
  python m10_stages.py --analyze

Every random choice seeded from M10_SEED. Strict parsing. Results
never overwritten. Budget logging is the laptop driver's job.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np

import m8_force as m8
import m10_families as fam
from m10_pipeline import (DEPTH_FRACTIONS, Handle, ReportLayout,
                          SteererRow, load_handle, log, M10_SEED,
                          SPECS)

N_MAIN = 400
N_LADDER = [400, 300, 250]
CAL_SEED = M10_SEED + 5000        # disjoint from Stage-3 seeds
MAIN_SEED = M10_SEED              # item seed for the arms
ALPHAS = [-8, -4, -1, 0, 1, 4, 8]
N_RAND = 5
RAND_ALPHAS = [4, -4]
PARSE_GATE = 0.90
PIN_MODAL_FRAC = 0.95
CKPT_DIR = "checkpoints"
G0_FILE = "m10_g0.json"
CAL_FILE = "m10_calibration.json"

os.makedirs(CKPT_DIR, exist_ok=True)


def _jsonable(x, path="", offenders=None):
    """Coerce torch/numpy scalars that leak into checkpoint rows;
    record where they came from so the leak gets named in the log."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if offenders is not None:
                offenders.append(path)
            return x.item() if x.numel() == 1 else x.tolist()
    except ImportError:
        pass
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        if offenders is not None:
            offenders.append(path)
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v, f"{path}.{k}", offenders)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, f"{path}[{i}]", offenders)
                for i, v in enumerate(x)]
    return x


def ckpt_append(name, obj):
    offenders = []
    obj = _jsonable(obj, name, offenders)
    if offenders:
        log(f"ckpt sanitizer coerced tensor/array fields: "
            f"{offenders[:5]}")
    with open(os.path.join(CKPT_DIR, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def ckpt_load(path):
    if not path or not os.path.exists(path):
        return []
    return [json.loads(x) for x in
            open(path, encoding="utf-8") if x.strip()]


_H = {}


def handle():
    if "h" not in _H:
        _H["h"] = load_handle("v4flash")
    return _H["h"]


# ----------------------------- G0 GATES ---------------------------------------

def noise_floor(h, built, k=5):
    """k repeated identical no-hook forwards; returns (max |dlogit|,
    {(L,pos): max L2 dh}) relative to the first pass."""
    ref_f, ref_lp, ref_logits = h.capture(built)
    max_dlogit = 0.0
    dh = {key: 0.0 for key in ref_f}
    for _ in range(k - 1):
        f2, lp2, logits2 = h.capture(built)
        max_dlogit = max(max_dlogit, float(
            np.max(np.abs(logits2.numpy() - ref_logits.numpy()))))
        for key in ref_f:
            dh[key] = max(dh[key], float(np.linalg.norm(
                f2[key] - ref_f[key])))
    return max_dlogit, dh


def g0():
    log("M10 G0: pod environment gates (DeepSeek-V4-Flash)")
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    import torch
    h = handle()
    check(torch.cuda.is_available(), "CUDA available")
    check(h.n_layers == 43 and h.hidden == 4096,
          f"architecture as pinned ({h.n_layers} layers, {h.hidden})")
    log(f"  layers {h.layers}, primary L{h.primary_layer}")

    txt = h.greedy(h.ids_of(h.template_text(
        [{"role": "user", "content": "Say the word hello."}])), 8)
    check(len(txt.strip()) > 0, f"generation coherent ({txt!r})")

    # 3. noise-floor protocol
    body, true, _ = m8.gen_tracking(0, random.Random(77))
    built = h.build_forced(body, str(true))
    nf_logit, nf_h = noise_floor(h, built, k=5)
    worst = max(nf_h.values())
    log(f"  noise floor: max |dlogit| {nf_logit:.3e}, worst per-layer "
        f"L2 dh {worst:.3e}")
    g0_out = {"noise_floor_logit": nf_logit,
              "noise_floor_h": {f"{L}_{p}": v
                                for (L, p), v in nf_h.items()}}
    # greedy stability 20/20 exact
    stable = all(h.greedy(built["ids3"], 6)
                 == h.greedy(built["ids3"], 6) for _ in range(10))
    check(stable, "greedy answer stability (20 comparisons exact)")

    # 4. dual-path capture cross-check
    caps = {}
    hooks = []

    def mk(L):
        def fn(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            caps[L] = hs[0].detach()
        return fn
    for L in h.layers:
        hooks.append(h.block_for_hidden_index(L)
                     .register_forward_hook(mk(L)))
    out = h.forward(built["ids3"], hidden=True)
    for hk in hooks:
        hk.remove()
    dual_ok = True
    for L in h.layers:
        for pos_name, pos in [("pre", built["pos_pre"]),
                              ("ans", built["pos_ans"]),
                              ("rep", built["pos_rep"])]:
            a = caps[L][pos].float().cpu().numpy()
            b = out.hidden_states[L][0, pos].float().cpu().numpy()
            d = float(np.linalg.norm(a - b))
            lim = max(2 * nf_h[(L, pos_name)], 1e-6)
            if d > lim:
                dual_ok = False
                log(f"  dual-path mismatch L{L}/{pos_name}: {d:.3e} > "
                    f"{lim:.3e}")
    check(dual_ok, "dual-path capture: block hooks == hidden_states "
                   "within 2x noise floor")
    # lens final-layer identity
    hs_last = out.hidden_states[-1][0, built["pos_rep"]]
    with torch.no_grad():
        lens = h.model.lm_head(hs_last.to(
            h.model.lm_head.weight.device).to(
            h.model.lm_head.weight.dtype))[0] \
            if hs_last.dim() > 1 else h.model.lm_head(
            hs_last.to(h.model.lm_head.weight.device).to(
                h.model.lm_head.weight.dtype))
    real = out.logits[0, built["pos_rep"]].float()
    cos = float(torch.nn.functional.cosine_similarity(
        lens.float().cpu().flatten(), real.cpu().flatten(), dim=0))
    lens_mode = "raw_last" if cos > 0.999 else "norm_all"
    if cos <= 0.999:
        with torch.no_grad():
            normed = h.model.model.norm(hs_last)
            lens2 = h.model.lm_head(normed).float().cpu()
        cos = float(torch.nn.functional.cosine_similarity(
            lens2.flatten(), real.cpu().flatten(), dim=0))
    check(cos > 0.999, f"lens final-layer identity (mode {lens_mode}, "
                       f"cos {cos:.5f})")
    h.lens_mode = lens_mode
    g0_out["lens_mode"] = lens_mode

    # 5. injection propagation
    L = h.primary_layer
    steer = SteererRow(h, L)
    # feat_dim covers multi-stream residuals (streams detected during
    # the noise-floor captures above)
    vec = torch.ones(h.feat_dim, dtype=next(
        h.model.parameters()).dtype)
    starts = torch.tensor([built["rep_start"]])
    seqs = torch.tensor([len(built["ids3"])])
    big = 1000.0
    steer.configure(vec, big, starts, seqs)
    steer.set_offset(0)
    out_s = h.forward(built["ids3"], hidden=True)
    delta = (out_s.hidden_states[L][0, built["pos_rep"]].reshape(-1)
             .float().cpu().numpy()
             - out.hidden_states[L][0, built["pos_rep"]].reshape(-1)
             .float().cpu().numpy())
    check(np.abs(delta - big).max() < 0.2 * big,
          f"injection visible at hidden_states[{L}] "
          f"(max dev {np.abs(delta - big).max():.1f})")
    prop_ok = True
    for L2 in [x for x in h.layers if x > L] + [h.n_layers]:
        d = float(np.linalg.norm(
            out_s.hidden_states[L2][0, built["pos_rep"]].float()
            .cpu().numpy()
            - out.hidden_states[L2][0, built["pos_rep"]].float()
            .cpu().numpy()))
        ref = nf_h.get((L2, "rep"), max(nf_h.values()))
        if d <= 2 * ref:
            prop_ok = False
    dl = float(np.max(np.abs(
        out_s.logits[0, built["pos_rep"]].float().cpu().numpy()
        - out.logits[0, built["pos_rep"]].float().cpu().numpy())))
    check(prop_ok and dl > 2 * nf_logit,
          f"injection propagates to deeper layers + final logits "
          f"(dlogit {dl:.3e})")
    tgt = int(h.ids_of("banana")[0])
    u = h.model.lm_head.weight[tgt].detach().float()
    u = (u / u.norm()).cpu()
    if (h.streams or 1) > 1:
        # inject the unembedding direction into every stream; the
        # hc_head blend carries it to the single output stream
        u = u.repeat(h.streams) / (h.streams ** 0.5)
    u = u.to(next(h.model.parameters()).dtype)
    steer.configure(u, 4000.0, starts, seqs)
    steer.set_offset(0)
    out_u = h.forward(built["ids3"])
    lp0 = float(torch.log_softmax(
        out.logits[0, built["pos_rep"]].float(), -1)[tgt])
    lp1 = float(torch.log_softmax(
        out_u.logits[0, built["pos_rep"]].float(), -1)[tgt])
    check(lp1 > lp0 + 1.0, f"planted unembedding direction "
                           f"({lp0:.2f} -> {lp1:.2f})")
    changed = 0
    rngp = random.Random(88)
    for _ in range(3):
        b2, t2, _ = m8.gen_tracking(0, rngp)
        bu = h.build_forced(b2, str(t2))
        steer.configure(vec, 50000.0,
                        torch.tensor([bu["rep_start"]]),
                        torch.tensor([len(bu["ids3"])]))
        g1 = h.greedy(bu["ids3"], 6, steer=steer)
        steer.off()
        g0_txt = h.greedy(bu["ids3"], 6)
        changed += int(g1 != g0_txt)
    check(changed >= 1, f"extreme-scale injection changes greedy "
                        f"output on {changed}/3 probes")
    steer.remove()

    # 6. template/thinking gate
    tmpl_ok = 0
    rngt = random.Random(99)
    for _ in range(5):
        b3, t3, _ = m8.gen_tracking(0, rngt)
        bu = h.build_forced(b3, str(t3))
        free = h.greedy(bu["ids3"], 20)
        first_line = free.strip().split("\n")[0]
        if m8.parse_conf_strict(first_line) is not None:
            tmpl_ok += 1
        else:
            log(f"  template gate sample: {free!r}")
    check(tmpl_ok == 5, f"free-generation starts with parseable "
                        f"report 5/5 (got {tmpl_ok})")
    check(h.template_id_consistency(
        [{"role": "user", "content": "hi"}]),
        "apply_chat_template(tokenize=True) == string-then-tokenize")

    # 7. batch gate — arms run UNBATCHED (batch=1); the applicable
    # part is cached-generation per-step logits vs full-forward
    ids3 = built["ids3"]
    with torch.no_grad():
        full = h.model(ids3.unsqueeze(0).to(h._dev_in()))
        base_next = int(full.logits[0, -1].argmax())
        o1 = h.model(ids3.unsqueeze(0).to(h._dev_in()),
                     use_cache=True)
        step_next = int(o1.logits[0, -1].argmax())
    check(base_next == step_next,
          "cached vs full-forward next-token agreement (unbatched "
          "arms; batch gate reduces to cache check, logged)")

    # 8. throughput pilot
    t0 = time.time()
    for _ in range(5):
        h.capture(built)
    per_capture = (time.time() - t0) / 5
    t0 = time.time()
    for _ in range(3):
        h.greedy(built["ids3"], 6)
    per_gen = (time.time() - t0) / 3
    g0_out.update({"s_per_capture": per_capture, "s_per_gen": per_gen})
    proj = {
        "arm_a_h": N_MAIN * (per_gen * 2) / 3600,
        "arm_b_h": N_MAIN * 3 * (per_capture * 2) / 3600,
        "arm_c_h": N_MAIN * 2 * 17 * (per_capture * 2 + per_gen)
                   / 3600}
    g0_out["projections_h"] = proj
    log(f"  throughput: capture {per_capture:.2f}s, gen {per_gen:.2f}s"
        f" -> projections {json.dumps(proj)}")

    with open(G0_FILE, "w", encoding="utf-8") as f:
        json.dump(g0_out, f, indent=1)
    log(f"G0 {'GREEN' if ok else 'RED'}")
    return ok


# ----------------------------- STAGE 1: CALIBRATION ---------------------------

def calibrate():
    """Answer-only difficulty walk: tracking ops ladder, then
    whitelisted m4 families. GATE: acc in [0.30, 0.80], parse >= 0.9."""
    log("M10 Stage 1: calibration (answer-only)")
    h = handle()
    rng = random.Random(CAL_SEED)
    probe_n = 24

    LADDER = {"tracking": [0, 1, 2, 3, 4, 6, 8],
              "program_exec": [0, 1, 2, 3, 4],
              "constraint_count": [0, 1, 2, 3, 4],
              "arith_long": [0, 1, 2, 3, 4]}

    def probe(family, level):
        n_ok = n_parse = n_right = 0
        pairs = Counter()
        for _ in range(probe_n):
            body, true, meta = fam.gen_feasible_item(
                family, level, rng, tokcount_fn=h.tokcount)
            ids = h.ids_of(h.template_text(
                [{"role": "user",
                  "content": m8.ANSWER_PROMPT.format(body=body)}]))
            raw = h.greedy(ids, 10)
            pred = m8.parse_int_strict(raw)
            n_ok += 1
            if pred is not None:
                n_parse += 1
                n_right += int(pred == true)
                pairs[str(true)] += 1
        acc = n_right / max(1, n_parse)
        parse = n_parse / n_ok
        log(f"  {family} L{level}: acc={acc:.2f} parse={parse:.2f}")
        return acc, parse

    plan = [("tracking", lv) for lv in LADDER["tracking"]]
    for f in fam.WHITELIST:
        if f != "tracking":
            plan += [(f, lv) for lv in LADDER[f]]
    landed = None
    for family, level in plan:
        acc, parse = probe(family, level)
        if parse >= PARSE_GATE and 0.30 <= acc <= 0.80:
            landed = {"family": family, "level": level, "acc": acc,
                      "parse": parse}
            break
        if acc < 0.30 and family == plan[-1][0]:
            break
    out = {"landed": landed, "probe_n": probe_n, "seed": CAL_SEED}
    with open(CAL_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    if landed:
        # matched-answer pair count estimate for the P1(b) branch:
        # draw the frozen main item list and count same-answer pairs
        items = build_main_items(landed["family"], landed["level"])
        vals = Counter(str(it["true"]) for it in items)
        est_pairs = sum(v * (v - 1) for v in vals.values())
        out["matched_pair_estimate"] = est_pairs
        with open(CAL_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        log(f"Stage 1 GREEN: landed {landed} "
            f"(matched-pair est {est_pairs})")
        return True
    log("Stage 1 RED: no family/level in band")
    return False


def build_main_items(family, level, n=N_MAIN):
    """The frozen item list: seed MAIN_SEED, feasible + unique bodies,
    foils per m10_families rules using M8's gemma error offsets as the
    prior (documented: V4's own error distribution replaces it only if
    the frozen prereg says so — it does not; offsets are a preference
    order, not a validity condition)."""
    h = handle()
    rng = random.Random(MAIN_SEED)
    st1 = json.load(open(m8.latest_path(m8.STAGE1_FILE),
                         encoding="utf-8"))
    err_offsets = {int(k): v for k, v in st1["err_offsets"].items()}
    frng = random.Random(MAIN_SEED * 7 + 3)
    items, seen = [], set()
    counter = {"rejected": 0, "dupes": 0, "foil_fail": 0,
               "cc_width_reject": 0}
    while len(items) < n:
        body, true, meta = fam.gen_feasible_item(
            family, level, rng, counter, h.tokcount)
        if body in seen:
            counter["dupes"] += 1
            continue
        try:
            fm, fp = fam.pick_foils(true, err_offsets, h.tokcount,
                                    frng)
        except fam.FoilError:
            counter["foil_fail"] += 1
            continue
        # constraint_count guard (m10_families open issue): a foil
        # above the maximum possible count is surface-detectable
        cc = meta.get("params", {}).get("cc_width")
        if cc is not None and fp > cc:
            counter["cc_width_reject"] += 1
            continue
        seen.add(body)
        items.append({"item": len(items), "body": body, "true": true,
                      "Fminus": fm, "Fplus": fp, "meta": meta})
    log(f"  items built: {len(items)} ({json.dumps(counter)})")
    return items


# ----------------------------- ARM A ------------------------------------------

def arm_a(resume=None):
    log("ARM A: live-reporter baseline (descriptive)")
    h = handle()
    cal = json.load(open(CAL_FILE, encoding="utf-8"))["landed"]
    items = build_main_items(cal["family"], cal["level"])
    done = {r["item"] for r in ckpt_load(resume)}
    rows = ckpt_load(resume)
    for it in items:
        if it["item"] in done:
            continue
        msgs = [{"role": "user",
                 "content": m8.ANSWER_PROMPT.format(body=it["body"])}]
        raw_a = h.greedy(h.ids_of(h.template_text(msgs)), 10)
        pred = m8.parse_int_strict(raw_a)
        conf_raw = None
        conf = None
        if pred is not None:
            msgs2 = msgs + [{"role": "assistant",
                             "content": raw_a.strip()},
                            {"role": "user", "content": m8.CONF_PROMPT}]
            conf_raw = h.greedy(h.ids_of(h.template_text(msgs2)), 8)
            conf = m8.parse_conf_strict(conf_raw)
        row = {"item": it["item"], "true": it["true"],
               "own_raw": raw_a, "own_pred": pred,
               "own_correct": (pred == it["true"])
               if pred is not None else None,
               "conf_raw": conf_raw, "conf": conf}
        rows.append(row)
        ckpt_append("arm_a.jsonl", row)
        if (it["item"] + 1) % 25 == 0:
            log(f"  arm A {it['item'] + 1}/{len(items)}")
    confs = [r["conf"] for r in rows if r["conf"] is not None]
    modal, modal_n = Counter(confs).most_common(1)[0] if confs \
        else (None, 0)
    pinned = bool(confs) and modal_n / len(confs) >= PIN_MODAL_FRAC
    out = {"n": len(rows), "conf_parsed": len(confs),
           "modal": modal, "modal_frac": modal_n / max(1, len(confs)),
           "pinned": pinned, "acc": float(np.mean(
               [r["own_correct"] for r in rows
                if r["own_correct"] is not None]))}
    with open("m10_arm_a.json", "w", encoding="utf-8") as f:
        json.dump({"summary": out, "rows": rows}, f, indent=1)
    log(f"ARM A done: {json.dumps(out)}")
    if pinned:
        log("PINNED reporter -> running 5-variant diagnostic battery")
        diag_battery(h, cal)
    return True


def diag_battery(h, cal):
    rng = random.Random(M10_SEED + 777)
    variants = {
        "A_verbatim": m8.CONF_PROMPT,
        "B_endpoint_90": m8.CONF_PROMPT.replace("0-100", "0-90")
        .replace("0 to 100", "0 to 90"),
        "C_inverted": ("On a scale from 0 to 100 where 0 means you "
                       "are CERTAIN your answer is correct and 100 "
                       "means you are CERTAIN it is wrong, how would "
                       "you rate your answer? Reply with ONLY an "
                       "integer from 0 to 100."),
        "D_impossible": ("How confident are you that your answer "
                         "equals the secret number I am thinking of "
                         "right now, on a 0-100 scale? Reply with "
                         "ONLY an integer from 0 to 100."),
        "E_sanity": ("How many numbers appear in the problem "
                     "statement? Reply with ONLY the integer."),
    }
    res = {}
    for name, prompt in variants.items():
        replies = []
        for _ in range(20):
            body, true, meta = fam.gen_feasible_item(
                cal["family"], cal["level"], rng)
            msgs = [{"role": "user",
                     "content": m8.ANSWER_PROMPT.format(body=body)}]
            own = h.greedy(h.ids_of(h.template_text(msgs)), 10)
            msgs2 = msgs + [{"role": "assistant",
                             "content": own.strip()},
                            {"role": "user", "content": prompt}]
            replies.append(h.greedy(
                h.ids_of(h.template_text(msgs2)), 8).strip())
        res[name] = dict(Counter(replies))
        log(f"  diag {name}: {res[name]}")
    with open("m10_diag_battery.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


# ----------------------------- ARM B ------------------------------------------

def arm_b(resume=None):
    log("ARM B: forced-answer extraction")
    h = handle()
    cal = json.load(open(CAL_FILE, encoding="utf-8"))["landed"]
    items = build_main_items(cal["family"], cal["level"])
    done = {(r["item"], r["cond"]) for r in ckpt_load(resume)}
    layout = ReportLayout(h)
    conds = ["T", "Fminus", "Fplus"]
    for k, it in enumerate(items):
        for cond in conds:
            if (it["item"], cond) in done:
                continue
            v = it["true"] if cond == "T" else it[cond]
            built = h.build_forced(it["body"], str(v))
            feats, flogp, rep_logits = h.capture(built)
            np.savez_compressed(
                os.path.join(CKPT_DIR,
                             f"b_{it['item']}_{cond}.npz"),
                **{f"L{L}_{p}": feats[(L, p)].astype(np.float16)
                   for (L, p) in feats},
                rep_logits=rep_logits.numpy().astype(np.float16))
            row = {"item": it["item"], "cond": cond, "forced": v,
                   "true": it["true"], "correct": cond == "T",
                   "body": it["body"],
                   "pos_pre": built["pos_pre"],
                   "pos_ans": built["pos_ans"],
                   "pos_rep": built["pos_rep"],
                   "rep_start": built["rep_start"],
                   "ans_span": list(built["ans_span"]),
                   "seq_len": len(built["ids3"]),
                   "forced_logp": flogp}
            ckpt_append("arm_b.jsonl", row)
        if (k + 1) % 25 == 0:
            log(f"  arm B {k + 1}/{len(items)}")
    log("ARM B extraction done")
    return True


# ----------------------------- ARM C ------------------------------------------

def fit_direction(rows_b, layer):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for r in rows_b:
        z = np.load(os.path.join(
            CKPT_DIR, f"b_{r['item']}_{r['cond']}.npz"))
        X.append(z[f"L{layer}_rep"].astype(np.float64))
        y.append(1 if r["correct"] else 0)
    X = np.stack(X)
    y = np.array(y)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000, C=0.1).fit(
        sc.transform(X), y)
    w = lr.coef_[0] / sc.scale_
    unit = w / np.linalg.norm(w)
    wrong = -unit
    proj = X @ wrong
    m_gap = float(proj[y == 0].mean() - proj[y == 1].mean())
    assert m_gap > 0
    return wrong.astype(np.float32), m_gap


def arm_c(resume=None):
    import torch
    log("ARM C: steering + lens")
    h = handle()
    cal = json.load(open(CAL_FILE, encoding="utf-8"))["landed"]
    items = build_main_items(cal["family"], cal["level"])
    rows_b = ckpt_load(os.path.join(CKPT_DIR, "arm_b.jsonl"))
    L = h.primary_layer
    wrong_dir, m_gap = fit_direction(rows_b, L)
    g0d = json.load(open(G0_FILE, encoding="utf-8"))
    nf = max(v for k, v in g0d["noise_floor_h"].items()
             if k.endswith("_rep"))
    if nf > 0.10 * m_gap:
        log(f"ARM C RED: noise floor {nf:.3e} > 10% of m_L "
            f"{m_gap:.3e} (noise-to-dose gate)")
        return False
    log(f"  direction fitted at L{L}: m_gap {m_gap:.2f}, noise floor "
        f"{nf:.3e} OK")
    np.savez_compressed("m10_direction.npz", wrong=wrong_dir,
                        m_gap=np.array([m_gap]))
    rs = np.random.RandomState(M10_SEED * 13 + 1)
    rand_dirs = [v / np.linalg.norm(v) for v in
                 [rs.randn(len(wrong_dir)) for _ in range(N_RAND)]]
    layout = ReportLayout(h)
    dt = next(h.model.parameters()).dtype
    steer = SteererRow(h, L)
    plan = [("probe", a) for a in ALPHAS] + \
        [(f"rand{i + 1}", a) for i in range(N_RAND)
         for a in RAND_ALPHAS]
    done = {(r["item"], r["cond"], r["dir"], r["alpha"])
            for r in ckpt_load(resume)}
    conds = [("T", "true"), ("F", None)]
    for k, it in enumerate(items):
        # per-item F = the sign-alternating single foil (mission Arm C
        # uses T + one matched foil; reuse arm-B foils alternating)
        fv = it["Fminus"] if it["item"] % 2 == 0 else it["Fplus"]
        for cond, v in [("T", it["true"]), ("F", fv)]:
            built = h.build_forced(it["body"], str(v))
            starts = torch.tensor([built["rep_start"]])
            seqs = torch.tensor([len(built["ids3"])])
            for dname, alpha in plan:
                key = (it["item"], cond, dname, alpha)
                if key in done:
                    continue
                vec = torch.tensor(
                    wrong_dir if dname == "probe"
                    else rand_dirs[int(dname[4:]) - 1], dtype=dt)
                steer.configure(vec, alpha * m_gap, starts, seqs)
                if alpha == 0:
                    steer.off()
                steer.set_offset(0)
                out = h.forward(built["ids3"])
                lp_first = torch.log_softmax(
                    out.logits[0, built["pos_rep"]].float(), -1)
                cond_lp = {}
                for pref in layout.needed_conditionals():
                    seq2 = torch.cat([built["ids3"],
                                      torch.tensor(list(pref))])
                    steer.set_offset(0)
                    o2 = h.forward(seq2)
                    row_lp = torch.log_softmax(
                        o2.logits[0, len(built["ids3"])
                                  + len(pref) - 1].float(), -1)
                    cond_lp[pref] = {t: float(row_lp[t]) for t in
                                     {p[len(pref)] for p in
                                      layout.paths.values()
                                      if len(p) > len(pref)
                                      and p[:len(pref)] == pref}}
                ph, pl = layout.class_masses(lp_first, cond_lp)
                sampled = h.greedy(built["ids3"], 8, steer=steer)
                first_tok = int(lp_first.argmax())
                row = {"item": it["item"], "cond": cond,
                       "correct": cond == "T", "forced": v,
                       "dir": dname, "alpha": alpha,
                       "p_high": ph, "p_low": pl,
                       "margin": math.log(max(ph, 1e-30))
                       - math.log(max(pl, 1e-30)),
                       "greedy_first_token": first_tok,
                       "lp_first_of_greedy": float(
                           lp_first[first_tok]),
                       "sampled_raw": sampled,
                       "sampled": m8.parse_conf_strict(sampled)}
                if dname == "probe" and alpha in (0, 4):
                    outh = h.forward(built["ids3"], hidden=True)
                    stack = []
                    for i in range(h.n_layers + 1):
                        v = outh.hidden_states[i][0, built["pos_rep"]]
                        if v.dim() == 2:
                            # multi-stream mid-layer: stream mean for
                            # the (descriptive) lens; labeled in
                            # results
                            v = v.mean(0)
                        stack.append(v.float().cpu().numpy()
                                     .astype(np.float16))
                    np.savez_compressed(
                        os.path.join(
                            CKPT_DIR,
                            f"c_lens_{it['item']}_{cond}_a{alpha}"
                            f".npz"),
                        h=np.stack(stack))
                ckpt_append("arm_c.jsonl", row)
        if (k + 1) % 10 == 0:
            log(f"  arm C {k + 1}/{len(items)}")
    steer.remove()
    log("ARM C done")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for flag in ["g0", "calibrate", "arm-a", "arm-b", "arm-c"]:
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--resume-from", type=str, default=None)
    args = ap.parse_args()
    if args.g0:
        sys.exit(0 if g0() else 1)
    if args.calibrate:
        sys.exit(0 if calibrate() else 1)
    if getattr(args, "arm_a"):
        sys.exit(0 if arm_a(args.resume_from) else 1)
    if getattr(args, "arm_b"):
        sys.exit(0 if arm_b(args.resume_from) else 1)
    if getattr(args, "arm_c"):
        sys.exit(0 if arm_c(args.resume_from) else 1)
    ap.print_help()
