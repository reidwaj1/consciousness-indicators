#!/usr/bin/env python3
"""
M8: forced-answer correctness representation and the report-position readout
============================================================================
Mission: M8_MISSION.md. Teacher-force answers (true T, undershoot F-,
overshoot F+) into gemma-3-1b-it conversations that are otherwise
IDENTICAL, capture the residual stream at three positions per pass:

  pos-PRE = final prompt token before the answer turn (the position whose
            logits would emit the first answer token)
  pos-ANS = final token of the forced answer
  pos-REP = last position of the full templated conversation (the position
            whose logits generate the first token of the confidence reply)

FORCE arm: does the residual encode in-context answer correctness beyond
prompt features and beyond linear answer-surface features?
REPORT arm: is that signal still linearly present at the report position,
and do the pos-REP logits carry a graded signal that greedy decoding
flattens into the constant "100"?

Stage order (each gates the next; a red gate stops the mission):
  --gate0        G1 environment: model on cuda, greedy text, bit-identical
                 hook capture, tokenizer alignment on 1/2/3-token planted
                 forced answers, digit-tokenization dump for the prereg
  --verify-gen   G2: 2,000 generated items re-parsed from rendered text
                 and re-solved by independent code; 2000/2000 or stop
  --selftest     G3 (offline, no model): generator re-parse, parsers,
                 foil matcher, planted-signal probe synthetics, grouped-CV
                 leakage check
  --stage1       model's own greedy answers on the n=800 item set;
                 GATE accuracy in [0.30, 0.80]
  --pilot        G6: n=40 fresh items end-to-end (foils, passes, position
                 asserts, timing); prints 5 raw conversations for the
                 hand check
  --stage2       G7: main extraction, 800 x 3 = 2,400 passes
  --posttest     Stage 3 selftests on the saved artifacts
  --analyze      Stage 4 preregistered analysis -> m8_results.json

Inherited ground rules honored in code: result files are never
overwritten (fresh _runN suffixes); every random choice is seeded from
the run seed; raw strings saved; strict fullmatch parsing only.

M7 compatibility notes (binding): gen_tracking ported verbatim at level
0; ANSWER_PROMPT and CONF_PROMPT verbatim; chat template rendered to a
STRING first, then tokenized (apply_chat_template's return_tensors path
returns a BatchEncoding that downstream code rejects); bfloat16 + eager
attention (the M7/AST-1 stack) fully on the GPU, no quantization.

Foil feasibility (derived a priori, in the prereg): level-0 true answers
lie in [3, 14]. The matching rules (same digit count, same roundness
class, F- < T < F+, value >= 1) are unsatisfiable for T in {9, 10, 11}
(no one-digit value above 9; no two-digit round value below 10; no
two-digit non-round value below 11). Item generation therefore
rejection-samples until T is feasible, with the reject count recorded.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from statistics import mean, pstdev

import numpy as np

MODEL_ID = "google/gemma-3-1b-it"
SEED = 811                     # M8 run seed, fresh for this mission
N_MAIN = 800
N_PILOT = 40
N_VERIFY = 2000
LEVEL = 0                      # M7's level (add-only variant), acc 0.65 there
LAYERS = [4, 8, 12, 16, 20, 24]
PRIMARY_LAYER = 24             # fixed a priori from M7's exploratory sweep
CO_LAYER = 12                  # co-reported (M7's preregistered primary)
ACC_BAND = (0.30, 0.80)
INFEASIBLE_T = {9, 10, 11}     # derived above; verified by selftest
POSITIONS = ["pre", "ans", "rep"]
CONDS = ["T", "Fminus", "Fplus"]
N_BOOT = 1000
HIDDEN = 1152

# M7's frozen prompts, verbatim (m7_internal.py lines 85-91)
ANSWER_PROMPT = ("{body}\n\nAnswer immediately from your best judgment. "
                 "Do NOT work through steps. Reply with ONLY the integer "
                 "answer.")
CONF_PROMPT = ("How confident are you that your answer is exactly correct, "
               "on a 0-100 scale where N means you would expect answers "
               "like this to be exactly correct about N% of the time? "
               "Reply with ONLY an integer from 0 to 100.")

STAGE1_FILE = "m8_stage1.json"
ROWS_FILE = "m8_rows.json"
ACT_FILE = "m8_activations.npz"
REPLOGIT_FILE = "m8_replogits.npz"
RESULTS_FILE = "m8_results.json"
PILOT_FILE = "m8_pilot_run.json"
LOG_FILE = "m8_run.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fresh_path(path):
    """Never overwrite a result file: return path, or path_runN."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 2
    while os.path.exists(f"{stem}_run{k}{ext}"):
        k += 1
    return f"{stem}_run{k}{ext}"


def latest_path(path):
    """Reader-side twin of fresh_path (G4 review, confirmed finding 1):
    resolve to the HIGHEST existing _runN variant, or the canonical name.
    Readers must log the resolved path; analysis artifacts additionally
    carry a run_id that is cross-checked on load."""
    stem, ext = os.path.splitext(path)
    best = path
    k = 2
    while os.path.exists(f"{stem}_run{k}{ext}"):
        best = f"{stem}_run{k}{ext}"
        k += 1
    return best


def save_json(obj, path):
    p = fresh_path(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    log(f"saved {p}")
    return p


# ----------------------------- GENERATION -------------------------------------
# gen_tracking ported VERBATIM from the bundle's m7_internal.py (lines
# 100-145). Do not edit. Only the level-0 branch is exercised in M8.

def gen_tracking(n_ops, rng):
    """Small-model tracking problem: 3 boxes, n_ops operations, ask about
    a box that at least one op touched (no dead-box freebies). Level 0 is
    the add-only easy variant (single Add, values <= 9) for a model this
    small."""
    if n_ops == 0:
        boxes = {"red": rng.randint(2, 9), "green": rng.randint(2, 9),
                 "blue": rng.randint(2, 9)}
        x = rng.choice(["red", "green", "blue"])
        n = rng.randint(1, 5)
        body = (f"Red box starts with {boxes['red']} items, green box "
                f"with {boxes['green']}, blue box with {boxes['blue']}. "
                f"Add {n} items to the {x} box."
                f"\n\nHow many items end up in the {x} box?")
        return body, boxes[x] + n, x
    boxes = {"red": rng.randint(3, 12), "green": rng.randint(3, 12),
             "blue": rng.randint(3, 12)}
    touched = set()
    lines = [f"Red box starts with {boxes['red']} items, green box with "
             f"{boxes['green']}, blue box with {boxes['blue']}."]
    names = ["red", "green", "blue"]
    for _ in range(n_ops):
        kind = rng.choice(["add", "remove", "move"])
        if kind == "add":
            x = rng.choice(names); n = rng.randint(1, 6)
            boxes[x] += n; touched.add(x)
            lines.append(f"Add {n} items to the {x} box.")
        elif kind == "remove":
            cand = [b for b in names if boxes[b] > 0]
            x = rng.choice(cand)
            n = rng.randint(1, min(6, boxes[x]))
            boxes[x] -= n; touched.add(x)
            lines.append(f"Remove {n} items from the {x} box.")
        else:
            cand = [b for b in names if boxes[b] > 0]
            x = rng.choice(cand)
            y = rng.choice([b for b in names if b != x])
            n = rng.randint(1, min(6, boxes[x]))
            boxes[x] -= n; boxes[y] += n
            touched.update((x, y))
            lines.append(f"Move {n} items from the {x} box to the "
                         f"{y} box.")
    ask = rng.choice(sorted(touched)) if touched else rng.choice(names)
    body = (" ".join(lines)
            + f"\n\nHow many items end up in the {ask} box?")
    return body, boxes[ask], ask


def gen_item(rng, counter=None):
    """Draw a level-0 item, rejection-sampling until T admits both matched
    foils. Deterministic given the rng state; rejects counted."""
    while True:
        body, true, ask = gen_tracking(LEVEL, rng)
        if true not in INFEASIBLE_T:
            return body, true, ask
        if counter is not None:
            counter["rejected"] += 1


def build_items(n, rng, counter):
    """Item set for a study run: feasible-T AND globally unique bodies
    (the level-0 space is small; a duplicated body would put the same
    (prompt, forced answer, label) in two different CV groups — leakage
    the grouped CV is meant to exclude). Deterministic given rng."""
    items, seen = [], set()
    while len(items) < n:
        body, true, ask = gen_item(rng, counter)
        if body in seen:
            counter["rejected_duplicate"] += 1
            continue
        seen.add(body)
        items.append({"item": len(items), "body": body, "true": true,
                      "ask": ask})
    return items


# ----------------------------- INDEPENDENT VERIFIER ---------------------------
# Re-parses the RENDERED problem text and re-solves it with code that
# shares nothing with gen_tracking (G2). Handles the level-0 grammar and
# the general grammar; extracts the asked box from the question line
# itself, never from generator internals.

def independent_solve(body):
    m = re.match(r"Red box starts with (\d+) items, green box with "
                 r"(\d+), blue box with (\d+)\.", body)
    if not m:
        return None
    boxes = {"red": int(m.group(1)), "green": int(m.group(2)),
             "blue": int(m.group(3))}
    for mm in re.finditer(
            r"(Add|Remove) (\d+) items? (?:to|from) the (\w+) box\."
            r"|Move (\d+) items? from the (\w+) box to the (\w+) box\.",
            body):
        if mm.group(1) == "Add":
            boxes[mm.group(3)] += int(mm.group(2))
        elif mm.group(1) == "Remove":
            boxes[mm.group(3)] -= int(mm.group(2))
        else:
            boxes[mm.group(5)] -= int(mm.group(4))
            boxes[mm.group(6)] += int(mm.group(4))
    q = re.search(r"How many items end up in the (\w+) box\?", body)
    if not q or q.group(1) not in boxes:
        return None
    return boxes[q.group(1)]


def verify_generator():
    """G2: 2,000 fresh items; independent re-parse+re-solve must match the
    generator's answer 2000/2000, and all bodies unique."""
    rng = random.Random(SEED + 1000)     # verification-only seed, recorded
    counter = {"rejected": 0}
    bad = 0
    bodies = set()
    for i in range(N_VERIFY):
        body, true, ask = gen_item(rng, counter)
        bodies.add(body)
        got = independent_solve(body)
        if got != true:
            bad += 1
            log(f"G2 MISMATCH at item {i}: generator={true} "
                f"independent={got}\n{body}")
    n_unique = len(bodies)
    log(f"G2 generator verification: {N_VERIFY - bad}/{N_VERIFY} matched; "
        f"{n_unique} unique bodies; {counter['rejected']} infeasible-T "
        f"rejects during sampling")
    ok = bad == 0
    log(f"G2 {'GREEN' if ok else 'RED'}")
    return ok


# ----------------------------- PARSERS (verbatim M7 ports) --------------------

def parse_int_strict(text):
    if text is None:
        return None
    m = re.fullmatch(r"\s*\**(-?\d+)\**\s*\.?\s*", text.strip())
    return int(m.group(1)) if m else None


def parse_conf_strict(text):
    if text is None:
        return None
    m = re.fullmatch(r"\s*\**(\d{1,3})\s*%?\**\s*\.?\s*", text.strip())
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 100 else None


# ----------------------------- FOIL MATCHING ----------------------------------

def n_digits(v):
    return len(str(v))


def is_round(v):
    return v % 10 == 0


class FoilError(Exception):
    pass


def foil_candidates(true, side, tokcount_fn):
    """All values 1..99 matching T's digit count, roundness class, and
    tokenizer token count, strictly on the requested side of T."""
    out = []
    for c in range(1, 100):
        if c == true:
            continue
        if side == "minus" and c >= true:
            continue
        if side == "plus" and c <= true:
            continue
        if n_digits(c) != n_digits(true):
            continue
        if is_round(c) != is_round(true):
            continue
        if tokcount_fn(str(c)) != tokcount_fn(str(true)):
            continue
        out.append(c)
    return out


def pick_foil(true, side, err_offsets, tokcount_fn, rng):
    """Choose a matched foil. Preference order (preregistered): candidates
    at offsets seen in the model's own error distribution, sampled with
    the observed frequencies; else the smallest-offset candidate (seeded
    tie-break). Raises FoilError if the cell is structurally empty."""
    cands = foil_candidates(true, side, tokcount_fn)
    if not cands:
        raise FoilError(f"no matched {side} foil for T={true}")
    weights = [err_offsets.get(abs(c - true), 0) for c in cands]
    if sum(weights) > 0:
        return rng.choices(cands, weights=weights, k=1)[0]
    lo = min(abs(c - true) for c in cands)
    close = [c for c in cands if abs(c - true) == lo]
    return rng.choice(close)


# ----------------------------- MODEL PLUMBING ---------------------------------

_M = {}


def load_model():
    if "m" in _M:
        return _M["m"], _M["t"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), "G1 RED: CUDA not available"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        attn_implementation="eager")
    model.to("cuda").eval()
    _M["m"], _M["t"], _M["d"] = model, tok, "cuda"
    return model, tok


def tokcount(s):
    _, tok = load_model()
    return len(tok(s, add_special_tokens=False)["input_ids"])


def template_text(msgs, gen_prompt=True):
    _, tok = load_model()
    return tok.apply_chat_template(msgs, add_generation_prompt=gen_prompt,
                                   tokenize=False)


def ids_of(text):
    import torch
    _, tok = load_model()
    return tok(text, return_tensors="pt",
               add_special_tokens=False)["input_ids"][0]


class AlignError(Exception):
    pass


def build_forced(body, forced_str):
    """Construct the forced conversation's token ids and the three capture
    positions, with prefix-stability asserts at both the string and the
    token level (the known silent-failure surface; any violation raises).

    Returns (ids3, pos_pre, pos_ans, pos_rep, ans_span) where ans_span is
    the half-open token range [start, end) of the forced answer."""
    u1 = {"role": "user", "content": ANSWER_PROMPT.format(body=body)}
    t1_text = template_text([u1], gen_prompt=True)
    t2_text = t1_text + forced_str
    msgs3 = [u1, {"role": "assistant", "content": forced_str},
             {"role": "user", "content": CONF_PROMPT}]
    t3_text = template_text(msgs3, gen_prompt=True)
    if not t3_text.startswith(t2_text):
        raise AlignError("template does not extend prompt+answer as a "
                         f"string prefix\n{t3_text[:400]!r}")
    ids1, ids2, ids3 = ids_of(t1_text), ids_of(t2_text), ids_of(t3_text)
    n1, n2, n3 = len(ids1), len(ids2), len(ids3)
    if not (n1 < n2 < n3):
        raise AlignError(f"length order broken: {n1} {n2} {n3}")
    if not (ids2[:n1] == ids1).all():
        raise AlignError("token prefix broken at answer boundary")
    if not (ids3[:n2] == ids2).all():
        raise AlignError("token prefix broken at report boundary")
    _, tok = load_model()
    got = tok.decode(ids3[n1:n2])
    if got != forced_str:
        raise AlignError(f"answer span decodes to {got!r}, "
                         f"expected {forced_str!r}")
    return ids3, n1 - 1, n2 - 1, n3 - 1, (n1, n2)


REPORT_CONT = "100"    # the constant report; its token path is teacher-
                       # forced in a SEPARATE logits-only pass so the
                       # exact P("100") completion is computable for the
                       # margin secondary (G4 review, confirmed finding
                       # 2). The CAPTURE pass appends nothing: captured
                       # states and pos-REP logits are exactly the
                       # quantities greedy generation computes. (An
                       # earlier one-pass design appended the tokens to
                       # the capture pass; gate0 measured bf16 kernel-
                       # tiling deviations of ~2% on captures from the
                       # sequence-length change, so it was split.)


def _margin_sets_cached():
    if "sets" not in _M:
        _, tok = load_model()
        _M["sets"] = margin_token_sets(tok)
    return _M["sets"]


def forced_pass(ids3, pos_pre, pos_ans, pos_rep, ans_span):
    """Capture pass — no continuation appended. Returns
    ({(layer,pos): vec}, forced_logp, rep_logits_fp16)."""
    import torch
    model, tok = load_model()
    with torch.no_grad():
        out = model(ids3.unsqueeze(0).to("cuda"),
                    output_hidden_states=True)
    feats = {}
    for L in LAYERS:
        h = out.hidden_states[L][0]
        feats[(L, "pre")] = h[pos_pre].float().cpu().numpy()
        feats[(L, "ans")] = h[pos_ans].float().cpu().numpy()
        feats[(L, "rep")] = h[pos_rep].float().cpu().numpy()
    logits = out.logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    forced_logp = 0.0
    for p in range(*ans_span):
        forced_logp += float(lp[p - 1, ids3[p]])
    rep_logits = logits[pos_rep].cpu().numpy().astype(np.float16)
    return feats, forced_logp, rep_logits


def cont_pass(ids3, pos_rep, cont_str=REPORT_CONT):
    """Logits-only pass with cont_str teacher-forced after pos-REP.
    Returns per-digit logp at the second and third report-token
    positions (conditioned on the canonical continuation ids)."""
    import torch
    model, tok = load_model()
    cont_ids = ids_of(cont_str)
    seq = torch.cat([ids3, cont_ids])
    with torch.no_grad():
        out = model(seq.unsqueeze(0).to("cuda"))
    lp = torch.log_softmax(out.logits[0].float(), dim=-1)
    sets = _margin_sets_cached()

    def digit_lp(pos):
        row = lp[pos]
        return {str(d): float(torch.logsumexp(
            row[torch.tensor(sets[d])], dim=0)) for d in range(10)}
    return {"cont_ids": cont_ids.tolist(),
            "digit_lp_t2": digit_lp(pos_rep + 1),
            "digit_lp_t3": digit_lp(pos_rep + 2)}


def chat_generate(msgs, max_new):
    """Greedy generation via the chat template (M7 convention)."""
    import torch
    model, tok = load_model()
    ids = ids_of(template_text(msgs, gen_prompt=True)).unsqueeze(0).to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def surface_feats(v, tokcount_fn):
    return [float(v), float(abs(v)), float(n_digits(v)),
            1.0 if is_round(v) else 0.0, float(tokcount_fn(str(v)))]


# ----------------------------- G1: ENVIRONMENT GATE ---------------------------

def gate0():
    import torch
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    log("G1 environment gate")
    model, tok = load_model()
    check(next(model.parameters()).is_cuda, "model on cuda")
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    check(getattr(text_cfg, "hidden_size", None) == HIDDEN,
          f"hidden size == {HIDDEN}")

    txt = chat_generate(
        [{"role": "user", "content": "Say the word hello."}], 8)
    check(len(txt.strip()) > 0, f"greedy generation produces text ({txt!r})")

    # hook-capture determinism: same input twice -> bit-identical states;
    # changed input -> different states
    body, true, _ = gen_tracking(LEVEL, random.Random(123))
    ids3, p_pre, p_ans, p_rep, span = build_forced(body, str(true))
    f1, lp1, r1 = forced_pass(ids3, p_pre, p_ans, p_rep, span)
    f2, lp2, r2 = forced_pass(ids3, p_pre, p_ans, p_rep, span)
    same = all(np.array_equal(f1[k], f2[k]) for k in f1)
    check(same and lp1 == lp2 and np.array_equal(r1, r2),
          "repeated pass bit-identical (hidden states, logp, rep logits)")
    c1 = cont_pass(ids3, p_rep)
    c2 = cont_pass(ids3, p_rep)
    check(c1 == c2, "repeated continuation pass bit-identical")
    ids3b, pb, ab, rb, spanb = build_forced(body, str(true + 1))
    f3, _, _ = forced_pass(ids3b, pb, ab, rb, spanb)
    diff = any(not np.array_equal(f1[k], f3[k]) for k in f1)
    check(diff, "changed input changes captured states")

    # CAUSALITY check: two different SAME-LENGTH continuations appended
    # after pos-REP run under identical kernel schedules, so any
    # difference in states at positions <= pos-REP would be genuine
    # information leakage from future tokens (not bf16 tiling noise).
    import torch as _t
    capture_upto = p_rep + 1
    diffs = []
    for cs in ["100", "999"]:
        seq = _t.cat([ids3, ids_of(cs)])
        with _t.no_grad():
            o = model(seq.unsqueeze(0).to("cuda"),
                      output_hidden_states=True)
        diffs.append((
            np.stack([o.hidden_states[L][0, :capture_upto].float()
                      .cpu().numpy() for L in LAYERS]),
            o.logits[0, :capture_upto].float().cpu().numpy()))
    causal = (np.array_equal(diffs[0][0], diffs[1][0])
              and np.array_equal(diffs[0][1], diffs[1][1]))
    check(causal, "attention causal: same-length different continuations "
                  "leave all states/logits at <= pos-REP bit-identical")
    # informational: bf16 length-tiling deviation of the (now unused)
    # one-pass design, logged for the prereg record
    with _t.no_grad():
        o_long = model(_t.cat([ids3, ids_of("100")]).unsqueeze(0)
                       .to("cuda"), output_hidden_states=True)
    mx = max(float(np.max(np.abs(
        o_long.hidden_states[L][0, p_rep].float().cpu().numpy()
        - f1[(L, "rep")]))) for L in LAYERS)
    log(f"  INFO bf16 length-tiling deviation at pos-REP (with vs "
        f"without appended tokens): max {mx:.3e} — capture pass "
        f"therefore appends nothing")

    # tokenizer alignment on 1-, 2-, 3-token forced answers
    for forced in ["7", "13", "123"]:
        ntoks = tokcount(forced)
        ids3, p_pre, p_ans, p_rep, span = build_forced(body, forced)
        span_len = span[1] - span[0]
        check(span_len == ntoks,
              f"forced {forced!r}: span length {span_len} == tokenizer "
              f"count {ntoks}")
        check(p_ans == span[1] - 1, f"forced {forced!r}: pos-ANS is final "
                                    "answer token")
        check(p_rep == len(ids3) - 1, f"forced {forced!r}: pos-REP is "
                                      "final position")
        # the recorded pos-REP must be the position that generates the
        # first reply token: greedy-generate one token and compare with
        # the argmax of the forward logits at pos-REP
        with torch.no_grad():
            gen = model.generate(ids3.unsqueeze(0).to("cuda"),
                                 max_new_tokens=1, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            fw = model(ids3.unsqueeze(0).to("cuda"))
        nxt_gen = int(gen[0, len(ids3)])
        nxt_fw = int(fw.logits[0, p_rep].argmax())
        check(nxt_gen == nxt_fw, f"forced {forced!r}: pos-REP logits "
                                 "generate the first reply token")

    # digit-tokenization dump for the prereg margin token sets
    log("digit tokenization dump (for m8_prereg.md):")
    for s in [str(d) for d in range(10)] + ["10", "13", "100", "85"]:
        enc = tok(s, add_special_tokens=False)["input_ids"]
        log(f"  {s!r} -> {enc} -> {[tok.decode([i]) for i in enc]}")
    sets = margin_token_sets(tok)
    log(f"margin token sets (decoded-form rule): "
        f"{ {d: len(v) for d, v in sets.items()} } ids per digit")
    for d in sets:
        log(f"  digit {d}: ids {sets[d]} decode "
            f"{[tok.decode([i]) for i in sets[d]]}")

    log(f"G1 {'GREEN' if ok else 'RED'}")
    return ok


def margin_token_sets(tok):
    """Preregistered RULE: for each digit d in 0..9, the set of vocab ids
    whose decoded form, after stripping whitespace, equals str(d). Scanning
    the vocab is deterministic in the tokenizer."""
    sets = {d: [] for d in range(10)}
    for i in range(len(tok)):
        s = tok.decode([i]).strip()
        # ASCII digits only: str.isdigit() is True for Unicode digit
        # characters ('¹', '٣', ...) that are NOT confidence-report digits
        if len(s) == 1 and s in "0123456789":
            sets[int(s)].append(i)
    return sets


# ----------------------------- STAGE 1 ----------------------------------------

def stage1():
    log(f"Stage 1: n={N_MAIN} items at level {LEVEL}, seed {SEED}")
    rng = random.Random(SEED)
    counter = {"rejected": 0, "rejected_duplicate": 0}
    items = build_items(N_MAIN, rng, counter)
    # independent re-verify THIS item set too (belt over G2's braces)
    bad = sum(1 for it in items
              if independent_solve(it["body"]) != it["true"])
    if bad:
        log(f"Stage 1 RED: {bad} items fail independent solve")
        return False
    t0 = time.time()
    n_parsed = n_right = 0
    for it in items:
        raw = chat_generate(
            [{"role": "user",
              "content": ANSWER_PROMPT.format(body=it["body"])}], 8)
        pred = parse_int_strict(raw)
        it["own_raw"] = raw
        it["own_pred"] = pred
        if pred is not None:
            n_parsed += 1
            it["own_correct"] = (pred == it["true"])
            n_right += it["own_correct"]
        else:
            it["own_correct"] = None
        if (it["item"] + 1) % 100 == 0:
            log(f"  {it['item'] + 1}/{N_MAIN} "
                f"acc so far {n_right / max(1, n_parsed):.3f}")
    acc = n_right / max(1, n_parsed)
    drops = N_MAIN - n_parsed
    dt = time.time() - t0
    log(f"Stage 1: acc={acc:.3f} over {n_parsed} parsed "
        f"({drops} drops) in {dt / 60:.1f} min; "
        f"{counter['rejected']} infeasible-T rejects")
    errs = [abs(it["own_pred"] - it["true"]) for it in items
            if it["own_pred"] is not None and not it["own_correct"]]
    gate_ok = ACC_BAND[0] <= acc <= ACC_BAND[1]
    out = {"model": MODEL_ID, "seed": SEED, "level": LEVEL, "n": N_MAIN,
           "acc": acc, "answer_drops": drops,
           "rejected_infeasible_T": counter["rejected"],
           "rejected_duplicate_body": counter["rejected_duplicate"],
           "err_offsets": dict(Counter(errs)),
           "gate_acc_band": list(ACC_BAND), "gate_ok": gate_ok,
           "items": items}
    save_json(out, STAGE1_FILE)
    log(f"Stage 1 gate (acc in {ACC_BAND}): {'GREEN' if gate_ok else 'RED'}")
    return gate_ok


# ----------------------------- FORCED EXTRACTION CORE -------------------------

def make_foils(items, err_offsets, rng):
    """Attach F-/F+ values to every item. err_offsets: {offset: count}
    from the model's own observed errors (stage 1)."""
    for it in items:
        it["Fminus"] = pick_foil(it["true"], "minus", err_offsets,
                                 tokcount, rng)
        it["Fplus"] = pick_foil(it["true"], "plus", err_offsets,
                                tokcount, rng)
    return items


def extract(items, spot_reports=0, tag="main"):
    """Run the 3 forced passes for every item. Returns (rows, acts,
    rep_logits). Row order: item-major, cond order T, F-, F+."""
    acts = {(L, p): [] for L in LAYERS for p in POSITIONS}
    rep_logits = []
    rows = []
    t0 = time.time()
    for k, it in enumerate(items):
        for cond in CONDS:
            v = it["true"] if cond == "T" else it[cond]
            forced_str = str(v)
            ids3, p_pre, p_ans, p_rep, span = build_forced(
                it["body"], forced_str)
            feats, flogp, rlog = forced_pass(ids3, p_pre, p_ans,
                                             p_rep, span)
            cont = cont_pass(ids3, p_rep)
            for key in acts:
                acts[key].append(feats[key])
            rep_logits.append(rlog)
            row = {"row": len(rows), "item": it["item"], "cond": cond,
                   "forced": v, "forced_str": forced_str,
                   "correct": cond == "T",
                   "true": it["true"], "body": it["body"],
                   "ask": it["ask"],
                   "own_pred": it.get("own_pred"),
                   "own_correct": it.get("own_correct"),
                   "pos_pre": p_pre, "pos_ans": p_ans, "pos_rep": p_rep,
                   "ans_span": list(span), "seq_len": int(len(ids3)),
                   "forced_logp": flogp,
                   "cont": cont,
                   "surface": surface_feats(v, tokcount)}
            rows.append(row)
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            log(f"  {tag}: {k + 1}/{len(items)} items "
                f"({el / (k + 1):.2f} s/item)")
    # sampled-report spot-check (preregistered split: T on the first
    # spot_reports//2 items, F- and F+ on the next spot_reports//4 each)
    if spot_reports:
        nT = spot_reports // 2
        nF = spot_reports // 4
        plan = ([(i, "T") for i in range(nT)]
                + [(nT + j, "Fminus") for j in range(nF)]
                + [(nT + nF + j, "Fplus") for j in range(nF)])
        for i, cond in plan:
            it = items[i]
            v = it["true"] if cond == "T" else it[cond]
            msgs = [{"role": "user",
                     "content": ANSWER_PROMPT.format(body=it["body"])},
                    {"role": "assistant", "content": str(v)},
                    {"role": "user", "content": CONF_PROMPT}]
            raw = chat_generate(msgs, 6)
            for r in rows:
                if r["item"] == it["item"] and r["cond"] == cond:
                    r["sampled_report_raw"] = raw
                    r["sampled_report"] = parse_conf_strict(raw)
                    break
    return rows, acts, rep_logits


# ----------------------------- G6: PILOT --------------------------------------

def pilot():
    log(f"G6 pilot: n={N_PILOT} fresh items end-to-end")
    s1path = latest_path(STAGE1_FILE)
    log(f"reading stage 1 from {s1path}")
    st1 = json.load(open(s1path, encoding="utf-8"))
    err_offsets = {int(k): v for k, v in st1["err_offsets"].items()}
    rng = random.Random(SEED + 500)
    counter = {"rejected": 0, "rejected_duplicate": 0}
    items = build_items(N_PILOT, rng, counter)
    frng = random.Random(SEED * 7 + 4)
    make_foils(items, err_offsets, frng)
    t0 = time.time()
    rows, acts, rep_logits = extract(items, spot_reports=8, tag="pilot")
    dt = time.time() - t0
    per_item = dt / N_PILOT
    log(f"pilot timing: {dt:.1f}s total, {per_item:.2f}s/item -> "
        f"main run estimate {per_item * N_MAIN / 60:.0f} min")
    # print 5 raw conversations for the hand check
    _, tok = load_model()
    hand = []
    for it in items[:5]:
        ids3, p_pre, p_ans, p_rep, span = build_forced(
            it["body"], str(it["true"]))
        txt = tok.decode(ids3)
        hand.append({"item": it["item"], "true": it["true"],
                     "Fminus": it["Fminus"], "Fplus": it["Fplus"],
                     "pos_pre_token": tok.decode(ids3[p_pre:p_pre + 1]),
                     "ans_span_decoded": tok.decode(ids3[span[0]:span[1]]),
                     "last_token": tok.decode(ids3[p_rep:p_rep + 1]),
                     "full_text": txt})
        print("\n----- RAW CONVERSATION (T condition), item "
              f"{it['item']} -----")
        print(txt)
        print(f"[pos-PRE token: {tok.decode(ids3[p_pre:p_pre+1])!r} | "
              f"answer span: {tok.decode(ids3[span[0]:span[1]])!r} | "
              f"final token: {tok.decode(ids3[p_rep:p_rep+1])!r}]")
    spot = [(r["cond"], r.get("sampled_report_raw"))
            for r in rows if "sampled_report_raw" in r]
    log(f"pilot sampled reports: {spot}")
    out = {"n": N_PILOT, "seed_items": SEED + 500,
           "seed_foils": SEED * 7 + 4,
           "timing_s_per_item": per_item,
           "rejected_infeasible_T": counter["rejected"],
           "sampled_reports": [
               {"cond": c, "raw": r} for c, r in spot],
           "hand_check": hand,
           "rows": rows}
    save_json(out, PILOT_FILE)
    log("G6 pilot complete — hand-check the 5 conversations above, then "
        "run --stage2")
    return True


# ----------------------------- G7: MAIN RUN -----------------------------------

def stage2():
    log(f"G7 main extraction: {N_MAIN} items x 3 = {3 * N_MAIN} passes")
    s1path = latest_path(STAGE1_FILE)
    log(f"reading stage 1 from {s1path}")
    st1 = json.load(open(s1path, encoding="utf-8"))
    if not st1["gate_ok"]:
        log("G7 RED: stage 1 gate not green")
        return False
    err_offsets = {int(k): v for k, v in st1["err_offsets"].items()}
    items = st1["items"]
    frng = random.Random(SEED * 7 + 3)
    make_foils(items, err_offsets, frng)
    run_id = f"m8-{SEED}-{time.strftime('%Y%m%d-%H%M%S')}"
    rows, acts, rep_logits = extract(items, spot_reports=100, tag="main")
    # save order (G4 review, confirmed finding 1): both npz files first,
    # rows JSON LAST as the commit record naming the exact artifact
    # paths and the shared run_id
    act_arrays = {f"L{L}_{p}": np.stack(acts[(L, p)]).astype(np.float16)
                  for (L, p) in acts}
    ap = fresh_path(ACT_FILE)
    np.savez_compressed(ap, run_id=np.array([run_id]), **act_arrays)
    log(f"saved {ap} "
        f"({os.path.getsize(ap) / 1e6:.0f} MB)")
    rp = fresh_path(REPLOGIT_FILE)
    np.savez_compressed(rp, run_id=np.array([run_id]),
                        rep_logits=np.stack(rep_logits))
    log(f"saved {rp} ({os.path.getsize(rp) / 1e6:.0f} MB)")
    save_json({"model": MODEL_ID, "seed": SEED, "level": LEVEL,
               "run_id": run_id,
               "n_items": N_MAIN, "n_rows": len(rows),
               "seed_foils": SEED * 7 + 3,
               "act_file": ap, "replogit_file": rp,
               "stage1_file": s1path,
               "cond_order": CONDS, "layers": LAYERS,
               "positions": POSITIONS,
               "rows": rows}, ROWS_FILE)
    log("G7 complete")
    return True


def load_run():
    """Resolve and cross-verify the analysis artifact set (G4 review,
    confirmed findings 1 and 3): rows JSON is the commit record; the npz
    files it names must carry the same run_id; row counts must match."""
    rpath = latest_path(ROWS_FILE)
    meta = json.load(open(rpath, encoding="utf-8"))
    ap = meta["act_file"]
    lp = meta["replogit_file"]
    log(f"loading run {meta['run_id']}: rows={rpath} acts={ap} "
        f"replogits={lp}")
    data = np.load(ap)
    rep = np.load(lp)
    rid = meta["run_id"]
    assert str(data["run_id"][0]) == rid, \
        f"activation run_id {data['run_id'][0]} != rows run_id {rid}"
    assert str(rep["run_id"][0]) == rid, \
        f"replogit run_id {rep['run_id'][0]} != rows run_id {rid}"
    n = len(meta["rows"])
    assert meta["n_rows"] == n
    for L in LAYERS:
        for p in POSITIONS:
            assert data[f"L{L}_{p}"].shape[0] == n, \
                f"L{L}_{p} rows {data[f'L{L}_{p}'].shape[0]} != {n}"
    assert rep["rep_logits"].shape[0] == n
    # position-consistency asserts (review hardening)
    for r in meta["rows"]:
        assert r["pos_rep"] == r["seq_len"] - 1
        assert r["pos_ans"] == r["ans_span"][1] - 1
        assert r["pos_pre"] == r["ans_span"][0] - 1
    return meta, data, rep


# ----------------------------- STATS ------------------------------------------

def auroc(scores, labels):
    """Rank-based AUROC (ties get half credit)."""
    from scipy.stats import rankdata
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(scores)
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pair_frac(pairs):
    """Fraction of (pos_score, neg_score) pairs with pos > neg, ties 0.5."""
    if not pairs:
        return None
    w = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a, b in pairs)
    return w / len(pairs)


def cluster_boot(stat_fn, items_list, n_boot=N_BOOT, seed=0):
    """Item-clustered bootstrap: stat_fn(multiset of item ids) -> float.
    Returns (lo, hi) 2.5/97.5 percentiles, list of resample stats."""
    rs = np.random.RandomState(seed)
    vals = []
    items_arr = np.array(items_list)
    for _ in range(n_boot):
        pick = items_arr[rs.randint(0, len(items_arr), len(items_arr))]
        v = stat_fn(pick)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None, []
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)), vals)


def oof_scores(X, y, groups, seed=0):
    """Grouped 5-fold out-of-fold decision scores; returns (scores, folds)
    where folds[i] is the fold index of row i."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    scores = np.zeros(len(y))
    folds = np.zeros(len(y), int)
    for fi, (tr, te) in enumerate(
            GroupKFold(n_splits=5).split(X, y, groups)):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=0.1).fit(
            sc.transform(X[tr]), y[tr])
        scores[te] = m.decision_function(sc.transform(X[te]))
        folds[te] = fi
    return scores, folds


def leakage_check(groups, folds):
    """No item id may appear in more than one fold."""
    seen = {}
    for g, f in zip(groups, folds):
        if g in seen and seen[g] != f:
            return False
        seen[g] = f
    return True


# ----------------------------- STAGE 3: POST-RUN SELFTESTS --------------------

def probe_synthetics():
    """Planted-signal synthetics through the ACTUAL grouped-CV pipeline:
    aligned signal recovered; orthogonal signal not recovered by the
    correctness probe; shuffled labels at chance."""
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    rs = np.random.RandomState(7)
    n_items, d = 300, 64
    X = rs.randn(3 * n_items, d)
    groups = np.repeat(np.arange(n_items), 3)
    w = np.zeros(d); w[0] = 1.0
    y_aligned = (X @ w + 0.5 * rs.randn(len(X)) > 0).astype(int)
    s, folds = oof_scores(X, y_aligned, groups, seed=1)
    a = auroc(s, y_aligned)
    check(a > 0.85, f"aligned planted signal recovered (AUC {a:.2f})")
    check(leakage_check(groups, folds), "grouped CV: no item spans folds")
    w2 = np.zeros(d); w2[1] = 1.0
    y_orth = (X @ w2 + 0.5 * rs.randn(len(X)) > 0).astype(int)
    cross = auroc(s, y_orth)
    check(0.4 < cross < 0.6,
          f"orthogonal signal NOT recovered by the probe ({cross:.2f})")
    y_shuf = y_aligned.copy()
    np.random.RandomState(11).shuffle(y_shuf)
    s2, _ = oof_scores(X, y_shuf, groups, seed=1)
    a2 = auroc(s2, y_shuf)
    check(abs(a2 - 0.5) < 0.06, f"shuffled-label null at chance ({a2:.2f})")
    return ok


def posttest():
    log("Stage 3 selftests on saved artifacts")
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    check(probe_synthetics(), "probe synthetics block")

    meta, data, _rep = load_run()
    rows = meta["rows"]
    check(True, f"artifact set cross-verified (run {meta['run_id']}: "
                "run_ids match, row counts match, positions consistent)")

    # foil-audit bot: re-verify every foil's matching from the saved rows
    bad = 0
    for it_id in {r["item"] for r in rows}:
        grp = {r["cond"]: r for r in rows if r["item"] == it_id}
        T = grp["T"]["forced"]
        fm, fp = grp["Fminus"]["forced"], grp["Fplus"]["forced"]
        conds = [fm < T < fp,
                 n_digits(fm) == n_digits(T) == n_digits(fp),
                 is_round(fm) == is_round(T) == is_round(fp),
                 grp["T"]["surface"][4] == grp["Fminus"]["surface"][4]
                 == grp["Fplus"]["surface"][4],
                 grp["T"]["correct"] is True,
                 grp["Fminus"]["correct"] is False,
                 grp["Fplus"]["correct"] is False,
                 grp["T"]["forced"] == grp["T"]["true"],
                 independent_solve(grp["T"]["body"]) == grp["T"]["true"]]
        if not all(conds):
            bad += 1
            log(f"  foil audit fail at item {it_id}: {conds}")
    check(bad == 0, f"foil audit: {len({r['item'] for r in rows}) - bad}/"
                    f"{len({r['item'] for r in rows})} items clean")

    # shuffled-label null on the real primary matrix
    X = data[f"L{PRIMARY_LAYER}_ans"].astype(float)
    y = np.array([1 if r["correct"] else 0 for r in rows])
    groups = np.array([r["item"] for r in rows])
    y_shuf = y.copy()
    np.random.RandomState(SEED).shuffle(y_shuf)
    s, folds = oof_scores(X, y_shuf, groups, seed=1)
    a = auroc(s, y_shuf)
    check(abs(a - 0.5) < 0.06,
          f"shuffled-label null on real L{PRIMARY_LAYER} pos-ANS "
          f"({a:.3f})")
    check(leakage_check(groups, folds), "grouped CV leakage check on "
                                        "real data")

    # pos-REP parser re-verification on 20 planted conversations
    import torch
    model, tok = load_model()
    rng = random.Random(SEED + 900)
    good = 0
    for _ in range(20):
        body, true, _ = gen_item(rng)
        ids3, p_pre, p_ans, p_rep, span = build_forced(body, str(true))
        with torch.no_grad():
            gen = model.generate(ids3.unsqueeze(0).to("cuda"),
                                 max_new_tokens=1, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            fw = model(ids3.unsqueeze(0).to("cuda"))
        good += int(gen[0, len(ids3)]) == int(fw.logits[0, p_rep].argmax())
    check(good == 20, f"pos-REP generates first reply token (20/20, "
                      f"got {good})")

    log(f"Stage 3 {'GREEN' if ok else 'RED'}")
    return ok


# ----------------------------- STAGE 4: ANALYSIS ------------------------------

def slice_within_item(scores, rows):
    """Same-item T-vs-foil pairs from OOF scores."""
    by_item = {}
    for r, s in zip(rows, scores):
        by_item.setdefault(r["item"], {})[r["cond"]] = s

    def stat(item_multiset):
        pairs = []
        for it in item_multiset:
            g = by_item[it]
            pairs.append((g["T"], g["Fminus"]))
            pairs.append((g["T"], g["Fplus"]))
        return pair_frac(pairs)
    return stat


def slice_matched_answer(scores, rows):
    """Cross-item pairs with the SAME forced numeral string, correct in
    one item, wrong in the other."""
    def stat(item_multiset):
        chosen = Counter(item_multiset)
        pool = {}
        for r, s in zip(rows, scores):
            reps = chosen.get(r["item"], 0)
            if reps:
                pool.setdefault(r["forced_str"], []).append(
                    (r["correct"], s, reps, r["item"]))
        pairs = []
        for fs, lst in pool.items():
            pos = [(s, reps, it) for c, s, reps, it in lst if c]
            neg = [(s, reps, it) for c, s, reps, it in lst if not c]
            for ps, pr, pit in pos:
                for ns, nr, nit in neg:
                    if pit == nit:
                        continue
                    pairs.append(((ps, ns), pr * nr))
        if not pairs:
            return None
        w = sum(m * (1.0 if a > b else (0.5 if a == b else 0.0))
                for (a, b), m in pairs)
        return w / sum(m for _, m in pairs)
    return stat


def cross_sign_transfer(X, rows, groups, train_cond, test_cond, seed=1):
    """Train on T-vs-train_cond rows of train items; report same-item
    T-vs-test_cond pair fraction on held-out items."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    y = np.array([1 if r["correct"] else 0 for r in rows])
    conds = np.array([r["cond"] for r in rows])
    items = np.array([r["item"] for r in rows])
    pair_by_item = {}
    for fi, (tr, te) in enumerate(
            GroupKFold(n_splits=5).split(X, y, groups)):
        tr_mask = np.isin(np.arange(len(y)), tr) & np.isin(
            conds, ["T", train_cond])
        sc = StandardScaler().fit(X[tr_mask])
        m = LogisticRegression(max_iter=2000, C=0.1).fit(
            sc.transform(X[tr_mask]), y[tr_mask])
        te_mask = np.isin(np.arange(len(y)), te) & np.isin(
            conds, ["T", test_cond])
        s = m.decision_function(sc.transform(X[te_mask]))
        for idx, sv in zip(np.arange(len(y))[te_mask], s):
            pair_by_item.setdefault(items[idx], {})[conds[idx]] = sv

    def stat(item_multiset):
        pairs = [(pair_by_item[it]["T"], pair_by_item[it][test_cond])
                 for it in item_multiset if test_cond in pair_by_item.get(
                     it, {}) and "T" in pair_by_item.get(it, {})]
        return pair_frac(pairs)
    item_ids = sorted(pair_by_item)
    point = stat(item_ids)
    lo, hi, _ = cluster_boot(stat, item_ids, seed=seed)
    return point, lo, hi


def analyze():
    log("Stage 4 preregistered analysis")
    meta, data, rep_npz = load_run()
    rows = meta["rows"]
    y = np.array([1 if r["correct"] else 0 for r in rows])
    groups = np.array([r["item"] for r in rows])
    item_ids = sorted(set(groups))
    res = {"model": MODEL_ID, "seed": SEED, "n_items": len(item_ids),
           "n_rows": len(rows), "primary_layer": PRIMARY_LAYER,
           "co_layer": CO_LAYER}

    surf = np.array([r["surface"] for r in rows], float)
    logp = np.array([[r["forced_logp"]] for r in rows], float)

    def concat_baseline(L):
        return np.column_stack([data[f"L{L}_pre"].astype(float),
                                surf, logp])

    # shallow-observer battery (REPORT cue bar): lengths, magnitudes,
    # digit counts, roundness of body and answer — never solved features
    _bl = [[len(r["body"]), r["seq_len"]] for r in rows]
    shallow = np.column_stack([np.array(_bl, float), surf])

    def analyze_position(pos, L):
        X = data[f"L{L}_{pos}"].astype(float)
        s, folds = oof_scores(X, y, groups, seed=1)
        assert leakage_check(groups, folds)
        overall = auroc(s, y)
        out = {"overall_auc": overall}
        # (a) within-item
        stat_a = slice_within_item(s, rows)
        pa = stat_a(item_ids)
        lo, hi, _ = cluster_boot(stat_a, item_ids, seed=2)
        out["a_within_item"] = {"auc": pa, "ci": [lo, hi],
                                "pass": lo is not None and lo > 0.5}
        # (b) matched-answer cross-item
        stat_b = slice_matched_answer(s, rows)
        pb = stat_b(item_ids)
        lo_b, hi_b, _ = cluster_boot(stat_b, item_ids, seed=3)
        out["b_matched_answer"] = {"auc": pb, "ci": [lo_b, hi_b],
                                   "pass": lo_b is not None and lo_b > 0.5}
        # (c) beats the concatenation baseline (paired clustered bootstrap)
        sb, foldsb = oof_scores(concat_baseline(L), y, groups, seed=1)
        base_auc = auroc(sb, y)
        idx_of = {}
        for i, r in enumerate(rows):
            idx_of.setdefault(r["item"], []).append(i)

        def diff_stat(item_multiset):
            take = []
            for it in item_multiset:
                take.extend(idx_of[it])
            take = np.array(take)
            a1 = auroc(s[take], y[take])
            a2 = auroc(sb[take], y[take])
            if a1 is None or a2 is None:
                return None
            return a1 - a2
        point = overall - base_auc
        _, _, diffs = cluster_boot(diff_stat, item_ids, seed=4)
        p_c = (sum(1 for d in diffs if d <= 0) + 1) / (len(diffs) + 1)
        out["c_vs_concat_baseline"] = {
            "probe_auc": overall, "baseline_auc": base_auc,
            "diff": point, "p_one_sided": p_c, "pass": p_c < 0.05}
        # (d) cross-sign transfer
        d1, d1lo, d1hi = cross_sign_transfer(X, rows, groups,
                                             "Fminus", "Fplus", seed=5)
        d2, d2lo, d2hi = cross_sign_transfer(X, rows, groups,
                                             "Fplus", "Fminus", seed=6)
        out["d_cross_sign"] = {
            "train_Fminus_test_Fplus": {"auc": d1, "ci": [d1lo, d1hi]},
            "train_Fplus_test_Fminus": {"auc": d2, "ci": [d2lo, d2hi]},
            "pass": (d1lo is not None and d1lo > 0.5
                     and d2lo is not None and d2lo > 0.5)}
        # shallow-cue bar at this position (context for REPORT wording)
        ss, _ = oof_scores(shallow, y, groups, seed=1)
        out["shallow_battery_auc"] = auroc(ss, y)
        return out

    log("FORCE arm (pos-ANS)")
    res["force"] = {}
    res["force"][f"L{PRIMARY_LAYER}"] = analyze_position(
        "ans", PRIMARY_LAYER)
    res["force"][f"L{CO_LAYER}"] = analyze_position("ans", CO_LAYER)
    f = res["force"][f"L{PRIMARY_LAYER}"]
    force_pass = (f["a_within_item"]["pass"] and
                  f["b_matched_answer"]["pass"] and
                  f["c_vs_concat_baseline"]["pass"] and
                  f["d_cross_sign"]["pass"])
    res["force"]["primary_pass"] = force_pass
    log(f"FORCE primary (all four at pos-ANS L{PRIMARY_LAYER}): "
        f"{'PASS' if force_pass else 'FAIL'}")

    log("REPORT arm (pos-REP)")
    res["report"] = {}
    res["report"][f"L{PRIMARY_LAYER}"] = analyze_position(
        "rep", PRIMARY_LAYER)
    res["report"][f"L{CO_LAYER}"] = analyze_position("rep", CO_LAYER)
    r = res["report"][f"L{PRIMARY_LAYER}"]
    report_pass = (r["a_within_item"]["pass"] and
                   r["b_matched_answer"]["pass"] and
                   r["c_vs_concat_baseline"]["pass"])
    res["report"]["primary_pass"] = report_pass
    log(f"REPORT primary ((a)+(b)+(c) at pos-REP L{PRIMARY_LAYER}): "
        f"{'PASS' if report_pass else 'FAIL'} "
        f"(interpretation clause applies)")

    # SECONDARY: position-decay curve, all layers x positions
    log("secondaries")
    decay = {}
    for L in LAYERS:
        decay[L] = {}
        for pos in POSITIONS:
            s, _ = oof_scores(data[f"L{L}_{pos}"].astype(float), y,
                              groups, seed=1)
            decay[L][pos] = auroc(s, y)
        log(f"  L{L:>2}: " + "  ".join(
            f"{pos}={decay[L][pos]:.3f}" for pos in POSITIONS))
    res["position_decay"] = {str(k): v for k, v in decay.items()}

    # SECONDARY: T-rows on greedy-wrong items (in-context reading check)
    tmask = np.array([r["cond"] == "T" and r["own_correct"] is not None
                      for r in rows])
    X24 = data[f"L{PRIMARY_LAYER}_ans"].astype(float)
    s_all, _ = oof_scores(X24, y, groups, seed=1)
    t_scores = s_all[tmask]
    t_own = np.array([1 if r["own_correct"] else 0
                      for r in rows
                      if r["cond"] == "T" and r["own_correct"] is not None])
    res["force_secondary_T_greedy"] = {
        "auc_own_correct_vs_wrong": auroc(t_scores, t_own),
        "note": "expected ~0.5 if the probe reads in-context correctness; "
                ">0.5 means it partly reads item difficulty / the model's "
                "own trajectory"}

    # SECONDARY: report margin — high- vs low-confidence continuation
    # mass (corrected construct per G4 review, confirmed finding 2).
    # First-token digit masses from the saved full pos-REP logits; the
    # exact "100" path completed with the teacher-forced continuation
    # logps saved per row. HIGH = reports 80-100 = P(lead 8) + P(lead 9)
    # + P("100" path). LOW = reports 0-59 = P(lead 0,2,3,4,5) +
    # (P(lead 1) - P("100" path)), i.e. the 1/10-19 remainder of the
    # lead-1 mass. Leads 6,7 (reports 60-79) are excluded, stated.
    rep = rep_npz["rep_logits"].astype(np.float32)
    _, tok = load_model()
    sets = margin_token_sets(tok)
    lse = np.log(np.exp(
        rep - rep.max(axis=1, keepdims=True)).sum(axis=1)) + rep.max(axis=1)
    lp_digit = {d: np.log(np.clip(
        np.exp(rep[:, sets[d]] - lse[:, None]).sum(axis=1), 1e-30, None))
        for d in range(10) if sets[d]}
    p1 = np.exp(lp_digit[1])
    lp_t2_0 = np.array([r["cont"]["digit_lp_t2"]["0"] for r in rows])
    lp_t3_0 = np.array([r["cont"]["digit_lp_t3"]["0"] for r in rows])
    p_100 = np.clip(p1 * np.exp(lp_t2_0) * np.exp(lp_t3_0), 1e-30, 1.0)
    p_high = np.exp(lp_digit[8]) + np.exp(lp_digit[9]) + p_100
    p_low = (np.exp(lp_digit[0]) + np.exp(lp_digit[2])
             + np.exp(lp_digit[3]) + np.exp(lp_digit[4])
             + np.exp(lp_digit[5]) + np.clip(p1 - p_100, 1e-30, None))
    margin = np.log(np.clip(p_high, 1e-30, None)) - np.log(
        np.clip(p_low, 1e-30, None))
    m_auc = auroc(margin, y)

    idx_of_m = {}
    for i, r in enumerate(rows):
        idx_of_m.setdefault(r["item"], []).append(i)

    def margin_stat(item_multiset):
        take = []
        for it in item_multiset:
            take.extend(idx_of_m[it])
        take = np.array(take)
        return auroc(margin[take], y[take])
    mlo, mhi, _ = cluster_boot(margin_stat, item_ids, seed=8)
    res["report_margin"] = {
        "definition": "log P(report in 80-100) - log P(report in 0-59) "
                      "from pos-REP first-token digit masses plus the "
                      "teacher-forced '100' path (P(1)*P(0|1)*P(0|10)); "
                      "leads 6,7 (60-79) excluded; AUC vs forced-answer "
                      "correctness, item-clustered CI",
        "auc": m_auc, "ci": [mlo, mhi],
        "p100_mean_correct": float(p_100[y == 1].mean()),
        "p100_mean_wrong": float(p_100[y == 0].mean()),
        "p_high_mean_correct": float(p_high[y == 1].mean()),
        "p_high_mean_wrong": float(p_high[y == 0].mean())}
    log(f"  report margin AUC {m_auc:.3f} [{mlo:.3f}, {mhi:.3f}]")

    # descriptive extra (NOT the preregistered secondary): the old
    # leading-digit-identity contrast, kept under its own name
    lead1_margin = lp_digit[1] - np.log(np.clip(
        sum(np.exp(lp_digit[d]) for d in lp_digit if d != 1),
        1e-30, None))
    res["descriptive_lead_digit1_vs_rest_auc"] = auroc(lead1_margin, y)

    # SECONDARY: sampled-report spot-check tally
    spot = [(r["cond"], r.get("sampled_report"))
            for r in rows if "sampled_report_raw" in r]
    res["sampled_reports"] = {
        "n": len(spot),
        "values": dict(Counter(str(v) for _, v in spot)),
        "by_cond": {c: dict(Counter(str(v) for cc, v in spot if cc == c))
                    for c in CONDS}}
    log(f"  sampled reports: {res['sampled_reports']['values']}")

    res["m7_probe_transfer"] = {
        "status": "NOT RUN",
        "reason": "m7_activations.npz stayed on the MacBook; the frozen "
                  "L24 direction is not in the transfer bundle and cannot "
                  "be recomputed here. Flagged for the Mac-side session."}

    save_json(res, RESULTS_FILE)
    log("Stage 4 complete")
    return True


# ----------------------------- G3: OFFLINE SELFTESTS --------------------------

def selftest_offline():
    log("G3 offline selftests (no model)")
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    # generator + independent verifier on 200 items, uniqueness
    rng = random.Random(4)
    bodies = set()
    good = 0
    for _ in range(200):
        body, true, ask = gen_item(rng)
        bodies.add(body)
        good += independent_solve(body) == true
    check(good == 200, f"generator vs independent solve ({good}/200)")
    check(len(bodies) == 200, "bodies unique")
    check(all(3 <= t <= 14 and t not in INFEASIBLE_T for t in
              [gen_item(random.Random(s))[1] for s in range(300)]),
          "level-0 T range and feasibility filter")

    # the exact main-set construction: 800 items, unique bodies, all
    # feasible, all independently solvable
    c800 = {"rejected": 0, "rejected_duplicate": 0}
    its = build_items(N_MAIN, random.Random(SEED), c800)
    check(len({it["body"] for it in its}) == N_MAIN,
          f"main item set: {N_MAIN} unique bodies "
          f"({c800['rejected_duplicate']} duplicate rejects, "
          f"{c800['rejected']} infeasible-T rejects)")
    check(all(independent_solve(it["body"]) == it["true"] for it in its),
          "main item set: all independently solvable")
    check(all(it["true"] not in INFEASIBLE_T for it in its),
          "main item set: all foil-feasible")

    # parsers (verbatim M7 behavior)
    check(parse_int_strict("13") == 13
          and parse_int_strict("**13**") == 13
          and parse_int_strict("13 items") is None
          and parse_int_strict("the answer is 13") is None,
          "strict int parser")
    check(parse_conf_strict("85") == 85
          and parse_conf_strict("100") == 100
          and parse_conf_strict("confidence: 85") is None
          and parse_conf_strict("150") is None,
          "strict conf parser")

    # foil matcher with a digit-split fake tokenizer
    fake_tok = len
    ok_foil = True
    rng2 = random.Random(9)
    offsets = {1: 5, 2: 3, 7: 1}
    for T in [t for t in range(3, 15) if t not in INFEASIBLE_T]:
        fm = pick_foil(T, "minus", offsets, fake_tok, rng2)
        fp = pick_foil(T, "plus", offsets, fake_tok, rng2)
        ok_foil &= fm < T < fp
        ok_foil &= n_digits(fm) == n_digits(T) == n_digits(fp)
        ok_foil &= is_round(fm) == is_round(T) == is_round(fp)
        ok_foil &= fake_tok(str(fm)) == fake_tok(str(T)) == fake_tok(str(fp))
    check(ok_foil, "foil matcher properties over all feasible T")
    for T in sorted(INFEASIBLE_T):
        side = "plus" if T == 9 else "minus"
        try:
            pick_foil(T, side, offsets, fake_tok, rng2)
            check(False, f"T={T} {side} should be infeasible")
        except FoilError:
            check(True, f"T={T} {side} correctly infeasible")

    # error-distribution preference is actually used
    fm_pref = pick_foil(8, "minus", {2: 100}, fake_tok, random.Random(1))
    check(fm_pref == 6, f"foil respects observed error offsets (got "
                        f"{fm_pref}, want 6)")

    # probe pipeline synthetics (same code path as posttest)
    check(probe_synthetics(), "probe synthetics block")

    log(f"G3 offline {'GREEN' if ok else 'RED'}")
    return ok


# ----------------------------- MAIN -------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for flag in ["gate0", "verify-gen", "selftest", "stage1", "pilot",
                 "stage2", "posttest", "analyze"]:
        ap.add_argument(f"--{flag}", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest_offline() else 1)
    if getattr(args, "verify_gen"):
        sys.exit(0 if verify_generator() else 1)
    if args.gate0:
        sys.exit(0 if gate0() else 1)
    if args.stage1:
        sys.exit(0 if stage1() else 1)
    if args.pilot:
        sys.exit(0 if pilot() else 1)
    if args.stage2:
        sys.exit(0 if stage2() else 1)
    if args.posttest:
        sys.exit(0 if posttest() else 1)
    if args.analyze:
        sys.exit(0 if analyze() else 1)
    ap.print_help()
