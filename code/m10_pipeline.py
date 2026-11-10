#!/usr/bin/env python3
"""
M10: model-parameterized pipeline (port of m8_force + m9_steer)
===============================================================
Mission: M10_MISSION.md (project copy, incl. dated amendments).
Everything model-specific is resolved AT RUNTIME from the loaded model:
layer indices from depth fractions, decoder-block hook points, digit /
report-token layout from the actual tokenizer, chat-template prefixes
verified per conversation. Nothing inherits gemma constants.

Local (Stage -1) entry points:
  --replay        REGRESSION GATE: run the literal M8 rows (bodies +
                  forced values from m8_rows.json) through this
                  pipeline on gemma-3-1b; answers/parses must match
                  M8 exactly; probe AUCs within bootstrap noise.
  --rehearse      HETEROGENEOUS-MAP REHEARSAL: gemma-3-1b 4-bit with a
                  FORCED cuda:0+cpu split; capture, alignment,
                  injection, and lens selftests through this pipeline.
  --dryrun-v4     META-DEVICE DRY RUN: V4-Flash config+tokenizer at
                  the pinned revision on the meta device; resolve
                  layers, hook points, template prefixes; verify the
                  report-token layout machinery on the real tokenizer.
  --layout MODEL  REPORT-TOKEN LAYOUT GATE: tokenize every report
                  string 0-100, assert decode round-trips, derive the
                  margin trie, run the planted-report selftest.

Pod (Stage 0+) entry points (require --model v4flash):
  --g0            environment gates incl. noise floor, dual-path
                  capture cross-check, injection propagation,
                  template/thinking gate, batch gate, throughput pilot
  --calibrate     Stage 1 difficulty walk (answer-only prompts)
  --arm-a / --arm-b / --arm-c   the frozen arms, checkpointed,
                  --resume-from supported

Ground rules honored: never overwrite results (fresh _runN); every
random choice seeded; strict parsing; per-stage checkpoints; budget
logging is m10_pod.py's job, called at stage boundaries.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

import m8_force as m8

DEPTH_FRACTIONS = [0.15, 0.30, 0.45, 0.60, 0.80, 0.90, 0.95]
PRIMARY_FRACTION = 0.90          # panel order; gemma evidence at 24/26
SECONDARY_FRACTIONS = [0.80, 0.95]
M10_SEED = 1031
LOG_FILE = "m10_run.log"

V4_REPO = "deepseek-ai/DeepSeek-V4-Flash"
V4_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"

SPECS = {
    "gemma": {"repo": "google/gemma-3-1b-it", "revision": None,
              "load": {"torch_dtype": "bfloat16",
                       "attn_implementation": "eager"}},
    "gemma4bit": {"repo": "google/gemma-3-1b-it", "revision": None,
                  "load": {"quant4bit": True,
                           "attn_implementation": "eager"}},
    # The checkpoint is NATIVELY FP8 (~149 GB, FP4 experts) and
    # transformers refuses to stack bnb on a FineGrainedFP8Config;
    # A100s (sm_80) cannot execute FP8 at all. The pod is therefore
    # 4x H100 (sm_90): native-precision load, sharded, no requant.
    "v4flash": {"repo": V4_REPO, "revision": V4_REVISION,
                "load": {"torch_dtype": "auto", "device_map": "auto"},
                "encoder": "dsv4"},
}


def _v4_encoding_dir():
    return os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--deepseek-ai--DeepSeek-V4-Flash", "snapshots",
        V4_REVISION, "encoding")


class DSV4Encoder:
    """V4-Flash ships NO tokenizer chat template; conversations are
    rendered by the repo's own reference module (encoding_dsv4.py at
    the pinned revision), which ships 4 official input/output test
    vectors that the dry-run gate verifies exactly. PINNED invocation:
    encode_messages(msgs, thinking_mode='chat') — thinking OFF; in
    chat mode the generation prompt ends with the assistant token
    followed by the thinking_end token."""

    MODE = "chat"

    def __init__(self):
        import importlib.util
        p = os.path.join(_v4_encoding_dir(), "encoding_dsv4.py")
        spec = importlib.util.spec_from_file_location(
            "encoding_dsv4", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.m = mod
        self.gen_suffix = (mod.ASSISTANT_SP_TOKEN
                           + mod.thinking_end_token)

    def render(self, msgs, gen_prompt=True):
        txt = self.m.encode_messages(msgs, thinking_mode=self.MODE)
        if gen_prompt:
            return txt
        if not txt.endswith(self.gen_suffix):
            raise m8.AlignError(
                f"encoder generation suffix missing: {txt[-80:]!r}")
        return txt[: -len(self.gen_suffix)]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------- MODEL HANDLE -----------------------------------

class Handle:
    """Everything model-specific, resolved from the loaded objects."""

    def __init__(self, name, model, tok, meta_only=False,
                 encoder=None):
        self.name = name
        self.model = model
        self.tok = tok
        self.meta_only = meta_only
        self.encoder = encoder
        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        self.hidden = text_cfg.hidden_size
        self.blocks = self._find_blocks()
        self.n_layers = len(self.blocks)
        # depth fraction f -> hidden_states index ceil(f * n_layers):
        # ceiling (not rounding) is the mapping under which the panel's
        # 0.9 fraction lands on gemma's L24 evidence (0.9*26=23.4 ->
        # 24), and no fraction can resolve to the embedding index 0
        import math as _math
        self.layers = sorted({min(self.n_layers,
                                  _math.ceil(f * self.n_layers))
                              for f in DEPTH_FRACTIONS})
        self.primary_layer = min(self.n_layers,
                                 _math.ceil(PRIMARY_FRACTION
                                            * self.n_layers))
        self.lens_mode = None    # resolved by lens gate on live models
        # multi-stream residual support (V4's DSpark design carries 4
        # parallel streams per position mid-stack, collapsed by an
        # hc_head before the output): detected at first capture;
        # captures flatten (K, H) -> (K*H,), the steerer reshapes back
        self.streams = None

    @property
    def feat_dim(self):
        return self.hidden * (self.streams or 1)

    def _find_blocks(self):
        m = self.model
        for path in ["model.layers", "model.model.layers",
                     "transformer.h", "model.decoder.layers"]:
            obj = m
            ok = True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        raise RuntimeError(f"{self.name}: cannot locate decoder blocks")

    def block_for_hidden_index(self, L):
        """hidden_states[L] is the OUTPUT of decoder block L-1
        (index 0 = embeddings). Verified per-model by the dual-path
        capture cross-check gate."""
        assert 1 <= L <= self.n_layers
        return self.blocks[L - 1]

    # ---- template + positions (ports of m8.build_forced, per-model) ----

    def template_text(self, msgs, gen_prompt=True):
        if self.encoder is not None:
            return self.encoder.render(msgs, gen_prompt)
        return self.tok.apply_chat_template(
            msgs, add_generation_prompt=gen_prompt, tokenize=False)

    def ids_of(self, text):
        import torch
        return self.tok(text, return_tensors="pt",
                        add_special_tokens=False)["input_ids"][0]

    def tokcount(self, s):
        return len(self.tok(s, add_special_tokens=False)["input_ids"])

    def template_id_consistency(self, msgs):
        """Mission Stage-0 gate 6 helper: tokenize=True ids must equal
        string-then-tokenize ids exactly. For encoder-based models
        there is no tokenize=True path — integrity is carried by the
        official-test-vector gate instead (logged)."""
        import torch
        if self.encoder is not None:
            return True
        a = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt")
        if not hasattr(a, "shape"):
            a = a["input_ids"]
        b = self.ids_of(self.template_text(msgs, gen_prompt=True))
        return a.shape[-1] == len(b) and bool(
            (a[0].cpu() == b.cpu()).all())

    def build_forced(self, body, forced_str, answer_prompt=None,
                     conf_prompt=None):
        ap = answer_prompt or m8.ANSWER_PROMPT
        cp = conf_prompt or m8.CONF_PROMPT
        u1 = {"role": "user", "content": ap.format(body=body)}
        t1 = self.template_text([u1], gen_prompt=True)
        t2 = t1 + forced_str
        msgs3 = [u1, {"role": "assistant", "content": forced_str},
                 {"role": "user", "content": cp}]
        t3 = self.template_text(msgs3, gen_prompt=True)
        if not t3.startswith(t2):
            raise m8.AlignError("template prefix broken (string)")
        ids1, ids2, ids3 = (self.ids_of(t1), self.ids_of(t2),
                            self.ids_of(t3))
        n1, n2, n3 = len(ids1), len(ids2), len(ids3)
        if not (n1 < n2 < n3):
            raise m8.AlignError(f"length order {n1} {n2} {n3}")
        if not (ids2[:n1] == ids1).all() or not (ids3[:n2] == ids2).all():
            raise m8.AlignError("template prefix broken (token)")
        got = self.tok.decode(ids3[n1:n2])
        if got != forced_str:
            raise m8.AlignError(f"answer span {got!r} != {forced_str!r}")
        t_nogen = self.template_text(msgs3, gen_prompt=False)
        if not t3.startswith(t_nogen):
            raise m8.AlignError("no-gen template not a string prefix")
        ids_ng = self.ids_of(t_nogen)
        if not (ids3[:len(ids_ng)] == ids_ng).all():
            raise m8.AlignError("no-gen template not a token prefix")
        return {"ids3": ids3, "pos_pre": n1 - 1, "pos_ans": n2 - 1,
                "pos_rep": n3 - 1, "ans_span": (n1, n2),
                "rep_start": len(ids_ng)}

    # ---- forward + capture ----

    def _dev_in(self):
        import torch
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, ids, hidden=False):
        import torch
        with torch.no_grad():
            return self.model(ids.unsqueeze(0).to(self._dev_in()),
                              output_hidden_states=hidden)

    def capture(self, built):
        """Capture pass (nothing appended): residuals at the resolved
        layers x {pre, ans, rep}, forced logp, pos-REP logits fp32."""
        import torch
        out = self.forward(built["ids3"], hidden=True)
        feats = {}
        for L in self.layers:
            h = out.hidden_states[L][0]
            probe = h[built["pos_pre"]]
            if self.streams is None:
                self.streams = probe.shape[0] if probe.dim() == 2 else 1
            for pos_name, pos in [("pre", built["pos_pre"]),
                                  ("ans", built["pos_ans"]),
                                  ("rep", built["pos_rep"])]:
                feats[(L, pos_name)] = (h[pos].reshape(-1).float()
                                        .cpu().numpy())
        logits = out.logits[0].float().cpu()
        lp = torch.log_softmax(logits, dim=-1)
        flogp = 0.0
        for p in range(*built["ans_span"]):
            flogp += float(lp[p - 1, built["ids3"][p]])
        return feats, flogp, logits[built["pos_rep"]]

    def greedy(self, ids, max_new=8, steer=None):
        """Manual cached greedy with per-step absolute-position
        bookkeeping for the (optional) steerer."""
        import torch
        n0 = ids.shape[0]
        if steer is not None:
            steer.set_offset(0)
        with torch.no_grad():
            out = self.model(ids.unsqueeze(0).to(self._dev_in()),
                             use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
        toks = []
        for step in range(max_new):
            toks.append(nxt)
            d = self.tok.decode([nxt])
            if nxt == self.tok.eos_token_id or "end_of_turn" in d \
                    or "<|end" in d:
                break
            if step == max_new - 1:
                break
            if steer is not None:
                steer.set_offset(n0 + step)
            with torch.no_grad():
                out = self.model(
                    torch.tensor([[nxt]], device=self._dev_in()),
                    past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
        return self.tok.decode(toks, skip_special_tokens=True)


def load_handle(spec_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    spec = SPECS[spec_name]
    kw = dict(spec["load"])
    tok = AutoTokenizer.from_pretrained(spec["repo"],
                                        revision=spec["revision"])
    encoder = DSV4Encoder() if spec.get("encoder") == "dsv4" else None
    if kw.pop("quant4bit", False):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    if kw.get("torch_dtype") == "bfloat16":
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], **kw)
    if "quantization_config" not in kw and "device_map" not in kw:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return Handle(spec_name, model, tok, encoder=encoder)


# ----------------------------- PER-ROW STEERER --------------------------------

class SteererRow:
    """Forward hook adding scale * vec at absolute positions >=
    start[b] PER ROW. `starts` is always a 1-D tensor (batch,), never
    a scalar; per-row asserts fire on configure."""

    def __init__(self, handle, L):
        import torch
        self._torch = torch
        self.handle = handle
        self.L = L
        self.vec = None
        self.scale = 0.0
        self.starts = None
        self.offset = 0
        block = handle.block_for_hidden_index(L)
        self.hook = block.register_forward_hook(self._fn)

    def configure(self, vec, scale, starts, seq_lens):
        t = self._torch
        starts = t.as_tensor(starts)
        seq_lens = t.as_tensor(seq_lens)
        assert starts.dim() == 1, "starts must be per-row, never scalar"
        assert (starts >= 0).all(), "negative steer start"
        assert (starts < seq_lens).all(), \
            f"steer start beyond sequence ({starts} vs {seq_lens})"
        self.vec = vec
        self.scale = float(scale)
        self.starts = starts

    def set_offset(self, off):
        self.offset = off

    def off(self):
        self.scale = 0.0

    def remove(self):
        self.hook.remove()

    def _fn(self, module, args, output):
        if self.scale == 0.0 or self.vec is None or self.starts is None:
            return
        t = self._torch
        hs = output[0] if isinstance(output, tuple) else output
        if hs.dim() == 4:                    # multi-stream (B,S,K,H)
            B, S, K, H = hs.shape
            vec = self.vec.reshape(K, H).to(hs.device, hs.dtype)
        else:
            B, S, H = hs.shape
            vec = self.vec.to(hs.device, hs.dtype)
        assert B == len(self.starts), \
            f"batch {B} != starts {len(self.starts)}"
        pos = t.arange(self.offset, self.offset + S, device=hs.device)
        for b in range(B):
            mask = pos >= int(self.starts[b])
            if mask.any():
                hs[b, mask] += self.scale * vec
        return


# ----------------------------- REPORT-TOKEN LAYOUT ----------------------------

class ReportLayout:
    """Runtime-derived margin machinery. Tokenizes every report string
    "0".."100" with THIS tokenizer, asserts decode round-trips, and
    builds the value->token-path map plus the shared-prefix trie that
    determines which conditional positions must be teacher-forced to
    compute exact class masses. Never uses hard-coded token constants."""

    HIGH = set(range(80, 101))
    LOW = set(range(0, 60))       # 60-79 excluded, stated in prereg

    def __init__(self, handle):
        self.h = handle
        self.paths = {}
        for v in range(101):
            s = str(v)
            ids = handle.tok(s, add_special_tokens=False)["input_ids"]
            rt = handle.tok.decode(ids)
            if rt != s:
                raise RuntimeError(
                    f"report string {s!r} does not round-trip "
                    f"({rt!r}) — layout gate RED")
            self.paths[v] = tuple(ids)
        # shared prefixes that need conditional expansion: any proper
        # prefix of some path that is also shared across values with
        # different class membership
        self.prefixes = set()
        for v, p in self.paths.items():
            for k in range(1, len(p)):
                self.prefixes.add(p[:k])
        self.max_path_len = max(len(p) for p in self.paths.values())

    def first_token_classes(self):
        """first token id -> set of values starting with it."""
        out = {}
        for v, p in self.paths.items():
            out.setdefault(p[0], set()).add(v)
        return out

    def _cls(self, v):
        return "H" if v in self.HIGH else ("L" if v in self.LOW
                                           else "X")

    def _subtree_classes(self, pref):
        return {self._cls(v) for v, p in self.paths.items()
                if p[:len(pref)] == pref}

    def needed_conditionals(self):
        """Prefixes that must be teacher-forced for exact class
        masses: exactly those whose subtree spans MORE THAN ONE class
        (H/L/excluded). A class-homogeneous subtree contributes its
        whole first-token mass without a split — the generalization
        of M8's frozen formula (lead-1 mass split by the '100' path;
        homogeneous leads read directly)."""
        return [pref for pref in sorted(self.prefixes)
                if len(self._subtree_classes(pref)) > 1]

    def class_masses(self, lp_first, cond_lp):
        """Exact P(HIGH), P(LOW) by recursive subtree walk. Mass of a
        class-homogeneous subtree = its prefix mass; mixed subtrees
        split via the teacher-forced conditionals; mass of a value
        whose path ends at a mixed prefix = prefix mass minus the
        continuing children's mass (M8's 'P(1) - P(100 path)'
        semantics, generalized)."""
        import math
        out = {"H": 0.0, "L": 0.0, "X": 0.0}

        def walk(pref, mass):
            classes = self._subtree_classes(pref)
            if not classes:
                return
            if len(classes) == 1:
                out[classes.pop()] += mass
                return
            if pref not in cond_lp:
                raise KeyError(f"missing conditional for {pref}")
            cond = cond_lp[pref]
            used = 0.0
            nxt = {p[len(pref)] for v, p in self.paths.items()
                   if len(p) > len(pref) and p[:len(pref)] == pref}
            for t in nxt:
                q = math.exp(float(cond[t]))
                used += q
                walk(pref + (t,), mass * q)
            ends = [v for v, p in self.paths.items() if p == pref]
            if ends:
                out[self._cls(ends[0])] += mass * max(0.0, 1.0 - used)

        firsts = {p[0] for p in self.paths.values()}
        for t in sorted(firsts):
            walk((t,), math.exp(float(lp_first[t])))
        return out["H"], out["L"]

    def selftest_planted(self):
        """Mission Stage -1 item 5: plant 1-, 2-, 3-token report
        strings through a synthetic distribution and assert the
        machinery reads logp at the right positions. Uses a fake
        lp table, no model."""
        import math
        # find example values with 1, 2, 3 token paths if they exist
        by_len = {}
        for v, p in self.paths.items():
            by_len.setdefault(len(p), v)
        results = {}
        for L, v in sorted(by_len.items()):
            if L > 3:
                continue
            # synthetic: put 0.7 on v's exact path, spread the rest
            lp_first = {}
            for tid in {p[0] for p in self.paths.values()}:
                lp_first[tid] = math.log(0.3 / 200)
            lp_first[self.paths[v][0]] = math.log(0.7) if L == 1 else \
                math.log(0.9)
            cond = {}
            for pref in self.needed_conditionals():
                cond[pref] = {}
                nxts = {p[len(pref)] for p in self.paths.values()
                        if len(p) > len(pref) and p[:len(pref)] == pref}
                for t in nxts:
                    cond[pref][t] = math.log(0.05 / max(1, len(nxts)))
                pth = self.paths[v]
                if len(pth) > len(pref) and pth[:len(pref)] == pref:
                    cond[pref][pth[len(pref)]] = math.log(0.85)
            ph, pl = self.class_masses(lp_first, cond)
            hi = v in self.HIGH
            lo = v in self.LOW
            ok = (ph > pl) if hi else ((pl > ph) if lo else True)
            results[L] = ok
        return all(results.values()), results


# ----------------------------- STAGE -1 GATES ---------------------------------

def replay_gate():
    """Feed literal M8 rows through this pipeline on gemma; exact
    answer/parse match; primary-layer AUC within bootstrap noise of
    the frozen M8 result."""
    log("Stage -1 REPLAY regression gate (gemma-3-1b, M8 literal rows)")
    meta, data, _ = m8.load_run()
    rows = meta["rows"]
    h = load_handle("gemma")
    assert h.n_layers == 26 and h.hidden == 1152
    assert 24 in h.layers and h.primary_layer == 24, \
        f"depth fractions must include gemma L24 (got {h.layers})"
    # sample: all rows of the first 60 items + 20 random others (seeded)
    items = sorted({r["item"] for r in rows})
    rng = random.Random(7)
    pick = set(items[:60]) | set(rng.sample(items[60:], 20))
    sub = [r for r in rows if r["item"] in pick]
    n_ok = 0
    feats_new = {}
    for r in sub:
        built = h.build_forced(r["body"], r["forced_str"])
        ok = (built["pos_pre"] == r["pos_pre"]
              and built["pos_ans"] == r["pos_ans"]
              and built["pos_rep"] == r["pos_rep"]
              and list(built["ans_span"]) == r["ans_span"]
              and len(built["ids3"]) == r["seq_len"])
        if not ok:
            log(f"  REPLAY MISMATCH positions at row {r['row']}")
            return False
        feats, flogp, rep_logits = h.capture(built)
        if abs(flogp - r["forced_logp"]) > 0.05:
            log(f"  REPLAY MISMATCH forced_logp at row {r['row']}: "
                f"{flogp:.4f} vs {r['forced_logp']:.4f}")
            return False
        feats_new[r["row"]] = feats[(24, "ans")]
        n_ok += 1
    # activation agreement with the frozen npz (fp16 storage tolerance)
    max_dev = 0.0
    for row_id, vec in feats_new.items():
        frozen = data["L24_ans"][row_id].astype(np.float32)
        max_dev = max(max_dev, float(np.max(np.abs(
            vec - frozen))))
    log(f"  replay: {n_ok}/{len(sub)} rows positions+logp exact; "
        f"max L24-ans activation deviation vs frozen fp16: "
        f"{max_dev:.4f}")
    ok = max_dev < 1.0        # fp16 storage quantum at these scales
    # AUC within bootstrap noise: refit on the frozen matrices with
    # this pipeline's grouped CV (identical code path to M8 via m8)
    X = data["L24_ans"].astype(float)
    y = np.array([1 if r["correct"] else 0 for r in rows])
    groups = np.array([r["item"] for r in rows])
    s, folds = m8.oof_scores(X, y, groups, seed=1)
    auc = m8.auroc(s, y)
    frozen_auc = json.load(open(m8.latest_path(m8.RESULTS_FILE),
                                encoding="utf-8"))[
        "force"]["L24"]["overall_auc"]
    log(f"  replay AUC {auc:.4f} vs frozen {frozen_auc:.4f}")
    ok = ok and abs(auc - frozen_auc) < 0.02
    log(f"REPLAY gate {'GREEN' if ok else 'RED'}")
    return ok


def rehearse_gate():
    """Heterogeneous device_map rehearsal: gemma 4-bit forced onto
    cuda:0 + cpu; capture/alignment/injection/lens selftests through
    THIS pipeline."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    log("Stage -1 HETEROGENEOUS-MAP rehearsal (gemma 4-bit, "
        "cuda:0+cpu split)")
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    # explicit split map built by WALKING the real module tree (a
    # hand-list misses gemma-3 extras like rotary_emb_local; an
    # auto-map with a small GPU budget routes quantized layers into
    # accelerate's offload hooks, which bnb tensors cannot survive).
    # Quantized decoder layers stay on GPU; the second half runs fp32
    # on CPU via llm_int8_enable_fp32_cpu_offload. Tied embed/lm_head
    # stay together on GPU.
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("google/gemma-3-1b-it")
    with init_empty_weights():
        skel = AutoModelForCausalLM.from_config(cfg)
    dm = {}
    for name, child in skel.named_children():
        if name == "model":
            for n2, c2 in child.named_children():
                if n2 == "layers":
                    n_blocks = len(c2)
                    for i in range(n_blocks):
                        dm[f"model.layers.{i}"] = 0 \
                            if i < n_blocks // 2 else "cpu"
                else:
                    dm[f"model.{n2}"] = 0
        else:
            dm[name] = 0
    del skel
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it", device_map=dm,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # required by bnb for quantized models with CPU-dispatched
            # modules (the rehearsal's forced split); CPU-side modules
            # stay fp32 — fine, we rehearse dispatch+hooks, not speed
            llm_int8_enable_fp32_cpu_offload=True),
        attn_implementation="eager")
    model.eval()
    h = Handle("gemma-split", model, tok)
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    check(h.n_layers == 26, "blocks located across devices")
    devs = set(str(v) for v in getattr(model, "hf_device_map",
                                       {}).values())
    check(any("cpu" in d for d in devs)
          and any(d in ("0", "cuda:0") for d in devs),
          f"device map genuinely split (devices: {sorted(devs)})")
    body, true, _ = m8.gen_tracking(0, random.Random(11))
    built = h.build_forced(body, str(true))
    feats, flogp, rep_logits = h.capture(built)
    check(all(np.isfinite(v).all() for v in feats.values()),
          "capture finite on split map")
    f2, l2, r2 = h.capture(built)
    dev = max(float(np.max(np.abs(feats[k] - f2[k]))) for k in feats)
    check(dev == 0.0 or dev < 1e-3,
          f"repeated split-map capture stable (max dev {dev:.2e})")
    # injection through the per-row steerer on a CPU-side layer
    L_cpu = h.layers[-1]
    steer = SteererRow(h, L_cpu)
    vec = torch.ones(h.hidden, dtype=torch.bfloat16)
    starts = torch.tensor([built["rep_start"]])
    seq = torch.tensor([len(built["ids3"])])
    steer.configure(vec, 1000.0, starts, seq)
    steer.set_offset(0)
    out = h.forward(built["ids3"], hidden=True)
    delta = (out.hidden_states[L_cpu][0, built["pos_rep"]]
             .float().cpu().numpy() - feats[(L_cpu, "rep")])
    check(np.abs(delta - 1000.0).max() < 150.0,
          f"injection lands on split map at L{L_cpu} "
          f"(max dev {np.abs(delta - 1000.0).max():.1f})")
    steer.off()
    # per-row assert fires
    try:
        steer.configure(vec, 1.0,
                        torch.tensor([len(built['ids3']) + 5]), seq)
        check(False, "per-row start assert should have fired")
    except AssertionError:
        check(True, "per-row start assert fires on bad start")
    steer.remove()
    # lens on split map (norm may live on cpu)
    hstack = out.hidden_states[-1][0, built["pos_rep"]].float()
    with torch.no_grad():
        lens = model.lm_head(hstack.to(
            model.lm_head.weight.device).to(model.lm_head.weight.dtype))
    real = out.logits[0, built["pos_rep"]].float()
    cos = float(torch.nn.functional.cosine_similarity(
        lens.float().cpu().flatten(), real.cpu().flatten(), dim=0))
    check(cos > 0.999, f"lens identity on split map (cos {cos:.5f})")
    log(f"REHEARSAL {'GREEN' if ok else 'RED'}")
    return ok


def dryrun_v4():
    """Meta-device dry run: V4-Flash architecture resolved without
    weights; layer/hook/template/report-layout machinery verified."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, \
        AutoTokenizer
    log("Stage -1 META-DEVICE dry run: DeepSeek-V4-Flash @ "
        + V4_REVISION[:12])
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    cfg = AutoConfig.from_pretrained(V4_REPO, revision=V4_REVISION)
    tok = AutoTokenizer.from_pretrained(V4_REPO, revision=V4_REVISION)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    enc = DSV4Encoder()
    h = Handle("v4flash-meta", model, tok, meta_only=True,
               encoder=enc)
    # OFFICIAL ENCODING TEST SUITE: run the repo's own
    # test_encoding_dsv4.py verbatim in our pinned environment — the
    # strongest form of the template gate (their harness, their
    # vectors, our interpreter)
    import subprocess
    env = dict(os.environ, PYTHONUTF8="1")   # their file opens the
    # vectors without encoding=; Windows cp1252 chokes on the unicode
    # special tokens; the pod (Linux) defaults to UTF-8 anyway
    r = subprocess.run(
        [sys.executable, os.path.join(_v4_encoding_dir(),
                                      "test_encoding_dsv4.py")],
        capture_output=True, text=True, cwd=_v4_encoding_dir(),
        timeout=120, env=env)
    n_pass = r.stdout.count("[PASS]")
    check(r.returncode == 0 and n_pass >= 4,
          f"official encoding test suite passes verbatim "
          f"(exit {r.returncode}, {n_pass} [PASS] lines)")
    if r.returncode != 0:
        log(f"  suite output tail: {(r.stdout + r.stderr)[-400:]}")
    check(h.n_layers == 43, f"43 decoder blocks located "
                            f"(got {h.n_layers})")
    check(h.hidden == 4096, f"hidden size 4096 (got {h.hidden})")
    log(f"  depth fractions -> layers {h.layers}, primary "
        f"L{h.primary_layer}")
    check(len(h.layers) == len(DEPTH_FRACTIONS),
          "depth fractions resolve to distinct layers")
    for L in h.layers:
        blk = h.block_for_hidden_index(L)
        check(blk is not None, f"hook point resolves for L{L}")
    # template machinery on the real tokenizer (no weights needed)
    body, true, _ = m8.gen_tracking(0, random.Random(5))
    try:
        built = h.build_forced(body, str(true))
        check(built["rep_start"] <= built["pos_rep"],
              f"template prefixes + positions resolve "
              f"(rep_start {built['rep_start']}, pos_rep "
              f"{built['pos_rep']})")
    except m8.AlignError as e:
        check(False, f"template prefix machinery: {e}")
    msgs = [{"role": "user", "content": "hi"}]
    check(h.template_id_consistency(msgs),
          "apply_chat_template(tokenize=True) == string-then-tokenize")
    # report-token layout on the real tokenizer
    layout = ReportLayout(h)
    ftc = layout.first_token_classes()
    conds = layout.needed_conditionals()
    log(f"  report paths: max token length "
        f"{layout.max_path_len}; {len(ftc)} distinct first tokens; "
        f"{len(conds)} conditional prefixes needed: "
        f"{[h.tok.decode(list(p)) for p in conds][:10]}")
    planted_ok, per_len = layout.selftest_planted()
    check(planted_ok, f"planted-report margin selftest "
                      f"(path lengths {sorted(per_len)})")
    import transformers
    pin = {"transformers": transformers.__version__,
           "torch": torch.__version__,
           "repo": V4_REPO, "revision": V4_REVISION,
           "n_layers": h.n_layers, "hidden": h.hidden,
           "layers": h.layers, "primary_layer": h.primary_layer,
           "report_max_path_len": layout.max_path_len,
           "report_conditional_prefixes":
               [list(p) for p in conds]}
    with open("m10_v4_pin.json", "w", encoding="utf-8") as f:
        json.dump(pin, f, indent=1)
    log(f"  saved m10_v4_pin.json")
    log(f"DRY RUN {'GREEN' if ok else 'RED'}")
    return ok


def layout_gate(spec_name):
    log(f"Report-token layout gate for {spec_name}")
    h = load_handle(spec_name)
    layout = ReportLayout(h)
    ok, per_len = layout.selftest_planted()
    log(f"  round-trips 0-100 OK; planted selftest "
        f"{'GREEN' if ok else 'RED'} (path lengths {sorted(per_len)})")
    return ok


# ----------------------------- MAIN -------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--dryrun-v4", action="store_true")
    ap.add_argument("--layout", type=str)
    args = ap.parse_args()
    if args.replay:
        sys.exit(0 if replay_gate() else 1)
    if args.rehearse:
        sys.exit(0 if rehearse_gate() else 1)
    if getattr(args, "dryrun_v4"):
        sys.exit(0 if dryrun_v4() else 1)
    if args.layout:
        sys.exit(0 if layout_gate(args.layout) else 1)
    ap.print_help()
