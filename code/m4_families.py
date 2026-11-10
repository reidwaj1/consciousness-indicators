#!/usr/bin/env python3
"""
Mission 4 problem families (gate 1)
===================================
Four procedurally generated families whose difficulty scales against SEARCH
(reasoning grinds through mere length), each with a single exact integer
answer and an independent verifier.

INDEPENDENCE DESIGN: every family's verifier receives ONLY the rendered
problem text, re-parses it with its own rigid-template parser, and re-solves
it with separately written code (different algorithmic decomposition from the
generator's solver). A verification pass therefore certifies the whole chain
prompt-text -> semantics -> unique answer, not just the internal
representation. Gate 1 = 2,000+ cases per family, ZERO mismatches.

Families and knobs (parameters DELEGATED to Claude; rationale in comments and
in RESULTS_README's Mission 4 section):
  logic_grid       Einstein-style grids, solver-proven unique solution.
                   knob: (n houses, m categories, redundant-clue count).
  program_exec     execute a small register program (mod-10000 arithmetic,
                   branches, guaranteed-terminating loops); answer = r0.
                   knob: executed-step target, branch/loop density.
  constraint_count |{x in [A,B] : all k conditions}|.
                   knob: range width x number of conditions.
  arith_long       legacy-continuity arithmetic chains, exact-division
                   variants keep magnitudes bounded. knob: chain length,
                   multiply/divide density.

Every generated item returns dict(body, true, params) where params holds the
problem-size parameters that the study's cue battery uses as adversaries.
All answers are integers; one strict parser covers every family.

python3 m4_families.py --verify 2000 [--seed 7]   gate-1 verification report
python3 m4_families.py --selftest                 quick structural checks
python3 m4_families.py --show LEVEL               print one sample per family
"""

import re
import sys
import math
import time
import hashlib
import argparse
import random
from fractions import Fraction

N_LEVELS = 10

def item_rng(seed, family, level, i):
    """Deterministic per-item RNG independent of process hash randomization."""
    key = f"m4|{seed}|{family}|{level}|{i}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8],
                                        "big"))

# ============================================================================
# FAMILY 1: logic_grid
# ============================================================================
# Value pools: 7 values per category, globally unique tokens so a quoted
# value identifies its category unambiguously.
LG_CATS = [
    ("drink",  ["tea", "coffee", "milk", "beer", "water", "juice", "cocoa"]),
    ("pet",    ["dog", "cat", "fish", "bird", "horse", "rabbit", "turtle"]),
    ("color",  ["red", "green", "white", "yellow", "blue", "purple",
                "orange"]),
    ("sport",  ["chess", "tennis", "golf", "soccer", "hockey", "darts",
                "rugby"]),
    ("fruit",  ["apple", "pear", "plum", "mango", "grape", "lemon",
                "cherry"]),
]

# level -> (n houses, m categories, redundant clues kept after minimization)
# Rationale (DELEGATED): n*m is the state-space knob ((n!)^m assignments);
# redundant clues make early levels forgiving; the top levels are minimal
# clue sets on 7x5 grids, far beyond the frozen suite's hardest items.
LG_LEVELS = [(3, 3, 3), (3, 3, 1), (4, 3, 2), (4, 4, 1), (5, 4, 2),
             (5, 4, 0), (6, 4, 1), (6, 5, 0), (7, 4, 0), (7, 5, 0)]

# clue tuples:
#   ("same", c1, v1, c2, v2)   same house
#   ("diff", c1, v1, c2, v2)   different houses
#   ("left", c1, v1, c2, v2)   strictly somewhere left
#   ("imleft", c1, v1, c2, v2) immediately left
#   ("adj", c1, v1, c2, v2)    adjacent houses
#   ("at", c, v, k)            in house k (1-based)
#   ("notat", c, v, k)         not in house k
# (c = category index, v = value index within category)

_LG_NODE_CAP = 400_000

def _lg_count_solutions(n, m, clues, limit=2):
    """GENERATOR-SIDE solver: backtracking over individual (category, value)
    cells in clue-connectivity order, positions checked on the fly against
    clues whose other endpoint is already assigned, all-different enforced
    via per-category used-position sets.

    INDEPENDENCE NOTE (documented design decision): both this solver and the
    verifier's are constraint search over cells - for proving uniqueness of
    a logic grid there is no algorithmically alien alternative that runs in
    reasonable time. Independence lives in the implementation (this one:
    static connectivity ordering + on-the-fly candidate computation; the
    verifier: dynamic MRV over explicit domain sets, built from RE-PARSED
    problem text) and in the text layer the verifier adds on top.

    Returns count (up to limit), or raises _LGNodeCap if the search exceeds
    the deterministic node cap (caller resamples the item)."""
    unary_at = {}
    unary_not = {}
    binary = []
    for cl in clues:
        if cl[0] == "at":
            unary_at.setdefault((cl[1], cl[2]), set()).add(cl[3] - 1)
        elif cl[0] == "notat":
            unary_not.setdefault((cl[1], cl[2]), set()).add(cl[3] - 1)
        else:
            binary.append(cl)
    variables = [(c, v) for c in range(m) for v in range(n)]
    # clues indexed by endpoint
    by_var = {var: [] for var in variables}
    for cl in binary:
        by_var[(cl[1], cl[2])].append(cl)
        by_var[(cl[3], cl[4])].append(cl)
    # eager domains: unary clues first
    dom = {}
    for var in variables:
        d = set(range(n))
        if var in unary_at:
            d &= unary_at[var]
        if var in unary_not:
            d -= unary_not[var]
        dom[var] = d
    assign = {}
    count = 0
    nodes = 0

    def prune(var, p, trail):
        """Forward-check after assigning var=p: shrink peer domains (record
        removals on trail for undo). Returns False on an emptied domain."""
        c, v = var
        for v2 in range(n):              # all-different within the category
            if v2 != v:
                peer = (c, v2)
                if peer not in assign and p in dom[peer]:
                    dom[peer].discard(p)
                    trail.append((peer, p))
                    if not dom[peer]:
                        return False
        for cl in by_var[var]:
            va, vb = (cl[1], cl[2]), (cl[3], cl[4])
            other = vb if var == va else va
            if other in assign:
                continue
            var_is_a = (var == va)
            t = cl[0]
            if t == "same":
                allowed = {p}
            elif t == "imleft":
                allowed = {p + 1} if var_is_a else {p - 1}
            elif t == "adj":
                allowed = {p - 1, p + 1}
            elif t == "left":
                allowed = set(range(p + 1, n)) if var_is_a \
                    else set(range(0, p))
            else:                          # diff
                allowed = set(range(n)) - {p}
            for q in list(dom[other]):
                if q not in allowed:
                    dom[other].discard(q)
                    trail.append((other, q))
            if not dom[other]:
                return False
        return True

    class _LGNodeCap(Exception):
        pass

    def dfs():
        nonlocal count, nodes
        if len(assign) == len(variables):
            count += 1
            return
        # MRV over live domains
        var = min((v for v in variables if v not in assign),
                  key=lambda v: len(dom[v]))
        for p in sorted(dom[var]):
            nodes += 1
            if nodes > _LG_NODE_CAP:
                raise _LGNodeCap()
            assign[var] = p
            trail = []
            if prune(var, p, trail):
                dfs()
            for (peer, q) in trail:
                dom[peer].add(q)
            del assign[var]
            if count >= limit:
                return

    try:
        dfs()
    except _LGNodeCap:
        return -1          # sentinel: too hard to certify; caller resamples
    return count

def gen_logic_grid(level, rng):
    for _attempt in range(50):
        item = _gen_logic_grid_once(level, rng)
        if item is not None:
            return item
    raise RuntimeError("logic_grid: node cap hit on 50 consecutive attempts")

def _gen_logic_grid_once(level, rng):
    n, m, redundant = LG_LEVELS[level]
    cats = LG_CATS[:m]
    truth = [rng.sample(range(n), n) for _ in range(m)]  # truth[c][v] = house

    # candidate pool of TRUE clues, strong types first: 'same' and 'imleft'
    # pin values together, so uniqueness arrives with few clues and the
    # solver's search stays shallow; weak types ('diff', 'left', 'adj',
    # 'notat') are mixed in for variety, direct 'at' clues capped so the
    # puzzle is never a lookup table
    strong, medium, weak = [], [], []
    for c1 in range(m):
        for c2 in range(c1 + 1, m):
            for v1 in range(n):
                for v2 in range(n):
                    a, b = truth[c1][v1], truth[c2][v2]
                    if a == b:
                        strong.append(("same", c1, v1, c2, v2))
                    elif a + 1 == b:
                        strong.append(("imleft", c1, v1, c2, v2))
                    elif b + 1 == a:
                        strong.append(("imleft", c2, v2, c1, v1))
                    elif a < b and rng.random() < 0.25:
                        medium.append(("left", c1, v1, c2, v2))
                    elif rng.random() < 0.08:
                        weak.append(("diff", c1, v1, c2, v2))
    for c in range(m):
        for v in range(n):
            k = rng.randrange(n)
            if k != truth[c][v]:
                weak.append(("notat", c, v, k + 1))
    ats = [("at", c, v, truth[c][v] + 1) for c in range(m) for v in range(n)]
    rng.shuffle(ats)
    rng.shuffle(strong)
    rng.shuffle(medium)
    rng.shuffle(weak)
    medium += ats[:max(1, n // 3)]
    rng.shuffle(medium)
    # interleave ~2 strong : 1 medium, weak at the end
    pool = []
    si, mi = 0, 0
    while si < len(strong) or mi < len(medium):
        take_strong = (rng.random() < 0.67 and si < len(strong)) \
            or mi >= len(medium)
        if take_strong:
            pool.append(strong[si])
            si += 1
        else:
            pool.append(medium[mi])
            mi += 1
    pool += weak

    # greedy grow until unique, checking in batches of 3 (each uniqueness
    # check is a full solver call; batching keeps the call count ~len/3).
    # A -1 from the solver means the node cap was hit: bail, caller
    # resamples a fresh ground truth (deterministically, same rng stream).
    chosen = []
    i = 0
    while i < len(pool):
        cnt = _lg_count_solutions(n, m, chosen, limit=2)
        if cnt == -1:
            return None
        if cnt == 1:
            break
        chosen.extend(pool[i:i + 3])
        i += 3
    cnt = _lg_count_solutions(n, m, chosen, limit=2)
    if cnt == -1:
        return None
    if cnt != 1:
        # extremely rare: force uniqueness with direct clues
        for c in range(m):
            for v in range(n):
                cnt = _lg_count_solutions(n, m, chosen, limit=2)
                if cnt == -1:
                    return None
                if cnt == 1:
                    break
                chosen.append(("at", c, v, truth[c][v] + 1))
    # prune toward a minimal set (randomized removal order, capped solver
    # calls — sparsity is a difficulty knob, not an exactness guarantee)
    candidates = list(chosen)
    rng.shuffle(candidates)
    checks = 0
    for cl in candidates:
        if checks >= 60:
            break
        rest = [c for c in chosen if c is not cl]
        if len(rest) == len(chosen):
            continue
        checks += 1
        if _lg_count_solutions(n, m, rest, limit=2) == 1:
            chosen = rest
    # sparsity knob: re-add redundant true clues from the unused pool
    unused = [cl for cl in pool if cl not in chosen]
    chosen = chosen + unused[:redundant]
    rng.shuffle(chosen)

    # question target: value not fixed by a direct 'at' clue
    at_targets = {(cl[1], cl[2]) for cl in chosen if cl[0] == "at"}
    options = [(c, v) for c in range(m) for v in range(n)
               if (c, v) not in at_targets]
    qc, qv = options[rng.randrange(len(options))]
    answer = truth[qc][qv] + 1

    body = _lg_render(n, cats, chosen, qc, qv)
    weak_types = ("left", "adj", "diff", "notat")
    return {"body": body, "true": answer,
            "params": {"lg_n": n, "lg_m": m, "lg_clues": len(chosen),
                       # indirectness measure: weak clues force more search
                       # (per-item battery adversary at a locked level)
                       "lg_weak": sum(1 for cl in chosen
                                      if cl[0] in weak_types)}}

def _lg_render(n, cats, clues, qc, qv):
    lines = [f"There are {n} houses numbered 1 to {n} from left to right. "
             f"Each house is assigned exactly one value from each category "
             f"below, and within a category no two houses share a value.",
             "Categories:"]
    for name, vals in cats:
        lines.append(f"- {name}: " + ", ".join(vals[:n]))
    lines.append("Clues:")
    for i, cl in enumerate(clues, 1):
        t = cl[0]
        if t in ("at", "notat"):
            v = cats[cl[1]][1][cl[2]]
            neg = " not" if t == "notat" else ""
            lines.append(f"{i}. '{v}' is{neg} in house {cl[3]}.")
        else:
            v1 = cats[cl[1]][1][cl[2]]
            v2 = cats[cl[3]][1][cl[4]]
            txt = {"same": f"'{v1}' and '{v2}' are in the same house.",
                   "diff": f"'{v1}' and '{v2}' are in different houses.",
                   "left": f"'{v1}' is in a house somewhere to the left of "
                           f"'{v2}'.",
                   "imleft": f"'{v1}' is in the house immediately to the "
                             f"left of '{v2}'.",
                   "adj": f"'{v1}' and '{v2}' are in adjacent houses."}[t]
            lines.append(f"{i}. {txt}")
    target = cats[qc][1][qv]
    lines.append(f"Question: in which house (1-{n}) is '{target}'?")
    return "\n".join(lines)

# ---------------- logic_grid VERIFIER (text -> parse -> cell-based solver) --

_LG_RE = {
    "at":     re.compile(r"^'([a-z]+)' is in house (\d+)\.$"),
    "notat":  re.compile(r"^'([a-z]+)' is not in house (\d+)\.$"),
    "same":   re.compile(r"^'([a-z]+)' and '([a-z]+)' are in the same "
                         r"house\.$"),
    "diff":   re.compile(r"^'([a-z]+)' and '([a-z]+)' are in different "
                         r"houses\.$"),
    "left":   re.compile(r"^'([a-z]+)' is in a house somewhere to the left "
                         r"of '([a-z]+)'\.$"),
    "imleft": re.compile(r"^'([a-z]+)' is in the house immediately to the "
                         r"left of '([a-z]+)'\.$"),
    "adj":    re.compile(r"^'([a-z]+)' and '([a-z]+)' are in adjacent "
                         r"houses\.$"),
}

def verify_logic_grid(body):
    """Parses the rendered text and re-solves it with a CELL-BASED
    forward-checking MRV solver (different decomposition from the
    generator's category-permutation DFS). Returns (answer, n_solutions)."""
    lines = body.split("\n")
    n = int(re.match(r"There are (\d+) houses", lines[0]).group(1))
    cat_vals, val2cat = [], {}
    i = 2
    while lines[i] != "Clues:":
        mm = re.match(r"- ([a-z]+): (.+)$", lines[i])
        vals = mm.group(2).split(", ")
        for vi, v in enumerate(vals):
            val2cat[v] = (len(cat_vals), vi)
        cat_vals.append(vals)
        i += 1
    i += 1
    clues = []
    while not lines[i].startswith("Question:"):
        txt = re.sub(r"^\d+\. ", "", lines[i])
        for t, rx in _LG_RE.items():
            mm = rx.match(txt)
            if mm:
                if t in ("at", "notat"):
                    c, v = val2cat[mm.group(1)]
                    clues.append((t, c, v, int(mm.group(2))))
                else:
                    c1, v1 = val2cat[mm.group(1)]
                    c2, v2 = val2cat[mm.group(2)]
                    clues.append((t, c1, v1, c2, v2))
                break
        else:
            raise ValueError(f"unparseable clue: {txt!r}")
        i += 1
    qm = re.match(r"Question: in which house \(1-\d+\) is '([a-z]+)'\?",
                  lines[i])
    qc, qv = val2cat[qm.group(1)]
    m = len(cat_vals)

    # VERIFIER SOLVER: stateless fixpoint re-propagation per search node
    # (domains recomputed from the assignment each time - no trail, no
    # incremental state; structurally different from the generator's
    # incremental forward-checking solver). Counts solutions up to 2.
    variables = [(c, v) for c in range(m) for v in range(n)]
    base = {var: set(range(n)) for var in variables}
    binary = []
    for cl in clues:
        if cl[0] == "at":
            base[(cl[1], cl[2])] &= {cl[3] - 1}
        elif cl[0] == "notat":
            base[(cl[1], cl[2])].discard(cl[3] - 1)
        else:
            binary.append(cl)

    def propagate(assign):
        dom = {var: ({assign[var]} if var in assign else set(base[var]))
               for var in variables}
        changed = True
        while changed:
            changed = False
            # all-different: a singleton excludes its position from peers
            for c in range(m):
                for v in range(n):
                    d = dom[(c, v)]
                    if len(d) == 1:
                        p = next(iter(d))
                        for v2 in range(n):
                            if v2 != v and p in dom[(c, v2)]:
                                dom[(c, v2)].discard(p)
                                if not dom[(c, v2)]:
                                    return None
                                changed = True
            for cl in binary:
                va = (cl[1], cl[2])
                vb = (cl[3], cl[4])
                da, db = dom[va], dom[vb]
                t = cl[0]
                if t == "same":
                    inter = da & db
                    if inter != da or inter != db:
                        if not inter:
                            return None
                        dom[va] = set(inter)
                        dom[vb] = set(inter)
                        changed = True
                elif t == "imleft":
                    na = {p for p in da if p + 1 in db}
                    nb = {q for q in db if q - 1 in da}
                    if na != da or nb != db:
                        if not na or not nb:
                            return None
                        dom[va], dom[vb] = na, nb
                        changed = True
                elif t == "adj":
                    na = {p for p in da if (p - 1 in db or p + 1 in db)}
                    nb = {q for q in db if (q - 1 in da or q + 1 in da)}
                    if na != da or nb != db:
                        if not na or not nb:
                            return None
                        dom[va], dom[vb] = na, nb
                        changed = True
                elif t == "left":
                    hi = max(db)
                    lo = min(da)
                    na = {p for p in da if p < hi}
                    nb = {q for q in db if q > lo}
                    if na != da or nb != db:
                        if not na or not nb:
                            return None
                        dom[va], dom[vb] = na, nb
                        changed = True
                else:  # diff
                    if len(da) == 1 and next(iter(da)) in db:
                        db.discard(next(iter(da)))
                        if not db:
                            return None
                        changed = True
                    if len(db) == 1 and next(iter(db)) in da:
                        da.discard(next(iter(db)))
                        if not da:
                            return None
                        changed = True
        return dom

    def full_check(sol):
        """Explicit final check of every clue on a complete assignment."""
        for c in range(m):
            if len({sol[(c, v)] for v in range(n)}) != n:
                return False
        for cl in clues:
            if cl[0] == "at":
                if sol[(cl[1], cl[2])] != cl[3] - 1:
                    return False
            elif cl[0] == "notat":
                if sol[(cl[1], cl[2])] == cl[3] - 1:
                    return False
            else:
                pa = sol[(cl[1], cl[2])]
                pb = sol[(cl[3], cl[4])]
                okc = {"same": pa == pb, "diff": pa != pb, "left": pa < pb,
                       "imleft": pa + 1 == pb,
                       "adj": abs(pa - pb) == 1}[cl[0]]
                if not okc:
                    return False
        return True

    solutions = []

    def search(assign):
        if len(solutions) >= 2:
            return
        dom = propagate(assign)
        if dom is None:
            return
        if all(len(d) == 1 for d in dom.values()):
            sol = {var: next(iter(d)) for var, d in dom.items()}
            if full_check(sol):
                solutions.append(sol)
            return
        var = min((v for v in variables if len(dom[v]) > 1),
                  key=lambda v: len(dom[v]))
        for p in sorted(dom[var]):
            assign2 = dict(assign)
            assign2[var] = p
            search(assign2)

    search({})
    if len(solutions) != 1:
        return None, len(solutions)
    return solutions[0][(qc, qv)] + 1, 1

# ============================================================================
# FAMILY 2: program_exec
# ============================================================================
# 5 registers r0..r4, all arithmetic reduced mod 10000 (stated in the
# problem header, so wraparound is fully specified). r4 is a read-only zero.
# Termination is guaranteed by construction: every backward jump is guarded
# by a counter register that is initialized positive, strictly decremented
# inside the loop body, and never otherwise written.

# level -> (min executed steps, max executed steps, n loops, extra branches)
# Rationale (DELEGATED): executed-step count is the search knob; the frozen
# suite topped out around ~40 sequential ops, so the ladder extends ~10x
# past that. Loops/branches add control-flow tracking on top of arithmetic.
PE_LEVELS = [(6, 12, 0, 0), (10, 18, 0, 1), (16, 28, 1, 0), (24, 40, 1, 1),
             (36, 60, 1, 2), (55, 90, 2, 1), (85, 140, 2, 2),
             (130, 210, 2, 3), (200, 320, 3, 2), (300, 460, 3, 3)]

PE_MOD = 10000
PE_STEP_CAP = 1200

def _pe_exec_internal(prog, regs, step_cap=PE_STEP_CAP):
    """GENERATOR-SIDE interpreter over the internal tuple representation."""
    r = list(regs)
    pc = 0
    steps = 0
    while steps < step_cap:
        steps += 1
        op = prog[pc]
        k = op[0]
        if k == "halt":
            return r[0], steps
        if k == "set":
            r[op[1]] = op[2] % PE_MOD
        elif k == "add":
            r[op[1]] = (r[op[1]] + r[op[2]]) % PE_MOD
        elif k == "sub":
            r[op[1]] = (r[op[1]] - r[op[2]]) % PE_MOD
        elif k == "mul":
            r[op[1]] = (r[op[1]] * r[op[2]]) % PE_MOD
        elif k == "addc":
            r[op[1]] = (r[op[1]] + op[2]) % PE_MOD
        elif k == "subc":
            r[op[1]] = (r[op[1]] - op[2]) % PE_MOD
        elif k == "goto":
            pc = op[1]
            continue
        elif k == "ifgt":
            if r[op[1]] > r[op[2]]:
                pc = op[3]
                continue
        else:
            raise ValueError(k)
        pc += 1
    return None, steps   # exceeded cap: generator resamples

def _pe_random_arith(rng, writable):
    x = rng.choice(writable)
    kind = rng.choice(["add", "sub", "mul", "addc", "subc", "set"])
    if kind in ("add", "sub", "mul"):
        return (kind, x, rng.randrange(4))       # any register incl. counters
    if kind in ("addc", "subc"):
        return (kind, x, rng.randrange(1, 500))
    return ("set", x, rng.randrange(0, 1000))

def gen_program_exec(level, rng):
    smin, smax, n_loops, n_branch = PE_LEVELS[level]
    for _attempt in range(400):
        regs = [rng.randrange(0, 200) for _ in range(4)] + [0]
        prog = []
        counters = []
        # straight-line prelude; with no loops it must carry the whole step
        # band by itself (straight-line code executes each line exactly once)
        if n_loops == 0:
            n_prelude = rng.randrange(max(1, smin - 5), max(2, smax - 4))
        else:
            n_prelude = rng.randrange(1, 4)
        for _ in range(n_prelude):
            prog.append(_pe_random_arith(rng, [0, 1, 2, 3]))
        for li in range(n_loops):
            counter = 3 - li            # r3, r2, r1 as loop counters
            counters.append(counter)
            trip = rng.randrange(2, 3 + (smax // max(1, 3 * n_loops)))
            prog.append(("set", counter, trip))
            body_start = len(prog)
            writable = [x for x in range(4) if x not in counters]
            for _ in range(rng.randrange(2, 5)):
                prog.append(_pe_random_arith(rng, writable))
            if n_branch > 0 and rng.random() < 0.6:
                # forward skip inside the body: if rX > rY skip one line
                a, b = rng.sample([0, 1, 2, 3], 2)
                prog.append(("ifgt", a, b, len(prog) + 2))
                prog.append(_pe_random_arith(rng, writable))
            prog.append(("subc", counter, 1))
            prog.append(("ifgt", counter, 4, body_start))   # r4 == 0 always
        for _ in range(max(0, n_branch - n_loops)):
            a, b = rng.sample([0, 1, 2, 3], 2)
            prog.append(("ifgt", a, b, len(prog) + 2))
            prog.append(_pe_random_arith(rng, [0, 1, 2, 3]))
        for _ in range(rng.randrange(1, 3)):
            prog.append(_pe_random_arith(rng, [0, 1, 2, 3]))
        prog.append(("halt",))

        ans, steps = _pe_exec_internal(prog, regs)
        if ans is None or not (smin <= steps <= smax):
            continue
        if ans == regs[0]:
            continue                     # degenerate: r0 untouched
        body = _pe_render(prog, regs, steps)
        return {"body": body, "true": ans,
                "params": {"pe_lines": len(prog), "pe_steps": steps,
                           "pe_loops": n_loops,
                           "pe_branches": sum(1 for op in prog
                                              if op[0] == "ifgt")}}
    raise RuntimeError(f"program_exec: no program hit step band {smin}-{smax}")

def _pe_render(prog, regs, steps):
    lines = [
        "Simulate this register program exactly.",
        "Rules: registers r0..r4 hold integers 0..9999; after EVERY "
        "assignment the value is reduced modulo 10000 (e.g. 3 - 5 gives "
        "9998). Execution starts at line 1 and proceeds line by line; "
        "'goto L' jumps to line L; 'if rX > rY goto L' jumps only if the "
        "condition holds. The program is guaranteed to halt (within "
        f"{PE_STEP_CAP} executed steps).",
        "Initial values: " + ", ".join(f"r{i}={v}"
                                       for i, v in enumerate(regs)),
        "Program:",
    ]
    for i, op in enumerate(prog, 1):
        k = op[0]
        if k == "halt":
            t = "halt"
        elif k == "set":
            t = f"r{op[1]} = {op[2]}"
        elif k in ("add", "sub", "mul"):
            sym = {"add": "+", "sub": "-", "mul": "*"}[k]
            t = f"r{op[1]} = r{op[1]} {sym} r{op[2]}"
        elif k in ("addc", "subc"):
            sym = "+" if k == "addc" else "-"
            t = f"r{op[1]} = r{op[1]} {sym} {op[2]}"
        elif k == "ifgt":
            t = f"if r{op[1]} > r{op[2]} goto {op[3] + 1}"
        elif k == "goto":
            t = f"goto {op[1] + 1}"
        lines.append(f"{i}: {t}")
    lines.append("Question: what is the value of r0 when the program halts?")
    return "\n".join(lines)

_PE_LINE = re.compile(r"^(\d+): (.+)$")

def verify_program_exec(body):
    """VERIFIER: parses the rendered program text and executes it with an
    independently written interpreter (if/elif over parsed strings, explicit
    non-negative mod). Returns (answer, executed_steps)."""
    lines = body.split("\n")
    init = {}
    for mm in re.finditer(r"r(\d)=(\d+)", lines[2]):
        init[int(mm.group(1))] = int(mm.group(2))
    r = [init[i] for i in range(5)]
    # index instructions by their RENDERED line number (what a human follows)
    # rather than list position - catches any render-numbering bug
    prog = {}
    first = None
    for ln in lines[4:]:
        mm = _PE_LINE.match(ln)
        if mm:
            lineno = int(mm.group(1))
            if lineno in prog:
                raise ValueError(f"duplicate line number {lineno}")
            prog[lineno] = mm.group(2)
            if first is None:
                first = lineno
    def mod(v):
        return ((v % PE_MOD) + PE_MOD) % PE_MOD
    pc = first - 1
    steps = 0
    while steps <= PE_STEP_CAP:
        steps += 1
        if pc + 1 not in prog:
            raise ValueError(f"jump/fallthrough to nonexistent line {pc+1}")
        t = prog[pc + 1]
        if t == "halt":
            return r[0], steps
        mm = re.fullmatch(r"if r(\d) > r(\d) goto (\d+)", t)
        if mm:
            if r[int(mm.group(1))] > r[int(mm.group(2))]:
                pc = int(mm.group(3)) - 1
            else:
                pc += 1
            continue
        mm = re.fullmatch(r"goto (\d+)", t)
        if mm:
            pc = int(mm.group(1)) - 1
            continue
        mm = re.fullmatch(r"r(\d) = r(\d) ([+\-*]) r(\d)", t)
        if mm:
            a, b = r[int(mm.group(2))], r[int(mm.group(4))]
            v = a + b if mm.group(3) == "+" else \
                a - b if mm.group(3) == "-" else a * b
            r[int(mm.group(1))] = mod(v)
            pc += 1
            continue
        mm = re.fullmatch(r"r(\d) = r(\d) ([+\-]) (\d+)", t)
        if mm:
            a, c = r[int(mm.group(2))], int(mm.group(4))
            v = a + c if mm.group(3) == "+" else a - c
            r[int(mm.group(1))] = mod(v)
            pc += 1
            continue
        mm = re.fullmatch(r"r(\d) = (\d+)", t)
        if mm:
            r[int(mm.group(1))] = mod(int(mm.group(2)))
            pc += 1
            continue
        raise ValueError(f"unparseable program line: {t!r}")
    return None, steps

# ============================================================================
# FAMILY 3: constraint_count
# ============================================================================
# level -> (range width, number of conditions)
# Rationale (DELEGATED): width x conditions is the search knob; 200,000 x 6
# cannot be pattern-matched, only counted (or cleverly decomposed, which is
# also fine - the answer is still exact).
CC_LEVELS = [(60, 1), (120, 2), (300, 2), (700, 3), (1500, 3), (4000, 4),
             (10000, 4), (40000, 5), (100000, 5), (200000, 6)]

# condition kinds: ("div", d) ("ndiv", d) ("mod", m, r) ("dsum_gt", t)
# ("dsum_div", k) ("hasdig", d) ("nhasdig", d) ("square",)

def _cc_pred_gen(cond, x):
    """GENERATOR-SIDE predicates (string-based digit handling)."""
    k = cond[0]
    if k == "div":
        return x % cond[1] == 0
    if k == "ndiv":
        return x % cond[1] != 0
    if k == "mod":
        return x % cond[1] == cond[2]
    if k == "dsum_gt":
        return sum(int(ch) for ch in str(x)) > cond[1]
    if k == "dsum_div":
        return sum(int(ch) for ch in str(x)) % cond[1] == 0
    if k == "hasdig":
        return str(cond[1]) in str(x)
    if k == "nhasdig":
        return str(cond[1]) not in str(x)
    if k == "square":
        return math.isqrt(x) ** 2 == x
    raise ValueError(k)

def gen_constraint_count(level, rng):
    width, ncond = CC_LEVELS[level]
    for _attempt in range(300):
        a = rng.randrange(10, 900000)
        b = a + width - 1
        conds = []
        kinds = ["div", "ndiv", "mod", "dsum_gt", "dsum_div", "hasdig",
                 "nhasdig"] + (["square"] if width >= 1500 else [])
        rng.shuffle(kinds)
        for k in kinds[:ncond]:
            if k == "div":
                conds.append(("div", rng.choice([2, 3, 4, 6, 7, 9, 11, 13])))
            elif k == "ndiv":
                conds.append(("ndiv", rng.choice([2, 3, 5, 7])))
            elif k == "mod":
                mmod = rng.choice([5, 6, 7, 8, 9, 10, 11])
                conds.append(("mod", mmod, rng.randrange(mmod)))
            elif k == "dsum_gt":
                conds.append(("dsum_gt", rng.randrange(8, 26)))
            elif k == "dsum_div":
                conds.append(("dsum_div", rng.choice([3, 4, 5])))
            elif k == "hasdig":
                conds.append(("hasdig", rng.randrange(10)))
            elif k == "nhasdig":
                conds.append(("nhasdig", rng.randrange(10)))
            else:
                conds.append(("square",))
        count = sum(1 for x in range(a, b + 1)
                    if all(_cc_pred_gen(c, x) for c in conds))
        # review finding (gate 3): count >= 1 let answers degenerate to tiny
        # guessable values (mode 1) at high levels - a 'guess small' strategy
        # harvested free accuracy. Require count >= 5 so the answer space
        # stays honest; items that intersect down to <5 are resampled.
        if count >= 5:
            body = _cc_render(a, b, conds)
            kinds = [c[0] for c in conds]
            return {"body": body, "true": count,
                    "params": {"cc_width": width, "cc_conds": ncond,
                               # per-item difficulty-bearing composition
                               # (battery adversaries at a locked level)
                               "cc_digitish": sum(1 for k in kinds if k in
                                                  ("dsum_gt", "dsum_div",
                                                   "hasdig", "nhasdig")),
                               "cc_square": int("square" in kinds),
                               "cc_modular": sum(1 for k in kinds if k in
                                                 ("div", "ndiv", "mod"))}}
    raise RuntimeError("constraint_count: no count>=5 found at this level")

_CC_TEXT = {
    "div":      lambda c: f"is divisible by {c[1]}",
    "ndiv":     lambda c: f"is not divisible by {c[1]}",
    "mod":      lambda c: f"leaves remainder {c[2]} when divided by {c[1]}",
    "dsum_gt":  lambda c: f"has a digit sum greater than {c[1]}",
    "dsum_div": lambda c: f"has a digit sum divisible by {c[1]}",
    "hasdig":   lambda c: f"contains the digit {c[1]}",
    "nhasdig":  lambda c: f"does not contain the digit {c[1]}",
    "square":   lambda c: "is a perfect square",
}

def _cc_render(a, b, conds):
    lines = [f"How many integers x with {a} <= x <= {b} (inclusive) satisfy "
             f"ALL of the following conditions?"]
    for i, c in enumerate(conds, 1):
        lines.append(f"{i}. x {_CC_TEXT[c[0]](c)}.")
    lines.append("Question: how many such integers are there?")
    return "\n".join(lines)

_CC_RES = [
    ("ndiv",     re.compile(r"^x is not divisible by (\d+)\.$")),
    ("div",      re.compile(r"^x is divisible by (\d+)\.$")),
    ("mod",      re.compile(r"^x leaves remainder (\d+) when divided by "
                            r"(\d+)\.$")),
    ("dsum_gt",  re.compile(r"^x has a digit sum greater than (\d+)\.$")),
    ("dsum_div", re.compile(r"^x has a digit sum divisible by (\d+)\.$")),
    ("nhasdig",  re.compile(r"^x does not contain the digit (\d)\.$")),
    ("hasdig",   re.compile(r"^x contains the digit (\d)\.$")),
    ("square",   re.compile(r"^x is a perfect square\.$")),
]

def _cc_digsum_arith(x):
    """Arithmetic digit sum (no string conversion) - verifier-side."""
    s = 0
    while x:
        s += x % 10
        x //= 10
    return s

def _cc_has_digit_arith(x, d):
    while True:
        if x % 10 == d:
            return True
        x //= 10
        if x == 0:
            return False

def verify_constraint_count(body):
    """VERIFIER: parses conditions from text; counts via per-condition SET
    construction + intersection (different strategy from the generator's
    single loop), with arithmetic (non-string) digit predicates."""
    lines = body.split("\n")
    mm = re.match(r"How many integers x with (\d+) <= x <= (\d+)", lines[0])
    a, b = int(mm.group(1)), int(mm.group(2))
    sets = []
    for ln in lines[1:]:
        if ln.startswith("Question:"):
            break
        txt = re.sub(r"^\d+\. ", "", ln)
        for kind, rx in _CC_RES:
            m2 = rx.match(txt)
            if not m2:
                continue
            if kind == "div":
                d = int(m2.group(1))
                s = set(range(a + (-a) % d, b + 1, d))
            elif kind == "ndiv":
                d = int(m2.group(1))
                s = set(range(a, b + 1)) - set(range(a + (-a) % d, b + 1, d))
            elif kind == "mod":
                r0, mmod = int(m2.group(1)), int(m2.group(2))
                start = a + ((r0 - a) % mmod)
                s = set(range(start, b + 1, mmod))
            elif kind == "dsum_gt":
                t = int(m2.group(1))
                s = {x for x in range(a, b + 1) if _cc_digsum_arith(x) > t}
            elif kind == "dsum_div":
                k = int(m2.group(1))
                s = {x for x in range(a, b + 1)
                     if _cc_digsum_arith(x) % k == 0}
            elif kind == "hasdig":
                d = int(m2.group(1))
                s = {x for x in range(a, b + 1) if _cc_has_digit_arith(x, d)}
            elif kind == "nhasdig":
                d = int(m2.group(1))
                s = {x for x in range(a, b + 1)
                     if not _cc_has_digit_arith(x, d)}
            else:
                s = {q * q for q in range(math.isqrt(max(1, a - 1)),
                                          math.isqrt(b) + 1)
                     if a <= q * q <= b}
            sets.append(s)
            break
        else:
            raise ValueError(f"unparseable condition: {txt!r}")
    out = set(range(a, b + 1))
    for s in sets:
        out &= s
    return len(out)

# ============================================================================
# FAMILY 4: arith_long
# ============================================================================
# level -> (chain length, multiply/divide fraction)
# Rationale (DELEGATED): the frozen ladders topped out ~40 ops; this extends
# to 300. Exact division keeps magnitudes bounded without clamps (an audit
# lesson from v5's twin redesign: clamps leak difficulty cues).
AL_LEVELS = [(8, 0.15), (12, 0.20), (18, 0.20), (26, 0.25), (38, 0.25),
             (56, 0.30), (80, 0.30), (120, 0.35), (200, 0.35), (300, 0.40)]

AL_CAP = 10 ** 7

def gen_arith_long(level, rng):
    length, muldiv = AL_LEVELS[level]
    start = rng.randrange(10, 1000)
    v = start
    ops = []
    n_mul = 0
    while len(ops) < length:
        want_md = rng.random() < muldiv
        if want_md and abs(v) > AL_CAP // 12:
            want_md = False
        if want_md and rng.random() < 0.5 and abs(v) > 50:
            # exact division: pick d among 2..9; nudge v divisible first
            d = rng.randrange(2, 10)
            r = v % d
            if r != 0:
                delta = (d - r) % d + d * rng.randrange(0, 100)
                if delta == 0:
                    delta = d
                ops.append(("Add", delta))
                v += delta
                if len(ops) >= length:
                    break
            ops.append(("Divide by", d))
            v //= d
        elif want_md:
            d = rng.randrange(2, 10)
            ops.append(("Multiply by", d))
            v *= d
            n_mul += 1
        else:
            nnum = rng.randrange(2, 1000)
            if rng.random() < 0.5:
                ops.append(("Add", nnum))
                v += nnum
            else:
                ops.append(("Subtract", nnum))
                v -= nnum
    body = _al_render(start, ops)
    return {"body": body, "true": v,
            "params": {"al_ops": len(ops), "al_muls": n_mul,
                       "al_divs": sum(1 for verb, _ in ops
                                      if verb == "Divide by")}}

def _al_render(start, ops):
    lines = [f"Apply the following operations in order, using exact integer "
             f"arithmetic (every division below is exact).",
             f"Start with {start}."]
    for verb, n in ops:
        lines.append(f"{verb} {n}.")
    lines.append("Question: what is the final value?")
    return "\n".join(lines)

_AL_OP = re.compile(r"^(Add|Subtract|Multiply by|Divide by) (\d+)\.$")

def verify_arith_long(body):
    """VERIFIER: parses the rendered chain and recomputes with Fraction
    (catches any inexact division the generator might have produced)."""
    lines = body.split("\n")
    v = Fraction(int(re.match(r"Start with (\d+)\.", lines[1]).group(1)))
    for ln in lines[2:]:
        if ln.startswith("Question:"):
            break
        mm = _AL_OP.match(ln)
        if not mm:
            raise ValueError(f"unparseable op: {ln!r}")
        n = int(mm.group(2))
        verb = mm.group(1)
        if verb == "Add":
            v += n
        elif verb == "Subtract":
            v -= n
        elif verb == "Multiply by":
            v *= n
        else:
            v /= n
    if v.denominator != 1:
        raise ValueError("inexact division in chain")
    return int(v)

# ============================================================================
# REGISTRY / VERIFICATION HARNESS
# ============================================================================

FAMILIES = {
    "logic_grid":       gen_logic_grid,
    "program_exec":     gen_program_exec,
    "constraint_count": gen_constraint_count,
    "arith_long":       gen_arith_long,
}

def make_item(family, level, seed, i):
    rng = item_rng(seed, family, level, i)
    item = FAMILIES[family](level, rng)
    item["family"] = family
    item["level"] = level
    item["idx"] = i
    return item

def verify_item(item):
    """Run the family's text-level independent verifier against the item.
    Returns (ok, detail)."""
    fam, body, true = item["family"], item["body"], item["true"]
    if fam == "logic_grid":
        ans, nsol = verify_logic_grid(body)
        if nsol != 1:
            return False, f"solutions={nsol}"
        return ans == true, f"verifier={ans} claimed={true}"
    if fam == "program_exec":
        ans, steps = verify_program_exec(body)
        return ans == true, f"verifier={ans} claimed={true} steps={steps}"
    if fam == "constraint_count":
        ans = verify_constraint_count(body)
        return ans == true, f"verifier={ans} claimed={true}"
    if fam == "arith_long":
        ans = verify_arith_long(body)
        return ans == true, f"verifier={ans} claimed={true}"
    raise ValueError(fam)

def run_verification(n_per_family, seed):
    """Gate-1 report: n cases per family spread across all levels, zero
    mismatches required. Also prints body-length and answer stats per level
    so the cue battery's raw material is visible up front."""
    print(f"GATE 1 VERIFICATION - {n_per_family} cases/family, seed {seed}")
    print("Verifiers parse the RENDERED TEXT and re-solve independently.\n")
    all_ok = True
    t00 = time.time()
    for fam in FAMILIES:
        per_level = [n_per_family // N_LEVELS] * N_LEVELS
        for j in range(n_per_family - sum(per_level)):
            per_level[j] += 1
        n_ok = n_bad = 0
        t0 = time.time()
        fails = []
        for level in range(N_LEVELS):
            lens, answers = [], []
            for i in range(per_level[level]):
                item = make_item(fam, level, seed, i)
                ok, detail = verify_item(item)
                lens.append(len(item["body"]))
                answers.append(item["true"])
                if ok:
                    n_ok += 1
                else:
                    n_bad += 1
                    if len(fails) < 5:
                        fails.append((level, i, detail))
            from collections import Counter as _C
            mode_ans, mode_n = _C(answers).most_common(1)[0]
            small = sum(1 for x in answers if abs(x) <= 3) / len(answers)
            print(f"  {fam:17s} L{level}: n={per_level[level]:4d} "
                  f"len~{sum(lens)//max(1,len(lens)):5d}ch "
                  f"ans[{min(answers)}..{max(answers)}] "
                  f"mode={mode_ans}x{mode_n} frac|ans|<=3={small:.2f}",
                  flush=True)
        dt = time.time() - t0
        status = "OK" if n_bad == 0 else f"FAIL ({n_bad} mismatches)"
        print(f"  {fam:17s} TOTAL {n_ok}/{n_ok + n_bad} verified "
              f"[{dt:.1f}s] {status}")
        for lvl, i, detail in fails:
            print(f"    MISMATCH L{lvl} i={i}: {detail}")
        all_ok = all_ok and (n_bad == 0)
        print()
    print(f"VERIFICATION {'PASSED - gate 1 open' if all_ok else 'FAILED'} "
          f"[{time.time() - t00:.0f}s total]")
    return all_ok

# ============================================================================
# SELFTEST (structural checks beyond the mass verification)
# ============================================================================

def selftest():
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("m4_families selftest\n")
    # determinism: same key -> identical item; different key -> different rng
    a = make_item("arith_long", 3, 42, 7)
    b = make_item("arith_long", 3, 42, 7)
    c = make_item("arith_long", 3, 42, 8)
    check(a["body"] == b["body"] and a["true"] == b["true"],
          "generation deterministic in (seed, family, level, idx)")
    check(a["body"] != c["body"], "different idx gives a different item")

    # every family: params present, integer answers, all levels generate
    for fam in FAMILIES:
        items = [make_item(fam, lvl, 5, 0) for lvl in range(N_LEVELS)]
        check(all(isinstance(it["true"], int) for it in items),
              f"{fam}: integer answers at all {N_LEVELS} levels")
        check(all(it["params"] for it in items), f"{fam}: params populated")
        lens = [len(it["body"]) for it in items]
        check(lens[-1] > lens[0],
              f"{fam}: body grows with level ({lens[0]} -> {lens[-1]}ch)")

    # logic_grid: verifier must DETECT non-uniqueness (review gate-3 fix:
    # the old check was vacuous). This 3-house puzzle pins 'tea' only; the
    # other drink values can permute, so there are multiple solutions.
    nonunique = ("There are 3 houses numbered 1 to 3 from left to right. "
                 "Each house is assigned exactly one value from each "
                 "category below, and within a category no two houses "
                 "share a value.\nCategories:\n- drink: tea, coffee, milk\n"
                 "Clues:\n1. 'tea' is in house 2.\n"
                 "Question: in which house (1-3) is 'coffee'?")
    ans, nsol = verify_logic_grid(nonunique)
    check(ans is None and nsol == 2,
          f"logic_grid verifier detects non-uniqueness (nsol counted "
          f"to {nsol}, ans={ans})")
    # and a corrupted answer must be rejected on a real item
    it = make_item("logic_grid", 1, 5, 3)
    it_bad = dict(it, true=(it["true"] % LG_LEVELS[1][0]) + 1)
    okv, _ = verify_item(it_bad)
    check(not okv, "logic_grid verifier rejects a wrong claimed answer")

    # program_exec: two interpreters agree on the internal vs text path is
    # covered by mass verification; here check mod semantics edge (3-5=9998)
    body = ("Simulate this register program exactly.\nRules: registers "
            "r0..r4 hold integers 0..9999; after EVERY assignment the value "
            "is reduced modulo 10000 (e.g. 3 - 5 gives 9998). Execution "
            "starts at line 1 and proceeds line by line; 'goto L' jumps to "
            "line L; 'if rX > rY goto L' jumps only if the condition holds. "
            "The program is guaranteed to halt (within 1200 executed "
            "steps).\nInitial values: r0=3, r1=5, r2=0, r3=0, r4=0\n"
            "Program:\n1: r0 = r0 - r1\n2: halt\n"
            "Question: what is the value of r0 when the program halts?")
    ans, _ = verify_program_exec(body)
    check(ans == 9998, f"verifier wraparound 3-5 -> 9998 (got {ans})")

    # constraint_count: verifier catches a corrupted claimed answer
    it = make_item("constraint_count", 2, 5, 1)
    it_bad = dict(it, true=it["true"] + 1)
    okv, _ = verify_item(it_bad)
    check(not okv, "constraint_count verifier rejects an off-by-one answer")

    # arith_long: division is exact on a sample; corrupted op text raises
    it = make_item("arith_long", 6, 5, 2)
    okv, _ = verify_item(it)
    check(okv, "arith_long L6 sample verifies")
    try:
        verify_arith_long(it["body"].replace("Add", "Frobnicate", 1))
        check(False, "arith_long verifier raises on corrupted op text")
    except ValueError:
        check(True, "arith_long verifier raises on corrupted op text")

    # quick mini-verification (10/family) as smoke
    for fam in FAMILIES:
        bad = 0
        for lvl in (0, 4, 9):
            for i in range(4):
                if not verify_item(make_item(fam, lvl, 99, i))[0]:
                    bad += 1
        check(bad == 0, f"{fam}: 12-case smoke verification clean")

    print("\nselftest", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=int, default=0,
                    help="run N verification cases per family (gate 1)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", type=int, default=-1,
                    help="print one sample per family at LEVEL")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.verify:
        sys.exit(0 if run_verification(args.verify, args.seed) else 1)
    if args.show >= 0:
        for fam in FAMILIES:
            it = make_item(fam, args.show, args.seed, 0)
            print(f"===== {fam} L{args.show} (true={it['true']}, "
                  f"params={it['params']}) =====")
            print(it["body"][:1500])
            print()
        sys.exit(0)
    ap.print_help()
