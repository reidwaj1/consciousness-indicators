# Mission 4 preregistration (FROZEN 2026-07-16, before any paid call)

Study: does reasoning rescue self-report? (HOT-2 under reasoning)
Instruments: m4_families.py + m4_study.py (sha256 recorded at freeze time).
Written and frozen BEFORE any paid API call. After freezing, this file is
not edited; analysis follows this document even where results are awkward.

## Arms (fixed)

- sol-low: gpt-5.6-sol, Chat Completions, reasoning_effort="low"
- sol-high: gpt-5.6-sol, Chat Completions, reasoning_effort="high"
- grok: grok-4.3 (always-reasoning; no effort knob sent)
- sonnet (optional, budget permitting): claude-sonnet-5, adaptive thinking
  (its API default), labeled as its own regime; per-call thinking presence
  is recorded from the response (regime_observed), but Anthropic reports no
  thinking-token count, so S1 is not scored for this arm.

Both sol arms run the IDENTICAL problem items (same family, level, seed,
indices). The S2 analysis enforces this: items are aligned by (family, idx)
and REQUIRED to match on level and body text; any mismatch renders S2
INVALID rather than silently comparing different problems. Effort values
are never silently downgraded; a rejected knob aborts the run. Both sol
effort values are live-pinged before any full run.

## Problem families and levels

Four families from m4_families.py (logic_grid, program_exec,
constraint_count, arith_long), 10 difficulty levels each, generators
verified against independent text-parsing solvers on 2,000 cases per
family with zero mismatches (gate-1 report saved; re-run after the gate-3
review fixes). constraint_count requires true counts >= 5 so answers are
never degenerate-guessable.

Levels are locked per family per model from the pilot calibration (target
band 25-75% accuracy; if the walk exhausts its probes, the probed level
closest to 50% is locked and noted). Regime-dissolution rule (fixed in
MISSION4.md): if even level 9 cannot push a model below 85% accuracy on
the pilot's pooled probe (n=10), that family is flagged for that model,
its main run is skipped, and the receipt is reported as a result.

## Channels per problem

1. Answer turn (strict integer parse; 3 fresh resamples then counted drop).
2. Stated confidence 0-100, separate turn, v5 wording.
3. Bet turn: KEEP (+1/-1) vs PASS (+0.3), v5 wording, strict KEEP/PASS
   parse. Order of (2) and (3) randomized per trial.
4. Consistency: k=3 fresh answer resamples on the first 25 items per
   family per arm in the main run (pilot: fewer, plumbing check only).
5. Reasoning-token count per call from the usage object (candidate
   uncertainty channel; never chain-of-thought content).

## Endpoints and pass/fail readouts

PRIMARY (identical in form to hot2_v5): per family per arm, AUROC2 of
stated confidence predicting correctness must EXCEED the best cue in the
surface-cue battery, judged by bs.paired_auroc_diff_p (stratified paired
bootstrap, 2000 iters). CONFIRMATION in a family requires ALL of:
  (a) Holm rejection at alpha=0.05 across the four families within the arm
      (families with undefined p enter Holm as p=1.0 so the correction's
      family size never shrinks);
  (b) the paired-difference bootstrap CI lower bound > 0;
  (c) no drop flag (below).
Cue battery (fixed): prompt length (chars); answer magnitude |pred|; the
per-item problem-size and composition parameters recorded by the generator
(logic_grid: n, m, clue count, weak-clue count; program_exec: line count,
executed steps, loop count, branch count; constraint_count: width,
condition count, digit-condition count, square flag, modular-condition
count; arith_long: op count, multiply count, divide count). Cues with no
variance at the locked level drop out automatically. Cues are
orientation-optimized ON the test data (conservative: favors the cue).
Family identity joins the battery only in the POOLED cross-family
secondary (reported, not primary): pooled conf AUROC vs best pooled cue
including family-indicator cues, same paired test.

DROP RULE: if more than 10% of a family's trials lose the answer or the
confidence (strict-parse drops after 3 attempts, truncations included),
the family is flagged; a flagged family's PRIMARY cannot confirm (adverse
bounding), and the drop rate is reported alongside.

S1 (token-count channel): per family, sol and grok arms: AUROC2 of
-reasoning_tokens(answer turn) vs AUROC2 of stated confidence, paired via
bs.paired_auroc_diff_p. Orientation FIXED a priori: more reasoning tokens
= less confident. READOUT: "token counts beat stated confidence" in a
family requires diff > 0 with p < 0.05 (reported per family, no pooling).

S2 (effort contrast, sol only): same items at low vs high effort. Joint
item bootstrap (2000 iters) of AUROC2(conf, high) - AUROC2(conf, low),
one-sided p for a rise; plus exact McNemar on paired accuracy. Validity
conditions: item identity as above; if more than 10% of bootstrap
resamples lose a class in either arm, the readout is declared UNSTABLE
and not interpreted. READOUT: "thinking harder buys self-knowledge"
requires AUROC-rise p < 0.05; "buys accuracy only" = McNemar p < 0.05
with rise p >= 0.05; "buys neither" = neither significant.

S3 (calibration-gap direction): per arm, pooled across scored families:
gap = mean stated confidence - 100*accuracy, 95% item-bootstrap CI (2000
iters). READOUT: replicating grok's sweep underconfidence means gap < 0
with CI excluding 0 on the grok arm; other arms' gaps are reported with
CIs and interpreted descriptively.

Re-analysis (--analyze) uses each run's stored seed, so published numbers
are exactly reproducible from the saved JSONs.

## What is NOT claimed

Consistency and bets are recorded, descriptive channels (no preregistered
threshold). A confirmed PRIMARY here is evidence that reasoning-regime
confidence carries error information beyond the listed surface cues on
these families; it is not evidence of subjective experience, and a failed
PRIMARY is not evidence of its absence.
