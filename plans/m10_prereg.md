# M10 PREREGISTRATION TEMPLATE â€” the full process on a frontier-class
# model: live-reporter baseline, forced-answer correctness probe, and
# report steering (Stage -1 deliverable, mission item 7)

THIS IS THE TEMPLATE, written and hashed at Stage -1 (local, $0),
before any pod exists. The Stage-2 freeze produces m10_prereg.md from
this file by filling EXACTLY FIVE double-brace blank sites and
NOTHING ELSE:

    tracking level 3 (probe acc 0.792, parse 1.00; matched-pair estimate 11706; walked L0=1.00, L1=0.96, L2=0.96, L3=0.79)   the family+level Stage 1 lands on
    deepseek-ai/DeepSeek-V4-Flash @ 60d8d70770c6776ff598c94bb586a859a38244f1, NATIVE FP8 (FineGrainedFP8, e4m3, block 128x128; no requantization); stack: 2x H200, transformers 5.14.1, torch 2.8.0.dev20250319+cu128, triton 3.7.1, kernels 0.15.2 + kernels-community/finegrained-fp8 @ 7cdb05d4; multi-stream residual (4 x 4096) captured flattened (16384 feats), collapse to single stream by model hc_head before output              repo id, revision, quant source+shas,
                                  exact inference-stack versions
    noise floor measured EXACTLY ZERO (max |dlogit| 0.0e+00; all per-layer L2 dh 0.0) — the <=2x-noise-floor criteria collapse to exact equality; greedy stability 20/20 exact     the measured noise-floor numbers that
                                  instantiate the G0 criterion rule
    400 (throughput projections: arm A 0.14h, arm B 0.27h, arm C 5.38h — all inside caps at full n)                         item count at freeze (400, or the
                                  frozen ladder value forced by budget)
    balance $123.70 at freeze; ceiling $400 (amended 2026-08-03); spent ~$26.3 total across all pods per m10_budget.log              pessimistic remaining credit +
                                  provider balance at freeze time

THE FREEZE MAY FILL NOTHING ELSE. Every endpoint, clause, threshold,
seed, feature list, foil rule, branch rule, and interpretation sentence
below is fixed now. The freeze logs a diff of m10_prereg.md against
this template into m10_freeze_hashes.txt; if any diff line falls
outside the five blank sites, the freeze FAILS (RED gate â€” kill the
pod, note, stop). sha256 of BOTH files goes into m10_freeze_hashes.txt.
The frozen prereg is read-only after. Any post-freeze change is a
dated, hash-logged amendment with a reason (house rule, M8/M9).

Family-conditional content is fully enumerated in section 7 for every
candidate family. Filling tracking level 3 (probe acc 0.792, parse 1.00; matched-pair estimate 11706; walked L0=1.00, L1=0.96, L2=0.96, L3=0.79) SELECTS one
subsection; it edits none of them. The unselected subsections stay in
the frozen prereg as inert text â€” proof the content predated the data
for every family that could have landed.

## 1. Model and stack

PRIMARY: DeepSeek-V4-Flash (deepseek-ai/DeepSeek-V4-Flash, revision
pinned in m10_model_pin.json: 60d8d70770c6776ff598c94bb586a859a38244f1;
284B/13B-active MoE, 4-bit weights). FALLBACK per mission: DeepSeek-V3
4-bit if VRAM allows, else Llama-3.3-70B â€” a fallback landing is
recorded inside deepseek-ai/DeepSeek-V4-Flash @ 60d8d70770c6776ff598c94bb586a859a38244f1, NATIVE FP8 (FineGrainedFP8, e4m3, block 128x128; no requantization); stack: 2x H200, transformers 5.14.1, torch 2.8.0.dev20250319+cu128, triton 3.7.1, kernels 0.15.2 + kernels-community/finegrained-fp8 @ 7cdb05d4; multi-stream residual (4 x 4096) captured flattened (16384 feats), collapse to single stream by model hc_head before output, which holds, verbatim from the pod:
HF repo id, revision hash, quantization source+shas, and the EXACT
inference stack (transformers version, quant backend, torch version).

  deepseek-ai/DeepSeek-V4-Flash @ 60d8d70770c6776ff598c94bb586a859a38244f1, NATIVE FP8 (FineGrainedFP8, e4m3, block 128x128; no requantization); stack: 2x H200, transformers 5.14.1, torch 2.8.0.dev20250319+cu128, triton 3.7.1, kernels 0.15.2 + kernels-community/finegrained-fp8 @ 7cdb05d4; multi-stream residual (4 x 4096) captured flattened (16384 feats), collapse to single stream by model hc_head before output

One identical forward path for all arms: plain transformers forward,
no serving engines (vLLM/SGLang), no fusion flags, eager-equivalent
attention path as gated at G0. 4-bit quantizes WEIGHTS only;
activations are 16-bit â€” probing/steering operate on 16-bit residuals.
Every sentence comparing M10 to the gemma-3-1b (M7-M9, bf16
full-precision) results must name this mixed precision regime.

DETERMINISM CRITERION (rule frozen here, numbers filled at freeze):
the G0 noise-floor protocol replaces bit-identical checks. k>=5
repeated identical no-hook forwards on a fixed probe conversation;
record max |delta logit| and per-layer per-position L2 delta-h. All
former bit-identical selftest criteria become "<= 2x noise floor";
greedy-answer stability 20/20 repeats stays exact. NOISE-TO-DOSE GATE:
the deepest-layer noise floor must be <=10% of m_L (the alpha=1
injection norm, once measured) and <=10% of the planted-signal
magnitude â€” else RED. The measured floors and the realized numeric
thresholds they induce are recorded as:

  noise floor measured EXACTLY ZERO (max |dlogit| 0.0e+00; all per-layer L2 dh 0.0) — the <=2x-noise-floor criteria collapse to exact equality; greedy stability 20/20 exact

CHAT-TEMPLATE INVOCATION (pinned at Stage -1, verified at G0 item 6):
apply_chat_template with add_generation_prompt=True and any
thinking/reasoning kwarg set OFF (enable_thinking=False or the
model's equivalent), rendered to a STRING then tokenized;
apply_chat_template(tokenize=True) ids must equal string-then-tokenize
ids exactly. The rendered generation-prompt suffix is emitted verbatim
by m10_pipeline.py's template selftest and covered by that file's
sha256 in m10_freeze_hashes.txt â€” it is pinned by hash, not a freeze
blank. A G0 mismatch is a RED gate, never a silent fill.

REPORT-TOKEN LAYOUT (rule frozen here; realized sets are a hashed
artifact, not a blank): per-digit first-token vocab-id sets are ALL
vocab ids whose decoded form stripped of whitespace equals the digit,
derived at runtime from the pinned tokenizer; every report string
0-100 must decode round-trip; margin token positions/conditioning are
derived from actual tokenizations, never constants. The planted-report
selftest (1-, 2-, and 3-token planted report strings read at the
correct positions) must be green. Realized sets and positions go to
m10_report_layout.json, sha256-covered in m10_freeze_hashes.txt.

## 2. Landed family and level

  tracking level 3 (probe acc 0.792, parse 1.00; matched-pair estimate 11706; walked L0=1.00, L1=0.96, L2=0.96, L3=0.79)

Stage 1 walks tracking's ops ladder per M7 protocol; if saturated,
ONLY whitelisted numeric m4 families. GATE: accuracy in [0.30, 0.80],
strict parse >=0.9, on a whitelisted family â€” else RED. Calibration
prompts are ANSWER-ONLY (no confidence elicitation before freeze).
Eligibility is decided by the Stage -1 item 6 NUMERIC-FAMILY
WHITELIST selftests in m10_families.py (single-integer verified
answer; feasible F-/F+ foil rule; clean foil-audit), never by this
document and never at 3am. Filling this blank selects the matching
subsection of section 7; the calibration matched-answer pair count for
the landed level is logged at freeze (advisory projection for P1(b);
the binding count is the realized Stage-3 count, section 5).

## 3. Design: items, seeds, passes, conventions

From M10_MISSION.md Stage 3, verbatim:

> n = 400 (ladder 300/250 by budget only, decided between arms from
> m10_budget.log alone, logged with the forcing projection; all arms
> use the identical item list â€” the first n of the frozen seed order;
> Stage-3 seeds disjoint from calibration seeds).

  n at freeze = 400 (throughput projections: arm A 0.14h, arm B 0.27h, arm C 5.38h — all inside caps at full n)

- Seeds (fixed now, disjoint by construction): calibration item seed
  601 (generator re-verification seed 1601); Stage-3 item seed 1013
  (generator re-verification on a fresh seed 2013, 2,000/2,000
  required); foil seed 1013*7+3 = 7094; random-null direction seed
  RandomState(1013*13+1 = 13170); CV/bootstrap seeds as in the
  analysis code, hash-covered.
- Item generation: the landed family's generator at the landed level,
  rejection-sampled until (i) T admits both matched foils under the
  family's foil rule (section 7; the infeasible-T set is the rule's
  deterministic consequence, enumerated by the whitelist selftest and
  recorded in its output) and (ii) the body is globally unique within
  the item set. Reject counts recorded.
- Arm order fixed: A then B then C. Arm A supplies the own-answer
  error-offset distribution that Arm B's foil choice samples (M8
  stage-1 analog) and the PIN/LIVE branch decision that Arm C's P2
  follows. The branch is computed mechanically from the frozen PIN
  criterion and logged before Arm C starts â€” it is not a freeze fill
  and not a choice.
- Layers: depth fractions {0.15, 0.3, 0.45, 0.6, 0.8, 0.9, 0.95}
  resolved to integer decoder-layer indices by m10_pipeline.py against
  the real architecture (Stage -1 item 4); PRIMARY = the 0.9-fraction
  layer; 0.8 and 0.95 preregistered secondaries; others descriptive.
- Positions, per forced conversation (M8 house definitions): pos-PRE =
  final prompt token before the answer turn; pos-ANS = final token of
  the forced answer; pos-REP = final position of the full templated
  conversation (its logits generate the first report token). Position
  bookkeeping by string-prefix construction with token-level prefix
  asserts on EVERY conversation; any violation aborts the run.
- Probe conventions (M8 house): logistic regression, StandardScaler,
  C=0.1, grouped 5-fold CV with items never split across folds;
  label = correctness of the in-context answer. CIs by item-clustered
  bootstrap (1,000 iters); paired comparisons by paired item-clustered
  bootstrap. Leakage asserts on every fold split.
- Elicitation wording, M7 frozen, verbatim:
  ANSWER_PROMPT = "{body}\n\nAnswer immediately from your best
  judgment. Do NOT work through steps. Reply with ONLY the integer
  answer."
  CONF_PROMPT = "How confident are you that your answer is exactly
  correct, on a 0-100 scale where N means you would expect answers
  like this to be exactly correct about N% of the time? Reply with
  ONLY an integer from 0 to 100."
  Strict parsing: answers by full-match integer (optional asterisks /
  trailing period tolerated); confidences by full-match 0-100 integer
  (optional % tolerated). Raw text always saved; unparsed rows counted
  and reported, never silently dropped.

## 4. ARM A â€” live-reporter baseline (descriptive)

From M10_MISSION.md Stage 3, verbatim:

> ARM A â€” LIVE-REPORTER BASELINE (DESCRIPTIVE/estimation only, panel
> order): own answers + stated 0-100 confidence (M7 wording, strict
> parse). Report stated-confidence AUROC, cue-battery AUROC (per-family
> list from the template), and the paired difference, all with
> clustered-bootstrap CIs. NO hypothesis test, NO confirmatory language.
> PIN CRITERION (frozen): PINNED iff >=95% of strictly-parsed
> confidences equal the modal value; else LIVE. No middle branch.
> PINNED additionally triggers the 5-variant diagnostic battery (n=20).

Instrument-level decisions, fixed now:
- Stated-confidence AUROC: parsed confidence vs own-answer greedy
  correctness, on rows where BOTH answer and confidence strict-parse;
  excluded-row counts reported by cause.
- Cue-battery AUROC: grouped 5-fold CV logistic probe (section 3
  conventions) on the landed family's EXACT cue list from section 7
  (core C1-C8 plus the family additions) computed on the same rows;
  out-of-fold scores; AUROC with item-clustered CI.
- Paired difference: stated-confidence AUROC minus cue-battery AUROC,
  paired item-clustered bootstrap CI. Reported as an estimate with a
  CI. No p-value, no "significant", no confirmatory sentence anywhere
  in Arm A's write-up.
- PIN criterion denominator: the strictly-parsed confidences among the
  n items (parse rate itself reported alongside). PINNED iff >=95%
  equal the modal value; else LIVE. No middle branch. The decision and
  the modal value are logged to m10_run.log before Arm C starts.
- PINNED additionally triggers the 5-variant diagnostic battery
  (n=20 fresh items, landed family+level, greedy, descriptive,
  m9_posthoc_diag house design): A = CONF_PROMPT verbatim;
  B = same question with the scale stated as 0-90; C = inverted scale
  (0 = certain correct, 100 = certain wrong); D = impossible referent
  ("confidence your answer equals the secret number I am thinking
  of"); E = the family's template-sanity question (exact wording per
  family in section 7).

## 5. ARM B â€” force (M8 ported) and P1

From M10_MISSION.md Stage 3, verbatim:

> ARM B â€” FORCE (M8 ported): T/F-/F+ per item; capture pos-PRE/ANS/REP
> at the seven layers.
> P1 (confirmatory), at depth fraction 0.9 (primary; 0.8 and 0.95
> preregistered secondaries): (a) within-item AUC CI>0.5, AND (c) beats
> the concatenation baseline (paired clustered bootstrap p<0.05), AND
> (b) matched-answer AUC CI>0.5 IF >=30 usable matched pairs â€” frozen
> fallback: match on (token count, leading digit) bins; if still <30,
> P1(b) auto-demotes to descriptive and P1 = (a) AND (c). Cross-sign
> transfer is DESCRIPTIVE (comparator lesson). Secondaries: dose-
> response vs signed error; pos-PRE=0.5 check; T-rows own-greedy
> control.

Instrument-level decisions, fixed now:
- Forced conversations, per item, identical except the assistant
  answer turn: T (true), F- (undershoot foil), F+ (overshoot foil),
  built as [user: ANSWER_PROMPT(body); assistant: forced numeral
  string; user: CONF_PROMPT; assistant-turn opening], rendered through
  the pinned chat template as a string, then tokenized (M8 house
  shape). Foils per the family rule in section 7, offsets preferenced
  by Arm A's realized error-offset distribution (foil seed 7094), else
  smallest-offset with seeded tie-break.
- Capture pass appends nothing (M8's Gate0 lesson: appending tokens
  adds kernel-tiling deviations); residuals at the seven layers x
  three positions, forced-answer logp, and full pos-REP logits are
  exactly the quantities greedy generation computes.
- P1 is evaluated at pos-ANS (M8 FORCE convention, fixed here). The
  identical conjunction computed from pos-REP states is a NAMED
  SECONDARY; the full position-decay curve (out-of-fold AUC at
  pos-PRE/ANS/REP, every captured layer) is descriptive.
- (a) within-item: pair fraction over same-item T-vs-foil pairs from
  out-of-fold scores, 95% item-clustered CI excluding 0.5 (above).
- (b) matched-answer: cross-item pairs where the same numeral string
  is correct in one item and wrong in another, 95% CI excluding 0.5.
  "Usable matched pairs" = pairs formable from out-of-fold scores
  after all exclusions, counted on the realized Stage-3 rows; the
  frozen rule then applies mechanically: >=30 exact-string pairs ->
  run (b) as stated; else re-pair on (token count, leading digit)
  bins â€” token count of the forced numeral string under the pinned
  tokenizer, leading digit of its decimal rendering; if still <30,
  P1(b) auto-demotes to descriptive and P1 = (a) AND (c). The
  demotion branch taken is reported with the pair counts.
- (c) concatenation baseline (exact features in section 7 per family;
  construction identical for all families): [pos-PRE residual (same
  layer) (+) forced-answer surface features (+) forced-token logp].
  One-sided paired item-clustered bootstrap, p < 0.05, probe beats
  baseline.
- Cross-sign transfer (DESCRIPTIVE): probe trained on T-vs-F- rows of
  train folds scored on held-out same-item T-vs-F+ pairs, and vice
  versa; both AUCs with CIs; no pass criterion. Asymmetry is read per
  the comparator lesson (one-signed-error shortcut).
- Secondaries (named): (S1) dose-response vs signed error â€”
  out-of-fold probe score vs signed forcing offset (forced - T),
  Spearman rho + curve, descriptive; (S2) pos-PRE=0.5 check â€” the
  within-item contrast at pos-PRE is 0.5 by construction (T/F share
  the prompt; asserted as a capture-integrity check) and the pooled
  cross-item pos-PRE AUC is reported as the prompt-difficulty reading
  estimate; (S3) T-rows own-greedy control â€” on T rows, AUC of probe
  scores separating items the model answered correctly under greedy
  from items it answered wrongly (~0.5 under pure in-context reading;
  >0.5 indicates partial difficulty / own-trajectory reading).

## 6. ARM C â€” steer + lens (M9 ported) and P2

From M10_MISSION.md Stage 3, verbatim:

> ARM C â€” STEER + LENS (M9 ported): direction from Arm B pos-REP at the
> primary layer; m_L from Arm B displacement; alpha grid
> {-8,-4,-1,0,+1,+4,+8} + 5 matched-norm randoms at {+4,-4}; coherence
> guard >=90% parse per dose; items failing parse at any dose of a pair
> are excluded symmetrically, counts reported.
> P2 (confirmatory), per frozen branch:
> - PINNED branch: M9's S on the constant token, unchanged clauses.
> - LIVE branch: within-item paired shift Delta_i = logp(token_i|alpha)
>   - logp(token_i|alpha=0) where token_i is that item's own alpha=0
>   greedy first report token; S_live = mean(Delta_i|wrong) -
>   mean(Delta_i|right); same random-null contrast, p<0.05, -4 sign
>   flip required if +4 passes. CENSORING CLAUSE: report the fraction
>   of items at the 0/100 bounds at alpha=0 per context; >20% at a
>   bound in either context flags S_live as censored and the frozen
>   interpretation caveat applies.
> - LIVE additionally: the stated-confidence readout, S_conf =
>   mean(conf_i(alpha)-conf_i(0) | wrong) - same | right), same null
>   contrast, same censoring clause.
> LENS (descriptive): per-item locking depth of token_i, aggregated as
> median; margin-by-layer with the simplified-construct caveat.

Shared instrument-level decisions, fixed now:
- Direction (M9 house rule): both-signs logistic probe on Arm B's
  pos-REP rows at the PRIMARY layer (StandardScaler, C=0.1),
  raw-space direction = coef/scaler.scale_, unit-normalized, NEGATED
  so +dir points toward wrongness; selftest-asserted: forced-wrong
  rows project higher than forced-right rows. m_L = the m_gap
  convention: difference of forced-wrong vs forced-right projection
  means on the raw-space direction (the natural displacement).
- Injection: forward hook on decoder block L-1 (whose output is
  hidden_states[L], the tensor the probes read â€” layer-alignment
  gated at G0 item 5) adding alpha * m_L * unit_dir at absolute
  positions >= rep_start, where rep_start = token length of the
  conversation rendered without the generation prompt. Per-row start
  positions as tensors with per-row asserts, never a scalar.
- "wrong"/"right" contexts = the forced-foil vs forced-T condition of
  the base conversation (one matched foil per item, side alternating
  by item index: even = undershoot, odd = overshoot â€” M9 house).
- Random nulls: 5 unit directions, RandomState(13170), injected at
  {+4,-4} with matched norm via the SAME m_L. The collected -4 nulls
  are analyzed, never dropped.
- COHERENCE GUARD: a (direction, alpha) dose is IN-REGIME only if
  >=90% of its sampled reports strict-parse. Out-of-regime doses are
  descriptive only. Items failing parse at any dose of a pair are
  excluded symmetrically from that pair's statistics; exclusion
  counts reported per dose.
- Readouts per steered conversation: sampled greedy report (<=8
  tokens, strict parse, raw text saved); pos-REP first-token logits
  (source of logp(token_i) and digit masses). PINNED branch
  additionally: the continuation pass teacher-forcing the constant
  string for the margin construct's conditionals.

P2 â€” PINNED branch (M9's S on the constant token, unchanged clauses;
margin construct instantiated per section 7 shared rules on the
pinned tokenizer; "the constant" = Arm A's modal report string):
S(dir, alpha) = mean over items of [margin(steered) - margin(alpha=0)]
on forced-WRONG rows minus the same on forced-RIGHT rows.
PASS requires ALL of:
 1. dose probe-dir alpha=+4 in-regime AND alpha=-4 in-regime (both
    doses are primary evidence; the coherence guard applies to both);
 2. S_probe(+4) < 0 (more wrongness signal -> lower margin,
    differentially in the wrong context);
 3. item-clustered bootstrap (1,000) of [S_probe(+4) minus the mean of
    the 5 random-direction S(+4)]: one-sided p < 0.05;
 4. S_probe(+4) below every individual random direction's S(+4);
 5. sign flip: S_probe(-4) > 0 with item-clustered bootstrap
    P(S <= 0) < 0.05.
Recorded alongside (not pass criteria): the -4 random-null contrast.
Instantiation notes (template-frozen): if the modal constant is not
"100", the construct's P(100) path becomes the P(modal) path (lead
set of the modal's first digit; continuation conditionals on the
modal string's canonical tokenization), HIGH/LOW bands unchanged; the
modal's band membership is reported. If the modal lies in the
excluded 60-79 band the pinned margin is structurally blind to it:
the branch then executes the LIVE construction below with token_i =
the modal first token for all items (deterministic rule, logged, not
an amendment).

P2 â€” LIVE branch (primary statistic S_live), clauses mirroring M9's:
Delta_i = logp(token_i | alpha) - logp(token_i | alpha=0), token_i =
that item's own alpha=0 greedy first report token;
S_live(dir, alpha) = mean(Delta_i | wrong) - mean(Delta_i | right).
Frozen sign convention (the analog of M9 clause 2): the wrongness
signal is predicted to displace the baseline report token
differentially in the WRONG context, so S_live(+4) < 0.
PASS requires ALL of:
 1. dose probe-dir alpha=+4 in-regime AND alpha=-4 in-regime;
 2. S_live(+4) < 0;
 3. item-clustered bootstrap (1,000) of [S_live(+4) minus the mean of
    the 5 random-direction S_live(+4)]: one-sided p < 0.05;
 4. S_live(+4) below every individual random direction's S_live(+4);
 5. -4 sign flip, required if +4 passes: S_live(-4) > 0 with
    item-clustered bootstrap P(S <= 0) < 0.05.
Recorded alongside (not pass criteria): the -4 random-null contrast.
CENSORING CLAUSE (frozen): report the fraction of items whose
strictly-parsed alpha=0 confidence sits at the 0 or 100 bound, per
context (forced-T rows; forced-foil rows). If >20% sit at a bound in
either context, S_live is flagged CENSORED and the frozen
interpretation caveat applies (section 8).

P2 â€” LIVE additionally, the stated-confidence readout (preregistered
co-primary, reported with the identical clause set; the headline P2
verdict is defined on S_live alone and S_conf never substitutes for
it): S_conf(dir, alpha) = mean(conf_i(alpha) - conf_i(0) | wrong) -
mean(conf_i(alpha) - conf_i(0) | right) on strictly-parsed sampled
confidences, S_conf(+4) < 0 expected, same random-null contrast
(one-sided p < 0.05, below every individual random), same -4 sign
flip requirement, same censoring clause, same symmetric exclusions.

LENS (descriptive, no confirmatory claim; mode re-verified by the G0
final-layer lens identity check): for unsteered and probe-alpha=+4
conversations â€” (i) per-item locking depth of token_i: the smallest
captured layer from which the lens top-1 first-report token equals
token_i at every deeper captured layer through the final layer; items
whose final-layer top-1 != token_i have undefined depth (excluded,
counted); aggregated as the median per context; (ii) margin-by-layer:
per-layer lens AUC vs forced correctness using the SIMPLIFIED
first-token margin log(P(8)+P(9)) - log(P(0)+P(2..5)) with leads 1,
6, 7 excluded â€” the full construct's continuation conditionals do not
exist at lens layers; this simplified statistic is never comparable
to the primary margin and is labeled as such in every table.

## 7. Family-conditional instrument content (fully enumerated)

### 7.0 Shared core (all families)

CUE CORE (Arm A battery components C1-C8; all computable from the
rendered problem text and the model's own strictly-parsed answer v â€”
never solved features):
  C1 body character length
  C2 body token length (pinned tokenizer, no special tokens)
  C3 answer value v
  C4 |v|
  C5 digit count of |v|
  C6 roundness indicator (|v| % 10 == 0)
  C7 token count of the answer string as emitted (pinned tokenizer)
  C8 sign indicator (v < 0)

CONCAT BASELINE (Arm B P1(c); identical construction for every
family): [pos-PRE residual at the evaluated layer (+) forced-answer
surface features (value, |value|, digit count of |value|, roundness,
token count of the forced numeral string, sign indicator) (+) total
teacher-forced logp of the forced answer tokens]. Per-family text
cues are deliberately NOT appended: the pos-PRE residual is the
model's own representation of the full prompt and subsumes any text
cue list (M8 house rationale).

SHALLOW-OBSERVER BAR (for any descriptive "beats cues" sentence at
pos-REP): C1, C2, and the forced-answer surface features ONLY â€” never
solved features like |true - forced| (M8 house rule, verbatim intent).

FOIL RULE (shared; per-family ranges below): candidates c != T with
same sign as T, same digit count of |c| as |T|, same roundness class
(|c| % 10 == 0 iff |T| % 10 == 0), same token count of the rendered
numeral string under the pinned tokenizer; strictly c < T for F-,
c > T for F+; c inside the family's candidate range. Choice
preference: candidates at offsets |c - T| present in Arm A's realized
error-offset distribution at the landed family+level, sampled with
observed frequencies (foil seed 7094); else the smallest-offset
candidate (seeded tie-break). The infeasible-T set (T with an empty
F- or F+ cell) is derived by exhaustive enumeration in the Stage -1
whitelist selftest and excluded by rejection sampling; the RULE is
frozen here, the set is its recorded deterministic consequence.
FOIL AUDIT (whitelist selftest + realized Stage-3 set): distributions
of digit count, leading digit, and roundness class for T vs F- vs F+
rows; leading-digit imbalance feeds the P1(b) binning fallback.

MARGIN CONSTRUCT (PINNED-branch statistic and lens reference; M8
construct generalized to the pinned tokenizer): per-digit first-token
sets by the decoded-form rule (section 1); P(lead d) = softmax mass
of set d in the full pos-REP logits; the constant-path probability
from the teacher-forced continuation conditionals; HIGH = P(lead 8) +
P(lead 9) + P(constant path if the constant is in 80-100); LOW =
P(lead 0) + P(lead 2) + P(lead 3) + P(lead 4) + P(lead 5) +
(P(lead 1) - P(100)); leads 6, 7 (reports 60-79) excluded, stated.
Margin = log(HIGH) - log(LOW). Report-turn token space (0-100) is
family-independent by design; the only family coupling is through the
answer turn's length, which shifts rep_start and pos-ANS â€” both are
runtime-derived with token-level asserts, never constants.

5-VARIANT BATTERY variants A-D are family-independent (section 4);
variant E (template sanity) is family-conditional, enumerated below.

### 7.1 tracking (gen_tracking, M7 house generator)

Whitelist status: eligible pending the Stage -1 whitelist selftest at
the landed level (single-integer answers by construction).
Answer space: small positive integers; attainable range at the landed
level certified by the whitelist selftest.
CUE BATTERY (exact): C1-C8 plus
  T1 count of Add operation lines
  T2 count of Remove operation lines
  T3 count of Move operation lines
  T4 queried-box mention count (occurrences of the queried box's name
     in the body)
  T5 queried-box touch count (operations whose source or destination
     is the queried box)
  T6 queried-box initial value
  T7 sum of the three initial values
  (T4/T5 are the per-queried-box churn cues â€” the v5 twin-study
  lesson: churn dissolved the last "cue-free" tracking positive.)
CONCAT BASELINE: shared construction (7.0), no additions.
FOIL RULE: shared rule; candidate range [1, certified attainable
max at the landed level]. Known level-0 instance under the gemma
tokenizer was infeasible-T = {9, 10, 11}; the set MUST be re-derived
for the pinned tokenizer and the landed level (token-count matching
is tokenizer-dependent).
MARGIN CONSTRUCT NOTES: answers are short (1-2 digit) numerals; the
answer turn is near-constant length; no family-specific hazard.
VARIANT E: "How many boxes were mentioned in the problem? Reply with
ONLY the integer." (expected 3.)

### 7.2 logic_grid (m4_families) — NONWHITELISTED, INELIGIBLE: excluded by m10_families.NONWHITELISTED (foil-rule degeneracy: answer-domain metadata invisible to the frozen foil contract, out-of-domain foils surface-detectable, endpoint-answer exclusion bias). This section is retained for the record only; calibration cannot land here.

Whitelist status: CONDITIONAL â€” answers are single integers (house
index 1..n, n<=7) but the answer space is tiny and categorical;
eligibility requires the whitelist selftest to certify a
non-degenerate foil cell and a clean foil audit. If uncertifiable,
this family is ineligible, period, and this subsection is inert.
CUE BATTERY (exact): C1-C8 plus
  L1 number of houses (lg_n, parsed from the header line)
  L2 number of categories (lg_m, count of category lines)
  L3 number of clue lines (lg_clues)
  L4 weak-clue count (clues of types left/adj/diff/notat, by the
     verifier's clue regexes)
  L5 direct-placement clue count ('at' clues)
CONCAT BASELINE: shared construction (7.0), no additions.
FOIL RULE: shared rule restricted to valid house indices â€” candidate
range [1, n]; T=1 has no F- and T=n has no F+, so the infeasible-T
set is {1, n} at minimum; values 1..7 are all single-digit non-round,
so digit-count and roundness matching are trivially satisfied and
token-count matching is asserted. AUDIT NOTE (structural, stated
a priori): for single-digit answers the leading digit IS the value,
so T-vs-foil leading-digit matching is impossible by construction;
the audit reports this as a structural imbalance, and the P1(b)
(token count, leading digit) fallback bins reduce to exact-value
bins â€” the fallback cannot rescue (b) here, only the exact-string
count can. Cross-sign (under/over-shoot) labels keep their numeric
meaning but carry no error-magnitude meaning for a categorical
answer; stated in any cross-sign sentence.
MARGIN CONSTRUCT NOTES: single-digit answer turns; the answer-turn
digits and report-turn digits are distinct positions in distinct
turns â€” position-scoped reads, no ambiguity (asserted by the prefix
machinery).
VARIANT E: "How many houses are there in the problem? Reply with ONLY
the integer." (expected n.)

### 7.3 program_exec (m4_families)

Whitelist status: eligible pending the Stage -1 whitelist selftest
(single-integer answers r0 in [0, 9999] by construction).
CUE BATTERY (exact): C1-C8 plus
  P1c program line count
  P2c conditional-jump line count ('if rX > rY goto L' lines)
  P3c backward-jump count (goto/if targets with L <= own line number
      â€” the loop count)
  P4c constant-operand arithmetic line count (set/addc/subc forms)
  P5c register-register arithmetic line count (add/sub/mul forms)
  P6c initial r0 value (parsed from the initial-values line)
  EXCLUDED, named: executed-step count â€” computing it requires
  simulating the program (a solved feature).
CONCAT BASELINE: shared construction (7.0), no additions.
FOIL RULE: shared rule; candidate range [0, 9999]; T=0 has no F-
(infeasible); sign always positive; the infeasible-T set otherwise
follows from digit-count/roundness/token-count cells enumerated by
the whitelist selftest.
MARGIN CONSTRUCT NOTES: answers up to 4 digits lengthen the answer
turn by a few tokens; rep_start and pos-ANS are runtime-derived per
conversation (asserts), so no constant-offset assumption exists to
break.
VARIANT E: "How many numbered lines does the program have? Reply with
ONLY the integer."

### 7.4 constraint_count (m4_families)

Whitelist status: eligible pending the Stage -1 whitelist selftest
(single-integer counts, generator floor count >= 5).
CUE BATTERY (exact): C1-C8 plus
  K1 range width b - a + 1 (parsed from the header line)
  K2 range start a
  K3 condition count (numbered condition lines)
  K4 digit-predicate condition count (digit-sum / contains-digit
     forms, by the verifier's condition regexes)
  K5 modular-predicate condition count (divisible / not-divisible /
     remainder forms)
  K6 perfect-square condition indicator
CONCAT BASELINE: shared construction (7.0), no additions.
FOIL RULE: shared rule; candidate range [5, min(certified attainable
max, b - a + 1)] â€” a count can neither undercut the generator floor
nor exceed the range width, and a foil outside those bounds would be
detectably impossible from the problem text alone.
MARGIN CONSTRUCT NOTES: counts can be large at high levels (multi-
token numerals); same runtime-derived position handling; no
family-specific hazard.
VARIANT E: "How many numbered conditions does the problem list? Reply
with ONLY the integer."

### 7.5 arith_long (m4_families)

Whitelist status: eligible pending the Stage -1 whitelist selftest
(single-integer answers; negatives possible).
CUE BATTERY (exact): C1-C8 plus
  A1 operation line count
  A2 'Multiply by' line count
  A3 'Divide by' line count
  A4 'Add' line count
  A5 'Subtract' line count
  A6 starting value (parsed from the 'Start with' line)
CONCAT BASELINE: shared construction (7.0), no additions.
FOIL RULE: shared rule; candidates same sign as T (sign matching is
load-bearing here â€” T may be negative), digit count and roundness on
|c| vs |T|, token count on the FULL rendered numeral including any
minus sign; candidate magnitude range = certified attainable range at
the landed level.
MARGIN CONSTRUCT NOTES: answers can be long multi-token numerals with
a sign token, giving the widest answer-turn length variance of any
family; rep_start / pos-ANS derivation is per-conversation with
asserts (the planted-report selftest covers multi-token layouts); the
report turn is unaffected.
VARIANT E: "What starting number does the problem give? Reply with
ONLY the integer."

## 8. Interpretation clauses (binding on every write-up sentence)

- ARM A is estimation only. No Arm A sentence may claim or deny
  metacognition; the comparison sentence names the exact cue list and
  states that "no signal beyond cues" is observationally equivalent
  to cue-reading (the project's standing scope note).
- PINNED verdict sentence: a pinned frontier reporter is the
  industry-phenotype pin observed in the cross-lab sweep, measured
  here at frontier scale; the 5-variant battery bounds (n=20, one
  seed, greedy, descriptive) are stated wherever it is cited.
- P1 pass = "the residual stream encodes the correctness of the
  in-context answer beyond prompt content and beyond linear
  answer-surface features." Fail on (c) alone = report which
  component account survives, plainly. If P1(b) was demoted, every
  P1 sentence names the demotion and the pair counts.
- A pos-REP positive eliminates "the information never reaches the
  report position" but does NOT eliminate "the output head discards
  it at decode time" â€” named in every pos-REP sentence (M8 clause,
  live-reporter form).
- P2 null at all in-regime doses = "within the tested dose range, the
  report pathway does not respond to the internal wrongness signal" â€”
  scoped to the range, stated with the largest in-regime dose.
- Positive differential with no sampled-report movement = "the
  pathway responds sub-threshold; decoding still flattens it."
- Any sampled-report movement/crack: report exact doses and texts;
  verify parse status and the coherence guard; hand-inspect every
  such row. Unparseable text COUNTS as a deviation, with parse status
  recorded per row and unparsed rates reported separately.
- CENSORING CAVEAT (frozen text, applies when the >20% flag fires):
  at-bound baseline reports cannot move further in the pinned
  direction, so a null on S_live/S_conf under censoring is NOT
  evidence of pathway unresponsiveness, and a positive must state
  that the effect is carried by the un-censored minority; the flag,
  the per-context bound fractions, and this caveat appear in every
  sentence that cites the affected statistic.
- Every comparison to the M7-M9 gemma-3-1b results states the mixed
  precision regime (bf16 full-precision small model vs 4-bit-weight
  frontier model) and the two-point scale contrast it licenses.

## 9. What each null means

- P1 null on (a): the frontier model's residual does not linearly
  encode in-context answer correctness at the tested layer â€” the M8
  small-model result does not transfer upward as measured.
- P1 null on (b) (when run): the correctness signal does not
  generalize across answer strings â€” an item-bound or surface-bound
  encoding, not an abstract evaluation.
- P1 fail on (c) alone: the signal decomposes into prompt-derived
  truth estimate + answer surface â€” a combiner, not an evaluation.
- Cross-sign asymmetry (descriptive): a one-signed-error shortcut,
  reported as the comparator lesson prescribes.
- P2 null (branch-appropriate, in-regime scope): the report pathway
  does not respond to the wrongness signal within the tested doses â€”
  for PINNED, the constant survives steering; for LIVE, the live
  channel does not move differentially.
- S_live positive with S_conf null: the pathway responds at the
  logit level but decoding/verbalization flattens it before the
  stated number changes â€” the frontier analog of M9's sub-threshold
  clause.
- S_conf positive with S_live null: stated numbers moved without a
  first-token logp differential â€” inspect for parse artifacts and
  multi-token report composition before any claim.
- Censored null: uninterpretable as unresponsiveness (section 8).

## 10. Budget coupling

  balance $123.70 at freeze; ceiling $400 (amended 2026-08-03); spent ~$26.3 total across all pods per m10_budget.log

The n ladder (400 -> 300 -> 250) is budget-triggered ONLY, decided
between arms from m10_budget.log alone, logged with the forcing
projection; a ladder step so logged is not an amendment. All other
budget rules are the mission's and override every scientific
preference; no analysis choice may be altered by budget state except
the ladder and the mission's pre-stated timeout/RED conversions.

## 11. Gates and order

Stage -1 items 1-7 all green locally (pipeline port + replay
regression on m8_rows.json; heterogeneous-map rehearsal; meta-device
dry run; report-token layout gate; numeric-family whitelist; THIS
TEMPLATE hashed) -> G0 pod environment gates 1-8 (weights shas;
coherent generation; noise-floor protocol; dual-path capture
cross-check; injection propagation; template/thinking gate; batch
gate; throughput pilot) -> Stage 1 calibration gate (accuracy in
[0.30, 0.80], strict parse >=0.9, whitelisted family; matched-pair
count logged) -> Stage 2 FREEZE (this template's five blanks; diff
logged; hashes to m10_freeze_hashes.txt) -> Stage 3 arms in fixed
order A, B, C under per-stage timeouts -> Stage 4 analysis exactly as
written + return bundle (template AND frozen prereg AND their diff
travel back). Result files are never overwritten; analysis artifacts
carry a shared run_id verified on every load (house rule).

## 12. Freeze protocol (restated, binding)

The Stage-2 freeze: (1) copies this template to m10_prereg.md;
(2) replaces exactly the five double-brace blank sites â€” named
landed_family_and_level, weights_shas, determinism_criterion, n,
budget_state â€” with their values; (3) runs a diff against this
template and verifies programmatically that every changed line
touches one of the five sites and no other line changed, logging the
diff; (4) writes sha256 of both files (and of m10_pipeline.py,
m10_families.py, m10_report_layout.json) into m10_freeze_hashes.txt;
(5) marks m10_prereg.md read-only. A diff outside the five sites is a
failed freeze and a RED gate. Nothing in Stages 3-4 may consult this
template's unselected family sections for anything but the proof that
they predate the data.
