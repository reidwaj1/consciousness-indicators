# M10B PREREGISTRATION (frozen verbatim from M10B_MISSION.md before any analysis)

## Preregistration (freeze m10b_prereg.md from this section verbatim
## BEFORE running any analysis; sha256 into m10b_freeze_hashes.txt)

Data: M10 arm B pos-ANS captures at L39 (primary; L35/L41
co-reported), all 400 items x {T, F-, F+}. Split items by Arm A
own_correct (expected ~334 right / ~66 wrong; report exact counts).

Probe: to avoid the probe learning from the disagreement set itself,
fit ONLY on model-RIGHT items (grouped 5-fold within that set for its
own sanity AUC), then apply the frozen fit to ALL model-wrong rows.
Same probe settings as M10 (StandardScaler, LR C=0.1).

PRIMARY: within-item T-vs-foil AUC on model-WRONG items (probe
trained on model-right items only), item-clustered bootstrap CI,
compared side by side with the same statistic on held-out model-RIGHT
items. Decision language, frozen now:
- CI overlapping the model-right value and excluding 0.5 by a wide
  margin (point estimate >= 0.85): TRUTH-TRACKING.
- Point estimate <= 0.65 or CI including 0.5: BELIEF-TRACKING
  (collapse); below 0.5: belief-tracking with inversion.
- Between 0.65 and 0.85: MIXED - report as partial, no strong claim.

SECONDARIES (named, all on model-wrong items):
1. Belief-label AUC: score every row against the label "forced answer
   == model's own Arm-A answer" instead of truth; report truth-AUC vs
   belief-AUC side by side (whichever is higher is what the signal
   tracks).
2. The foil==belief subset: rows where a planted foil happens to
   EQUAL the model's own wrong answer (truth says wrong, belief says
   right - the purest cells). Report their raw probe scores against
   matched T rows; count the subset first, and if n < 10 report
   descriptively without a claim.
3. Per-layer sweep of the primary (all seven captured layers).
4. pos-REP repeat of the primary (does whatever the answer-position
   shows survive to the speaking position?).

CONFOUND GUARD (binding): model-wrong items are harder items. The
primary is a WITHIN-ITEM contrast, which difficulty cannot explain
(same item, same difficulty, T vs foil). Any cross-item statistic
must carry the difficulty caveat explicitly.


(Selftests and report-back per the mission; frozen 2026-08-03 16:58:18)