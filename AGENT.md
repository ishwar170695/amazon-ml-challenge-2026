# AGENT.md — Autonomous Local ML Engineering Loop

## 0. Mission

Solve the Amazon ML Challenge 2026 Business Entity Resolution problem.

The objective is:

> Continuously improve genuine local generalization of the entity-resolution pipeline while detecting and preventing leakage, synthetic-data coupling, overfitting, blocking failures, threshold artifacts, and false merges.

This is an autonomous engineering task.

Do NOT merely implement a model and report its metric.

The agent should repeatedly:

    inspect
      ↓
    establish baseline
      ↓
    analyze errors
      ↓
    form hypothesis
      ↓
    make ONE controlled change
      ↓
    evaluate
      ↓
    audit
      ↓
    keep / revert / investigate
      ↓
    formulate next hypothesis
      ↓
    repeat

Continue this loop until:

- useful improvements plateau,
- remaining problems require human judgment,
- a new unknown failure mode appears,
- evaluation integrity cannot be established,
- or another human-review condition below is reached.

Do not wait for the user after every normal iteration.

---

# 1. Local-Only Boundary

EVERYTHING MUST REMAIN LOCAL.

The agent must NEVER:

- submit to the competition
- upload predictions
- access a competition leaderboard
- query external business databases
- query Google or search engines for business information
- use external entity-resolution APIs
- use commercial geocoding/business APIs
- send competition data to external services
- upload datasets, predictions, logs, or model artifacts
- authenticate into competition submission systems
- publish results externally
- use external business information to improve predictions

Allowed:

- locally installed Python/Go/Rust/etc. packages
- offline/open-source libraries
- local documentation
- local datasets
- local synthetic generators
- local evaluation
- local model training
- local experiment artifacts

The project has NO automatic submission stage.

If a submission-format file is generated for local validation, it must remain local.

NEVER execute submission/upload commands.

---

# 2. Problem

There are three sources:

- Source 1: deduplicated reference businesses
- Source 2: noisy business records
- Source 3: noisy business records

For each Source-1 entity, identify every Source-2/Source-3 record referring to the same real-world business.

Valid outcomes include:

    S1 → no matches
    S1 → one match
    S1 → multiple matches

Therefore the system must handle:

- one-to-zero matching
- one-to-one matching
- one-to-many matching
- singleton detection

Important components:

    normalization
        ↓
    blocking / candidate generation
        ↓
    pairwise features
        ↓
    matching model
        ↓
    threshold / decision logic
        ↓
    one-to-many + singleton handling
        ↓
    local evaluation

The primary competition metric is F0.5.

Precision is therefore especially important.

---

# 3. Core Principle

Do NOT optimize the metric blindly.

Optimize:

    TRUSTWORTHY GENERALIZATION

not:

    validation score

and especially not:

    synthetic validation score

A metric increase is accepted as a genuine improvement only after the relevant evaluation and audit checks pass.

Always ask:

> What would make this result look good for a reason unrelated to actual generalization?

Try to falsify the result.

---

# 4. Mandatory Experiment Discipline

Every meaningful iteration must have ONE primary change.

Before changing anything, record:

    Observation:
    Hypothesis:
    Proposed change:
    Expected effect:
    Main risk:
    Falsifying experiment:

Example:

    Observation:
    Many false negatives involve abbreviated addresses.

    Hypothesis:
    Character-level address similarity does not capture
    abbreviation variation sufficiently.

    Proposed change:
    Add character n-gram similarity.

    Expected effect:
    Reduce address-related false negatives.

    Main risk:
    Increase false positives between nearby businesses.

    Falsifying experiment:
    Remove the new feature while keeping every other
    component identical and check whether the improvement disappears.

Do not simultaneously change:

- generator
- parser
- feature set
- model
- threshold
- blocking

unless explicitly conducting a controlled combined experiment.

Prefer one-variable experiments.

---

# 5. Continuous Engineering Loop

For every iteration:

## Step 1 — Inspect

Inspect the current:

- code
- model
- features
- normalization
- blocking
- thresholds
- evaluation
- train/validation/test construction
- generator, if synthetic data is involved
- experiment history

Do not assume the current implementation is correct.

## Step 2 — Baseline

Know the current best reproducible result.

Record:

- git commit
- dataset version
- split
- blocking configuration
- candidate count
- blocking recall
- precision
- recall
- F0.5
- threshold
- model
- feature set
- runtime
- memory if relevant

## Step 3 — Analyze errors

Inspect false positives and false negatives.

Identify the highest-value systematic weakness.

## Step 4 — Form hypothesis

Write the hypothesis and falsifying experiment.

## Step 5 — Run cheap test

If the hypothesis can be tested cheaply, test it before running a full expensive experiment.

## Step 6 — Make ONE controlled change

Implement the smallest change that tests the hypothesis.

## Step 7 — Evaluate

Run the required evaluation protocol.

## Step 8 — Audit

Check:

- split integrity
- generator/pipeline coupling
- feature leakage
- threshold behavior
- blocking recall
- feature dominance
- false positives
- false negatives
- subgroup/country behavior
- adversarial robustness where applicable

## Step 9 — Decide

Use exactly one:

    KEEP
    REVERT
    INVESTIGATE
    HUMAN_REVIEW

## Step 10 — Continue

If valid:

    keep change
    inspect remaining errors
    formulate next hypothesis

If invalid:

    revert
    understand why
    formulate different hypothesis

Do not repeatedly try random features.

---

# 6. Evaluation Gate

Every meaningful model iteration MUST evaluate at least:

- blocking recall
- candidate count
- precision
- recall
- F0.5
- threshold sweep
- sampled false positives
- sampled false negatives
- feature importance / SHAP where appropriate

Do not report an iteration as an improvement until this evaluation is complete.

---

# 7. Entity-Level Splitting

Train/validation/test separation MUST be entity-level.

Never use a naive pair-level random split.

The same underlying real-world entity must not appear across train/validation/test through:

- duplicate records
- normalized duplicates
- synthetic variants
- cross-source copies
- generated corruptions

Verify this explicitly.

If split integrity is uncertain:

    STOP
    INVESTIGATE

Do not report model metrics as trustworthy until resolved.

---

# 8. Threshold Selection

Threshold selection MUST be separated from final test evaluation.

Correct:

    train
      ↓
    validation
      ↓
    choose threshold
      ↓
    freeze threshold
      ↓
    test
      ↓
    report

Incorrect:

    test
      ↓
    choose threshold
      ↓
    report test

The test set must not be used to optimize:

- threshold
- margin
- model hyperparameters
- feature selection
- blocking parameters
- post-processing

## 8.1 Enforcement Guardrail (mandatory)

This rule has been violated in practice by ad hoc diagnostic scripts (e.g. a
one-off sweep computing F0.5 across a range of thresholds directly against a
held-out/test split "just to look"). A rule that is only documented and not
enforced will be violated again under time pressure.

Any script, notebook cell, or one-liner that computes precision / recall /
F0.5 against a named split MUST:

- take an explicit `split_name` argument or variable, and
- refuse to execute a threshold/margin/hyperparameter sweep against any
  variable, file, or split named (or aliased to) `test`, `held_out`,
  `heldout`, or equivalent, unless an explicit override flag is passed
  (e.g. `--i-know-this-contaminates-test`), and
- log any such override to the experiment ledger, since a test split that
  has been swept over for tuning purposes is no longer valid evidence and
  must be flagged as contaminated going forward.

If a test split is ever contaminated this way, do not silently keep using
it as if it were still clean. Either re-draw a fresh held-out split or
explicitly document the contamination and its scope in the ledger.

---

# 9. Threshold Sweep

Inspect the complete threshold behavior.

Record:

- threshold
- precision
- recall
- F0.5
- predicted-match count
- singleton rate
- average matches per S1

Flag:

- suspiciously flat sweeps
- suspiciously sharp peaks
- sudden discontinuities
- implausible match-count changes

Do not assume that a non-flat sweep proves validity.

Threshold behavior is diagnostic evidence, not proof of generalization.

Threshold sweeps for the purpose of *selecting* a threshold must only ever
be run against validation data (see Section 8.1). Sweeps run against test
data are diagnostic-only, must be explicitly labeled as such, and must not
feed back into any parameter choice.

---

# 10. Suspicious Result Gate

Immediately STOP normal optimization if:

- any metric exceeds 0.95 unexpectedly,
- F0.5 jumps by >0.05 in one iteration,
- precision/recall suddenly becomes near-perfect,
- blocking recall suddenly becomes near-perfect,
- candidate count unexpectedly collapses,
- a single feature dominates,
- a country/segment suddenly improves dramatically,
- a result looks substantially better than expected from the actual error analysis.

Do NOT report the result as an improvement.

Enter investigation mode.

---

# 11. Investigation Must Be Falsifiable

A plausible explanation is NOT an investigation.

For every suspicious result, write:

    Suspicious observation:
    Proposed explanation:
    Falsifiable prediction:
    Controlled experiment:
    Result:
    Conclusion:

Example:

    Observation:
    France F0.5 jumped from 0.81 to 0.97.

    Explanation:
    Improved street-number parsing.

    Falsifiable prediction:
    Removing ONLY street-number parsing should materially
    reduce the improvement.

    Experiment:
    Freeze generator, model, threshold, split, features,
    and all other preprocessing. Remove only street-number parsing.

    Result:
    ...

    Conclusion:
    ...

If the explanation cannot be tested, it is not sufficient.

Never clear a suspicious result merely because the explanation sounds reasonable.

---

# 12. Suspicious Result Isolation

When investigating, freeze:

- dataset
- split
- random seed
- model
- threshold
- blocking
- feature set
- generator
- evaluation code

Then change exactly ONE relevant variable.

Examples:

    baseline
    baseline + feature X
    baseline - feature X

or:

    generator A + pipeline A
    generator B + pipeline A

or:

    parser A + generator A
    parser B + generator A

The purpose is to isolate causality.

If multiple variables change, the investigation is inconclusive.

---

# 13. Synthetic Generator Coupling

If synthetic data is used, verify that the generator and pipeline do not share hidden structure.

Inspect for shared:

- vocabulary lists
- abbreviation lists
- suffix lists
- typo rules
- corruption rules
- transliteration rules
- locality lists
- field-splitting logic
- positional assumptions
- formatting assumptions
- generated prefixes/suffixes
- noise probabilities

Example of dangerous coupling:

    generator:
        Road → Rd

    feature:
        detects Road ↔ Rd

The model may be learning the generator rather than entity resolution.

---

# 14. Adversarial Noise Requirement

A synthetic result is not trusted solely because it works on the original generator.

Create or use a modified/adversarial generator that changes some combination of:

- abbreviation rules
- typo distributions
- token ordering
- missing fields
- address corruption
- transliteration
- noise intensity
- vocabulary
- corruption combinations

The modified generator must not simply reproduce the exact rules used to create the training/evaluation data.

A claimed improvement should survive the modified generator before being treated as robust.

---

# 15. False Positive / False Negative Analysis

Every model iteration MUST sample:

    N false positives
    N false negatives

For each sample record:

    S1 record
    candidate record
    score
    important features
    predicted decision
    likely reason
    human interpretation
    error category

Possible categories:

- abbreviation
- typo
- token reorder
- transliteration
- address variation
- missing address
- missing name
- locality mismatch
- postcode mismatch
- same-name different-location business
- duplicate branch
- generic name
- landmark address
- blocking failure
- classifier failure
- threshold failure
- singleton error
- one-to-many error
- **guard override failure** (a hand-written heuristic/veto suppressed or
  overrode a high-confidence, correct model prediction — see 15.1)
- unknown failure mode

Do not merely dump examples.

Reason about patterns.

## 15.1 Guard Override Failure Mode (mandatory check)

This has been observed in practice: a large majority of false negatives
(87.7% in one iteration) had model confidence P ≥ 0.88 — often P > 0.999 —
but were blocked by a rigid, hand-written string-similarity guard. The
guard was net-negative: it suppressed correct, high-confidence model
output rather than protecting against errors.

Whenever any manual guard, veto, or acceptance rule sits downstream of the
classifier (secondary-match guards, margin rules, house-number veto,
name/address agreement checks, etc.), every iteration that touches that
logic MUST additionally report:

- the fraction of false negatives that had model probability above a
  "should have been trusted" bar (e.g. P ≥ 0.90),
- whether those suppressions were concentrated in a specific guard or
  rule,
- what happens to precision if that guard is loosened or removed
  entirely, evaluated on validation (never test) before any change is
  kept.

A guard should be treated with the same suspicion as a dominant feature
(Section 19): understand *why* it exists, whether it is still earning its
keep, and whether it is now costing more recall than it protects in
precision.

---

# 16. Blocking Is a First-Class Model

Measure:

    Blocking Recall =
    true matches present in candidate set
    /
    total true matches

Also measure:

- candidates per S1
- total candidate pairs
- reduction ratio
- runtime
- memory

Separate:

    blocking false negative

from:

    classifier false negative

If the true pair never enters the candidate set, the classifier cannot recover it.

Do not optimize the classifier while ignoring blocking recall.

---

# 17. Blocking Strategy

Prefer multiple complementary blocking methods.

Possible methods:

- exact country
- normalized name tokens
- significant name tokens
- character n-grams
- phonetic keys
- locality/city
- postcode
- address tokens
- distinctive/long address tokens (excluding generic address stopwords —
  this recovered rebranded-business and phonetic-drift cases that
  standard name/address blockers missed)
- approximate retrieval
- MinHash/LSH where justified

Usually combine complementary blockers using UNION rather than intersection.

Measure the marginal contribution of each blocker.

For every blocker ask:

    How much blocking recall does it add?
    How many candidates does it add?
    What failure modes does it recover?
    Is the computational cost justified?

---

# 18. Feature Engineering

Potential name features:

- normalized exact match
- Levenshtein/edit distance
- normalized edit distance
- Jaro-Winkler
- token Jaccard
- token overlap
- token-sort similarity
- word TF-IDF cosine
- character n-gram TF-IDF cosine
- shared-token count
- length ratio
- phonetic skeleton similarity
- compact-brand similarity

Potential address features:

- normalized exact match
- character similarity
- token overlap
- TF-IDF cosine
- postcode agreement
- city/locality agreement
- state agreement
- numeric-token agreement
- address-component agreement
- length ratio
- missingness indicators
- unit/suite key agreement

Potential cross-field features:

- country equality
- country missingness
- name/address agreement combinations
- number agreement
- shared significant tokens
- disagreement indicators

Do not add features merely because they are available.

Each feature should address an observed failure mode or have a clear hypothesis.

---

# 19. Feature Dominance Audit

Every iteration MUST inspect feature importance.

Use:

- native model importance
- permutation importance where useful
- SHAP where useful

If one feature suddenly dominates:

    STOP
    INVESTIGATE

Ask:

- Is it legitimate?
- Is it leaking the label?
- Is it derived from generator logic?
- Is it a source/row-position proxy?
- Does it survive adversarial generation?
- Does removing it preserve the result?

A dominant feature is not automatically invalid, but it must be understood.

TF-IDF-derived features (word/char cosine) are fit on a specific corpus
vocabulary. When the underlying corpus changes (synthetic → real, or a
newly added open-set country's vocabulary is merged in), these features
must be refit and re-audited — a large importance weight carried over from
a stale vocabulary fit is itself a form of silent drift and should be
treated with the same suspicion as any other dominant feature.

---

# 20. Model Strategy

Use the simplest model appropriate for the evidence.

Potential progression:

    heuristic baseline
        ↓
    simple classifier
        ↓
    gradient boosted trees
        ↓
    more complex model only if justified

LightGBM/XGBoost/CatBoost are reasonable candidates for pairwise ER features.

Do not force a model progression merely for the sake of testing models.

A simpler model may be skipped if there is a documented reason.

Model complexity must justify itself through:

- genuine validation improvement
- better error behavior
- robustness
- acceptable runtime
- acceptable memory

## 20.1 Ensembling (mandatory checks when used)

If multiple models are combined (e.g. bagging/seed-averaging, varied
tree depth/leaves/column-subsampling), the following must be reported
in addition to the standard evaluation gate:

- per-model metrics individually, alongside the ensemble metric — never
  report only the ensemble number,
- whether the ensemble gain persists on an expanded held-out sample, not
  just the original (often small) held-out split — a gain measured on a
  few thousand entities can be within sampling noise,
- an explicit check of whether the gain is better attributed to variance
  reduction (probability averaging smoothing out individual-model noise)
  versus genuinely different signal captured by different models,
- the added runtime/memory cost of serving N models versus one, and
  whether that cost is justified by the measured gain (Section 30/31
  compute discipline still applies to ensembles).

An ensemble gain that cannot be shown to survive a larger sample should be
treated as INVESTIGATE, not KEEP, until re-confirmed.

---

# 21. One-to-Many Matching

Do not force:

    one S1 → one candidate

The output may contain:

    S1 → S2-A, S2-B, S3-C

Inspect:

- number of matches per S1
- score distribution
- best-vs-second-best score
- duplicate candidates
- same-name/different-location cases

Do not introduce extra-match heuristics without validation.

Prefer, where feasible, evaluating whether a global/collective assignment
step (e.g. min-cost bipartite matching over ambiguous multi-candidate
clusters) recovers cases that a purely greedy top-1-then-margin-based
secondary acceptance rule misses. A greedy resolver is a reasonable
starting point but is not guaranteed to find the globally optimal
one-to-many assignment.

---

# 22. Singleton Handling

If no candidate passes the acceptance criteria:

    predict singleton

Do not force a match.

Monitor:

- singleton rate
- false-positive rate among true singletons
- best candidate score
- score gap between top candidates

A suspiciously low singleton rate requires investigation.

---

# 23. Country / Open-Set Handling

Do not hard-code the training countries.

The system must handle unseen countries such as France.

Test:

- known countries
- unseen countries
- missing country
- formatting variations

Never implement logic equivalent to:

    if country not in known_training_countries:
        reject

unless explicitly justified by the problem.

## 23.1 Open-Set Validation Protocol (required practice)

Because an open-set country may have zero ground truth anywhere in the
training data, standard held-out evaluation is not possible for it. The
required protocol, established in practice for France and to be reused
for any future unseen country/segment, is:

1. Confirm via direct data audit whether the country appears in train at
   all (record counts, do not assume).
2. If absent, construct a synthetic stress test using real records and
   real address/name patterns *extracted from the actual test-set
   entities for that country* — not fully synthetic invented data — with
   manually constructed positive and negative pairs.
3. Explicitly label this evaluation as a synthetic proxy, not equivalent
   in strength to a real held-out numeric result (a small hand-built
   stress test, e.g. 8 pairs, demonstrates the model isn't confused by
   the country's orthography/legal forms — it does not establish that
   real recall/precision matches the numbers seen on countries with
   ground truth).
4. Scale the stress test up (hundreds+ pairs, not a handful) before
   treating a "100% accuracy" result on it as strong evidence rather than
   a sanity check.
5. Separately verify blocking recall for the open-set country using the
   same synthetic-pairs-against-real-distractors method, since blocking
   failures are invisible to a classifier-only stress test.

---

# 24. Real-Data First Pass

When the actual competition dataset becomes available, do NOT immediately optimize.

First perform a local data-sanity pass.

Inspect samples from every source.

Check:

- raw names
- raw addresses
- countries
- missingness
- Unicode/non-Latin scripts
- accented characters
- multilingual records
- postcode formats
- address structures
- punctuation
- unusual separators
- encoding artifacts
- source-specific formatting
- extremely long/short fields

Document:

    expected assumption
    observed reality
    mismatch
    consequence

If a genuinely new failure mode appears:

    STOP
    HUMAN_REVIEW

Do not silently modify the pipeline around an unknown failure.

---

# 25. Real Data Re-Tuning

Do NOT carry synthetic-tuned constants directly into real data.

Re-tune using real training/validation data:

- thresholds
- margins
- blocking parameters
- post-processing
- acceptance rules

Test blocking recall separately by:

- country
- source
- relevant segment
- data-quality regime

A strong classifier score does not compensate for poor blocking recall.

---

# 26. Regression Tests

Every discovered important failure mode should become a local regression test.

Examples:

- abbreviation
- typo
- transliteration
- address reorder
- missing postcode
- missing address
- same-name different-location
- singleton
- one-to-many
- unseen country
- duplicate branch
- previously discovered leakage pattern
- guard override (a high-confidence correct prediction suppressed by a
  hand-written heuristic/veto — see 15.1)
- test-set threshold contamination (a sweep or tuning step run against a
  supposedly-frozen test split — see 8.1)

Future changes must not silently reintroduce known failures.

---

# 27. Experiment Ledger

Maintain a local experiment ledger.

Every iteration records:

    iteration_id
    timestamp
    git_commit
    hypothesis
    change
    dataset
    split
    blocking
    candidate_count
    blocking_recall
    model
    features
    threshold
    precision
    recall
    F0.5
    runtime
    FP findings
    FN findings
    feature audit
    leakage audit
    adversarial result
    decision
    next hypothesis

Decision:

    KEEP
    REVERT
    INVESTIGATE
    HUMAN_REVIEW

Never lose experiment history.

Any test-split contamination event (Section 8.1) or guard-override finding
(Section 15.1) must be explicitly recorded in the ledger, not folded
silently into the general notes.

---

# 28. Fresh-Eye Review

Run a fresh-eye review:

- every 3 feature/model changes,
- OR whenever a metric improves by >0.05,
- OR whenever a suspicious result occurs,
- OR whenever the pipeline architecture changes materially.

Pretend you are seeing the system for the first time.

Explicitly inspect:

1. split assumptions
2. positive generation
3. negative generation
4. blocking
5. normalization
6. feature construction
7. thresholding
8. post-processing
9. generator/pipeline shared vocabulary
10. corruption rules
11. positional assumptions
12. field-splitting assumptions
13. source-specific artifacts
14. hidden label information

Then answer:

> What would make this result look good for a reason unrelated to genuine generalization?

Try to disprove the current result.

The fresh-eye review must be surfaced in the experiment record.

Do not silently skip it.

---

# 29. Required Fresh-Eye Output

Use:

    FRESH-EYE REVIEW

    Assumptions checked:
    - ...

    Possible coupling:
    - ...

    Leakage risks:
    - ...

    Falsification tests:
    - ...

    Findings:
    - ...

    Decision:
    CONTINUE / INVESTIGATE / REVERT / HUMAN_REVIEW

Even if nothing is found, say so explicitly.

---

# 30. Compute Discipline

Do not run expensive full-dataset experiments unnecessarily.

Prefer:

    hypothesis
       ↓
    cheap targeted test
       ↓
    promising?
       ↓
    medium experiment
       ↓
    promising?
       ↓
    full evaluation

Use the smallest experiment capable of falsifying the hypothesis.

Track:

- runtime
- memory
- candidate count
- feature-generation cost
- training cost

Optimize measured bottlenecks, not assumed bottlenecks.

---

# 31. Diminishing Returns

If repeated experiments produce:

- negligible gains,
- unstable gains,
- gains that disappear under adversarial testing,
- gains smaller than evaluation noise,
- increasing complexity without meaningful error reduction,

stop speculative feature accumulation.

Prioritize:

1. robustness
2. systematic error removal
3. blocking recall
4. subgroup failures
5. runtime/memory
6. reproducibility

Do not spend unlimited compute chasing tiny metric increases.

---

# 32. Do Not Optimize Against the Leaderboard

There is no leaderboard feedback loop.

Do not:

- submit
- upload
- check leaderboard
- change the model based on leaderboard feedback
- use external competition results

Optimize entirely against local evidence.

---

# 33. Human Escalation

Do not self-resolve when:

1. A new failure mode appears that does not fit known categories.

2. Two consecutive fresh-eye reviews disagree.

3. You cannot determine whether a result is valid.

4. The evaluation protocol itself appears flawed.

5. Leakage is suspected but cannot be resolved.

6. Real data contains a structural property not anticipated by the pipeline.

7. A major architecture change is required to handle an unresolved issue.

8. A test split is discovered to have been contaminated (Section 8.1) and
   it is unclear whether a fresh split can be drawn without re-leaking
   entities already seen.

When escalating, provide:

    Problem:
    Evidence:
    Experiments performed:
    What remains uncertain:
    Decision required:

Do not hide uncertainty.

---

# 34. Iteration Output

After EVERY meaningful iteration, output:

    ITERATION
    ID:
    Hypothesis:
    Change:

    METRICS
    Blocking recall:
    Candidates:
    Precision:
    Recall:
    F0.5:
    Threshold:

    ERROR ANALYSIS
    False positives sampled:
    Main FP categories:
    False negatives sampled:
    Main FN categories:
    Guard override findings (if applicable):

    FEATURE AUDIT
    Dominant features:
    Suspicious features:

    LEAKAGE AUDIT
    Split leakage:
    Generator coupling:
    Feature leakage:
    Test-split contamination status:

    ROBUSTNESS
    Adversarial evaluation:
    Result:

    DECISION
    KEEP / REVERT / INVESTIGATE / HUMAN_REVIEW

    NEXT HYPOTHESIS:
    ...

Do not omit sections simply because the result looks good.

Silence is not evidence that a check happened.

---

# 35. Definition of "Improvement"

A change is a genuine improvement only when:

- evaluation is valid,
- split integrity is intact,
- no important leakage is detected,
- the result is reproducible,
- error analysis supports the change,
- adversarial testing does not invalidate it where applicable,
- blocking has not regressed materially,
- subgroup regressions are understood,
- and the complexity/cost is justified.

A higher headline F0.5 alone is insufficient.

---

# 36. Default Autonomous Loop

Unless a human-review condition is reached, continuously execute:

    1. Inspect current state.
    2. Reproduce/confirm baseline.
    3. Inspect FP/FN errors.
    4. Identify highest-value weakness.
    5. Form one hypothesis.
    6. Design a falsifying experiment.
    7. Run the cheapest useful test.
    8. Make one controlled change.
    9. Run evaluation.
    10. Check threshold behavior.
    11. Sample N false positives.
    12. Sample N false negatives.
    13. Check feature importance / SHAP.
    14. Check blocking recall.
    15. Check split integrity.
    16. Check generator/pipeline coupling.
    17. Check suspicious-result conditions.
    18. Run adversarial evaluation when required.
    19. KEEP / REVERT / INVESTIGATE.
    20. Record experiment.
    21. Every 3 changes, run fresh-eye review.
    22. Form next hypothesis.
    23. Repeat.

Do not stop simply because the model trains.

Do not stop simply because the metric is high.

Do not stop simply because an experiment improved the metric.

Continue engineering until a stopping condition or human-review condition is reached.

---

# 37. Final State

The final output of this project is:

    BEST VALIDATED LOCAL PIPELINE

containing locally:

- preprocessing
- normalization
- blocking
- feature generation
- model
- threshold configuration
- evaluation
- adversarial testing
- regression tests
- experiment ledger
- error analysis
- fresh-eye reviews
- known limitations
- local prediction artifacts

There is NO submission step.

There is NO upload step.

There is NO external leaderboard step.

Everything remains local.

The final report should state:

    Best validated local result:
    Pipeline:
    Blocking recall:
    F0.5:
    Main strengths:
    Remaining failure modes:
    Known risks:
    Reproducibility information:
    Whether human review is required:

Never claim that the system is competition-ready merely because the local metric is high.