# Causal-transfer hypotheses after M29

This is an evidence-ranked research review, not posterior probabilities or a claim that these explanations are mutually exclusive. The task remains understanding reusable units of computation; output fidelity and intervention-response fidelity are necessary diagnostics, not definitions of semantic units. M20–M29 closed several optimization explanations enough to return to representation transfer. Preserve the broader task/featurizer/manifold/discovery loop.

| # | Hypothesis | Current assessment | Discriminating evidence or test |
|---|---|---|---|
|1|Causal roles depend on surrounding context.|Leading open explanation: same-input unused groups benefit more consistently than mixed evaluation panels.|Paired wrappers with source and task-span delta fixed.|
|2|Training response targets cover too few directions.|Plausible contributor; unseen-group gains are weaker, though positive in every model/region aggregate.|Existing M28 weakens complete group memorization; compare direction coverage after input effects are controlled.|
|3|M11 versus M18 generator mismatch explains the gap.|Contributes but insufficient: M29 changes few winners.|Completed generator-only comparison; no further generator sweep.|
|4|Log-probability response and answer-margin fidelity optimize different aspects of behavior.|Strongly supported metric distinction, not a complete mechanism.|Retain both metrics and inspect endpoint contributions; never select one after seeing results.|
|5|A few high-energy examples dominate aggregate gains.|Plausible; family-weighted and effect-weighted conclusions differ.|Contribution concentration and paired leave-one-family-out summaries, no refitting.|
|6|Inactive directions inflate apparent success.|Cannot explain strict wins alone, but affect effective sample sizes.|Noop counts and nonzero coverage already retained; compare nonzero-only accounting without dropping the primary score.|
|7|Adam first-moment carryover caused all failures.|Rejected as sufficient: local M22 causality did not yield M23 rescue.|No more momentum sweep.|
|8|Minibatch direction conflicts prevent joint improvement.|Plausible for sequential training; not a feasibility barrier at starts.|M24–M25 construct nearby joint improvement; repeated optimization remains unproven.|
|9|TopK selection changes cause all instability.|Rejected as sufficient: smooth damage and fixed-support gains both observed.|Existing trajectory and finite-step decomposition; retain gate changes in future causal tests.|
|10|Finite-step curvature defeats infinitesimal predictions.|Supported in selected directions; not universal.|M25 full versus tenth displacement, fixed supports, no further step search.|
|11|The current architecture lacks capacity for these targets.|Not established; nearby joint improvements weaken an immediate impossibility claim.|Only consider capacity intervention after context/direction failures are localized.|
|12|Loss balancing by initial values mismatches gradient influence.|Supported accounting fact; causal importance unresolved.|M24 gradient norms; do not infer remedy by coefficient tuning.|
|13|Native model incompetence corrupts evaluation interpretation.|Possible in some cohorts; not permission to discard failures.|Report paired competence and all native failures alongside effects.|
|14|Small panels and two seeds give unstable rankings.|Persistent limitation, especially six-pair joint holdout.|Broaden competent task coverage once the paired mechanism is identified.|
|15|KL-only improvements account for all causal benefits.|Rejected for training and same-input new-group aggregates; partly explains evaluation gains.|M27 matched objective control, retained in every future comparison.|
|16|The response loss or implementation is incorrect.|Reduced by finite differences, native identity checks, parameter replay and independent metric audits; not logically eliminated.|Target specific contradictory evidence rather than repeating broad audits.|
|17|Decoder scale or collapsed directions create spurious gains.|Reduced by fixed native deltas, normalized columns and explicit energy denominators.|Keep intervention vectors independent of updated candidates.|
|18|Lexical novelty rather than contextual novelty is the main obstacle.|Open; current nonpaired strata do not establish causality.|Paired wrapper crossover first, then matched lexical edits if warranted.|
|19|Stable geometric blocks need not be stable computational units.|Still central: geometric overlap/manifold claims have not passed broad causal transfer.|Require consistent cross-context interventions and ablation/transfer before naming units.|
|20|Useful units are conditional subspaces or shared computations rather than globally named features.|Plausible architectural hypothesis, not established by these updates.|If paired context effects persist, model context-conditioned causal roles; compare against existing scalar/block controls before adding hierarchy/DAG complexity.|

Prioritize1,4,5,18 and19 for the next information gains. The immediate fixed-delta wrapper crossover addresses1/18 while retaining4. A negative result will not justify an optimizer grid; a positive result will not by itself validate manifolds. After the crossover, use concentration accounting and the existing scalar/block comparison to decide whether wider competent tasks, conditional representations or manifold validation is the next useful deliverable.

## M30 update

Hypothesis1 gains direct paired evidence for some model/scale cases, but its effect is not universal and top1-competent subset conclusions differ. Binary incompetence alone (13) does not explain the selected block reversals. Family omission accounting partially addresses5 but does not exclude endpoint concentration. Before fitting conditional representations (20), test whether recomputing the same group component in the target context changes the conclusion relative to a fixed transported vector. Positional shifts and wrapper meaning remain unseparated.

## M31 update

Reencoding the same group in each context changes some effects but does not rescue the full-step block sample crossovers. Block component medians stay geometrically close, weakening a wholesale transported-vector mismatch account. This does not refute conditional units: native response dependence and log-probability scoring must be separated before assigning the residual failure to representation geometry. The next native-only response decomposition tests hypotheses1/4 without changing student parameters or intervention construction.


## M32 update

Identical MLP1 interventions have context-dependent relative-logit effects in the native model. Removing the common log-probability shift leaves 57–91% of the response-change energy; 60–91% of that centered change is orthogonal to the reference response. These are descriptive aggregates on familiar tasks, not semantic-unit validation or a universal failure of context transfer. The shift-only and gain-only accounts are insufficient for these aggregates. A legitimate reusable feature can still have context-dependent readers. Separate local sensitivity from finite-removal curvature before fitting conditional representations. See [report](NATIVE_CONTEXT_RESPONSE_GEOMETRY.md).
