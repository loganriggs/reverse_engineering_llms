# DGP variation as a test of weight-based interpretability

The main outcome is whether a method applied to **weights** recovers rules and reusable structure that a bilinear transformer actually learned from a known, variable, language-like data-generating process (DGP). Capability is a prerequisite for a particular rule's weight-level evaluation, not the target to maximize. The large probabilistic-grammar proposal is a source of components, not a requirement to build a perfect language oracle before training.

## Experimental unit

Sample complete documents from one joint DGP. Local statistics (bigram, skip trigram), copying/induction, bracket and quotation closure, entity reference, agreement, arithmetic, genre, and layout may coexist and interfere in a document. Record the sampled latent events and the rule responsible for each prediction site. Keep these labels separate from anything used to fit a supposedly unsupervised weight decomposition.

Vary DGP knobs with controlled seeds: lexical frequency tail and document burstiness; paraphrase and syntactic realization entropy; genre and discourse transitions; each rule's prevalence and context distance; entity/distractor count; and whether rules are independent, correlated, or condition one another. Vary model depth/width/training seed separately. Start with a small factorial covering lexical/surface variation, rule prevalence, and cross-rule coupling while holding total tokens and architecture fixed; extend only after the first comparison reveals which factor matters. Compare generated samples and independent FineWeb panels on word and n-gram frequencies, vocabulary growth, repetition by distance, sentence/paragraph length, layout, and qualitative coherence. Report gaps as continuous measurements, not a single pass/fail score.

## Gate each interpretation claim on learned behavior

For each rule family, use unseen generated documents, balanced alternatives, supported counterfactual edits, and held-out combinations. A rule counts as learned only if the model predicts its target beyond local/frequency shortcuts and responds in the specified direction when the relevant cause changes. Report uncertainty over documents and model seeds. If a rule is absent or weak, record that as a learning result; do not score the absence of a weight circuit as an interpretability failure. Also preserve untrained and shuffled-label controls.

An exact marginal next-token oracle helps quantify regret and contextual information when available, and small exact fixtures remain useful correctness tests. It is not needed to label sampled rule sites, run controlled edits, or compare weight methods. Do not substitute the generator's hidden sampled state for the visible-prefix predictive target when making oracle claims.

## Weight-method comparison

On each behaviorally learned rule, compare the original iterative L1/prune method with final **no-L1** fine-tuning; direct embedding-to-unembedding and QK/OV path inspection; weight-only canonical/bond bases and gauge/sparse rotations; and task-conditioned or data-weighted variants as explicitly separate methods. Match retained function fidelity, total nonzeros/edges, and analysis budget where possible. Score whether a proposed component is causally necessary/sufficient for the edit, specific to the rule rather than a correlated token class, stable across seeds and surface variants, and reusable on held-out rule combinations. Reconstruction or visual sparsity alone is not evidence of rule recovery. A task-conditioned method can use rule labels for fitting, but must be evaluated on distinct documents, edits, and rule combinations; label-free discovery uses labels only after fitting.

## Immediate next comparison

The [natural-prose spelling-convention pilot](CORPUS_SPELLING_STYLE_PILOT.md)
shows that retaining real surface text alone does not provide enough
learnable rule instances: 1,970 training windows produced strong training
target memorization but no held-out cue-dependent behavior in depth-1 or
depth-2 models. It remains a controlled realism anchor, not the main DGP.
The [corpus construction audit](DISCOURSE_CONSTRUCTION_FORM_AUDIT.md)
then tested a corpus-inspired V3 renderer: it preserved semantic and causal
checks and improved short-sentence length, but its phrase rates overshot
natural prose and repeated trigrams did not improve. V3 is a negative
surface control, not the next training distribution.
The [purpose-driven report candidate](PROJECT_REPORT_DGP_V1_AUDIT.md)
then verified a coherent three-rule event chain and four DGP-control cells,
but its fixed act sequence and tiny template bank produced 89–95% repeated
trigrams across documents, against .73% on a fresh FineWeb form panel.
It is a semantic/causal fixture, not the main training distribution.
The same report's corpus-carrier screen found only 79 superficially
topic-matched, repeated-project, multiline documents in 10,000 FineWeb
sources; this is an upper bound before semantic validation. Merely
attaching natural text would also hide formulaic rule clauses in aggregate
form metrics. The next surface method must vary and validate the
task-bearing prose itself.

Build a coherent generator revision from the existing sampled-rule machinery.
Retain a purpose-driven document/event graph rather than independent listings,
and source a substantially larger, variable realization space from natural
text rather than adding a few more hand-written phrases. Audit task-bearing
spans separately from any natural carrier so aggregate realism cannot hide
formulaic rule clauses. Include
(1) at least two surface/lexical-variation settings and (2) independent
versus coupled rule settings. Keep explicit per-site provenance and supported
edit pairs. The first comparison should jointly include at least two rule
families in each training document, with enough instances of each family for
learning and a frozen token budget. Diagnose whether the rules remain
predictive from visible context under the more variable realization. Compare
the sampled documents to held-out natural prose on lexical, layout, local
repetition, and discourse-form screens, and inspect actual examples. Train a
small fixed-budget depth-1/depth-2 pilot across settings, score learned-rule
panels, then compare weight methods on the same **behaviorally learned**
rule(s). Include the older V3 sampled generator as a controlled, less
realistic baseline. Do not resume PCFG-only scale sweeps until a specific
weight-recovery question needs them.

The subsequent [V4 clause-grammar control](PROPOSITION_CLAUSE_AST_V4_RESULT.md)
improved semantic phrasing while leaving a large full-document and rule-site
form gap. Stop adding finite sentence templates as the main route. The next
renderer must vary sampled document state and discourse alongside a
compositional, corpus-calibrated realization process. The [natural joint-rule
opportunity audit](NATURAL_JOINT_RULE_OPPORTUNITIES.md) also found that only
8.1–9.6% of 2,000-document preview panels contain two or more currently
detectable mechanical rule opportunities. Thus task prevalence must be an
explicit DGP knob: compare natural-rate and enriched regimes, preserve
source-disjoint holdouts, and report selection rate. This is a calibration
constraint, not evidence that a natural carrier supplies a full symbolic
oracle or that the semantic renderer is already realistic.
