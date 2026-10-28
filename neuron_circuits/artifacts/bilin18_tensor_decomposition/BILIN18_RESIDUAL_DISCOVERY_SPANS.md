## H1001 — Compact noun-contrast transfer, but fragment directions remain vocabulary-specific

Previous turn was progress: H1000 showed behavior-relevant variation outside the old six-task PCA128 span, while natural cluster-center spans failed to capture the effects. H1001 fits small residual contrast spans from discovery only and tests their transfer, completing1152finite interventions at MLP12.

Use the four discovery pair differences per family (16endpoint states total), subtract old task PCA128, and take uncentered SVD. Retain ranks1and4 per family. Pool both contrast matrices after equalizing family Frobenius norm and retain ranks1,4,8. All choices fixed before intervention. Controls: two seeded random residual bases at nested ranks1,4,8, identity, full donor, task128, full residual, and task128+pooled8 (136dimensions). Every basis acts on both families; no per-example correctness filtering or validation rank selection. No original whole-task holdout or list-marker data used.

| Evaluation cohort | Noun own rank1 effect/full | Noun own rank4 effect/full | Fragment own rank4 effect/full | Fragment own rank4 relative KL |
|---|---:|---:|---:|---:|
| Lexical |0.752|0.705|0.024|0.993|
| Structural |0.889|0.907|0.897|0.119|
| Joint |0.830|0.860|0.040|0.978|

Effect/full is ratio of mean signed answer-margin change to mean full-donor change, within recipient context. Relative KL is KL(full-donor-patched output || component-patched output), divided by identity baseline; lower is better. Finite effect ratios are not an additive causal partition or explained-variance fraction.

Noun rank1 captures a transferable contrast across these subjects and wordings. However the discovery and lexical cohorts reuse knife/cut versus key/unlock cues; structural wording uses sliced/opened. This could still be a stable lexical/domain contrast rather than abstract noun expectation. Rank1 aggregate relative KL0.429 shows that preserving the answer contrast does not preserve the entire output distribution.

Fragment rank4 captures~90%effect under rewording of the same fragments, but only2–4%for the four unseen fragment sources. The aggregate36%effect across validation cohorts hides this failure. This is evidence for source-specific continuation directions in the present fit, not a general morphology or word-completion component. More varied discovery vocabulary may help, but should be tested rather than presumed.

Cross-family rank4 transfer is negligible: fragment span contributes0.3%of noun full effect and noun span0.1%of fragment full effect. Four principal cosines between spans are0.113,0.088,0.054,0.008, indicating low geometric overlap. These two fitted task contrasts provide no positive evidence for shared computation. This does not rule out reuse elsewhere or under different tasks.

Adding pooled8directions to old task128 reduces aggregate relative full-patch KL from0.793to0.228 for nouns and0.851to0.548 for fragments. The augmented fragment representation still fails lexical/joint generalization (relative KL0.875/0.881), while structural KL0.057 is strong. Pooled rank1 predominantly serves the low-rank noun contrast despite equal family energy, illustrating why allocation based on variance can neglect a more distributed task. Random rank-matched directions capture little effect.

Discovery augmented136 KL is effectivelyzero by construction: its span includes all eight fitted discovery differences. This is an implementation invariant, not generalization evidence. Augmented136 uses more dimensions than old128; this experiment is not a parameter- or active-dimension-matched BSF comparison. No whole-state reconstruction or old-task retention claim follows from these paired interventions.

Decision: keep noun rank1 as a provisional causal contrast anchor, not a verified abstract semantic feature. Next prioritize factorial noun probes separating instrument, action and output category, including novel tools/actions and irrelevant cue controls, evaluated with the frozen direction. Preserve fragment rank4 as a diagnostic of lexical memorization; do not label it a broad morphology module. After nuisance controls, feed supported anchors into the joint featurizer and test old-task retention and natural-text fidelity. The broader bootstrap loop remains incomplete; do not resume an architecture sweep before resolving the supervision target.

Audits:1152/1152unique interventions. Identity logits2.57e-5maximumerror; decomposition2.84e-14; basis orthogonality1.55e-15. Frozen basis and input hashes unchanged, no exact training/evaluation prompt overlap. Independent saved-logit margin/KL and projection-norm replay errors0. Independent discovery-only basis-projector reconstruction error4.92e-17 from exactlyeightpairs/16states. All finite; no reserved tests. Natural-document pool remains26,112. H961 context-bridge evaluation remains pending; broad goal active.
