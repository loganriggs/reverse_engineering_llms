## H633 — Task means transfer across items better than across constructions

H632 was progress: nine frozen input probes showed that bilinear-context sensitivity extends beyond the original selected direction. H633 measures whether grouping by task is sufficient to predict native output responses, before fitting a more elaborate context-conditioned representation.

Freeze80 prompts: copying, capital facts, plural inflection, subject–verb agreement and single-digit addition; two templates and eight items per family. Reuse all nine equal-norm H632 directions at the final Attention5 output. For each direction, estimate full-logit derivatives by central differences at ±0.02. Fix the first four items of template0 in each family as discovery data (20 prompts). Predict the other60 contexts with either the global discovery mean or that task's discovery mean. Means are fitted separately for each direction. This tests unseen contexts, not unseen input directions, low-rank operators, or causal reference replacements.

The held-out template0 cell has four new items per family. Template1 has all eight items, including four identities shared with discovery. No output-based selection or prompt edits were made. Grammar labels alternate singular/plural; generic subject continuation permits grammatical alternatives beyond is/are. Numerical labels are fixed sums. Plural candidates explicitly include singular competitors. These first-token scores are not free-generation semantic accuracy.

**Task grouping helps across items in the same construction, but is generally insufficient across wording.** Averaged equally over family, probe and held-out item, mean relative derivative error is0.620 for task means versus0.904 for global means within template0. For template1, errors are1.244 versus1.078. Task means beat global means on all five same-template cells, but only agreement and addition on the changed-template cells. An error of1 is the error of predicting no derivative; values above1 are worse than that zero baseline. These are means of per-example probability-weighted centered-logit errors, not causal fractions or statistical confidence estimates.

| Family / template | Held-out N | Top1 / candidate accuracy | Global mean error | Task mean error | Own-tangent KL at ±0.1 | Task-mean KL | Zero-response KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| copy_0 | 4 | 0.00 / 1.00 | 0.795 | 0.332 | 1.1e-06 | 0.00011 | 0.0011 |
| copy_1 | 8 | 1.00 / 1.00 | 1.043 | 1.314 | 6e-07 | 0.00094 | 0.00037 |
| fact_0 | 4 | 1.00 / 1.00 | 0.856 | 0.680 | 6.5e-06 | 0.00048 | 0.0034 |
| fact_1 | 8 | 0.50 / 1.00 | 1.005 | 1.538 | 2.5e-06 | 0.0018 | 0.00079 |
| plural_0 | 4 | 0.00 / 0.00 | 0.835 | 0.571 | 3.5e-06 | 0.00016 | 0.00042 |
| plural_1 | 8 | 0.00 / 0.00 | 1.081 | 1.297 | 7.3e-07 | 0.00037 | 0.00014 |
| agreement_0 | 4 | 0.75 / 1.00 | 1.069 | 0.952 | 2.4e-07 | 9.2e-05 | 0.00015 |
| agreement_1 | 8 | 0.75 / 1.00 | 1.084 | 0.923 | 1.8e-07 | 0.00011 | 0.00015 |
| addition_0 | 4 | 0.00 / 0.00 | 0.965 | 0.568 | 1.3e-06 | 2.8e-05 | 9.1e-05 |
| addition_1 | 8 | 0.00 / 0.00 | 1.176 | 1.148 | 1e-06 | 0.00055 | 0.0003 |

KL columns use only the original vector at finite doses±0.1, averaging both signs. For each receiver, the predictor adds its own derivative or the frozen task mean to its native logits; the zero predictor uses unperturbed logits. KL is from the actually perturbed token distribution to the prediction. The native logits are known receiver-specific information in every method; this is a local surrogate, not a substitute language model. The full derivative-error columns include all nine probes. Absolute KL is small because doses are small; failure relative to zero still matters for explaining the response.

**Wording failure is not merely a consequence of failed tasks.** Explicit “Copy exactly: … / Copy:” has100% held-out top1 accuracy, yet its task-mean derivative error is1.314, and finite-dose task-mean KL exceeds the zero-response baseline. Capital question prompts have100% candidate accuracy and50% immediate top1 accuracy, but task-mean error1.538. The model's own context-specific tangent predicts finite-dose distributions much better, with mean KL around1e-7–7e-6 across cells. Thus a linear approximation can work locally while a shared task-average tangent fails to transfer.

**Competence must remain separate from the geometry.** The first copy template often predicts “ a”, despite ranking the intended word best among the eight copy candidates. Its candidate score cannot establish successful immediate copying. Both plural templates fail against singular competitors. Addition often repeats the first operand; all held-out addition candidates are wrong. Agreement candidate choice succeeds throughout, while native top1 sometimes selects another continuation. These failures are retained and prevent broad claims about mechanisms of successful inflection or arithmetic.

The experiment makes construction-conditioned transformations more plausible than task-label-only response units. It does not establish that all within-construction variation is low rank, that a context-conditioned predictor will generalize, or that the MLP6–9 bilinear mechanism causes every cross-task difference. Output means also conflate internal transformation with receiver-specific final readout.

Refreshed hypothesis set (20 alternatives; qualitative priority, not posterior probabilities):

1. Construction-dependent bilinear operating contexts alter several input directions. Strongest current mechanistic support from H626–632; test across these task contexts with internal factor interventions.
2. A small family of transformations indexed by construction generalizes across identities. Plausible from same-template transfer; require held-out context prediction, not clustering after seeing responses.
3. Task identity alone indexes reusable responses. Weakened by changed-template errors, including competent copying.
4. A task-independent fixed output direction suffices. Weakened by varied probe responses and global-mean errors.
5. The final readout accounts for much of apparent construction specificity. Earlier work rules out a complete explanation for the original vector; compare internal tangents and fixed/readout-swapped predictions on new tasks.
6. A scalar gain on a shared response is sufficient for some task cells. Still possible locally; compare direction-only cosine errors with amplitude errors, with any gain learned on discovery contexts.
7. Broad lexical emission or continuation preference explains much of the original vector. Plausible; compare target, competitor and non-answer probability changes.
8. Target identity changes the transformation even within a template. Supported as a residual possibility by nonzero within-template errors; hold out identities with matched context geometry.
9. Syntax rather than task semantics indexes transformations. Plausible; cross grammatical forms and task content independently.
10. Prefix or punctuation positions are the dominant context cues. Open; matched minimal-pair prompts can isolate them.
11. Current context replacement effects are generic mismatch rather than meaningful computation. Open; compare matched within-task references with cross-task references and native controls.
12. Several additive downstream changes explain sign reversals. Supported in H630; test attribution on new constructions without assuming irreducible synergy.
13. Strong nonlinear interactions between downstream operators are essential. Limited by additive sign predictions in H630, though interactions affect magnitude.
14. Normalization alone explains the context dependence. Weakened on H632 probes; remains a possible contributor in other tasks and doses.
15. Curvature explains poor transfer. Unlikely to explain these small-dose errors because own tangents predict±0.1 well; larger doses remain untested here.
16. The selected original direction is unusually sensitive, but controls are generic. Supported in magnitude on earlier language cells; sample more independently specified directions before ranking semantic specificity.
17. A shared low-dimensional output subspace exists even when mean vectors fail. Open and not tested by mean baselines; fit on discovery responses and evaluate held-out projection errors.
18. Compactness exists internally but final readout expands apparent diversity. Open; compare internal and probability-weighted output spectra using held-out data.
19. Poor competence creates spurious apparent task mechanisms. Material for plurals/addition, insufficient for explicit-copy transfer failure; report competent and failed cases separately without selecting away failures.
20. Candidate sets and token boundaries create misleading success. Demonstrated risk in copying/plurals; include competitors and distinguish immediate token choice from complete generated answers.

Next prioritize hypotheses1,2,5,11,17: capture internal transformations on this frozen panel, compare task/construction-conditioned predictors against fixed means, and include matched within-task versus cross-task context-replacement controls. A proper compactness test must fit its basis on discovery data and evaluate projection/reconstruction on held-out contexts; held-out directions need new independent probes or an explicitly justified linear representation. Keep competent copying/facts/agreement and failure cells visible. No rank search or outcome-based prompt revision has occurred in H633.

Audits:80 frozen prompts,20 discovery and60 held-out;2160 whole-model forwards; half-step convergence checked on original/span0/orthogonal0 over all80 prompts, maximum relative derivative difference0.001710; all recorded metrics finite; manifest/probes/runner hashes unchanged. Half-step agreement is a numerical convergence check, not an exact derivative proof. No new natural-text documents; FineWeb remains26,112. Broad research goal active and incomplete.
