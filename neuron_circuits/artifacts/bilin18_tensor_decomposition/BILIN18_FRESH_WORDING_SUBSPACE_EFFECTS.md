## H747 — Fresh narrative wording confirms causal transfer; list competence fails

All H746 candidate bases were frozen before generating this panel. Two new narrative templates use concert/message/talked-to (A) and library/book/wrote-to (B). Two list templates use giver/receiver (A) and source/destination (B). Name cohorts are reused from H744, so freshness concerns wording, not identities. Internal wording tags old/new denote A/B here; they do not denote the H744 templates. Rank remains four separately for query and key, with receiver-norm restoration and a surgical single-edge intervention at L13H0.

Of128 pairs, eight Harold/Doris cases fail the token-validity or endpoint-length rule;120 pairs/240 contexts remain. All120 binding/copy endpoint comparisons match source/final tokens and positions. Four narrative comparison cells pass both native task gates. Every list-binding cell fails the >=75% both-endpoint-choice threshold: A62.5%/71.4%, B12.5%/14.3%, old/new name cohorts respectively. Copying passes everywhere. No failed cell was removed or repaired after observing outputs.

| Task-competent narrative cell | Original operation | New-wording operation | Pooled operation | Identity control | Wording control | Random controls |
|---|---:|---:|---:|---:|---:|---:|
| Old names, A | 10.3% | 22.1% | 8.4% | 90.5% | 82.6% | 96.2–103.0% |
| Old names, B | 8.1% | 17.9% | 16.4% | 99.2% | 96.3% | 97.9–102.2% |
| New names, A | 10.7% | 21.8% | 10.7% | 91.9% | 84.4% | 96.0–103.0% |
| New names, B | 16.8% | 21.5% | 27.7% | 98.5% | 97.5% | 97.4–101.9% |

Values are downstream absent Kevin-versus-Brian contrast RMS error, normalized by the full-factor exchange's RMS change from native. All eligible contexts contribute to errors; the table does not select a winning basis per case. Original-value intended-margin errors for the original operation basis are11.5%,8.1%,11.1%,18.6% in the same order. Thus coefficient fidelity is reflected in actual downstream interventions on both original and absent values.

Sign transfer is less complete than RMS accuracy suggests. Among cases where the full intervention reverses the native absent-value contrast, the original operation basis transfers24/32,20/22,25/28,17/20; pooled transfers28/32,22/22,26/28,17/20. These correlated context-level counts are not independent statistical trials. Good aggregate approximation does not imply a reliable binary operation switch in every context.

All list results remain diagnostic. Original operation errors are29.3%/20.7% for A/B with old names and31.1%/27.1% with new names. Pooled errors are9.8%/23.9% and9.9%/18.3%. Strong approximation of a failing model's intervention cannot establish successful binding generalization.

Interpretation: the original operation-derived subspace has reproducible causal specificity over identity/wording/random controls across additional narrative templates. Adding wording diversity at fixed rank is not uniformly better. This supports a reusable but context-sensitive operation component, not a universal rank-four manifold, task classifier, or improved whole-model accuracy. Gates, query/key bases, and operation pairs remain supervised; unsupervised recovery is still outstanding.

Audits:8,160 model forwards; all manifest/basis/source hashes unchanged; native head factorization relative error5.41e-7; native-value identity error7.15e-6; zero contrast exactly zero; finite effects. No original reserved-test evaluations; natural document pool remains26,112.

Next: distinguish threshold-sensitive sign misses from substantial missing operations, then trace the successful compact component upstream. Keep whole-edge and task-competence failures visible. Expand control tasks and test necessity/selectivity, rather than optimizing one template's reconstruction.

### Hypothesis refresh: twenty alternatives and discriminating tests

1. A shared operation component is expressed across narratives. Supported by fresh downstream transfer; test selective removal across held-out lexical families.
2. Several template-conditioned components coexist. Supported by pooling tradeoffs; compare conditional subspaces with a matched total-rank global basis on fresh families.
3. Rank four is an arbitrary bottleneck. Compare a predeclared rank curve with matched nuisance controls, without choosing rank on test effects.
4. The strongest singular directions favor high-energy templates rather than causal utility. Compare train-only balanced or effect-weighted fits on held-out effects.
5. A small query component controls task mode while keys carry identity. Mixed support; do upstream separate-query/key mediation, not just output-edge substitution.
6. Query and key jointly encode the operation. Supported by joint versus one-sided swaps; test interaction-preserving versus interaction-destroying controls.
7. Template-specific key directions account for remaining errors. Supported for new list wording; selectively restore omitted key directions on competent fresh tasks.
8. Template-specific query directions account for remaining errors. Supported for new narrative wording; run the analogous query intervention.
9. RoPE or relative positions explain apparent sharing. Exact de-rotation and matched positions weaken this; vary positions independently with semantics fixed.
10. Token identity alone explains the subspace. Nuisance controls weaken this; extend to unseen single-token identities and identity-balanced contrasts.
11. Lexical cues correlated with operation explain it. Still live; construct minimal lexical contrasts and identical final-token controls with different required operations.
12. Global norm changes drive the effect. H745 weakens this; compare fixed-norm and native-norm paths with explicit gain controls.
13. The first attention factor determines operation sign. H741 weakens this within studied cases; test additional heads/tasks where its sign varies.
14. The second factor is a context-sensitive routing gate over a fixed token map. Strong within this edge; test other source positions and values without fitting signs.
15. The absent-name probe measures a special output association. Still live; use held-out absent values from several semantic classes and measure full-vocabulary changes.
16. Downstream nonlinearities create the apparent operation switch. Existing coefficient/effect agreement weakens this locally; map intervention amplitude and compare layerwise effects.
17. Sign misses are mostly near-zero margins. Plausible but untested here; stratify errors by full-target effect size before interpreting counts.
18. Other heads compensate when this head's operation label fails. Test selective head/component ablation and residual pathways on native-competent exceptions.
19. Failed list tasks reflect ambiguous or poor prompts rather than absent computation. Retain failures; use independently specified, competence-tested task families without retroactive template repair.
20. An unsupervised featurizer can recover these causal components without operation labels. Outstanding central question; compare sparse/overlapping/hierarchical representations against the now-verified paired-edit reference on held-out interventions.

Priority: (1),(2),(6),(14) best describe current evidence. (11),(15),(17),(18) are important unresolved alternatives. The next useful steps are effect-size-stratified sign diagnostics and selective upstream intervention, followed by unsupervised recovery comparisons. Improved compression alone would not distinguish these explanations.
