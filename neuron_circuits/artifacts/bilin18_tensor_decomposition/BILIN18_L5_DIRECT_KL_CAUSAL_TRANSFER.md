## H775 — Small natural-KL gains do not establish better causal preservation

Previous goal turn was progress: H774 found a small, uncertain natural-text KL gain with broader training coverage and retained all fits. H775 tests whether that change preserves actual computation under interventions.

All three larger-pool direct-KL fits, all three matched reconstruction fits, original rank-eight PCA and the full head edit are evaluated frozen at L5H5. The diverse suite covers agreement, capital recall, entity binding, sequence copying and sentiment in two formats, using final-query edits. The separate copying panel covers semicolon, newline, colon and equals formats with query-only, source-only and joint edits. Both panels include reciprocal answer-changing edits, nuisance edits and identity controls. Every compressed edit adds F(donor)-F(receiver) to the native activation; downstream computation remains live.

For each intervention, the full head edit supplies the reference answer-margin change and output distribution. We measure margin approximation error and KL(full edit||compressed edit). The latter is distinct from native-to-edit KL: faithfully preserving an important edit can require changing the native output substantially. Relative margin errors are reported only when native task competence passes and full-effect RMS>=.25; absolute errors remain available everywhere.

| Copying query task | Full margin RMS | PCA absolute RMS error | Reconstruction-fit RMS error | Direct-KL-fit RMS error |
|---|---:|---:|---:|---:|
| colon | 0.2528 | 0.2348 | 0.2355 | 0.2333 |
| equals | 0.2539 | 0.2339 | 0.2338 | 0.2353 |
| newline | 0.2616 | 0.2137 | 0.2145 | 0.2118 |
| semicolon | 0.1688 | 0.1753 | 0.1756 | 0.1799 |

Fit errors pool squared errors across all three schedules; they are not errors of an ensemble intervention. Direct-KL relative errors for the eligible colon, equals and newline task edits remain92.3%,92.7% and81.0%, compared with PCA92.9%,92.1% and81.7%. Changes are small and mixed. Semicolon's full effect is below the relative-error threshold; its absolute error worsens from.1753 to.1799. The small natural-replacement KL improvement has not solved causal preservation at rank eight.

Nuisance effects also do not consistently improve. For newline query nuisance edits, relative error changes from64.7% for PCA to67.1% for direct KL. On the older diverse-suite copying format A, task relative error improves slightly from63.1% to61.9%, but nuisance error stays around102%. Output-distribution fidelity does not uniformly track the margin change: in that diverse copying task, full-edit-to-compressed KL increases slightly despite the small margin improvement.

The diverse suite was genuinely evaluated across all five families, but its diagnostic power at this particular head/site is limited. Competent agreement, binding and capital-format-B full task effects have RMS only.0035–.0100, so their small absolute errors cannot establish broadly preserved computation. Capital-format-A and both sentiment formats retain their existing native competence failures. These are reported as diagnostics, not silently removed or promoted into successful generalization. Competence-gated relative results are consequently concentrated in copying.

Paired item bootstraps retain reciprocal directions and all schedules within each item. Most direct-KL versus PCA comparisons include zero. A few comparisons favor or disfavor a method under descriptive intervals, including worse semicolon query task error, but the panels are small, reused and multiply compared. No seed or task-format winner is selected. All source and joint-position results, including small-effect cells, are saved.

Audits:5,536 model forwards; full-reference margins exactly replay earlier runs; initial-PCA margin replay<=7.16e-6; identity-logit replay<=2.20e-5; full-donor vector error<=1.77e-5. All metrics finite and hashes unchanged. No refitting or original reserved-test use; natural document pool remains26,112.

Interpretation: these rank-eight natural-data fits are not verified featurizers for manifold interpretation. Neither reconstruction training nor direct-KL training demonstrated a substantive causal gain here. The result leaves two different possibilities: eight shared directions may be insufficient across these computations, or the natural-data objectives may fail to find the relevant directions. Existing higher-capacity curves show that more coordinates can help but do not resolve whether a suitably learned low-dimensional task representation exists.

Next: fit an explicitly task-informed rank-eight contrast subspace on a predetermined subset of copying activation differences, then test held-out items/formats, nuisance controls and actual causal effects against the same references. Use no output outcomes to select directions. Include random/ordinary PCA controls and report training versus held-out vector coverage. This is a diagnostic of representational capacity and transfer, not a mathematical causal upper bound or an unsupervised discovery. If it transfers, the learning objective/data are implicated; if it fails, do not conclude impossibility from one fitted subspace. Only after a reliable representation is established should its geometry be interpreted as a computational manifold.
