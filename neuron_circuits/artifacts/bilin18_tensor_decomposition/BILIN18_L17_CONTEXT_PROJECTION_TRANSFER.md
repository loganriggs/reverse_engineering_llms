## H444 — Unused-context transfer and component accounting

H443 suggested context-sensitive coefficients help learned and random correction spans, but comparisons were uncertain on a selected32-context panel. H444 randomly selects128 receiving documents from H438’s remaining480, excluding every earlier intervention-evaluated site. Their baseline states were previously collected for selection, but their intervention results were unused. This is held-out receiving-context transfer within an existing corpus panel, not a new set of corpus documents. Total unique documents remains12480.

Freeze the H442 learned span, the baseline-context full-Fisher projection rule and ridge1e-6. Use12 fresh natural donor draws,12 fresh Gaussian draws and four separately seeded random rank64 output spans. Retained union128 and exact radial terms are held fixed; explicit no-radial and retained-only conditions provide component accounting. Coefficients project analytic linear-map responses using baseline geometry, never full finite target errors. Sources are new draws conditional on the same donor calibration pool.

| Frozen contrast | Estimate | Joint95% interval | Lower > .02 |
|---|---:|---|---|
|learned_context_minus_fixed|+0.017908|[-0.012818, +0.048633]|False|
|learned_context_minus_random_mean|+0.022622|[-0.008104, +0.053347]|False|
|random_mean_minus_union_radial|+0.081979|[+0.051253, +0.112705]|True|

The mean random contextual correction adds8.20 percentage points of full finite-KL recovery beyond union+radial, joint95% interval[5.13,11.27] points, passing the >2-point gate. Learned-context improvement over learned-fixed and learned-over-random-context remain unresolved; both intervals include zero. No failed gate is relabeled. Bootstrap uses4000 paired source/receiving-document draws and resamples the four random spans. Four seeds do not exhaust the random-span population; inference remains conditional on the fixed calibration/model setup.

| Predictor | Full finite-KL recovery |
|---|---:|
|union_only|84.017%|
|union_radial|88.033%|
|learned_fixed|96.702%|
|learned_context|98.493%|
|random_context_0|96.210%|
|random_context_1|96.342%|
|random_context_2|96.192%|
|random_context_3|96.180%|
|learned_context_no_radial|83.345%|
|random_context_no_radial_0|83.338%|
|random_context_no_radial_1|83.343%|
|random_context_no_radial_2|83.532%|
|random_context_no_radial_3|83.449%|
|exact_compiled|99.170%|

The four contextual random-span scores range from96.18% to96.34% (mean96.23%), versus98.49% learned-context and99.17% exact shared16 compiled correction. Removing the radial term leaves83.35% learned-context and83.34–83.53% random-context, near or below the84.02% retained-only score. Union+radial alone reaches88.03%. Thus the radial term is necessary to preserve the tested combined predictor’s quality, but it does not account for the whole gain: random contextual correction contributes significantly on top of it. This is a controlled component result, not proof of independent semantic modules or a unique mediation fraction.

The random span is only the linear correction’s output span. The predictor still contains learned native retained factors, an exact baseline-derived radial write, dense analytic source-response information, baseline token-probability geometry and a per-context solve. It would be wrong to say64 random features explain the model. What transfers is a structured input-derived prediction procedure whose final correction can use a random output span fairly effectively in this scope.

Numerical audit: maximum relative solve residual5.65e-15, below1e-8; all arrays are finite; all128 selected receiving documents are distinct and disjoint from the32 excluded intervention cases. Frozen artifact hashes are checked. No exclusions are made after outcomes, no new corpus documents are counted, and no prior negative statistical results are changed.

Next identify which output constraints the contextual solve needs. Compare a fixed budget of baseline-most-likely token constraints with full-vocabulary Fisher scoring, plus probability-mass/identity controls, while retaining finite-KL evaluation and the component accounting above. Constraint selection must use baseline probabilities only. This extends earlier oracle-geometry findings to this input-derived finite predictor. If a small constraint set transfers, it may reduce the solve’s additional computation; the dense analytic linear-map construction remains a separate cost and must not be hidden. The broader goal still requires a compact, transferable and interpretable account beyond this layer/head and synthetic intervention family.
