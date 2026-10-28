## H630 — Four replacements are needed for unanimous sign change here; additive effects already predict it

The previous turn was progress: H629 localized a combined sign flip to normalized-MLP operators6–9 but could not distinguish accumulation from interaction. H630 tests every subset of those four replacements on the fixed72-prompt panel, retaining native attention and later operators. It also factors each early MLP operator analytically into the RMS derivative and the bilinear derivative, comparing shared bilinear operating context, shared RMS operator, product of mean factors, and mean of paired factors. The incoming tangent is propagated at every stage; no direction is refitted.

**Only the full four-stage subset makes all four held-out “speak” derivatives positive.** Smaller subsets flip at most three. The pair{7,9} has a positive mean but only two positive recipients, so a group mean alone would overstate its sufficiency. This is a finite-panel minimality statement for these particular reference substitutions, not proof of a minimal natural circuit or four independently necessary neurons/layers.

**The sign change does not require interaction terms.** Predict each subset by adding its single-replacement changes to the native derivative. For all four replacements, that additive prediction is positive on every held-out recipient. The predicted mean is+0.0605, while the actual mean is+0.1308. Interactions amplify the mean effect by+0.0702 but are not necessary to predict the sign in these cases. Thus “no single replacement flips it” should not be mistaken for evidence that synergy causes the sign change.

| Replaced MLPs | Actual mean derivative | Sum-of-singles prediction | Through-pairwise prediction | Positive recipients /4 |
|---|---:|---:|---:|---:|
|none|-0.23526|-0.23526|-0.23526|0/4|
|6|-0.18474|-0.18474|-0.18474|0/4|
|7|-0.11302|-0.11302|-0.11302|0/4|
|6,7|-0.02497|-0.06250|-0.02497|2/4|
|8|-0.23810|-0.23810|-0.23810|0/4|
|6,8|-0.16343|-0.18758|-0.16343|0/4|
|7,8|-0.12837|-0.11586|-0.12837|1/4|
|6,7,8|-0.01526|-0.06534|-0.01617|1/4|
|9|-0.10941|-0.10941|-0.10941|0/4|
|6,9|-0.05573|-0.05889|-0.05573|1/4|
|7,9|0.01064|0.01283|0.01064|2/4|
|6,7,9|0.09384|0.06335|0.10185|3/4|
|8,9|-0.04195|-0.11225|-0.04195|2/4|
|6,8,9|0.02280|-0.06173|0.03588|2/4|
|7,8,9|0.01656|0.00999|0.06558|3/4|
|6,7,8,9|0.13075|0.06051|0.18095|4/4|

Pairwise terms alone predict+0.1810 for the four-stage combination; third/fourth-order terms offset it by−0.0502, giving+0.1308. The stored Möbius expansion is exact on the sixteen evaluated subsets. These interactions arise from composing changed linearized operators. They do not require finite-amplitude nonlinearity in the original injected perturbation, and they are not additive causal shares of the unmodified model. Baseline, reference contexts, recipient states and scalar readout all define the decomposition.

**Bilinear operating context carries the key sign change in this region.** Sharing the normalized operating context in all four MLP bilinear derivatives while retaining each native RMS Jacobian gives mean derivative+0.2716, positive for all four recipients. Sharing only RMS Jacobians while retaining native bilinear contexts gives−0.2209, negative for all four, close to native−0.2353. The full paired normalized-MLP substitution gives+0.1308. Thus normalization moderates the effect in combination, but the context-only substitution is sufficient for this local sign change and normalization-only is not.

The bilinear derivative for normalized context y and incoming normalized tangent t is Down[(Left(y)·Right(t))+(Left(t)·Right(y))]. Because it is linear in y, averaging donor bilinear Jacobians equals evaluating the derivative at the donor mean y. The mean RMS Jacobian acts on each current incoming tangent; it is not a fixed response vector. An analytic native reconstruction matches the automatic-differentiation chain within9.35e-7, and the mean paired-factor reconstruction matches the original all-MLP substitution within7.99e-7.

| Held-out cell | Native derivative | Full normalized-MLP sharing | Bilinear context only | RMS Jacobian only | Product of mean factors |
|---|---:|---:|---:|---:|---:|
|capital/0|0.5448|0.5816|0.5835|0.5362|0.5814|
|capital/1|2.8760|3.2345|3.8293|2.5183|3.2322|
|capital/2|0.4922|0.4706|0.7102|0.1165|0.4696|
|language/0|0.1148|0.0250|-0.0102|0.1542|0.0244|
|language/1|-0.2353|0.1308|0.2716|-0.2209|0.1306|
|language/2|0.0904|0.0047|-0.0060|0.0993|0.0052|
|plural/0|0.2220|-0.0328|0.0093|0.1911|-0.0366|
|plural/1|0.3563|-0.0141|-0.0581|0.3428|-0.0177|
|plural/2|0.1178|0.1992|0.3224|0.0153|0.2006|

The factor distinction is not uniform across all prompts. For capital questions, RMS-only sharing substantially changes the derivative; for language declaratives and questions, context-only sharing can push it slightly negative. We therefore should not generalize “normalization is negligible” beyond the specific “speak” sign contrast. The paired average and product of means also need not agree mathematically, though they are close on this panel.

Current local explanation: the frozen perturbation encounters context-conditioned bilinear transformations in multiple early MLPs. Replacing those operating contexts with capital-reference contexts changes the accumulated response enough to reverse the intended-language derivative. Single effects explain the sign of the all-stage result on these four cases, while cross-stage composition changes its magnitude. This is a substantially more precise claim than a universal answer-emission feature or an irreducibly synergistic circuit.

The remaining major risk is adaptation to a small, repeatedly examined panel. No new country or scaffold was introduced in this turn. Subset minimality is based on these four held-out-from-fit items, already used in earlier research. A positive derivative is not proof that a finite intervention improves task accuracy, and plural results remain measurements on competence-failing prompts.

Next freeze the original vector, four capital reference contexts and the context-only/RMS-only replacement rules for a prospective panel of countries absent from the current eight-country set, crossed with familiar “People in … speak” and new resident/country-language wording. Retain every token-valid prompt regardless of competence, include an unchanged original-panel anchor, and report individual signs and candidate correctness. Test the precise local mechanism before doing further subset selection or fitting. After this independent-panel check, probe additional input directions to determine whether the operator contrast is specific to the discovered vector or generalizes to a broader response subspace.

Audits:288 native attention-stage reconstructions from cached states,2,060 batched JVP calls, no new whole-model native forwards; all hashes unchanged; all16 subsets present for every recipient; native factorization discrepancy2.89e-6; analytic native9.34e-7; paired factors7.98e-7; end-to-end finite-difference comparison max0.000563; finite responses. FineWeb remains26,112 natural-text documents. Broad research goal active and incomplete.
