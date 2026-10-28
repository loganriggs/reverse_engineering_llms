## H872 — Coefficient sign controls the L14H4 reversal; downstream context adds gain and residual variation

H871 showed a fixed country-bus contrast multiplied by a context-dependent signed coefficient. H872 interchanges those coefficients while holding recipient context, reader and source direction fixed. TheL14H4 copying/capital reversal follows coefficient sign. Equal-sized writes nevertheless have context-dependent downstream gain; late readers are more consistent across these contexts thanL8H3. This supports a reusable content-map-plus-routing description without claiming a context-independent whole-model computation.

### Intervention definition

All144 H870 endpoints and all3 frozen readers retained. Every arm adds gamma times the same donor–recipient country-bus contrast to the final-query head output. Arms use nativegamma, the other task's gamma for the same countrypair/format/endpoint, foreignsign withnative magnitude, foreignmagnitude withnative sign, and native signflip. Identity controls remain. Failedbinding rows are retained as diagnostics and use copying as their foreign-context source; conclusions below concern competent copying/capital.

These are counterfactual head-write perturbations. They interchange coefficients of the midpoint-contrast perturbation; they do not replace the entire native attention pattern or guarantee naturally realizable query/key states. The native head activation and all other model state remain in place before the added perturbation. No scale is fitted for these causal arms.

### Sign and magnitude controls

Entries are native paired answer-gap fractions removed. Negative values mean the intervention increases the intended distinction. Values above100% mean reversal of the aggregate distinction, not a probability or more-than-total mediation.

|Recipient task|Reader|Native coefficient|Foreign coefficient|Foreign sign only|Foreign magnitude only|Native sign flipped|
|---|---|---:|---:|---:|---:|---:|
|copy|L13H0|54.1%|119.6%|54.1%|119.6%|-49.4%|
|copy|L14H4|-46.5%|30.2%|31.0%|-34.8%|47.7%|
|copy|L8H3|26.7%|4.3%|26.7%|4.3%|-20.2%|
|capital|L13H0|49.8%|23.0%|49.8%|23.0%|-43.7%|
|capital|L14H4|8.7%|-11.9%|-9.6%|8.8%|-8.3%|
|capital|L8H3|1.5%|10.0%|1.5%|10.0%|-1.5%|

AtL14H4 in copying contexts, the native-coefficient perturbation increases the copying distinction46.5%; using capital coefficients instead removes30.2%. Foreignsign alone removes31.0%, while foreignmagnitude withnative sign still increases the distinction34.8%. In capital contexts, the native perturbation removes8.7%, but copying coefficients increase the distinction11.9%; foreignsign alone gives the same direction of reversal. Thus sign, not amplitude alone, is a decisive cause under these fixed-recipient interventions.

L13H0 andL8H3 have matching native coefficient signs across copying/capital, so their foreignsign arms reproduce native effects. Their foreign-coefficient changes follow magnitude: for exampleL13H0 capital coefficients applied to copying remove119.6% of its aggregate gap, whileL8H3's smaller capital coefficients reduce its copying effect from26.7% to4.3%. The former is a strong counterfactual perturbation, not evidence that natural copying routes all its computation through that reader.

### Same-write response across contexts

For each pair/format/endpoint/reader, define a shared amplitude sqrt(abs(gamma_copy × gamma_capital)). Apply both positive and negative versions in both recipient contexts. The source direction and amplitude then match across tasks to floating-point precision. Measure symmetric finite-amplitude responses for the same output contrast in both contexts: countryidentity or capitalassociation. These are not infinitesimal derivatives.

Raw correlations across both endpoint orientations can be inflated by a shared sign flip. Therefore orient every response to the same countryA→B contrast and average formats/orientations within each of12 lexicalpairs before correlating. Fit a single through-origin gain on8 discoverypairs and evaluate it on4 transferpairs. No gain was used in the causal interchange arms above.

|Reader|Output contrast|Canonical pair correlation|Discovery gain, capital context / copy context|Transfer relativeRMSE, gain1|Transfer relativeRMSE, frozen gain|
|---|---|---:|---:|---:|---:|
|L13H0|country|0.971|0.837|0.246|0.075|
|L13H0|capital|0.973|0.897|0.216|0.114|
|L14H4|country|0.780|0.852|0.226|0.085|
|L14H4|capital|0.880|0.797|0.366|0.126|
|L8H3|country|0.696|0.963|0.282|0.250|
|L8H3|capital|0.776|0.930|0.323|0.308|

ForL13H0/L14H4, one discovery-fit gain reduces transfer error to about7.5–12.6%, versus21.6–36.6% without gain adjustment. L8H3 retains25–31% error after gain adjustment. This is consistent with more downstream context sensitivity for the earlier reader, but depth is not experimentally isolated here: reader identity and computation differ too. Twelve lexicalpairs and four transferpairs are insufficient to establish a universal law.

### Updated interpretation and next step

The supported candidate computational unit is a static content direction, selected/scaled with a signed contextual coefficient, followed by a reader write and context-dependent downstream response. The observed sign reversal is causal under the coefficient controls. The downstream response is reusable approximately, not exactly, and cannot be replaced by an unqualified fixed linear map.

This matters for featurizers: a stable dictionary direction alone omits the computation that selects, signs and scales it. Conversely, representing the operation does not require treating the full activation cloud as an intrinsically high-dimensional semantic manifold. It remains unproven that a learned shared dictionary/group/DAG model can discover this relation task-blind, or that a common fit transfers beyond these country examples.

Freeze the content/routing definition and discovery gains. Next evaluate new country-pair combinations and continuation variants without refitting, with a competence/alignment screen reported independently of intervention results. Compare raw native-gamma prediction, frozen-gain response prediction and matched-operation controls. Failures should distinguish lexical-map dependence, routing changes and downstream context dependence. Do not train or name another semantic manifold solely because this adaptive panel fits a compact description. The broader diverse-task and unsupervised-recovery requirements remain open.

### Verification and scope

3,456 full-model forwards. Native logit replay and priorH871 native-edit margins match exactly. Matched copy/capital source contrasts differ by at most1.91e-05 raw units; common-amplitude edit norms differ by at most6.4e-06. Routing replay error2.38e-07; one reader edit perforward. Sources unchanged, outputs finite, no reserved-test use;26,112-document pool unchanged. Pointwise lexical-cluster intervals are saved; panel/splits have been adaptively reused, so transfer is withheld from gainfitting but not a fresh global research holdout. Binding remains a failedfullvocab diagnostic. Broad goal remains active.
