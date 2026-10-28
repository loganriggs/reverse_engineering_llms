## H778 — Value changes approximate copying edits better than routing changes alone

Previous goal turn was progress: H777 established that the frozen rank-eight contrast bases do not transfer faithfully across lexical changes. H778 moves from compressing whole head outputs to testing the actual attention factorization.

The inspected L5 attention is signed bilinear attention, not softmax attention. After headwise RMS normalization and RoPE, its routing matrix is A=(QK^T/128)⊙(Q2K2^T/128), with a causal mask. The head output is H=A V. V mixes the layer-local value projection with the persistent layer-zero value stream. At L5 the learned mixture coefficient is−.0563965, so this is a non-convex mixture; it must not be interpreted as a probability.

Capture the actual factors from the running model, then for receiver r and donor d decompose:

ΔH = (Ad−Ar)Vr + Ar(Vd−Vr) + (Ad−Ar)(Vd−Vr).

The three terms are routing-only, value-only and interaction, respectively. Their sum is algebraically the full head change. Intervene with each term, routing-plus-value, all terms together and the full native head difference, at query, source and both positions. These are receiver-centered interventions, not unique causal attribution fractions. Reciprocal answer edits are retained.

All256directed task edits and128identity edits are aligned. Thirty-two of128nuisance edits have different token lengths or source positions and are excluded from this factorwise analysis under the prespecified alignment requirement;96nuisance edits remain. Task comparisons therefore use the complete H777task panel, while nuisance aggregates differ from H777 and must not be compared as if they used identical rows.

| Query task cohort | Full effect RMS | Routing-only error | Value-only error | Routing+value error |
|---|---:|---:|---:|---:|
| Old keys / old values | .2373 | .2373 | .0583 | .0343 |
| New keys / old values | .2737 | .2545 | .0993 | .0381 |
| Old keys / new values | .3183 | .3229 | .0801 | .0269 |
| New keys / new values | .3023 | .3254 | .0861 | .0480 |

Errors are absolute RMS differences from the full edit's answer-margin effect. In the three novel cohorts, value-only relative errors are36.3%,25.2% and28.5%; routing-only errors are93.0%,101.4% and107.6%. Routing-plus-value errors are13.9%,8.4% and15.9%. Thus native receiver routing applied to changed values approximates the full task edit substantially better than changed routing applied to receiver values. The interaction still matters, and the result is not perfect causal preservation.

In vector space, value-only component RMS is96–98% of full task-difference RMS, routing25–34%, and interaction11–14%. These norm ratios do not sum to100% because vectors can align or cancel. Their corresponding margin effects also do not add exactly: the RMS discrepancy between full margin change and the sum of separately intervened component margin changes is.0088–.0207 across task cohorts. Applying all components together, however, reproduces the full edit numerically, as required.

This is evidence that the tested copying head edits are largely carried through changes in V, not merely changes in attention coefficients. It is compatible with a compact routing operation acting on variable content, but does not establish compact routing, high-dimensional semantic identity, or a reusable manifold. V already contains upstream contextual computation, and its changes can occur at unchanged tokens—including the query—as well as at copied-token positions. The value-only intervention retains all value dimensions, so its better fidelity is not a matched-rank comparison with the failed rank-eight bases.

Nuisance changes also involve both factors; they do not become a pure routing component. All aligned nuisance, source-only and joint-position results are saved, along with excluded rows. The component conclusions apply to this head, these copying tasks and the receiver-centered decomposition; no necessity or whole-model explanation is claimed.

Audits:9,024 model forwards; native logits and cached heads replay exactly; A V reconstructs native head outputs within3.55e-6; the component sum reconstructs full head differences within5.12e-6. Full-reference margins replay exactly; all-component versus full margin discrepancies<=8.59e-6; identity logits replay exactly. All metrics finite and source/model/input hashes unchanged.480aligned directed edits,32alignment exclusions. Original reserved test untouched; natural document pool remains26,112.

Next: localize the value changes before calling them transported word identity. Split the value-only edit by changed copied-token positions, final query position and remaining positions, and separately by layer-local versus persistent layer-zero value streams. Audit both partitions recombine exactly. Intervene with the pieces across the same lexical cohorts. This tests whether the dominant value term reflects direct access to copied content, contextual computation already present at the query, or a mixture—an important distinction that whole-head compression obscures.
