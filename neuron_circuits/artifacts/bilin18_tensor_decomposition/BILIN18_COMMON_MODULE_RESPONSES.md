## H829 — First-block MLP response variation drives local divergence, while later context dependence remains

Execute a factorial response-replacement experiment on all13 H827/H828 groups, both block and random directions, and all52 recipients. The preceding turn traced context divergence to the first downstream block but could not separate contributions causally.

### Controlled response experiment

Keep each recipient's native attention output, native MLP output and skip path. At the first downstream block's final token, replace BOTH module outputs by recipient native output plus signed amplitude times a stored H828 central response derivative. Independently choose each derivative to be the recipient's own or the common mean across its four-recipient group. Four arms are own/own, common attention, common MLP and common both. Input edit remains the identical H827 vector, with amplitudes +/-0.1. Zero-amplitude replacements verify baseline replay.

This controls module response contributions. It is NOT natural mediation: when attention's response changes, the replaced MLP response does not adapt to that change. Both are explicitly clamped to selected derivatives. The group-mean derivative can change response norm as well as direction. Conclusions therefore concern these controlled response contributions, not a general claim that attention is unimportant or a percentage of naturally mediated context effects.

### MLP response replacement strongly increases local alignment

Median cross-recipient cosine of the completed first downstream block response:

| Injection site | Direction | Own responses | Common attention | Common MLP | Common both |
|---|---|---:|---:|---:|---:|
| after block0 | block |.475|.490|.952|1.000|
| after block0 | random |.467|.485|.987|1.000|
| after block1 | block |.591|.637|.944|1.000|
| after block1 | random |.602|.625|.973|1.000|

Holding the MLP contribution fixed increases group-median alignment in all13 groups for both direction types. Common attention improves11/13 groups for block directions and12/13 for random directions, but with much smaller aggregate changes. This is stronger evidence than the H828 trace alone that variation in the MLP response contributes substantially to the first block's outgoing directional divergence. It does not establish whether that variation comes from the native MLP baseline, its incoming perturbation, RMS normalization, or their interaction.

Common-both alignment is imposed algebraically: the identical skip perturbation plus two identical module response vectors produces the same outgoing derivative. It is an implementation/control check, not a discovered invariant mechanism. Remaining finite-difference differences are at numerical precision.

### Equalizing the first block does not equalize final behavior

Final response cosines under the common probability-weighted metric:

| Injection site | Direction | Own responses | Common attention | Common MLP | Common both |
|---|---|---:|---:|---:|---:|
| after block0 | block |.265|.270|.186|.178|
| after block0 | random |.039|.029|-.037|-.057|
| after block1 | block |.058|.094|.097|.123|
| after block1 | random |.093|.115|.084|.146|

Uniformly centered final-logit cosines also remain low in common-both: .134/.026 for block/random after block0, and .061/.036 after block1. Thus later recipient-dependent computation transforms even an equalized first-block output response differently. Preserving native baselines means the full outgoing states are still different, as intended. This is evidence against explaining the entire effect with one early bottleneck or one context-free map from that outgoing perturbation to logits.

Final weighted alignment need not improve monotonically when an early contribution is equalized. Output sensitivity, later interactions and softcap filtering all remain. H828's softcap exception is a reason to retain uniform and common-weighted metrics rather than selecting the more favorable one.

### Faithfulness and correctness

884 whole-model forwards:832 signed nonzero replacements and52 zero-amplitude controls. Zero replacements reproduce native logits exactly. All input hashes unchanged, all metrics finite, no reserved-test evaluations or new documents; natural-document pool26,112.

The own/own response replacement closely reproduces natural H827 central derivatives. Target-residual relative error median2.94e-6, maximum1.11e-5; minimum cosine .99999999994. Final uniform-logit relative error median.000793, maximum.00355; common-metric error median.000739, maximum.00362. Minimum final-response cosine exceeds .9999938. This supports using the controlled local response approximation for the large alignment changes reported here, while preserving its finite-dose limitations. Full patched outputs and target residuals are saved.

### Updated mechanistic direction

The immediate computational hypothesis is a state-conditioned MLP operator: a feature direction's effect depends on the native state that multiplies it and on normalization. But the current clamp cannot distinguish that from already context-dependent incoming attention perturbations. The next high-information test crosses the raw MLP baseline and incoming perturbation separately in an exact RMS-plus-bilinear evaluation, retaining all13 groups and random directions. Compare own baseline/own perturbation, common baseline/own perturbation, own baseline/common perturbation, and common/common. Verify the analytic or high-precision response against captured central derivatives before attributing effects. After identifying which factor matters, test selected response crossings downstream with the skip path retained. Continue treating feature vectors, state-conditioned operators and task-specific subspaces as competing descriptions; no semantic ontology or hierarchy has yet been validated.
