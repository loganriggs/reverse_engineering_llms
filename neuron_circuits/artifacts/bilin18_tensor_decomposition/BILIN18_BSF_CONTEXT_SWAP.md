## H686 — Same-token block swaps carry some donor-context information

On all16 H685 residual candidates, select up to four deterministic pairs per block with identical token and sampled position, different documents, and both blocks active. Test both directions. Fifteen blocks supply4pairs; residual0 block111 supplies2. The resulting124 directed edits use122 unique endpoints. Replace only the selected block's decoded contribution. Compare training-PCA16 and fixed random16 projections of the full donor/receiver residual difference, rescaled to the learned patch norm; also swap the full residual as a reference. All later computation runs normally.

Positive donor KL gain means the edited receiver distribution moves toward the donor distribution. It is not task correctness, and the donor has a different prefix. The full residual edit is not an upper bound because earlier context and value bus remain those of the receiver.

| Site | Edit | Mean donor KL gain | Median gain | Positive / total | Mean receiver KL |
|---|---|---:|---:|---:|---:|
|residual0|block|0.01632|0.00074|41/60|0.00187|
|residual0|pca|0.00017|0.00100|37/60|0.00266|
|residual0|random|-0.00219|-0.00002|29/60|0.00052|
|residual0|full|0.08731|0.00949|41/60|0.00555|
|residual1|block|0.18753|0.03021|47/64|0.01073|
|residual1|pca|0.13793|0.00873|38/64|0.00683|
|residual1|random|0.01257|0.00612|34/64|0.00174|
|residual1|full|0.56446|0.06591|46/64|0.18634|

Learned-block edits outperform the controls on site-level mean gain, but effects are heterogeneous and not uniformly better per block or example. Residual1 block91 (often byte fragments) has large effects and contributes substantially to its site mean; residual0 median block gain is actually below PCA's. Do not infer semantic superiority or significance from aggregate means. Bidirectional examples and repeated endpoints are dependent.

The geometry check in decoded residual space confirms substantial same-token directional variation: mean unit-decoded within-token fractions0.670 at residual0,0.779 at residual1, versus approximately8.4e-11 for value0. Thus H685's coordinate result is not solely a nonorthogonal code-basis artifact. This still does not prove a smooth manifold.

Audits:618 full-model forwards; prefix states replay cached activations with maximum relative error1.960e-6; all source/checkpoint hashes unchanged, finite effects, test split unused. An initial syntax error was corrected before model execution. Exploratory validation reuse and only2–4pairs per block limit confirmation. Subsequent user direction adds overlapping shared dictionaries, implication hierarchies and computational DAG alternatives, with synthetic correctness/recovery tests before LLM comparisons. Broad goal incomplete.
