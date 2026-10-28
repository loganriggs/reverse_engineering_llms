## H659 — Predicting boundary and attention changes improves finite fidelity

H658 established exact recovery when the incoming block9 state and all later attention writes are replaced by task-prompt values. H659 asks whether discovery data can predict those values without held-out donor access.

For each family and form, fit the mean original-wording copy→task change using four discovery items: one1152-dimensional block9 residual change and eight1152-dimensional attention-output changes for layers10–17. Evaluate on four independent transfer items per family/form under the second task wording. All six families and96 held-out copy prompts remain. No output-based fitting, scaling search or task filtering occurs. The24 groups store248,832 scalar mean entries; this is not yet a low-dimensional or model-compressing representation.

Predicted boundary state equals the receiver's native block9 state plus the discovery mean. Predicted attention writes equal cached native receiver attention outputs plus the corresponding discovery mean. Those predicted writes replace the evolving attention outputs; MLPs run naturally. This baseline-dependent prediction needs a native receiver pass and the original network, but no fresh donor states. Exact component controls use donor states. Native attention in the factorial is recomputed from the evolving state. A separate live-additive variant adds attention means to those evolving attention outputs instead of the cached native baseline.

Run the full native/predicted/exact3×3 component factorial, plus an other-task control using pooled means from the other five families and the live-additive attention variant. All methods operate at fixed full strength. Source manifests are checked for matching items, targets and splits; original discovery and new transfer captures stay separate.

**Finite output KL(task||edited), nats:**

| Family | Native | Predicted I only | Predicted A only | Both predicted | Other-task pair | Predicted I + live-additive A |
|---|---:|---:|---:|---:|---:|---:|
|addition|0.1415|0.0731|0.0784|0.0406|0.0992|0.0425|
|agreement|0.1837|0.0880|0.0768|0.0585|0.1606|0.0587|
|antonym|0.1381|0.1161|0.0980|0.1086|0.0860|0.1102|
|capital|0.2041|0.1241|0.1135|0.0605|0.1528|0.0495|
|plural|0.1685|0.1508|0.0829|0.0674|0.1333|0.0788|
|successor|0.5062|0.2892|0.1663|0.0969|0.3557|0.1163|

Both predicted components improve mean KL over native copying in every family and outperform the other-task pair in five of six. Antonyms are the exception: same-task pair KL0.1086 is worse than the other-task pair0.0860, and worse than same-task attention-only0.0980. Thus the components are not uniformly complementary. Live-additive attention helps capitals but worsens most other cells; feedback through the evolving state cannot simply be presumed beneficial. Neither deployment convention is selected after the fact as a universal winner.

**Which prediction limits fidelity?** Keep one component predicted and make the other exact:

| Family | Both predicted | Predicted I + exact A | Exact I + predicted A | Both exact |
|---|---:|---:|---:|---:|
|addition|0.0406|0.0119|0.0165|0.0000|
|agreement|0.0585|0.0121|0.0318|0.0000|
|antonym|0.1086|0.0345|0.0491|0.0000|
|capital|0.0605|0.0135|0.0266|0.0000|
|plural|0.0674|0.0223|0.0231|0.0000|
|successor|0.0969|0.0443|0.0861|0.0000|

Exact attention produces lower KL than exact boundary state in all six families, although plurals are almost tied. This makes attention prediction the larger conditional limitation in this comparison. It does not assign independent causal shares or imply the boundary is unimportant: both errors interact, and full exact recovery is an architectural reference.

**Task loss and correctness:**

| Family | Native target CE | Both-predicted target CE | Other-task target CE |
|---|---:|---:|---:|
|addition|3.936|3.747|3.803|
|agreement|9.107|8.110|9.200|
|antonym|5.387|5.304|5.378|
|capital|6.132|5.569|6.010|
|plural|5.963|5.695|5.426|
|successor|3.911|2.777|3.681|

Both-predicted edits decrease target CE in every family, but correctness remains unchanged. Exact next-token accuracy is0% for every condition. Candidate accuracy remains addition0%, agreement50%, antonyms100%, capitals100%, plurals100%, successor0%. These are the held-out subset's rates. For plurals the other-task control has better target CE than the same-task pair despite worse output KL, again separating loss optimization from distributional fidelity. Preserving a natural task distribution that is itself often wrong is not task execution.

The supported result is a conditional finite surrogate: original-wording mean changes in a boundary state and attention writes predict part of the second-wording task distribution on new items while MLPs compute normally. It retains the full model, a native receiver baseline, known family/form, and large mean arrays. It is not a unique canonical form, a fundamental semantic unit, or an end-to-end compact model. Antonym failures limit broad task-specific interpretations.

Next distinguish wording mismatch from item-specific attention variation. Fit corresponding means using only the four second-wording discovery items, then cross original/new-wording boundary and attention means on the same held-out items, alongside the exact controls. This will show whether attention's remaining gap is mostly removable by matching the instruction wording or remains even within wording. No held-out fitting or rank/regularization search is needed. After that, target the residual limitation rather than adding predictor complexity indiscriminately. A future genuinely new item/wording panel remains necessary for prospective validation of this representation.

Audits:1056 whole-model forwards for11 conditions x96 prompts;2304 cached attention-stage evaluations to reconstruct original discovery and native receiver writes. Exact/native factorial conditions reproduce all four matching H658 logit arrays identically. Source/runner/locked-mean hashes unchanged; metrics finite. No MLP edits; all families and items retained. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
