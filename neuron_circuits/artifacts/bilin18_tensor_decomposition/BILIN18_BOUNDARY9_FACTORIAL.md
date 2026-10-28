## H638 — The block9 response and later operators must be matched; readout effects depend on the incoming computation

H637 was progress: cumulative MLP5–9 context replacement partly closed actual prompt-response gaps, leaving substantial unexplained differences. H638 localizes those differences with an exact local factorization rather than expanding layer replacements blindly.

Use the same frozen80-prompt panel, original/span2/orthogonal2 probes and two involutive donor maps: opposite wording with identical item, or same wording with item shifted by four. Factor the local full-logit derivative into three components: I, the residual tangent leaving block9; D, the product of downstream block10–17 native Jacobians; R, the native final RMS-normalization, unembedding and softcap derivative. Evaluate all eight receiver/donor choices, giving R_c D_b I_a. The injection itself remains the same direction. All tangent exchanges occur at the final position, consistent with causal propagation.

The donor map being an involution allows reciprocal mixed computations to be reused exactly. Incoming tangents are the centrally measured H635 values at±0.02; downstream maps are automatic derivatives at frozen native states. Mixed combinations need not be jointly realizable as prompts. These are local operator counterfactuals, not replacements of a complete finite forward computation.

**The all-donor endpoint validates the decomposition, rather than establishing a new semantic unit.** All-native and all-donor responses reproduce their original finite-difference responses within0.000983 and0.000928 relative error respectively. The largest endpoint error divided by the donor–receiver gap is0.001615. Recovery is expected when all components come from the same native donor; it is an implementation and numerical positive control, not evidence of compression.

**Across wording, the incoming block9 response and later operators both matter.** The table reports mean remaining gap relative to the native donor–receiver gap under receiver probabilities. Values below1 indicate movement toward the donor response; above1 indicate worsening. “I+D” uses donor incoming tangent and donor downstream transformations but retains the receiver readout.

| Family | I only | D only | R only | I+D | I+R | D+R | I+D+R |
|---|---:|---:|---:|---:|---:|---:|---:|
|copy|0.738|0.869|1.092|0.369|0.704|0.841|0.000|
|fact|0.648|0.783|0.984|0.317|0.637|0.781|0.000|
|plural|0.659|0.917|0.967|0.283|0.619|0.854|0.000|
|agreement|0.601|0.893|0.997|0.335|0.572|0.826|0.000|
|addition|0.564|0.897|0.983|0.275|0.510|0.854|0.000|

Changing I alone leaves mean ratios0.564–0.738. Changing D alone leaves0.783–0.917. Changing both leaves only0.275–0.369. Thus neither a universal downstream transformation acting on different incoming responses nor a fixed incoming response followed by context-dependent processing captures the full contrast. Both sources of variation contribute, and their compatibility matters. These are average results over three probes, eight items and both exchange directions per task; they are not independent statistical trials.

**Readout cannot be assigned a fixed importance independent of the internal response.** Swapping R alone leaves ratios0.967–1.092 across wording, offering little improvement and worsening copying. Yet with donor I and D, the receiver readout leaves a0.275–0.369 gap, and donor R removes it to numerical tolerance. Therefore “readout alone does little” must not be converted into “readout does not matter.” Its effect depends on the tangent produced by preceding computation. Nor does the final drop give a unique causal share: changing factors in another order gives different incremental reductions.

Within-wording item controls show another pattern:

| Family | I only | D only | R only | I+D | I+R | D+R |
|---|---:|---:|---:|---:|---:|---:|
|copy|0.738|0.697|1.060|0.390|0.748|0.631|
|fact|0.972|1.061|1.179|0.678|1.132|0.569|
|plural|0.884|0.835|1.191|0.703|0.954|0.762|
|agreement|0.706|0.804|1.071|0.519|0.650|0.825|
|addition|0.706|0.938|1.026|0.237|0.755|0.875|

For facts, donor incoming tangent alone barely helps (0.972), while donor downstream operators plus donor readout leave0.569. This differs from the cross-wording pattern and argues against interpreting the components solely by task or syntax labels. Item changes can alter how the later computation and readout use a perturbation. The within controls change lexical identity and are not perfect minimal pairs.

**Interaction is substantial and explicitly measurable.** The complete2×2×2 factorial permits exact vector inclusion–exclusion terms: single effects, I×D, I×R, D×R and I×D×R. Averaged across cross-wording cases, their interaction norms divided by original gap norm are0.616,0.299,0.226 and0.222 respectively. Summing just the three single effects leaves mean gap0.656 even though all components jointly recover the donor. Within-wording corresponding interaction norms are0.443,0.226,0.173 and0.103, and the additive remaining gap is0.496. Norms do not add and are not causal percentages. The interaction decomposition reconstructs the full vector within1.2e-16 absolute error. This is composition of context-dependent linear maps, not finite-dose nonlinearity.

The working account is increasingly a context-conditioned sequence of transformations over residual directions. What is available by block9 and how it is processed later cannot be treated as interchangeable independent semantic labels. But this factorial by itself is descriptive localization: donor Jacobians are large objects taken from the model, not an interpretable or compact recovered algorithm. Finding a reusable computational unit still requires reducing those operator differences to structured, predictive factors.

Next break D into interpretable alternatives: matched donor bilinear contexts in later MLPs versus normalized attention or full-MLP operator changes, keeping incoming tangent and readout choices explicit. The central comparison is whether bilinear-context replacement alone captures the benefit of full donor downstream operators, for both native and donor incoming tangents. Use full donor D as a validated ceiling and same-wording item donors as controls. This tests a mechanistic account of the large I×D interaction instead of merely restating that the model is context-dependent.

Competence limitations carry over: explicit copying succeeds; capital/agreement candidate scores perform well; plurals and addition fail their declared candidate tasks. None of the factorial response comparisons tests improved task accuracy. All cases remain included.

Audits:no new whole-model forwards;504 batched JVP calls using exact cached contexts; all source/runner hashes unchanged; finite metrics; endpoint controls and vector decomposition verified as above. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.
