## H669 — Input-value writes suppress the current token, without a specific next-item readout

H668 found month-sensitive next/previous coefficients, but swapping that coefficient did not switch task outputs. H669 tests what the input-value write actually does downstream. For every144 held-out copy/next/previous prompt, add alpha times its native projected layer14 head4 input value to the final attention output, then run downstream computation normally. Fixed alphas are−0.1,−0.05,−0.025,0,+0.025,+0.05,+0.1. No strengths are chosen by task performance. A shared seed669 random direction scaled to each receiver value norm is tested at±0.05 as a limited control.

This changes the scalar coefficient on the value direction. It is not a full head ablation or a test of whether the head is necessary for the task. Values and vector norms depend on the receiver; cross-domain slope magnitudes should not be interpreted as normalized effect sizes.

At each amplitude, compute the central finite slope (logit(+alpha)−logit(−alpha))/(2alpha). Margins compare next or previous with the current input token and are invariant to a common logit offset. Values below use alpha0.05; they are slopes per unit coefficient, not the actual logit change from a0.05 edit.

| Domain | Instruction | Current-token slope | Next-token slope | Previous-token slope | Next−current | Previous−current | Next−previous |
|---|---|---:|---:|---:|---:|---:|---:|
|month|copy|-2.3874|0.5224|-0.4693|2.9097|1.9180|0.9917|
|month|next|-2.4087|0.5351|-0.4384|2.9439|1.9703|0.9735|
|month|previous|-2.3103|0.5838|-0.3977|2.8941|1.9126|0.9814|
|number|copy|-0.8158|0.3120|0.1636|1.1278|0.9794|0.1484|
|number|next|-0.8151|0.2949|0.1533|1.1100|0.9683|0.1416|
|number|previous|-0.7893|0.2836|0.1461|1.0729|0.9353|0.1376|

Increasing this write suppresses the current-token logit and raises next-versus-current, previous-versus-current and next-versus-previous margins in every tested prompt at alpha0.05 (96 month and48 number contexts). The broad sign pattern persists across copying, next and previous instructions. In particular, a previous instruction does not reverse the next-versus-previous preference. This is more consistent with a content-linked output bias than an instruction-following next/previous algorithm.

The fixed random direction has much smaller mean margin slopes (month next−current roughly−0.015 to0.013; number0.025–0.029), and mixed signs across cases. It is only one seeded direction, not a statistical null distribution or significance test.

**Amplitude check, pooled across instructions:**

| Domain | Half-width | Next−current slope | Previous−current slope | Next−previous slope |
|---|---:|---:|---:|---:|
|month|0.025|2.9161|1.9338|0.9823|
|month|0.05|2.9159|1.9337|0.9822|
|month|0.1|2.9150|1.9331|0.9819|
|number|0.025|1.1036|0.9610|0.1426|
|number|0.05|1.1035|0.9610|0.1425|
|number|0.1|1.1034|0.9609|0.1425|

The slopes vary little over this fixed range, supporting a stable local readout effect. This does not imply global linearity at larger changes or other contexts.

**The forward bias is not specifically a next-month output.** A post-hoc readout audit uses the already-saved logits to examine all twelve month tokens. Offsets are cyclic relative to the input month;11 means the previous month.

| Cyclic output offset | Copy slope | Next-instruction slope | Previous-instruction slope |
|---|---:|---:|---:|
|0|-2.3874|-2.4087|-2.3103|
|1|0.5224|0.5351|0.5838|
|2|0.6932|0.7203|0.7606|
|3|0.7707|0.8259|0.8543|
|4|0.8195|0.8784|0.9033|
|5|0.6654|0.7480|0.7556|
|6|0.6993|0.7721|0.7850|
|7|0.4929|0.5857|0.5993|
|8|0.5087|0.5882|0.5967|
|9|0.2547|0.3168|0.3421|
|10|0.0601|0.1252|0.1610|
|11|-0.4693|-0.4384|-0.3977|

Several non-next months receive larger average increases than the next month; offset+4 is largest in these pooled averages. This rules out labeling the observed value direction a selective next-month vector on this evidence. The table is descriptive, combines different held-out inputs, and does not establish an offset+4 mechanism. The most defensible account is suppression of the input identity with a broad, asymmetric bias among alternatives. The next-minus-previous difference is real but insufficient for a successor-unit interpretation.

This also clarifies earlier prediction gains. A learned coefficient can control the strength of such a content-dependent bias and improve fidelity to natural task outputs without implementing the requested relation. H666's prospective fidelity gain remains valid; its mechanistic interpretation is now narrower. Native task competence limitations remain unchanged, and no correctness claim is inferred from selected-logit slopes.

Next test whether the current-token suppression generalizes beyond ordered domains, using the existing noun, adjective and country panels with matched copy/task contexts. Measure the raw input-token logit separately from the requested answer token, preserve signed finite probes and controls, and include months as a reference. If suppression is shared, a candidate description is identity-dependent suppression gated by context, not a succession operator. Distinguish direct output-projection effects from additional downstream transformations only after establishing that broader pattern.

Audits:1296 whole-model forwards (144 held-out prompts×9 conditions). Native replay maximum absolute logit error2.03e-5. Source/factor/runner hashes unchanged; finite slopes. All fixed amplitudes and all contexts retained. Full-calendar audit reuses stored outputs, adds no forwards, and is explicitly post-hoc. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
