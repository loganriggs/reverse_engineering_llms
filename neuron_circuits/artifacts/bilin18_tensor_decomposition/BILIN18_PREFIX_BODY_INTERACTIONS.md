## H610 — Removing early MLP interaction across the body barely changes the later target

H609 identified a direct MLP0 bilinear cross term but removed interactions only at the final token. This experiment addresses the omitted body positions. Use the same 128 boundary-matched prompts and exact body alignment. Capture MLP0 inputs/writes at every body token. Compute the full four-condition MLP0 interaction in float64 and decompose it into the direct bilinear cross term plus the input-nonadditivity remainder.

On the combined prefix+lowercase prompt, subtract each component at the first body token, final body token, other body tokens, or all body tokens. Prefix-only tokens are left untouched; they have no unprefixed counterpart. Include per-position norm-matched random subtractions and identity runs. There are 800 interventions: 32 recipients ×(3 components ×4 position sets ×2 directions + identity). All scopes were declared before results. Every intervention retains the intended copied name as full-vocabulary top1.

**The early interaction has little control of the later target under these removals, even when all body positions are included.** Removing the full MLP0 interaction across the body leaves mean MLP11 interaction norm at 0.993 of baseline in old wording and 0.971 in new wording. Removing the direct cross term leaves 0.996 and 0.976. Attention9's interaction likewise changes little. This is a negative result for the proposed early-MLP carrier, not a proof that MLP0 is irrelevant to the model or that no alternative intervention could have a larger effect.

| Wording | Component | Scope | Remaining MLP11 norm | Random control | Remaining Attention9 norm | Mean output KL |
|---|---|---|---:|---:|---:|---:|
|0|full|first|0.9927|0.9855|0.9702|0.0000547|
|0|full|final|1.0000|1.0000|1.0000|0.0000000|
|0|full|other|0.9999|0.9979|0.9990|0.0000015|
|0|full|all|0.9932|0.9835|0.9698|0.0000575|
|0|cross|first|0.9943|0.9874|0.9726|0.0000473|
|0|cross|final|1.0000|1.0000|1.0000|0.0000000|
|0|cross|other|1.0008|0.9979|0.9997|0.0000011|
|0|cross|all|0.9959|0.9853|0.9731|0.0000492|
|0|remainder|first|0.9959|0.9912|1.0005|0.0000272|
|0|remainder|final|1.0000|1.0000|1.0000|0.0000000|
|0|remainder|other|0.9991|0.9995|0.9993|0.0000001|
|0|remainder|all|0.9949|0.9907|0.9998|0.0000275|
|1|full|first|0.9726|1.0341|0.9932|0.0002706|
|1|full|final|0.9999|1.0000|0.9999|0.0000000|
|1|full|other|0.9966|0.9984|1.0044|0.0000041|
|1|full|all|0.9706|1.0328|1.0001|0.0003032|
|1|cross|first|0.9773|1.0221|0.9991|0.0001243|
|1|cross|final|0.9999|1.0000|0.9999|0.0000000|
|1|cross|other|0.9979|0.9986|1.0056|0.0000047|
|1|cross|all|0.9762|1.0209|1.0065|0.0001562|
|1|remainder|first|0.9930|1.0171|0.9978|0.0000353|
|1|remainder|final|1.0000|1.0000|1.0000|0.0000000|
|1|remainder|other|0.9987|0.9998|0.9988|0.0000006|
|1|remainder|all|0.9917|1.0171|0.9966|0.0000316|

The raw MLP0 interaction is large and concentrated at the first body token: mean full-interaction norms are about 20,791/27,180 for the two wordings, versus 9.72/19.69 at the final token. Those magnitudes do not translate into downstream importance. A separate application audit confirms that the all-body subtraction really reaches the post-MLP0 residual, within 2.92e-7 relative error on representative prompts, and leaves prefix positions unchanged.

The learned architecture supplies a concrete candidate explanation for this attenuation: the next block scales the incoming residual by 0.0126953125 while adding 8 times the original embedding residual. Thus the direct residual route from MLP0 is strongly reduced before the next attention/MLP computation. The persistent value pathway also remains live. These facts are consistent with the observed weak influence but do not quantify a causal share of the attenuation; downstream normalization, routing and regeneration remain involved.

Intervention ratios are conditional changes in one corner of the prefix×case factorial. They are not percentages of total model computation or additive mediation shares. Output KL is absolute KL from the unmodified combined prompt, not the normalized reference error used in earlier approximation experiments. The random controls use one seeded direction per relative body position; they test direction at a matched norm profile, not a full random-effect distribution.

Updated interpretation: an algebraically identifiable bilinear interaction is not automatically a behaviorally important computational unit. Here the local term is real and large, yet removing it across the body barely changes the downstream interaction or copying behavior. The next useful target is where interaction is reintroduced or amplified after the strong residual rescaling, rather than further decomposing this early term solely because it is mathematically clean.

Next sweep aligned-body interaction removals at later attention/MLP writes, especially after the first residual rescaling and around layers 2–5, while retaining later Attention9 as a readout. Compare those results with the final-position sweep to locate positional carriers. Then distinguish residual-stream transmission from fresh mixing through the persistent value pathway. This preserves the broader goal of finding causally useful units rather than equating large local products with important features.

Audits: 64 exact body alignments; 128 native prompts; 800 interventions; exact identity replay; float64 component identity <=7.94e-15; native float32 full-body interaction replay error <=7.50e-6; per-position random norm mismatch <=2.45e-4 absolute at write norms of tens of thousands; representative residual application error <=2.92e-7 and prefix change zero; frozen runner/manifest; all metrics finite and all patched prompts retain correct full-vocabulary top1. FineWeb remains 26,112 documents; broad goal active and incomplete.
