## H608 — Token-boundary differences seed an early interaction; later attention writes carry stronger final-position influence

Trace all four H607 prefix/capitalization conditions through final-position attention writes, MLP writes, residuals and normalized MLP inputs at all 18 layers. Use the same 128 prompts, previously fully competent in all eight cells. The target is the native four-condition interaction at normalized MLP11 input. This experiment does not inject the number/task/random signals from H607; it studies the upstream interaction that those experiments used as context.

Then, on each combined prefix+lowercase prompt, subtract the native four-condition interaction vector at one final-position write, allowing downstream computation to run normally. Test every attention/MLP write in layers 0–10 plus Attention11: all 23 sites are predeclared. Compare a seeded norm-matched random subtraction per site and unchanged replay. There are 1,504 interventions: 32 identity runs plus 32 recipients ×23 sites ×2 subtraction modes. No site is selected before evaluation; reported rankings are exploratory on this panel.

**The input manipulation contains a concrete token-boundary confound.** The first body token in the four conditions is:

| Wording | Native | Case only | Prefix only | Both |
|---|---|---|---|---|
| He sent… | He (1544) | he (258) | ĠHe (679) | Ġhe (339) |
| A letter… | A (32) | a (64) | ĠA (317) | Ġa (257) |

Ġ denotes a leading-space token, not an extra displayed character. The prefix adds four tokens; among aligned original body tokens, only the first changes. Thus capitalization is implemented with different token pairs at the sequence boundary and after the prefix. These are distinct embedding inputs, so the H607 factors are not two identical token-level operations in different contexts. This is a plausible source of their earliest interaction, not yet an isolated causal explanation of all later effects.

The final-position embedding interaction is exactly zero because the final token is unchanged. Nevertheless, the first attention output already has a nonzero interaction (mean norm 8.526/8.553 in old/new wording). MLP0's write interaction is much larger (428.3/479.9 raw norm). Raw norms vary with residual scaling and cannot be read as causal importance. Normalized-input interaction grows and shrinks through later blocks, reaching mean norm 2.728/2.409 at MLP11. Exact residual accounting, including learned skip coefficients, verifies that the trace is internally consistent.

**Early appearance is not the same as strong influence under a final-position removal.** Subtracting Attention0 or MLP0 interaction at the final position leaves MLP11's interaction norm almost unchanged. Other token positions remain live and may regenerate or transmit the interaction; this does not establish that the early layers are unimportant. Attention9 is the strongest tested single final-position removal in both wordings, leaving approximately 0.820 and 0.786 of the original target norm.

| Removed write | Old wording remaining norm | Old random control | New wording remaining norm | New random control |
|---|---:|---:|---:|---:|
|attn0|1.0001|0.9997|1.0000|0.9997|
|mlp0|1.0002|0.9998|1.0000|0.9999|
|attn1|0.9984|1.0014|0.9995|1.0010|
|mlp1|0.9989|0.9986|0.9991|0.9996|
|attn2|0.9951|0.9994|0.9951|1.0001|
|mlp2|0.9960|1.0010|0.9961|1.0029|
|attn3|0.9964|0.9979|0.9913|0.9959|
|mlp3|0.9980|0.9989|0.9917|0.9973|
|attn4|0.9870|1.0090|1.0018|1.0144|
|mlp4|0.9900|1.0097|0.9942|1.0082|
|attn5|0.8971|1.0145|0.9370|1.0175|
|mlp5|0.9688|1.0195|0.9826|1.0305|
|attn6|0.9096|0.9921|0.9355|1.0073|
|mlp6|0.9241|1.0163|0.9535|1.0208|
|attn7|0.9161|1.0268|0.8840|1.0820|
|mlp7|0.9404|1.0153|0.9368|1.0224|
|attn8|0.9501|1.0544|0.9582|1.0456|
|mlp8|0.9345|1.0164|0.9304|1.0131|
|attn9|0.8201|1.0207|0.7861|1.0389|
|mlp9|0.9136|1.0283|0.9183|1.0267|
|attn10|0.8260|1.0601|0.8649|1.0583|
|mlp10|0.9052|1.0410|0.9147|1.0433|
|attn11|0.8495|1.0346|0.8612|1.0565|

Remaining-norm ratios are not fractions of total computation explained. The intervention changes only the combined condition and only one write; each write may carry inherited interaction rather than generate it locally. Separate removals cannot be summed into causal shares. Norm-matched controls use one random direction per site, so they are directional controls rather than a random-distribution significance test.

For Attention9, projection onto the original target interaction is reduced by about 0.205/0.266 of its squared-norm scale in the two wordings, while the norm remains 0.820/0.786. These differ because the target can rotate as well as shrink. Mean output KL from the unmodified combined prompt is about 0.00179/0.00488. That KL is an absolute behavioral change, unlike the normalized reference-KL errors in H607. Target-token CE changes are separately recorded; no beneficial task effect is assumed.

Interpretation: interaction is present immediately, with a concrete tokenization asymmetry at the manipulated first word. Later final-position attention writes have greater measurable control of the interaction reaching MLP11 under this intervention. This supports a distributed propagation/amplification account rather than locating a unique "prefix×case neuron". It remains unclear how much is inherited from boundary-token differences and how much is newly created by nonlinear mixing.

Next equalize the first body token's boundary status: add a leading space to the unprefixed conditions so upper/lowercase use the same token IDs with and without the prefix. Repeat the native trace, preserving body-token alignment and checking competence. This directly tests whether Attention0's interaction disappears when the tokenization asymmetry is removed. Then test earlier token positions or selected paths to determine how any remaining interaction reaches the later attention writers. This should precede interpreting the current interaction as an abstract semantic feature combination.

Audits: same manifest unchanged; runner unchanged; 128 native prompts, 1,504 interventions; exact identity logit replay; final embedding interaction zero; full residual accounting maximum relative discrepancy 5.84e-6; finite metrics. No new natural-text documents; FineWeb remains 26,112. Broad goal remains active and incomplete.
