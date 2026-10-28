## H611 — Later body writes carry influential interaction, with opposing positional effects

Extend H610 from MLP0 to every attention/MLP write in layers 0–10 and Attention11. Use the same 128 boundary-matched prompts and all 64 exact body alignments. On the combined prefix/lowercase condition subtract each native four-condition write interaction at first, final, other, or all aligned body positions. Prefix-only positions remain untouched. Attention interventions alter the residual write, leaving the persistent value bus live. Include seeded per-position norm-matched random controls and identity runs.

All 23 sites and four scopes are predeclared, yielding 5,920 interventions. The target is the final-position normalized MLP11 input interaction. Output KL and copying correctness are separate outcomes. This is exploratory localization on the same two formats and eight name pairs, not fresh confirmation or a statistical test across independent tasks.

**Influence becomes much stronger after the earliest layers, especially in non-final positions.** Attention5 all-body removal leaves mean target norm 0.574 in old wording and 0.507 in new wording, versus random controls 1.072 and 1.036. Removing only other body positions leaves 0.681/0.668; final-only removal leaves 0.901/0.839. Thus final-position-only tests missed much of this write's influence. First-token-only Attention5 removal has little effect.

| Write | Old first | Old final | Old other | Old all | New first | New final | New other | New all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|attn0|0.989|1.000|1.000|0.989|0.987|1.000|1.000|0.987|
|mlp0|0.993|1.000|1.000|0.993|0.973|1.000|0.997|0.971|
|attn1|1.021|0.999|0.986|1.006|0.995|0.998|0.978|0.969|
|mlp1|0.976|0.999|0.994|0.972|0.867|0.999|0.978|0.853|
|attn2|0.990|1.003|0.911|0.903|0.972|1.005|0.901|0.870|
|mlp2|0.812|1.002|0.928|0.744|0.846|1.002|0.909|0.766|
|attn3|0.981|1.019|0.762|0.747|1.012|1.007|0.737|0.751|
|mlp3|0.781|1.017|0.785|0.568|1.751|1.006|0.727|1.499|
|attn4|0.994|0.997|0.702|0.698|1.000|0.995|0.706|0.695|
|mlp4|0.894|0.996|0.659|0.574|1.703|0.997|0.662|1.446|
|attn5|0.991|0.901|0.681|0.574|0.996|0.839|0.668|0.507|
|mlp5|0.951|0.978|0.821|0.747|0.997|0.960|0.833|0.796|
|attn6|1.011|0.878|0.934|0.830|1.000|0.927|0.869|0.804|
|mlp6|0.982|0.918|0.875|0.767|0.999|0.911|0.837|0.740|
|attn7|0.995|0.882|0.893|0.768|0.998|0.865|0.913|0.782|
|mlp7|0.990|0.918|0.907|0.813|0.999|0.914|0.896|0.805|
|attn8|0.996|0.950|0.980|0.926|1.000|0.945|0.977|0.923|
|mlp8|0.988|0.938|0.919|0.841|1.001|0.924|0.958|0.880|
|attn9|1.002|0.806|0.931|0.744|1.000|0.850|0.960|0.815|
|mlp9|0.999|0.910|0.962|0.871|0.999|0.890|0.977|0.864|
|attn10|1.000|0.867|0.998|0.866|1.000|0.927|0.995|0.922|
|mlp10|0.999|0.908|0.986|0.895|1.000|0.929|0.994|0.923|
|attn11|1.000|0.859|1.000|0.859|1.000|0.934|1.000|0.934|

Values are remaining target-interaction norms relative to native baseline. They are conditional effects of changing one factorial corner, not percentages of computation explained. Results from different layers or scopes cannot be added as causal shares, and a write may carry inherited interaction rather than generate it locally.

MLP3 and MLP4 expose opposing positional effects. In old wording, their all-body removals leave 0.568/0.574. In new wording, other-body removals reduce the target to 0.727/0.662, but first-token removals increase it to 1.751/1.703; all-body removals therefore leave 1.499/1.446. This is inconsistent with treating every interaction-bearing position as an interchangeable positive contributor. It supports amplification/cancellation that depends on position and wording. The removal geometry and downstream nonlinear response remain part of that conditional interpretation.

Attention5 is a more consistent candidate across both formats. Its all-body removal has mean absolute output KL 0.0046/0.0040 and retains perfect copying top1 in both wordings. Indeed every all-body interaction removal retains copying top1 in this panel, including the MLP3/4 conditions with enlarged internal interaction. Hence these results localize representational influence; they do not establish a necessary mechanism for producing the copied name. Internal norms, output probabilities and discrete task success remain distinct objectives.

The strongest old-wording internal reduction is MLP3, whereas new wording favors Attention5. Selecting Attention5 for follow-up is an adaptive decision based on consistent influence across the two wordings, not a preregistered winner. Random controls are one direction per relative position per operation, matching each case's norm profile; they are directional checks rather than significance estimates.

Next decompose Attention5's influential body write into routing and value contributions, distinguishing local contextual values from the persistent value path. Compare whether the four-condition interaction is already present in values or emerges when routing acts on them. Use matched cross-condition substitutions and explicit replay checks rather than attributing effects solely from activation size. Then test whether a smaller component predicts or reproduces the body-write intervention across wordings and examples. This connects localization back to the goal of finding useful computational units.

Audits: 128 native prompts, 64 exact body alignments, 5,920 interventions; exact identity replay; frozen source/runner; finite metrics. Applied subtraction error is <=4.69e-4 relative across all sites, with the largest relative sensitivity arising for small differences; random per-position norm mismatch <=0.00196 absolute. Removing non-final Attention11 outputs leaves the final MLP11 target exactly unchanged, as required by causal ordering, while later output behavior can still change. FineWeb remains 26,112 documents; broad goal active and incomplete.
