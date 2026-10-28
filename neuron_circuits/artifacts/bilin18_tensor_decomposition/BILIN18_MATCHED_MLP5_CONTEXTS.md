## H636 — Matched MLP5 context effects are strong but explain little of the full prompt-response gap

H635 was progress: residual passthrough and readout-aware projection controls left substantial internal context dependence. H636 returns to a local mechanistic test rather than another rank search.

Use the frozen80-prompt five-task panel. For every receiver, define two donors before evaluation: the opposite template with exactly the same item, and the same template with item shifted by four modulo eight. The latter preserves singular/plural number in the agreement task. It is an item-variation control, not a perfect causal minimal pair. Both directions of template exchange are retained. No task, prompt, donor or direction is selected on output.

At the first affected MLP (block5), the response to an injected direction q is B(y,t), where y is the normalized native residual, t is the derivative of RMS normalization applied to q, and B is the symmetric bilinear derivative of the MLP. Exchange donor y while keeping receiver t (“context”), exchange donor t while keeping receiver y (“normalization”), or exchange both. The interaction remainder is exactly B(y_donor−y_receiver,t_donor−t_receiver). The residual q is retained in all modes. Hybrids are well-defined local operator counterfactuals but need not correspond to a naturally occurring prompt.

Evaluate all nine probes locally. Predeclare original, span2 and orthogonal2 for downstream propagation through the receiver's native block6–17 Jacobians and final readout. These sample the selected direction, an existing learned-span direction and an orthogonal control. The two latter directions were withheld from the H634–635 basis fitting; this experiment fits no basis or response predictor.

**Bilinear operating context dominates normalization on these replacements.** Context replacement changes the local MLP response more than normalization replacement in all720 receiver–probe comparisons for each donor type. After downstream propagation, the same ordering holds in all240 comparisons for each donor type. These are repeated measurements on a structured panel, not independent statistical trials.

| Family | Local within-context change | Local cross-wording context change | Local cross-wording normalization change | Output within-context change | Output cross-wording context change | Output cross-wording normalization change |
|---|---:|---:|---:|---:|---:|---:|
|copy|0.252|1.151|0.067|0.153|0.633|0.026|
|fact|0.126|1.088|0.078|0.075|0.702|0.053|
|plural|0.322|1.084|0.063|0.205|0.541|0.020|
|agreement|0.657|0.634|0.105|0.379|0.344|0.058|
|addition|0.183|1.148|0.091|0.107|0.701|0.029|

Local numbers divide Euclidean response-change norm by native MLP-response norm, averaging all nine probes, items and both templates. Output numbers divide probability-weighted centered-logit response-change norm by the receiver's full native logit-response norm, averaging the three propagated probes. The two columns use different spaces and denominators; they cannot be read as a fraction transmitted through the network.

Cross-wording context replacement is substantially larger than the matched item replacement for copying, facts, plurals and addition. **Agreement is an exception:** same-wording item changes are as large or larger than changed-wording effects. Thus a universal “template determines the operator” account is too strong. Item-dependent operating context remains material even when grammatical number is held fixed.

**A large local effect does not mean it explains the full difference between prompts.** A supplemental descriptive audit compares the propagated replacement with the actual donor-versus-receiver difference in full logit derivatives, measured in the receiver's probability metric. The donor endpoint includes all native downstream and readout differences. This is a response-gap comparison, not a mediation percentage or task-accuracy measure. Oracle fitting is not involved.

| Family | Remaining cross-wording gap after context-only replacement | Remaining gap after replacing both factors | Context-change alignment cosine |
|---|---:|---:|---:|
|copy|0.957|0.957|0.316|
|fact|1.008|1.018|0.376|
|plural|0.945|0.950|0.350|
|agreement|0.907|0.912|0.457|
|addition|0.897|0.891|0.450|

A remaining-gap ratio of1 means no improvement over the native receiver; above1 means a larger mismatch. Despite positive average alignment with the donor gap, context-only replacement leaves roughly90–101% of its original norm on average across these cells. Replacing both MLP5 factors does not substantially improve this. Facts even move slightly farther away by this metric. The replacement can therefore change the response strongly while failing to reproduce the other prompt's full response. This directly weakens an account in which MLP5 alone explains the construction-dependent transformation.

The current supported picture is distributed context dependence: MLP5 bilinear operating context matters, normalization is a smaller contributor for this intervention, and later computation/readout differences remain essential to explaining the complete prompt contrast. Earlier H630 results support cumulative later-MLP effects on the narrower language panel, but that finding has not yet been established for these matched cross-task contrasts.

Competence limitations remain unchanged. Explicit-copy wording is fully successful on this panel; first-copy wording often predicts “ a” first. Capital and agreement candidate tests perform well, while plural and addition candidates fail. Neither a positive derivative nor a smaller response gap demonstrates a corrected answer. All failures remain included.

Next apply matched donor bilinear contexts to MLP6–9, separately and together with MLP5, retaining native normalization, attention, residual path and readout. Include the same within-template item controls. Compare both the intervention magnitude and the remaining donor-response gap. This tests whether cumulative context replacement explains the observed response contrast or merely produces another large but poorly aligned transformation. Only expand to additional layers if the fixed early-window contrast leaves informative residual structure.

Audits:80 native forwards;1764 batched downstream JVP calls; native logits replay exactly; analytic local response agrees with finite differences within0.000154; native chain agrees with H633 within0.001009; exact factor decomposition residual below1.7e-12; input/runner hashes unchanged; finite metrics. Supplemental gap audit uses existing captures and is separately hashed. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.
