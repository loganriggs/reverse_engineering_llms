## H920 — Counterbalanced binding supports candidate promotion plus repeated-sender suppression

H919 rejected tiny discovery-name difference spans as useful transfer representations. H920 therefore tests the operation applied to lexical values. It crosses list order with sender identity independently, using the same eight name pairs and two frozen templates. For example, both “John and Mary ... John sent a letter to” and “Mary and John ... John sent a letter to” request Mary. Copy controls are included once rather than duplicated for a nonexistent list-order factor. No outcomes are filtered.

### Native competence

All discovery cells have 100% exact top1. On transfer names, template A has 75% and template B 100%, identically for both list orders. Every pair has correct binary name discrimination in all cells. Copy controls have 100% top1. All cells pass the prior competence gate. This excludes a simple fixed-list-position answer rule for this panel; it does not establish broad syntax generalization or distinguish semantic roles from repetition heuristics.

### Occurrence ablations

At each of the three frozen readers, remove only the local persistent-value contribution from one name occurrence: initial-list sender, initial-list other name, or repeated sender. Other heads, current-layer values and downstream computation remain live. These are zero ablations of local contributions, not global value deletions or natural donor substitutions. No manifold claim follows from them alone.

The table gives transfer means across four pairs, two templates, both list orders and both sender identities. “Other logit” means the wrong answer, which is the sender. Positive margin drop means the ablation harms correct-versus-sender discrimination. Signed routing coefficients are not probabilities.

|Reader|Removed occurrence|Native signed routing|Correct logit change|Sender logit change|Margin drop|
|---|---|---:|---:|---:|---:|
|L13H0|list_sender|-0.296|-0.015|-0.677|-0.662|
|L13H0|list_other|-0.307|-0.588|+0.181|+0.770|
|L13H0|sender_repeat|0.159|-0.034|+0.318|+0.353|
|L14H4|list_sender|-0.237|-0.042|-1.338|-1.296|
|L14H4|list_other|-0.253|-1.156|+0.119|+1.275|
|L14H4|sender_repeat|0.321|-0.090|+1.615|+1.705|

At L14H4, removing the initial sender occurrence lowers the sender logit by 1.338, improving discrimination. Removing the initial other-name occurrence lowers the correct logit by 1.156, harming discrimination and lowering top1 from 87.5% to 68.75%. Removing the repeated sender occurrence raises the sender logit by 1.615 while changing the correct logit by only -0.090. Correct-versus-sender margin falls by 1.705, although top1 remains 87.5%. Thus accuracy alone hides a substantial suppression effect.

L13H0 shows the same qualitative pattern at smaller magnitude. L8H3 has weaker effects. Copying L14H4 has negative routing to both repeated-token occurrences; removing either reduces the correct token's logit, especially the first occurrence. This is compatible with a common lexical carrier used differently through signed routing.

The role of downstream computation matters: routing sign alone does not identify promotion or suppression. The ablations establish the net downstream effect for these contexts. Changes in log probability, as well as logits and full-distribution KL, are saved to avoid relying solely on raw-logit interpretations.

### List-order check at L14H4

|List order|Occurrence|Mean routing|Margin drop|
|---|---|---:|---:|
|0|list_sender|-0.233|-1.293|
|0|list_other|-0.256|+1.303|
|0|sender_repeat|0.323|+1.742|
|1|list_sender|-0.241|-1.298|
|1|list_other|-0.251|+1.248|
|1|sender_repeat|0.319|+1.668|

These are correlated endpoints from four transfer pairs, not independent large-sample statistical evidence. The counterbalance strengthens the operational hypothesis but does not establish an abstract recipient feature. Candidate promotion plus repeated-token suppression could solve these prompts without general semantic binding.

### Decision and next test

Prioritize a reusable signed-routing rule over a tiny lexical subspace: promote both listed candidates and suppress the repeated sender. Fit role-aligned routing coefficients on discovery names only, then replace native coefficients on transfer names while retaining recipient value vectors. Compare role/occurrence-matched coefficients with deliberately wrong mappings, native/self controls, and copying. Include separate evaluation of role mapping versus simple occurrence mapping: the current panel makes the repeated entity the sender in every task, so those explanations remain confounded. A later competence-screened syntax panel must separate repetition from semantic role before asserting abstract binding.

Measure fidelity to native behavior and full-vocabulary KL, not only whether the answer remains correct. A mean coefficient that preserves high accuracy may still fail to reproduce the original computation. Keep all discovery and transfer cells, including failures.

### Verification

1,152 forwards: 96 native plus each selected reader's identity and occurrence ablations. Identity replay is exactly zero; source hashes unchanged; all results finite. All name continuations are single tokens; occurrences are identified by token IDs with asserted counts. Native logits, persistent values, full routing vectors, per-endpoint logit/log-probability changes and KL are saved. No reserved test or added natural documents; document pool remains 26,112. Broad goal active.
