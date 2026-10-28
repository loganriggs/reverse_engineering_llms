## H921 — Discovery-fitted occurrence routing transfers across name identities

H920 supported candidate promotion and repeated-sender suppression through signed persistent-value routing. H921 replaces those routing coefficients with means fitted only on discovery names, retaining each recipient's own lexical values. This directly tests a reusable operation, following H919's failure to find a small transferable name-value span.

### Design and scope

Fit the coefficient for each name occurrence on the first four name pairs, balanced over sender identity and list order. Evaluate the other four name pairs across both templates and list orders, plus copying. Evaluate the three frozen readers separately and jointly. The fitted coefficients replace only persistent-value contributions at the named occurrences. Current-value contributions, other positions, other heads and downstream computation remain live. In joint edits, later readers' live routing is subtracted before adding the fitted contribution; the implementation does not subtract stale cached routing after an earlier edit.

Own-mean coefficients use the matching template; other-format uses the other template; pooled averages both. The occurrence model additionally ties both initial-list coefficients, so it needs no distinction between sender and recipient within the list. Wrong-occurrence swaps list-other and repeated-sender coefficients; sign-flip reverses all pooled coefficients. Zero removes all selected name-value contributions. Identity leaves them unchanged. No fitting or outcome selection uses transfer names, but these names are familiar from prior experiments and are not a pristine holdout.

### Joint three-reader transfer

Error ratios compare deviation from native behavior with zeroing the same name contributions. Low KL and error indicate fidelity; improved accuracy alone does not. Native binding accuracy is87.5%, copying100%.

|Family|Replacement|Margin RMS error|Error / zero|Native→edited KL|KL / zero|Top1|
|---|---|---:|---:|---:|---:|---:|
|copy|identity|0.000|0.000|0.00000|0.000|100.00%|
|copy|zero|1.084|1.000|0.14164|1.000|100.00%|
|copy|own_mean|0.274|0.252|0.00563|0.040|100.00%|
|copy|other_format|0.314|0.289|0.00752|0.053|100.00%|
|copy|pooled|0.288|0.266|0.00627|0.044|100.00%|
|copy|occurrence|0.288|0.266|0.00627|0.044|100.00%|
|copy|wrong_occurrence|0.288|0.266|0.00627|0.044|100.00%|
|copy|sign_flip|3.281|3.026|1.18640|8.376|62.50%|
|task|identity|0.000|0.000|0.00000|0.000|87.50%|
|task|zero|2.424|1.000|0.30841|1.000|40.62%|
|task|own_mean|0.268|0.110|0.00492|0.016|84.38%|
|task|other_format|0.392|0.162|0.03657|0.119|96.88%|
|task|pooled|0.295|0.122|0.01225|0.040|90.62%|
|task|occurrence|0.326|0.134|0.01465|0.047|90.62%|
|task|wrong_occurrence|8.049|3.320|1.52221|4.936|0.00%|
|task|sign_flip|4.864|2.006|0.99177|3.216|0.00%|

Joint own-mean binding error is0.268 logits, only11.0% of zeroing's2.424; KL is0.00492 versus0.3084 for zeroing. Pooled coefficients remain useful (error ratio.122, KL.01225), and tying initial-list roles is only somewhat worse (error ratio.134, KL.01465). Other-template transfer is less faithful than own-template transfer despite higher accuracy: its KL is0.03657 while top1 improves to96.875%. This is a concrete demonstration that correctness and mechanistic fidelity differ.

Wrong-occurrence and sign-flip joint edits produce0% binding top1. At L14H4 alone, own-mean margin-error ratio is.130 and KL.00445; wrong-occurrence also yields0% top1. This supports the importance of how lexical contributions are assigned and signed, not just their presence.

Copying own-mean joint error ratio is.252 and KL.00563. The copy wrong-occurrence control is exactly equivalent to pooled coefficients here: both positions carry the same token's persistent value, so swapping two coefficients preserves their sum. It is an algebraically degenerate control, not evidence of insensitivity to arbitrary wrong routing. Binding wrong-occurrence is nondegenerate because it exchanges coefficients attached to different names.

### Interpretation

These experiments support a reusable signed occurrence rule over relatively distinct lexical vectors. A small lexical span failed, while a handful of coefficients substantially preserved the causal contribution across names. Sharing an operation therefore need not mean sharing a small activation subspace. The occurrence-only fit further weakens the need to posit separate semantic sender/recipient features for these templates.

This remains a partial model. It uses externally annotated name positions, retains native values and the rest of the model, and does not explain how the model computes the routing or identifies occurrences. It neither replaces whole heads nor establishes the complete binding circuit. There are only four transfer name pairs; repeated endpoints and templates are correlated. Wrong controls can be strong off-native perturbations, so their failures complement rather than replace the positive fidelity test.

### Next information-gaining test

Break the current confound between repeated identity and semantic role. Use a fixed competence screen where the repeated entity is sometimes the expected answer: matched active/passive or subject/object query variants, with name-list order and identity counterbalanced. Retain all conditions before examining outcomes. Compare the frozen occurrence rule with actual routing on competent conditions. A fixed repetition-suppression rule predicts the wrong response when repetition and the required answer coincide; context-sensitive routing should change sign or allocation. Failed competence conditions cannot adjudicate the mechanism and must be reported separately rather than prompt-tuned until successful.

If competent counterexamples exist, test whether the new routing profile transfers across names and whether the existing local rule fails selectively there. This tests compositional computation directly and reconnects the result to choosing featurizers that represent operations as well as activation geometry.

### Verification

1,584 full-model forwards. Identity replay and native-cache replay both exactly zero. Source hashes unchanged; results finite. Discovery fit indices and coefficients, all endpoint results and all single/joint controls are saved. No reserved test or added natural documents; document pool remains26,112. Broad goal active.
