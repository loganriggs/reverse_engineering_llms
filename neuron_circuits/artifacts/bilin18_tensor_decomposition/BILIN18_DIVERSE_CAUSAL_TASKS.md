## H533 — Diverse tasks separate local MLP effects from routing questions

The user authorized resuming with direct mechanistic tests and diverse tasks. This run declares 128 synthetic prompt pairs across eight families, plus one intended answer-preserving edit per pair: 384 prompts. Every family has eight discovery and eight transfer pairs. Transfer changes wording and items; vocabulary is not completely disjoint (especially sequence colors and ambiguous-word probes). These are new measurements on constructed prompts, not claimed unseen model-training data. No new FineWeb documents; tracked total remains 26,112.

The manifest and gates were written before inference. A family passes on a split if at least four pairs tokenize, both endpoints choose the intended binary answer in at least 75% of pairs, and pooled intended full-vocabulary top-1 is at least 50%. Exact tokens are used for retrieval; fixed sets of sentiment/agreement continuations are inherited from the previous red team. These sets measure grammatical number or sentiment class, not literal answer accuracy. Context-association completions are ambiguous and exploratory. No individual pair is selected for successful behavior.

| Family | Discovery pairs with both contrasts correct | Transfer pairs with both contrasts correct | Discovery full-vocabulary top-1 | Transfer full-vocabulary top-1 | Both gates |
|---|---:|---:|---:|---:|---|
|agreement|100.0%|100.0%|56.2%|56.2%|pass|
|capital_recall|100.0%|87.5%|87.5%|87.5%|pass|
|context_association|75.0%|87.5%|0.0%|0.0%|fail|
|entity_binding|100.0%|100.0%|100.0%|100.0%|pass|
|negation|100.0%|100.0%|37.5%|31.2%|fail|
|relational_order|0.0%|0.0%|0.0%|0.0%|fail|
|sentiment|100.0%|100.0%|62.5%|68.8%|pass|
|sequence_copy|100.0%|100.0%|81.2%|56.2%|pass|

Five families pass on both splits. Negation and context association show binary contrast information but fail the full-vocabulary gate. Relational ordering fails these formats; this does not establish that the underlying capability is absent. Small structured samples do not support broad population-accuracy claims.

For each discovery-passing family, swap the entire last-position MLP output from the paired donor at each of 18 layers, in both directions. The recipient's bypass, other positions, attention and value bus are native until that intervention; all subsequent computation is recomputed. Select the layer maximizing mean donor-directed answer log-odds change on discovery only, then apply that frozen choice to all transfer pairs. Also test nuisance donors and self donors. Answer log-odds compares the two declared token sets, so it is distinct from top-1 behavior or KL fidelity.

| Family | Selected layer (0-based) | Transfer donor-directed log-odds change | Fraction of native endpoint contrast | Mean absolute nuisance change | Positive directions |
|---|---:|---:|---:|---:|---:|
|agreement|12|1.8437|21.8%|0.1252|16/16|
|capital_recall|13|0.8757|9.5%|0.3038|16/16|
|entity_binding|10|0.1454|1.9%|0.0244|14/16|
|sentiment|12|0.8239|31.7%|0.0631|16/16|
|sequence_copy|10|0.2886|3.6%|0.3202|15/16|

Single selected MLP writes move about 22% of the native agreement contrast and 32% of sentiment contrast on transfer; capital recall is about 9%. Binding and copying are only about 2% and 4%. These are ratios of aggregate contrast changes, not fractions of task accuracy, necessary computation or uniquely mediated information. Whole-output swaps are sufficiency probes under hybrid states, not necessity tests. Copying's nuisance effect is comparable to its task effect, so this localization is not selective evidence for a copying operation. Nuisance edits are not matched in edit count or activation norm, and only base-to-nuisance is tested; comparisons are descriptive.

This prioritizes reusable feature changes versus conditional operations for agreement/sentiment. It prioritizes attention, residual and value-bus paths for binding/copying. It does not establish that those paths are responsible: they have not been tested in this run. The source of a whole MLP output's information remains unidentified.

Audits: native custom forward matches the existing native forward exactly; self swaps change margins by at most 6.68e-06; all causal margins finite; manifest unchanged; 0 excluded pairs. The 3,040 intervention records include all discovery layers and frozen-layer transfer, nuisance and self controls. Full prompt manifest, per-prompt predictions and per-intervention margins are archived.

Next experiment: freeze the selected layers and dissect left/right bilinear operand swaps, including their exact cross term. Separately fit shared number/polarity write changes on discovery only and test transport to transfer contexts against nuisance and random controls. Neither experiment presumes that checkpoint operand coordinates are unique semantic features.

### Twenty competing explanations and discriminating tests

1. Reusable semantic changes are mixed across neuron coordinates. Test whether a discovery-mean number/polarity write transfers to unseen items, against matched nuisance and random directions.
2. A context-dependent bilinear operation is the useful unit. Swap left and right operands separately; test the finite cross term and transport across receivers.
3. Information routing dominates binding and copying. Compare attention-write swaps with MLP-write swaps at every layer on discovery and frozen selected layers on transfer.
4. Features become usable only at a downstream reader. Preserve a source change while replacing or varying the receiving state; test output predictions beyond reconstruction.
5. MLPs mostly amplify already-selected answers. Compare task contrast before/after each MLP and predict effects from native answer alignment; agreement with this alone is not semantic identification.
6. Whole-write effects come from generic confidence changes. Measure complete distribution KL, entropy, and effects on unrelated tokens, alongside answer contrasts.
7. Several layers implement redundant versions of the same operation. Compare isolated and jointly patched layers; look for subadditive effects.
8. Several layers implement a sequence of necessary operations. Compare combinations and path-specific interventions; look for order-dependent interactions.
9. The residual bypass carries most binding information. Compare residual-stream swaps, attention writes and MLP writes without conflating their differing scopes.
10. The persistent attention value bus carries important information missed by residual-only swaps. Intervene on it separately with native replay controls.
11. Within-task features transfer but cross-task universality is limited. Freeze task-specific transformations and evaluate semantically aligned cross-task conditions.
12. Apparent feature directions encode answer tokens. Hold the relation fixed while changing output vocabulary; test counterbalanced mappings.
13. Apparent relations encode particular prompt templates. Hold lexicon fixed across templates and template fixed across lexicons in a crossed design.
14. The competence gate hides brittle cases. Report every declared prompt and evaluate failure cases separately without using them to select mechanisms.
15. A low-dimensional feature is represented nonlinearly. Compare fixed vector transport with receiver-conditioned transformations on strictly separate fit and evaluation sets.
16. A useful unit is a distributed subspace rather than a neuron. Compare causal subspace interventions with matched controls; do not infer semantics from rank alone.
17. Changes in normalization create apparent operand interactions. Separate pre-normalization effects from algebraic interactions in the already normalized bilinear operands.
18. Semantically equivalent coordinates are non-unique. Test whether different fitted coordinates predict the same held-out interventions; interpret the equivalence class rather than naming axes.
19. Off-distribution hybrids exaggerate causal effects. Compare native donor swaps with interpolation paths, endpoint controls and norm-matched nuisance swaps.
20. The chosen local units are genuinely insufficient. Escalate to multi-position, multi-layer source-to-reader paths only if simpler frozen explanations fail on new examples.

Priority: hypotheses 1–4 are the leading working explanations, with 5, 12, 13, 17 and 19 as immediate alternatives/controls. This ranking is qualitative; the present observations do not identify a winner.
