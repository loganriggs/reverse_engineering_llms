# Fresh whole-combination transfer check

The 18-document pilot suggested that a depth-2 bilinear transformer trained on **separate** natural-text sources could carry its delimiter-closing rule into a document that also contains a spelling-style cue. This frozen-model check repeats that question on **164 new FineWeb documents**. It supports delimiter transfer in both seeds, but it still does not establish two-rule composition: the spelling rule fails its behavioral gate.

The [panel builder](build_joint_style_delimiter_panel.py) streamed FineWeb `sample-10BT` offsets 84,500–94,499 with a fixed seed. Of 10,000 documents, 320 admitted a US/UK spelling × round/square delimiter plan and 164 passed the complete four-arm semantic and token-alignment validator. None of these documents were used to train or select the checkpoints. They are disjoint by source offset from the spelling training range (53,500–83,499), delimiter training range (44,000–51,999), and the first 18-document joint panel (83,500–84,499). An exact screen found zero shared 64-token chunks among 2,575 unique panel chunks and 45,946 unique training-cache chunks; this does **not** exclude shorter or approximate duplicates.

The six checkpoints are exactly the two sports starting models and their matched 1,000-update continuations from the [original holdout](THREE_SOURCE_RULE_COMBINATION_HOLDOUT.md). They were loaded unchanged. The control continuation saw sports and spelling examples; the three-source continuation also saw delimiter examples. Each score below is an average over the four controlled arms, with 328 paired comparisons for each rule (two other-rule settings per document). “Paired two-way” requires both counterfactual arms to prefer their own target over the alternative target. “Full-vocabulary top-1” requires the target to beat every token in the vocabulary at its site.

| Seed / frozen checkpoint | Delimiter paired two-way | Delimiter full-vocab top-1 | Spelling paired two-way | Spelling full-vocab top-1 | Both rules in all four arms |
|---|---:|---:|---:|---:|---:|
| 41701 / sports start | .046 | .000 | .021 | .000 | .000 |
| 41701 / sports + spelling | .000 | .000 | .000 | .034 | .000 |
| 41701 / three sources | **.527** | **.488** | .012 | .015 | .000 |
| 41702 / sports start | .030 | .000 | .000 | .000 | .000 |
| 41702 / sports + spelling | .006 | .000 | .006 | .027 | .000 |
| 41702 / three sources | **.683** | **.605** | .000 | .011 | .000 |

Delimiter edit-direction correctness is .756/.841 in the three-source seeds, compared with .729/.692 for their controls. The higher paired and full-vocabulary scores make the transfer conclusion stronger than the edit-direction statistic alone. Spelling edit-direction correctness is .524/.445 in the three-source seeds, near chance, consistent with the failed paired test. No checkpoint gets both rules right across all four arms of any document. The first 18-document panel gave three-source delimiter paired scores .389/.722; the larger fresh panel gives .527/.683, so the qualitative result survives a substantially larger holdout without checkpoint selection.

As a **form screen**, the first 100 words of 128 randomly chosen documents from this offset range have top-ten word share .2114 and across-document repeated-trigram occurrence share .0188. The first 100 words of 128 selected original carriers have .2134 and .0217; their edited versions have .2134 and .0217. These near matches indicate that the controlled edits leave these coarse surface statistics intact. Selection is still rare (164/10,000), and neither a first-100-word screen nor source text itself proves that the task-bearing windows are a realistic *sampled* language distribution. The empirical carriers also lack a full symbolic next-token oracle.

This makes the bracket/delimiter models eligible for a targeted test of whether a proposed weight component is **causally reusable** across sports bracket and FineWeb delimiter closure. It does not make spelling eligible for a circuit-recovery claim. The next method comparison should test component ablation and matched controls on both rule families, while separately pursuing the broader sampled document DGP with realistic task-bearing windows.

Reproduce the local data with `build_joint_style_delimiter_panel.py --skip 84500 --n 10000 --seed 98502 --output language_task_suite/results/joint_style_delimiter_fresh_84500_94500.pt`, then run [the frozen evaluator](eval_three_source_joint_fresh.py). The source-token `.pt` panel and model checkpoints stay local; the [aggregate result](results/three_source_joint_fresh_84500_94500.json) can be shared.
