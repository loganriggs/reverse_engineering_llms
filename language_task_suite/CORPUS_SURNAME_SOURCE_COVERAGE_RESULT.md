# Natural-source coverage at matched training steps

The question is whether more independent natural-language carriers help a small bilinear model learn the *visible rules*, rather than merely fitting more generated templates. The [20,000-source cache](results/corpus_surname_joint_unedited_background_train20k_500000_test2k_manifest.json) contains 19,999 accepted FineWeb documents at offsets 500000–519999; one preview near-duplicate was rejected. The [10,000-source control](results/corpus_surname_joint_unedited_background_train10k_500000_test2k_manifest.json) uses the first half of the same source panel. The first 10,000 accepted source hashes and ordinary background `x`/`y` token rows are exactly equal across the two caches. All ordinary background documents are unedited. Diagnostic task rows still include controlled bracket and surname edits, and the task rates and site weights are research-chosen rather than natural frequencies.

Both models use depth 2, width 64, four heads, RMS normalization, 6.58 million parameters, seed 46451, pair-loss weight 2, batch size 8, and 12,000 AdamW steps. They have the same validation and reused development-test sources at offsets 120000–121999. Test evaluation occurred only at the final step. The sole intended difference is whether the training sampler draws from the first 10,000 or all 20,000 sources. This is one seed, not a population estimate. The run has a fixed update budget, so broader coverage also means fewer expected presentations of each source.

| Final development test (81 surname pairs, 131 bracket sites) | 10k sources | 20k sources |
| --- | ---: | ---: |
| Surname: both edited answers outrank the opposite candidate | .358 | .494 |
| Surname: both are full-vocabulary top-1 | .037 | .123 |
| Bracket-site full-vocabulary top-1 | .351 | .450 |
| Ordinary background token NLL | 7.320 | 7.269 |
| Literal-code / spelling-style site top-1 | 0 / 0 | 0 / 0 |

After both fixed checkpoints were saved, the [fresh-source evaluation](results/corpus_surname_joint_fresh_530000_5000_matched10k_vs20k.json) screened all 5,000 sources at offsets 530000–534999 against the training, old test, and earlier 100000–109999 source panels. No exact or preview-SimHash near duplicates were found. The same frozen evaluation law yielded 421 aligned surname pairs and 200 bracket counterfactuals:

| Fresh source panel | 10k sources | 20k sources |
| --- | ---: | ---: |
| Surname paired two-candidate ranking | .299 | .480 |
| Surname paired full-vocabulary top-1 | .019 | .083 |
| Bracket paired two-choice accuracy | .880 | .935 |
| Bracket paired full-vocabulary top-1 | .240 | .320 |
| Bracket paired correct-closer-character top-1 | .650 | .870 |
| Bracket edit moves answer margin correctly | .980 | .995 |

The matched control supports a **source-coverage benefit at fixed compute** for these checkpoints, particularly on bracket response and partial surname retrieval. It does not establish reliable surname copying: only about 8% of fresh pairs have both answers as the top token. Nor does it establish that code or style rules were learned. The original 10k-source/6k-step checkpoint is historical context, not the causal comparison; it had both fewer sources and fewer updates. Natural-carrier prose has no exact symbolic oracle for all tokens. Therefore the learned-bracket behavior is eligible for weight-method tests, while surname, code, and style should remain behavioral development targets. Next, repeat the source-coverage comparison across seeds or a stronger architecture and only then compare sparse/canonical representations at matched causal fidelity and actual edge count.

The [matched 10k history](results/corpus_surname_joint_unedited_train10k_500000_depth2_pair2_seed46451.json) and [20k history](results/corpus_surname_joint_unedited_train20k_depth2_pair2_seed46451.json) are committed. Raw source text, token caches, and model/optimizer checkpoints remain local in `/home/loganriggs/Coding/neuron_circuits/artifacts`.

## Second-seed and new-source replication

I repeated the matched 12,000-step training with seed 46452, using the identical 10k and 20k caches and no intervening task-law changes. The [seed-46452 10k](results/corpus_surname_joint_unedited_train10k_500000_depth2_pair2_seed46452.json) and [20k](results/corpus_surname_joint_unedited_train20k_depth2_pair2_seed46452.json) histories reached their fixed endpoints. A first launch of the 10k trainer exited with code 139 before saving a checkpoint; the fault-handler retry completed, and this unexplained transient should be kept in mind for reproducibility.

The new [5,000-source panel](results/corpus_surname_joint_fresh_540000_5000_2seed_10k_vs20k.json) at offsets 540000–544999 was screened against the 20k training panel, 2k reused development test, earlier 10k training sources, and the prior 530k fresh panel. All 5,000 passed exact-hash and preview-SimHash near-duplicate checks. Four checkpoints were frozen before scoring; the evaluator yielded 458 surname pairs and 200 bracket pairs. This panel is now evaluated, not reserved for model or method selection.

| Seed | Train sources | Surname paired rank | Surname paired full-vocab top-1 | Bracket paired two-choice | Bracket paired full-vocab top-1 | Bracket paired correct-character top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 46451 | 10k | .288 | .026 | .880 | .195 | .615 |
| 46451 | 20k | .467 | .070 | .945 | .365 | .855 |
| 46452 | 10k | .192 | .013 | .840 | .185 | .675 |
| 46452 | 20k | .574 | .148 | .940 | .310 | .885 |

The source-coverage advantage survives both model initializations and both independent fresh panels under the fixed architecture and update budget. It is still a small two-seed experiment on contiguous FineWeb convenience panels, so effect sizes should not be generalized to other model sizes or corpora without replication. Bracket *direction* response is already near saturation in both arms (.99–1.00 positive edit-response fraction on this panel); better answer accuracy does not by itself demonstrate better teacher-circuit fidelity or simpler internal weights. Only bracket currently clears a robust two-choice behavior gate; surname remains partial and code/style remain negative controls. The next mechanistic comparison should use a predeclared causal-effect magnitude and background-fidelity gate, actual nonzero edge count, and a fresh method-evaluation source panel.
