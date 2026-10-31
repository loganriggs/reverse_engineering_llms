# Rank-96 bond cuts: natural fidelity and branch-intervention fidelity diverge

The [full-function rank comparison](FULL_FUNCTION_RANK_AND_SCORE_RESULTS.md) found that a flat key dictionary is expensive for the 546M model's first-layer head 0. This panel returns to the **physical QK1 bond**: replace its 128-dimensional identity with a saved rank-96 matrix, leaving QK2 and OV intact. The rank-96 cuts were fitted with either a uniform token metric (weight-only/data-free) or a FineWeb frequency metric (dataset-aware). A previously fitted rank-64 sequence cut is an unequal-rank reference, not a capacity-matched competitor. No sparsity or semantic prior is used here.

The evaluation has three levels. First, `flagship_bond_metric_panel.py` scores the **complete head pair response** over all 50,304 key IDs, held-out query IDs from strict-uniform and mixed panels, and positions (7,8), (32,64), and (16,64). It also reuses the 512 natural pairs per corpus from the earlier panel. Second, `flagship_rank96_sequence_panel.py` scores the summed layer-0 head output on natural 128-token windows and under two QK2 score edits. Third, `flagship_full_model_eval.py` and `flagship_rank96_intervention_full_model.py` propagate those cuts through all 18 layers and compare next-token distributions. The full-model tests use 24 documents per seed, two seeds per corpus, and FineWeb test, WikiText-103, and TinyStories. Each cut within a condition sees the same document windows.

## Full-vocabulary and natural-text fidelity

For the **mixed held-out query panel at positions (7,8)**, rank-96 relative squared complete pair-response error is:

| Cut | Uniform keys | Corpus-weighted keys | FineWeb natural pairs | WikiText natural pairs | TinyStories natural pairs |
|---|---:|---:|---:|---:|---:|
| Uniform rank 96 | .0662 | .0316 | .0179 | .0465 | .0074 |
| Corpus rank 96 | .0700 | .0262 | .0183 | .0426 | .0028 |

The full-vocabulary ranking depends on the test metric. At the other two position pairs, uniform-weight errors are .0692/.0806 for the uniform cut and .0781/.0916 for the corpus cut; corpus-weighted errors are .0746/.0882 and .0757/.0907 respectively. The rank-64 cuts have substantially larger errors (typically about .18–.28 across these position/metric combinations). The exact pair-response identity gate is `5.37e-12`. Pair tests are diagnostic; the earlier natural-pair energy-concentration caveat still applies.

Mean **natural original-to-cut KL, nats per next-token position**, across two seeds:

| Corpus | Uniform rank 96 | Corpus rank 96 | Sequence fit rank 64 |
|---|---:|---:|---:|
| FineWeb test | 0.0000165 | 0.0000203 | 0.0000306 |
| WikiText-103 | 0.0000217 | 0.0000257 | 0.0000410 |
| TinyStories | 0.0000132 | 0.0000156 | 0.0000209 |

The **uniform rank-96 cut wins on natural downstream KL in both seeds of all three corpora**, despite the corpus cut having lower corpus-weighted pair error and, on FineWeb and TinyStories, lower summed-head relative MSE. This establishes that head-level squared error alone does not select the best full-model approximation. The replayed identity bond has zero max absolute logit difference. Next-token CE shifts are tiny and change sign, so these KL numbers describe fidelity, not a quality gain.

## QK2 branch edits

To test preservation of the model's computation under a controlled counterfactual, the evaluator replaces layer-0/head-0 QK2 scores with either a constant one or the original scores shifted one key column before causal masking. It then compares the full model with an identity QK1 bond to the same intervened model with each rank cut. All other heads and layers retain their checkpoint weights. These edits change internal scores, not the input text.

Mean **intervened-reference-to-cut KL**, nats/token:

| Corpus / QK2 edit | Uniform rank 96 | Corpus rank 96 | Sequence fit rank 64 |
|---|---:|---:|---:|
| FineWeb / constant | .00510 | **.00130** | .00451 |
| WikiText-103 / constant | .00172 | **.00133** | .00356 |
| TinyStories / constant | .00148 | **.000452** | .00180 |
| FineWeb / key shift | .00000834 | .00000774 | .0000301 |
| WikiText-103 / key shift | .00000768 | .00000828 | .0000185 |
| TinyStories / key shift | .00000328 | .00000350 | .00000726 |

The corpus rank-96 cut wins the **constant edit in both seeds of all three corpora**. The key-shift ranking is small and mixed across seeds and corpora. The constant edit is a substantial distribution shift: natural-to-intervened-reference KL averages about .05 on FineWeb and WikiText and .029 on TinyStories. Key shift is milder, about .0006–.0011. The intervened identity-bond logit gate is exactly zero. Thus, the constant-edit result is meaningful evidence of branch robustness under this edit, but it is not a claim about all plausible causal interventions.

The summed-head relative MSE supports the same direction under branch edits. On FineWeb, for example, uniform/corpus rank-96 errors are .0689/.0526 under constant QK2 and .0550/.0478 under key shift. Yet natural downstream KL favors uniform. Different layers and output directions amplify different errors; a scalar head MSE cannot capture this sensitivity.

## Decision for the legibility stage

There is no single metric winner. Use **uniform rank 96 as the natural-behavior fidelity control** and **corpus rank 96 as the QK2-constant causal-fidelity control**. Before applying weight, activation, or intervention sparsity to either cut, retain both gates: held-out natural full-model KL and at least the QK2-constant full-model KL. A new legibility objective should report its movement along both axes, rather than treating lower head MSE or sparser factors as evidence of a better circuit explanation.

The next high-information test is a small sweep of *downstream-sensitivity-aware* bond metrics and intervention mixtures, selected on separate documents, followed by the same natural and edited holdouts. If no mixed objective improves the two-axis frontier, keep the two cuts as distinct reference decompositions and test sparse gauges against each. Whole-corpus and position transfer remain necessary because the rank-96 panel covers only one head and these specific edits.

Reproduce from this directory with `flagship_bond_metric_panel.py`, `flagship_rank96_sequence_panel.py`, `flagship_full_model_eval.py --n-docs 24 --batch-size 8 --threads 4 --cuts uniform_fit_l128_wtg96 corpus_fit_l128_wtg96 sequence_fit_rank64_cw0 --output-name head0_full_model_rank96_panel_n24.json`, and `flagship_rank96_intervention_full_model.py`. Numerical records and test settings are in `full_width_results/head0_bond_metric_panel.json`, `head0_rank96_sequence_panel_n24.json`, `head0_full_model_rank96_panel_n24.json`, and `head0_rank96_intervention_full_model_n24.json`. The model and corpora are referenced externally, not copied.
