# Rank-96 sparse gauges improve coordinate sparsity, not established legibility

The [rank-96 fidelity panel](RANK96_METRIC_AND_CAUSAL_PANEL.md) identified two useful reductions of the 546M model's first-layer head 0: a uniform, weight-only QK1 bond cut that best preserves natural next-token behavior, and a FineWeb-weighted cut that better survives a strong QK2-constant edit. Here the **cut is fixed** and we rotate its factor coordinates with an orthogonal matrix. Every tested rotation leaves the entire attention function unchanged, so this experiment isolates legibility priors from the fidelity prior.

Four objectives are compared to the SVD gauge and random orthogonal gauges: factor-weight L1, sparse pair contributions on uniform and corpus pairs, the same activation objective with QK2-constant pair weighting, and a variant balanced with rare-token pairs. The activation objective minimizes the energy-weighted effective number of active QK1 factor products per pair. It is a **pair-coordinate proxy**, not sequence-level sparsity or a semantic label. Each cut uses 16,384 fit and 16,384 selection pairs per law, three optimization seeds, and 8,192 held-out pairs per law. FineWeb documents for fit, selection, and test are disjoint; WikiText-103 and TinyStories are external tests.

The derivative check has relative error `3.8e-10`. Across both cuts, rotation orthogonality errors remain below `1.5e-13`, and relative reconstructed-bond errors below `1.6e-14`. Thus the following changes are coordinate changes, not losses of model fidelity.

## Sparse scores and stability

Mean effective active coordinates on **held-out corpus pairs** (out of 96):

| Prior | Uniform cut | Corpus cut |
|---|---:|---:|
| SVD | 34.7 | 36.7 |
| Random rotation | 35.9 | 37.5 |
| Natural pair activation | 32.5 | 30.8 |
| Natural + QK2-constant activation | **29.9** | **29.8** |
| Rare-balanced activation | 30.2 | 30.3 |
| Factor-weight L1 | 35.6 | 34.0 |

The intervention-aware activation gauge also reduces this count on external text: on WikiText-103, 31.7 versus SVD 36.9 for the uniform cut and 32.0 versus 37.5 for the corpus cut; on TinyStories, 26.7 versus 34.2 and 26.5 versus 36.0. The gain nearly disappears on unseen-token pairs: about 37–38 active coordinates for every gauge. Factor-weight L1 falls from .138/.139 in the SVD gauge to .0596/.0625, but that does not translate into comparably sparse pair contributions. Different sparsity objectives select different geometry.

The independently optimized activation axes do **not** stabilize well. Mean matched absolute component-signature correlations across the three seeds are .186/.191 for natural-pair activation, .196/.192 for intervention-aware activation, and .191/.180 for rare-balanced activation (uniform/corpus cut). Random rotations give .105/.115. Factor-weight runs match perfectly because that objective and initialization are deterministic across nominal seeds; they provide no independent-seed stability evidence.

## Post-fit labels and causal red team

Only **after** fitting the gauges, we assign each of the ten largest components a coarse query-token-form → key-token-form label on separate FineWeb selection pairs. The labels include categories such as space-prefixed word, word fragment, digits, and punctuation. On FineWeb test, mean label enrichment in log2 units is:

| Prior | Uniform cut | Corpus cut |
|---|---:|---:|
| SVD | 1.21 | .67 |
| Random | 1.26 | .90 |
| Natural pair activation | .86 | .55 |
| Intervention-aware activation | .78 | .46 |
| Rare-balanced activation | .80 | .38 |
| Factor-weight L1 | 1.18 | .79 |

The sparse gauges also fail to beat SVD or random on mean enrichment in WikiText-103 and TinyStories. This is a **negative result for this coarse label assay**, not proof that the components lack any semantic content. Response energy is concentrated: 65,536 sampled pairs yield only about 107–250 effective energy-weighted pairs depending on corpus and cut. The label categories are dominated by broad word-form distinctions.

For causal effects, we partially delete the two selection-highest-energy axes per gauge and calibrate every edit to `.001` aggregate relative layer-0 head-output error on 64 separate FineWeb documents. The prior `.003` target is infeasible as a deletion-only edit for at least one rank-96 component, so `.001` is used for **both** cuts and all gauges. We then measure full-18-layer KL per document on two seeds × eight documents in each of FineWeb test, WikiText-103, and TinyStories. The model's native-forward gate is zero.

For the uniform cut, mean FineWeb test KL is `4.3e-7` for random axes and `9.3e-7` for intervention-aware activation axes; the corresponding head errors are .00132 and .00167 on the shifted test sample. Higher KL per head error indicates greater downstream sensitivity, **not** selective semantic effect. A localization statistic, the fraction of documents effectively carrying the squared KL, is .56 for random and .40 for intervention-aware on FineWeb, but .67/.76 on WikiText and .50/.82 on TinyStories. For the corpus cut, the analogous fractions are .62/.54, .51/.80, and .60/.77. Lower means more concentrated. The direction reverses on external corpora, so there is no robust localization advantage in this small assay. The corpus cut's KL levels also do not establish one: on WikiText, activation-gauge mean KL is `2.8e-7` versus random `6.6e-7`, with different test head-error levels.

## Research decision

Exact gauge sparsity is real, but **single-axis sparsity is not yet a human-legibility criterion** on this head. The result repeats the rank-64 warning at a fidelity level that better preserves natural model behavior. The next experiment should use a stronger target: groups of axes with reproducible *functional signatures* or interventions selected for downstream contrast, evaluated on whole-task and corpus holdouts. A downstream-sensitivity-aware reduction metric may also improve the natural/constant-edit frontier before the gauge is optimized. Any claim of task discovery needs new labels or hypotheses tested after fitting; the present coarse labels are only a red team.

Reproduce the two gauge fits with `flagship_exact_gauge_sparsity.py --cut-name uniform_fit_l128_wtg96` and `--cut-name corpus_fit_l128_wtg96`, each with `--n-pairs 16384 --n-ood 8192 --steps 300 --batch-size 2048 --seeds 0 1 2`. Run `flagship_component_labels.py` for each cut with `--n-pairs 65536 --top-components 10`, then `flagship_component_ablation.py` with `--n-docs 8 --seeds 23 29 --batch-size 4 --threads 4 --top-components 2 --target-head-error .001 --calibration-docs 64`. The output JSON and rotation NPZ files in `full_width_results/` carry the cut name. The original checkpoint and corpora are referenced by path rather than copied.
