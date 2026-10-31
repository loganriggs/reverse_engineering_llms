# A nonlinear count-language test of function reduction and sparse gauges

This experiment asks whether the weight-only reduction → sparse orientation → causal validation pipeline still works when the target is **not additive** over token counts. The model has two bilinear-attention heads whose outputs feed a bilinear MLP. It predicts two count-comparison bits, A and B, plus XOR(A,B). A common query reads A/B in ordinary order; a rare query swaps them. The rare-query A=0, B=1 *combination* is absent from training and validation. Each of ten seeds uses 50,000 training, 10,000 validation, and 20,000 independent test sequences with variable-length key multisets. Seeds 0–3 and 5–9 trained 1,500 steps; seed 4 trained 4,000 after its rare-holdout failure persisted. The test includes that failure.

The exact first-layer table has shape query 2 × key 6 × head 2. It is computed directly from embeddings and QK/value weights, including QK RMS normalization. For a sentence with key-count vector (c\), attention heads are (h=T_qc\). Passing those reconstructed heads through the *original* bilinear MLP matches direct model heads and logits to below (10^{-10}\) maximum absolute error in every seed. No sampled sentences are required to construct (T\). The full first-layer table is linear in counts; the downstream MLP makes its logits quadratic in counts. This is why the model can represent XOR even though the earlier additive count toy needed three independent rank directions for three independent output bits.

## Reduction metric: weights only versus unlabeled data

Flatten (T\) to a 4 × 6 query/head-by-key matrix. Its mean singular values are **8.15, 7.22, .304, .097**. The two large directions encode most of the learned function. Ten-seed means on separate 20,000-sequence tests:

| Rank-two fit | Full-test joint accuracy | Rare-combination joint accuracy | Relative head-output MSE | Relative final-logit MSE |
|---|---:|---:|---:|---:|
| Exact model | .99599 | .88938 | ~0 | ~0 |
| Uniform token-cell SVD, weights only | .99616 | .89489 | .00359 | .00443 |
| Training token-pair frequency | .99614 | .89838 | **.00243** | .00249 |
| Half uniform / pair frequency | .99617 | .89540 | .00261 | **.00241** |
| Training-sequence head-output fit | .99618 | .89589 | .00297 | .00418 |

The latter three use 50,000 **unlabeled training sentences** per seed; no rare held-out combination enters the fit. The sequence objective is expected squared head-output error under training count covariance. All rank-two fits have essentially the same task accuracy. The pair-frequency fit improves measured output fidelity, but the distinctions are small and do not establish a better interpretation. At rank one, all four metrics lose the task: full-test joint accuracy ranges **.442–.476**, with relative final-logit MSE **.730–.767**. Rank one cannot retain both independent count contrasts, so this is a severe capacity constraint, not evidence that one metric is generally best.

The exact model's mean rare accuracy is lower than its ordinary-test accuracy because **seed 4 gets 0% rare joint accuracy after 4,000 steps**, despite ~99.98% validation accuracy. Seed 2 gets 90.6%; the other eight are approximately 99.2–100%. A compression that preserves seed 4's wrong rare behavior is faithful to the model, not a solution to the intended task. Means of per-seed rare scores are reported, not pooled rare examples.

## Legibility prior: which sparsity recovers reusable variables?

Within the fixed rank-two function, an orthogonal rotation changes the component basis while leaving every attention output and downstream logit unchanged. Four gauges were selected: raw SVD; minimize L1 of the **key-token dictionary** using weights only; minimize L1 of the **query/head row factors** using weights only; or minimize L1 of **latent activations** on 10,000 unlabeled training sentences. All orientation search is label-free. A/B axis alignment and deletions are evaluated only afterward on independent 30,000-sequence tests.

| Gauge | Mean A/B axis cosine | After deleting matched A: A / B / XOR accuracy | After deleting matched B: A / B / XOR accuracy |
|---|---:|---:|---:|
| SVD | .904 | .559 / .933 / .588 | .935 / .567 / .589 |
| **Key-dictionary L1** | **.99870** | **.501 / .992 / .499** | **.999 / .499 / .503** |
| Query/head-factor L1 | .861 | .604 / .914 / .570 | .899 / .611 / .609 |
| Unlabeled activation L1 | .711 | .736 / .763 / .642 | .760 / .742 / .646 |

The key-L1 basis reaches at least **.9965** alignment in every seed; its key L1 falls from 3.666 (SVD) to 2.924 on average. Random orthogonal rotations have mean alignment .897. The deletion is of an explanatory component of the **reduced first-layer function**, before the unchanged nonlinear MLP. It is a causal intervention on this replacement model, not evidence of a corresponding isolated physical head or neuron. The matched A/B names use known token contrasts *after* fitting. For this count grammar, activation L1 selects a mixed basis: sparse use on the observed data is a different prior from sparse token rules.

Expanding the original bilinear MLP in the two sparse coordinates gives a direct compositional check. For each query and seed, the absolute coefficient of the XOR cross monomial \(z_Az_B\) exceeds the sum of the absolute two square coefficients by a factor **3.32–49.82**. Keeping only the learned signed cross monomial (zero threshold, no refit) predicts XOR with **100%** accuracy on a further independent 30,000 variable-count sentences per seed, including the rare combination. The full XOR logit also has linear, square, and constant terms; the cross-only diagnostic establishes a sufficient rule on this test distribution, not a globally exact symbolic identity on all real counts. Seed 4 illustrates the distinction sharply: its XOR cross-term generalizes while its rare A/B output ordering fails.

## Physical-branch red team

The ordinary semantic two-dimensional key space is not a complete explanation of the physical attention circuit. Replacing the QK1 or QK2 score with the constant 1 defines two branch interventions; doing both is a third, held-out combination. The following are ten-seed mean relative table MSEs of head-preserving key-space projections. The logit column passes projected heads through the original bilinear MLP and reports relative squared deviation from the corresponding *intervened* model's logits on 20,000 independent test sentences per seed.

| Key-space fit | Rank | Original table | QK1 constant | QK2 constant | Both constant | Both-constant logit MSE |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary function only | 2 | .0013 | .4945 | .5016 | .5461 | .8464 |
| Original + both single edits | 4 | .0009 | .0031 | .0035 | .0627 | .0747 |
| Original + both single edits | 5 | .0002 | .0007 | .0005 | .0395 | .0427 |
| All four conditions | 5 | .0009 | .0016 | .0021 | .0014 | .0031 |

Every fit uses exact weight-derived tables, no task labels, and no sampled inputs. Each fitted condition is normalized before stacking so its total scale does not dominate. The double edit is absent from the first three fits. Thus the nonlinear output does not rescue an ordinary-function dictionary from unseen branch interventions. The extra dimensions describe the specified counterfactual family; they do not prove the learned axes are physical nodes. The double-edit failure of the single-edit fit also cautions against treating individual branch coverage as compositional coverage.

This extends the earlier additive toy in a useful way: the weight-only sparse function basis reveals two reusable count variables, and the downstream bilinear MLP combines them to compute a third, nonlinear rule. It remains a small synthetic task with designed token families and a simple first-layer count table. It does not show that key L1 is the right prior for natural language or for the 546M model, nor that function-level axes identify physical QK branches. The next discriminating test is a natural-language one/two-layer model or the existing 546M head with matched weight-only/data-aware reductions, then predeclared component labels and held-out downstream/branch interventions. Keep ordinary-function fidelity, task prediction, and physical-intervention fidelity separate.

Reproduction: `train_xor_count_language.py` creates the ten checkpoints and training histories; `analyze_xor_count_language.py`, `xor_count_metric.py`, `xor_count_sparse.py`, `xor_count_prior_grid.py`, `xor_count_polynomial.py`, and `xor_count_intervention.py` create their matching JSON records. All scripts run on CPU with PyTorch, NumPy, and SciPy.
