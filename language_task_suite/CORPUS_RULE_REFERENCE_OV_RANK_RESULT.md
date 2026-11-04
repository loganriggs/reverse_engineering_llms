# OV reductions on ordinary-order rule transfer

The [fresh-source transfer panel](CORPUS_RULE_REFERENCE_TRANSFER_RESULT.md) established that the saved joint-trained bilinear models respond to list and bracket edits in ordinary FineWeb document order. This follow-up asks what a weight-only or data-aware rank reduction preserves on exactly those same panels. It is a functional-fidelity test of an OV subcircuit, not a full attention canonicalization or evidence that its coordinates are human-interpretable.

For each of two depth-2, width-64, four-head models, the [evaluator](eval_corpus_rule_reference_ov_rank.py) holds QK, MLP, embeddings, and unembedding fixed. It compares three reductions at **16 retained OV channels per attention layer**: rank four in each of four heads from ordinary weight SVD of the head's effective `O_h V_h` product; the same rank using an activation covariance estimated on separate-rule training windows; and one complete 16-channel head chosen by effective-OV Frobenius norm. Both fresh test panels (363k–368k and 368k–373k FineWeb source offsets) remain unseen by the covariance fit. No test text is used to construct the reductions. This channel budget does not equate computational latency or parameter-storage format.

Bracket results below are ordered by the two independent panels. Pair exact top-1 requires the correct full-vocabulary token in both natural and edited arms. Effect MAE compares, source by source, the reduction's natural-to-edited answer-logit-margin response with the dense teacher. Site KL compares complete next-token distributions at both answer sites; background KL is sampled every 32 positions in the selected natural rule-bearing chunks, excluding the answer. Lower MAE/KL means greater teacher fidelity, whereas higher exact top-1 means better answer accuracy.

| Seed | Method | Pair exact top-1 | Effect MAE | Site KL | Background KL |
|---:|---|---:|---:|---:|---:|
| 71503 | Dense | .326 / .300 | 0 / 0 | 0 / 0 | 0 / 0 |
| 71503 | Weight SVD | .435 / .431 | 4.39 / 4.58 | 1.43 / 1.38 | .482 / .483 |
| 71503 | Data-weighted SVD | .279 / .266 | 1.31 / 1.25 | .154 / .149 | .040 / .040 |
| 71503 | One full head | .284 / .286 | 9.43 / 9.21 | 2.44 / 2.43 | .613 / .627 |
| 71504 | Dense | .305 / .307 | 0 / 0 | 0 / 0 | 0 / 0 |
| 71504 | Weight SVD | .385 / .384 | 3.25 / 3.10 | .651 / .546 | .225 / .227 |
| 71504 | Data-weighted SVD | .286 / .278 | 1.67 / 1.64 | .136 / .134 | .030 / .032 |
| 71504 | One full head | .471 / .492 | 7.30 / 7.00 | 2.32 / 2.18 | .526 / .511 |

For lists, dense pair exact is .940/.984 and .940/.902 for seeds 71503/71504. Weight SVD gives .940/.967 and .760/.721; data-weighted SVD .900/.918 and .800/.754; one-head pruning .960/.918 and .780/.705. Thus no reduction consistently preserves even list accuracy. The full per-panel behavioral, character, two-choice, causal-direction, selected-chunk CE, effect-correlation, and KL measures are in the [aggregate JSON](results/corpus_rule_reference_ov_rank4_start363000_373000.json).

The two SVD objectives make a genuine empirical difference. Data weighting sharply reduces background KL and preserves bracket edit-response magnitudes much better. Weight SVD improves exact bracket-token accuracy in both seeds and panels, even though it departs more from the teacher distribution. In one seed, retaining a single whole head improves exact bracket accuracy further while giving especially poor teacher fidelity. Exact answer accuracy, imitation of the trained model, and causal-effect fidelity are therefore distinct objectives; an accuracy gain after truncation does not identify the original circuit. The correct-direction causal statistic stays near saturation for these models, so its magnitude and pairwise agreement are more informative.

This result has two important limits. First, the reduction only changes OV products; it does not resolve the QK×QK gauge, interactions among heads, or the sparse human-legibility objective. Second, these models have been trained on enriched rule windows and do not have a full symbolic oracle for natural prose. The next weight-method experiment should move beyond OV-only truncation and use predeclared acceptance thresholds for **both** exact task behavior and teacher causal fidelity before optimizing sparsity or local fan-in. It should run on at least one additional independently verified language rule family, so the choice of a legibility prior does not overfit lists and bracket punctuation alone.
