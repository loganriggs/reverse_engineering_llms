# Weight-only versus data-aware reduction of the exact token-pair function

The [first-layer construction](V8_FIRST_LAYER_TOKEN_FUNCTION.md) gives an
exact, data-free collapsed score matrix `S_h[a,b;31,7]` and OV value vector
`P_h[b,:]` for each head and all 368×368 token pairs. Here we compare three
rank-matched reductions of that **one position-pair function**:

1. **Score SVD:** best rank-`r` Frobenius approximation to `S`.
2. **Output SVD:** best rank-`r` approximation to
   `S diag(||P_b||)`; dividing the key factors by `||P_b||` produces a
   rank-`r` score approximation that minimizes uniform-vocabulary squared
   error of the full head output tensor `T[a,b,o]=S[a,b]P[b,o]`.
3. **Data-aware:** initialize from output SVD and optimize the same two
   `368×r` factors on token pairs actually sampled at positions `(31,7)`
   from the v8 stream. We compare no global prior with a fixed 0.1-weight
   uniform-output prior. Development data selects the update step; test and
   uniform-vocabulary pairs are untouched by fitting.

All methods use the same rank and storage budget for the score factors;
the OV map is held fixed. The [runner](eval_v8_token_function_reductions.py)
uses 4,096 training, 2,048 development, and 4,096 test streams, plus a
second independent 4,096-stream test panel. The training sample covers
2,030 distinct token pairs; 1,550 of 4,096 test occurrences belong to
**pairs unseen in training**. The split is by generated stream, not by
whole rule or section–color combination. These are the task-rich synthetic
v8 rates, not FineWeb prevalence.
The final runs use single-thread deterministic CPU math: for each teacher,
the fitted factor tensors are bitwise identical when rerun with the other
test seed. This checks that test selection does not feed back into fitting.

The table shows relative L2 error of the *head output contribution* on
sampled test pairs; lower is better. It uses head 1, where all methods move
meaningfully. A/B are the two independent test seeds 61001/61002:

| Teacher, rank | Score SVD | Output SVD | Data-aware, no prior | Data-aware, 0.1 prior |
|---|---:|---:|---:|---:|
| Seed 1, rank 8 | .388 / .355 | .376 / .356 | .310 / .291 | **.175 / .187** |
| Seed 1, rank 16 | .185 / .170 | .142 / .129 | .142 / .129 | **.089 / .084** |
| Seed 4, rank 8 | .094 / .092 | .041 / .041 | .047 / .038 | **.021 / .022** |
| Seed 4, rank 16 | .037 / .037 | .022 / .021 | .022 / .021 | **.012 / .013** |

The output-aware weight-only metric already helps, especially for seed 4.
The data-aware prior improves average sampled-pair test error in 7/8
head-rank combinations for seed 1 and 6/8 for seed 4 (ties in the
remainder), on both test panels. It does **not** dominate every head or
distribution. The unregularized version can overfit sampled pair support:
seed-1 head 1 rank 8 has test error .310 but unseen-pair error .938,
versus .345 for output SVD. Its uniform-vocabulary error is .503 versus
.240. The 0.1 prior lowers the sampled-pair test error further to .175,
but still has unseen-pair error .530 and uniform error .288—worse than the
weight-only output SVD. On seed-4 head 1 rank 8 the corresponding sampled
error improves .041→.021, while unseen-pair error is .059→.061 and
uniform-vocabulary error .127→.132. The second test panel shows the same
direction of this tradeoff.

This is a concrete **distribution-relative canonical reduction**: the
same exact functional tensor supports different useful low-rank summaries
depending on the metric. Fitting a data-weighted factorization can improve
the function where the generator puts probability mass, while making it
less faithful to unseen and uniformly weighted token pairs. The global
prior mitigates severe overfitting but does not remove the coverage cost.
The result is specific to a fixed first-layer position pair. It does not
show that the reduced attention preserves next-token KL, exact-oracle task
accuracy, key/relevant-color intervention effects, or interpretability of
individual factors.

The next experiment should place these rank-matched approximations back
into the first-layer attention computation at task-relevant query/key
positions, with position dependence handled explicitly. Evaluate downstream
model KL and supported key/color edits on held-out section–color
combinations, then compare whether sparsity in score factors, OV factors or
causal responses gives the most legible faithful representation. Avoid
using test-pair frequency to choose the metric.

The [seed-1 A](results/v8_token_reductions_seed1_q31_k7.json),
[seed-1 B](results/v8_token_reductions_seed1_q31_k7_test61002.json),
[seed-4 A](results/v8_token_reductions_seed4_q31_k7.json), and
[seed-4 B](results/v8_token_reductions_seed4_q31_k7_test61002.json)
records contain every head, both ranks, train/dev/test/unseen/uniform
errors, development curves, and links to the fitted factor checkpoints.
