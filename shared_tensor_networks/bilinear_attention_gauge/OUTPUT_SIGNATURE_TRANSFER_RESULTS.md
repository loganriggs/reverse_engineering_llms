# A punctuation-associated key axis does not yield a stable output rule

The [unseen-token finite test](FULL_CONTEXT_UNSEEN_TOKEN_RESULTS.md) found
that deleting full-context QK1-key axis 7 changes final logits much more
for punctuation keys than frequency-matched word keys. That establishes
conditional *effect size*, not what the effect says. This experiment asks
whether the deletion predicts a particular signed output direction or a
named output-token category.

Discovery uses 64 disjoint FineWeb **validation** documents and four of the
eight unseen punctuation/word token pairs, 16 documents per pair. The other
four pairs and all test corpora are held out. For each same-position input
pair, run the fixed rank-256 model with and without deleting axis 7, then
subtract the two deletion-induced final-position full-vocabulary logit
vectors. The mean of these difference vectors is frozen as an output
signature. A comparison uses the previously selected pair-response axis 16
at its training-calibrated partial deletion strength. No model weights,
rank-256 subspace, or coordinate rotations are refitted.

The discovery signature is weakly coherent. The squared norm of the mean
difference divided by mean per-document squared norm is **0.030** for
full-context axis 7 and **0.013** for pair-response axis 16. On two
32-document splits, the full-context mean directions have cosine `0.30`
(first/second half) or `0.38` (even/odd); the pair-response means have
cosine `−0.08` or `0.00`. The top positive and negative output tokens in
the saved discovery report do not form a clear, specific language rule.

Before looking at the test directions, we froze two readouts:

1. Cosine of each held-out difference vector with the discovery mean.
2. The sign of the deletion-induced **punctuation-output probability mass
   minus space-word-output probability mass**, contrasted between
   punctuation and word key variants. Axis 7's discovery sign is positive;
   pair-response axis 16's is negative.

On 12 documents per corpus using the four held-out token pairs:

| Corpus | Axis 7 mean cosine [95% document bootstrap] | Positive cosines | Axis 7 category-contrast mean [95% interval] |
|---|---:|---:|---:|
| FineWeb test | .114 [.029, .196] | 8/12 | +.000602 [+.000159, +.001183] |
| WikiText-103 | .054 [−.055, .157] | 8/12 | −.000165 [−.000672, +.000178] |
| TinyStories | .077 [−.054, .185] | 8/12 | −.000169 [−.000667, +.000133] |

The category contrast matches the discovery sign in 10/12, 5/12, and
7/12 documents respectively. The external-corpus intervals span zero for
both readouts, and the category point estimates reverse. The pair-response
axis also lacks a stable signed output direction: mean held-out cosines are
`−.022/−.031/+.010`, with intervals spanning zero in all three corpora.
Its category-contrast signs are inconsistent, despite its highly repeatable
punctuation-conditioned **response magnitude**.

The rank-256 reduction was previously checked on these same edited test
inputs: native-versus-reduced punctuation-minus-word response relative MSE
is `.00119/.00165/.00040` across the three corpora. Gauge rotations in
the discovery and holdout each change unedited logits by less than `3.0e-5`.
Thus the
failure to find a broad output rule is not explained by a grossly
unfaithful token-edit baseline. It may reflect context-dependent routing,
multiple output functions inside one coordinate, or simply an objective
that optimizes local sensitivity sparsity without a stable output meaning.
These are hypotheses, not conclusions from the current readout.

**Decision.** The evidence supports “punctuation-associated QK1-key
sensitivity” as a functional description of axis 7. It does not support
calling it a punctuation-prediction circuit or a unique physical component.
The punctuation-axis detour has met its stopping criterion: further gauge
sparsity or nearby token edits are unlikely to settle that identity. The
next canonicalization experiment should change the behavioral target:
discover candidate tasks with explicit output predictions and compare
weight-only, data-aware, and causal priors at matched fidelity and rank,
holding out whole tasks and token combinations. Alternatively, a specific
position-to-position path hypothesis would justify a targeted mediation
experiment on this axis.

Reproduce with `flagship_output_signature_discovery.py --n-docs 64 --seed
1431`, then `flagship_output_signature_holdout.py` and
`analyze_output_signature_transfer.py`. Saved discovery JSON/NPZ,
held-out per-document records, and bootstrap summary use the stem
`all_head_output_signature_discovery_n64_s1431_l64` under
`full_width_results/`.
