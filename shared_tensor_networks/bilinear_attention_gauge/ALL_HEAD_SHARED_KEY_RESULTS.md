# A shared key dictionary across all first-layer attention heads

The trained two-head toy showed that reducing the *joint function* can
preserve a rule lost by separate headwise cuts. The prior 546M experiments
reduced one physical head or one bond at a time. This experiment asks whether
the nine first-layer heads of the 546M bilinear model reuse enough key-token
structure to justify **one shared dictionary**.

## Object and controls

`flagship_shared_key_dictionary.py` extracts the complete normalized,
pre-RoPE QK1 key table \(K\in\mathbb{R}^{50304\times1152}\) directly from
the checkpoint. The 1,152 columns are nine 128-dimensional physical-head
key vectors. Every vocabulary token is included in the reduction. The
weight-only condition minimizes uniform table reconstruction; the
data-aware condition gives half its weight to uniform vocabulary coverage
and half to token frequency in FineWeb training documents 0–15,000. There
are no task or semantic labels in either fit.

An SVD of the entire table gives one shared rank-\(m\) token dictionary and
head-specific readouts. The control takes a separate rank-64 SVD for each
head. Both conditions use the same uniform or usage-weighted metric. The
factor-entry budget is \(Vm+1152m\) for a shared dictionary and
\(9(Vr+128r)\) for nine separate rank-\(r\) dictionaries. At \(m=256\)
versus \(r=64\), these are **13.17 million versus 29.05 million entries**:
the shared representation uses 45.3% as many. This budget counts lookup
factors and readouts, not runtime FLOPs or a gauge-invariant function rank.
It is a compression of the **expanded token-to-key table**, not a claim that
these factors use fewer parameters than the original embedding-plus-linear
weight implementation.

The projection implementation agrees with the existing per-head model path
to within 2e-5 on selected token IDs and three heads. Held-out pair tests
include RoPE, both QK branches, OV, the sum across all nine heads, and the
1,152-dimensional output projection. They use independent uniform token
pairs, FineWeb documents 17,500–20,000, WikiText-103, and TinyStories.

## Key geometry and held-out pair response

In the uniform key-table norm, shared rank 256 leaves 0.1871 relative
squared error, versus 0.1920 for nine separate rank-64 dictionaries, despite
the lower factor budget. Under the usage-weighted table norm, shared rank
256 leaves 0.1499, versus 0.1603 for the usage-weighted separate fits.
This is direct evidence of cross-head overlap in the native key table.

On a fresh 8,192-pair sample per law, relative squared error of the
**complete summed first-layer output** is:

| Test pairs | Shared uniform 256 | Separate uniform 64/head | Shared usage 256 | Separate usage 64/head |
|---|---:|---:|---:|---:|
| Uniform tokens | 0.02582 | 0.02553 | 0.02692 | 0.02664 |
| FineWeb | 0.01818 | 0.01456 | **0.00786** | 0.00912 |
| WikiText-103 | 0.03742 | 0.03272 | **0.01740** | 0.02201 |
| TinyStories | 0.02523 | 0.02180 | **0.00894** | 0.01172 |

The original independent 2,048-pair panel gives the same qualitative
usage-weighted ordering on all three natural corpora. Uniform fitting
better protects uniformly sampled pairs, while usage weighting substantially
improves natural-pair response. The shared usage rank-256 fit beats the
larger separate usage fit on every natural pair panel, but this pair metric
is only a local approximation measure.

## Full-model propagation

`flagship_shared_key_full_model.py` applies the same pre-RoPE dictionaries
to the layer-0 QK1 keys of all nine heads and propagates through all 18
layers. Its identity-basis round trip differs from the native model by at
most 2.9e-5 in logits. The evaluation uses 32 independently sampled
document windows per corpus, paired across methods. The reference is the
unaltered native model; no reduced-circuit baseline is involved.

The full-model result is **less favorable to sharing than the pair test**.
For usage-weighted rank 256 versus separate rank 64/head, the shared fit
wins 11/32 FineWeb windows, 22/32 WikiText windows, and 27/32 TinyStories
windows. Mean native-to-edited KL ratios (shared / separate) are 1.37,
0.93, and 0.89 respectively. On FineWeb, the lower pair error therefore
fails to predict downstream fidelity. WikiText has two extreme windows with
many `<|endoftext|>` boundaries, formula markup, or segmented entity text;
both methods incur large KL there, so median and paired results matter more
than its mean alone.

This is a positive compression result for the joint key table, but not a
uniform full-model advantage. A rank-384 shared usage dictionary improves
full-model KL further on all three corpora at a higher, though still lower
than separate-rank-64, factor budget (19.76 million entries, or 68.0% of
the separate budget). It wins 26/32, 29/32, and 31/32 paired windows on
FineWeb, WikiText, and TinyStories respectively. Mean CE differences and
argmax flips do not establish a comparable task-performance advantage;
this is a fidelity result. The full JSON retains every window
and method so the mean, median, and tails can be inspected separately.

A second full-model panel uses seeds 41 and 43 on **new document IDs**,
excluding every document in the first panel. The rank-384 shared usage
dictionary again beats the separate rank-64/head usage fit on 30/32 FineWeb,
30/32 WikiText, and 32/32 TinyStories windows. Its mean KL is 8.13e-5,
0.00230, and 4.66e-5, versus 1.50e-4, 0.00366, and 1.37e-4 for the
separate fit. Rank 256 wins 16/32, 27/32, and 27/32; its FineWeb mean KL
is slightly worse (1.61e-4 versus 1.50e-4). Across both disjoint panels,
rank 384 wins **56/64, 59/64, and 63/64** paired windows in corpus order.
WikiText still has a heavy right tail, although the later panel's maximum
KL is below the earlier extreme windows. CE changes vary in sign and are
small or outlier-sensitive relative to the task; no task-accuracy claim is
made from them.

| Corpus | First panel: shared 384 / separate 64 mean KL | New-doc panel: shared 384 / separate 64 mean KL | Shared wins across both panels |
|---|---:|---:|---:|
| FineWeb | 1.33e-4 / 1.73e-4 | 8.13e-5 / 1.50e-4 | 56/64 |
| WikiText-103 | 0.0176 / 0.0294 | 0.00230 / 0.00366 | 59/64 |
| TinyStories | 5.16e-5 / 1.15e-4 | 4.66e-5 / 1.37e-4 | 63/64 |

## Scope and next test

The shared dictionary is formed from the model's **native pre-RoPE QK1 key
coordinates**. It is not invariant under arbitrary branch swaps, head
redistribution, or non-orthogonal changes of QK factorization. Nor does
compression alone identify a human-readable computation. Its practical
value is a lower-dimensional common coordinate system that can be subjected
to explicit sparsity, group, and causal priors across several heads. The
negative FineWeb full-model result is a warning that key-table and pair
objectives do not fully determine downstream behavior.

The next experiment should compare a function-aware shared dictionary
against this token-table SVD at the **same factor budget**, train on diverse
pair or sequence laws, and hold out whole contexts and QK2 interventions.
Only then should a sparse orientation be interpreted as reusable
cross-head computation. All saved bases and evaluation metrics are under
`full_width_results/`.
