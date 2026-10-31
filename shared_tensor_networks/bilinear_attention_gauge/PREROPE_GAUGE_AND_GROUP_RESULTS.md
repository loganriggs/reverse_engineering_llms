# Exact-gauge legibility on the mixed pre-RoPE cut

This is the second stage of the attention-canonicalization experiment. The
first stage chose a rank-96 approximation of the 546M model's layer-0 head-0
QK1 bond by comparing uniform token pairs (weight-only), corpus pairs
(data-aware), and their mixture in pre- and post-RoPE coordinates. The
[metric factorial](ROPE_METRIC_FACTORIAL_RESULTS.md) found that the mixed
pre-RoPE cut gave the lowest full-model KL on the three tested corpora. Here
that cut is **fixed** while an exact orthogonal gauge rotates its 96 internal
coordinates. Every gauge computes the same reduced attention function; a
change in legibility metrics cannot be credited to improved reconstruction.

## Which prior makes components simpler?

`flagship_prerope_gauge_sparsity.py` compares the SVD gauge, Haar rotations,
factor-weight L1, ordinary pair-response activation sparsity, and a mixture
of ordinary and QK2-constant pair-response activation sparsity. Three starts
are fit on 32,768 uniform and 32,768 FineWeb training pairs, with independent
selection and test pairs. FineWeb test documents are disjoint from fit and
selection documents. Orthogonal rotations preserve the rank-96 bond to
relative error below 1e-9; an analytic gradient agrees with finite
differences to 3.8e-10 relative error.

On 32,768 held-out corpus pairs, the effective number of active coordinates
is 34.4 for SVD, 35.9 for a random gauge, 32.8 for factor-weight L1, 20.5
for activation sparsity, and 19.2 for the intervention-aware activation
prior. On held-out position gaps 32 and 48, the corresponding counts are
37.7 for SVD, 31.0 for activation, and 28.7 for intervention-aware
activation. Thus a data-aware legibility prior can make this *same reduced
function* much more concentrated. Low effective count alone is not an
interpretation: individual activation-sparse axes agree only about 0.20–0.21
in matched absolute correlation across independent starts, versus a 0.116
random-gauge comparison.

## Does a reusable functional group exist?

`flagship_prerope_response_grouping.py` groups axes by their complete-head
pair-response signatures. It uses 128 query anchors and 8,192 key anchors
per sampling law, and requires response correlation above the 99.9th
percentile of a random-gauge null. In the factor-weight gauge, the largest
corpus-response group has ten axes from the identity start; three Haar-start
optimizations yield groups of six, eight, and six axes. The ten-axis group
holds 25.8% of summed component-response energy. Activation-sparse gauges
usually produce zero or one null-calibrated graph edge, so the sparsest axes
are not automatically the best *grouped* explanation.

`flagship_prerope_group_projector.py` compares the corresponding subspaces,
independently of axis permutation and sign. The mean normalized projector
overlap between the identity-start group and the three Haar-start groups is
0.494; among Haar-start groups it is 0.490, compared with roughly 0.09–0.12
against random-gauge groups. Fixed pre-RoPE components also have near-zero
same-token response error under a global position translation (roughly
4e-6–6e-6 relative MSE), fixing a key failure of the previous post-RoPE
component analysis. This is evidence for a stable functional subspace
*within the selected reduced bond*, not a unique semantic mechanism.

## Post-fit label test

`flagship_prerope_group_exemplars.py` used only FineWeb test-half documents
17,500–18,750 to inspect high-energy pairs. Several prominent key tokens
were connective-like. After this inspection we fixed the set {as, because,
but, how, if, so, what, when, where, whether} and evaluated its key-side
pair-energy share on disjoint FineWeb documents 18,750–20,000 and separate
WikiText-103 and TinyStories corpora, with two random pair samples per
corpus. In `flagship_prerope_group_label_test.py`, the identity-start group
has 2.8x, 4.6x, and 2.6x the connective-key energy share of the complete
rank-96 cut on FineWeb, WikiText, and TinyStories respectively. The
Haar-start groups show smaller but consistent enrichment; random rank-10
subspaces do not show comparable enrichment.

These are **post-fit labels**, selected after reading discovery examples.
`flagship_prerope_group_frequency_null.py` compares the fixed set of 55
matching tokenizer IDs against 100 sets matched ID by ID for coarse token
form and nearby training-corpus frequency. For the identity-start group,
the observed enrichment is 2.81x versus a mean null of 1.62x on FineWeb,
2.61x versus 1.82x on TinyStories, and 4.56x versus 1.83x on WikiText.
One of 100 null sets beats the target on each FineWeb sample, 19 and three
beat it on the two TinyStories samples, and none beats it on either
WikiText sample. Other independent-start groups have weaker matched-null
rankings, especially on TinyStories. Thus the connective label has some
out-of-distribution support, but it is **not uniformly specific** against
matched words. Pair energy is not model-level causal importance or proof
that the group implements syntax. The two seeds share the same control
sets and corpus, so their null ranks are not independent p-values.

## Downstream intervention

`flagship_prerope_group_ablation.py` removes a fraction of each group
projector from the fixed pre-RoPE cut. The fraction is calibrated on separate
FineWeb pairs to give the same 0.001 relative natural pair-head error for
four discovered groups and three random rank-10 controls. It then runs each
edited head through all 18 layers of the 546M model and records per-document
KL relative to the unablated rank-96 cut. The larger run uses 24 document
windows per seed, two seeds, and three corpora (1,008 group/document records).
The four independently fit response groups have mean pairwise Pearson
correlation 0.957, 0.949, and 0.895 across their document-KL effect profiles
on FineWeb, WikiText, and TinyStories; the three random groups have 0.637,
0.506, and 0.436. Rank correlations are less consistent: 0.787, 0.493, and
0.871 for response groups versus 0.642, 0.584, and 0.484 for random groups.
Thus the strong Pearson agreement, particularly on WikiText, is partly
driven by high-effect windows. Mean KL is only about 3e-7–6e-7 per response
group, and the response groups do **not** consistently cause larger mean KL
than random groups. This is reproducible *effect geography*, not evidence
that the discovered group is more important or selectively causal. The
deletion is only matched in pair-head error; sequence-level head error and
full-model KL are outcomes, not matched quantities. See the JSON for the
complete per-document records.

## Interpretation and next gate

The evidence so far favors **a gauge-dependent sparse coordinate system
containing a more stable functional subspace**. It does not establish that
the subspace is a unique decomposition of attention, that individual axes
are reproducible, or that it has a selective downstream role. The next
decisive experiment should compare a connective-key-conditioned deletion
against frequency-matched key controls at the full-model level, and then
test a second independently discovered group. A learned one- or two-layer
toy language model with known multiple head rules would make selectivity and
missed rare rules directly measurable. Whole-task labels must remain hidden
from unsupervised fitting and selection.

All new scripts are CPU-only and read the existing checkpoint; the scripts
write JSON metrics (and gauge arrays where relevant) under
`full_width_results/`. The checkpoint is referenced rather than copied.
