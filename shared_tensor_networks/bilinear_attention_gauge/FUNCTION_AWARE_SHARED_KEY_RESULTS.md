# Function-aware reduction of the nine-head shared key dictionary

The preceding [all-head key dictionary experiment](ALL_HEAD_SHARED_KEY_RESULTS.md)
found substantial overlap in the native pre-RoPE QK1 key tables, but the
rank-256 data-weighted SVD gave inconsistent downstream improvement. Here the
same rank-256 shared projection is fitted to the *summed pair-output function*
of all nine heads. QK2, values, output weights, and the query side remain
frozen. This is a stronger functional metric on the same structured cut,
**not** a unique canonical form of the external attention function.

## The fit and its controls

`flagship_function_aware_shared_key.py` starts from the usage-weighted shared
SVD and optimizes an orthonormal 1,152-by-256 basis. For each token pair and
relative position, it measures squared error in the complete 1,152-output
dimension, summed across heads and normalized by original output energy.
The objective equally weights uniform vocabulary pairs and FineWeb pairs
from training documents 0–15,000. Selection uses uniform pairs and disjoint
FineWeb documents 15,000–17,500. Pair tests use uniform draws, FineWeb
documents 17,500–20,000, WikiText-103, and TinyStories. A compact 9-by-9
OV Gram computes the exact nine-head output norm without allocating the full
output for every training pair. Direct RoPE and OV checks pass at roughly
1e-5 absolute score and 2e-4 relative output-norm tolerance, respectively.

Two fit sizes were run at the same rank and learning rate: 8,192 training
pairs per law, selected at step 20; and 32,768 per law, selected at step 130.
The larger fit uses 8,192 validation and test pairs per law. On its held-out
pair panel, relative squared errors of the natural nine-head response are:

| Test law | Initial usage SVD 256 | Function-aware 256 |
|---|---:|---:|
| Uniform pairs | 0.03209 | 0.03217 |
| FineWeb | 0.00863 | 0.00717 |
| WikiText-103 | 0.01859 | 0.00987 |
| TinyStories | 0.00917 | 0.00599 |

The uniform pair cost stays approximately flat while natural-language pair
fidelity improves, including corpora absent from the fit. The small-data
fit improves less; fit scale and validation selection matter.

## Propagation through all 18 layers

`flagship_shared_key_full_model.py` inserts each basis in layer-0 QK1 for
all heads, then compares final logits with the unaltered 546M model on
independent 128-token windows. Identity projection reproduces native logits
to within 2.4e-5. No held-out window or document is used to fit a basis.
The comparison is paired within each panel, with document exclusions from
all earlier panels. On the first 32-window/corpus panel that compares both
fit sizes, mean native-to-edited KL per token is:

| Corpus | Usage SVD 256 | Large function-aware 256 | Usage SVD 384 |
|---|---:|---:|---:|
| FineWeb | 0.000171 | 0.000120 | 0.0000788 |
| WikiText-103 | 0.01676 | 0.00199 | 0.01032 |
| TinyStories | 0.0000959 | 0.0000647 | 0.0000418 |

The large function-aware rank-256 fit beats the usage-SVD rank-256 fit on
27/32 FineWeb, 31/32 WikiText, and 25/32 TinyStories windows. It also beats
the smaller function-aware fit on 30/32 windows in each corpus. The rank-384
SVD is better on most FineWeb and TinyStories windows, though it uses 50%
more shared basis dimensions. Both shared bases use fewer factor entries
than nine independent rank-64 headwise token dictionaries. A separate
32-window/corpus panel, using seeds 71 and 73 and excluding documents from
all three earlier panels, confirms the same-rank comparison: 26/32 FineWeb,
28/32 WikiText, and 29/32 TinyStories wins for function-aware rank 256.
Its mean KL values (function-aware / usage SVD rank 256) are 0.000102 /
0.000127, 0.000334 / 0.000929, and 0.0000797 / 0.000123 respectively.
No documents overlap the earlier function-evaluation panels. The WikiText
mean is sensitive to document composition, so the paired wins and medians
should be read alongside it.

These are **faithfulness** comparisons. They do not establish a change in
language-task quality or a legible component structure.

## Branch intervention: a limit of natural-response fitting

`flagship_function_aware_shared_key_interventions.py` draws another 8,192
held-out pairs per law and sets the QK2 score to one for every pair, leaving
QK1, OV, and head summation intact. This changes the causal role of QK1
without changing the fitted basis. Relative nine-head output error is:

| Test law | Usage SVD 256, natural | Function-aware 256, natural | Usage SVD 256, QK2=1 | Function-aware 256, QK2=1 |
|---|---:|---:|---:|---:|
| Uniform | 0.02987 | 0.02975 | 0.09725 | 0.09935 |
| FineWeb | 0.00915 | 0.00740 | 0.03427 | 0.03515 |
| WikiText-103 | 0.01609 | 0.00988 | 0.04875 | 0.04752 |
| TinyStories | 0.00935 | 0.00601 | 0.02820 | 0.02807 |

The natural-response gain largely disappears under this branch edit. The
rank-384 SVD is stronger still, as expected from its larger dimension.
Thus optimizing the observed combined function does not automatically retain
the branch behavior needed for intervention-faithful mechanistic claims.

A matched rank-256 follow-up puts half the training and validation weight
on the QK2-constant response. On a *fresh* 8,192-pair-per-law panel, compared
with the natural-only function fit, edited-response error falls from 0.09935
to 0.09752 on uniform, 0.03515 to 0.03384 on FineWeb, 0.04752 to 0.04601
on WikiText, and 0.02807 to 0.02700 on TinyStories. Natural-response error
also falls slightly on uniform/FineWeb, but rises slightly on WikiText and
TinyStories. This is a modest improvement, not an intervention-faithful
decomposition: QK2-constant errors remain much larger than natural errors,
and the edit used in training is known to the fitter. The full-model test
compares this branch-aware basis on a further disjoint document panel.

That full-model panel uses seeds 81 and 83, 32 windows per corpus, and no
documents from any prior panel. Mean KL for the natural-only / 50:50
intervention-aware rank-256 fits is 0.000144 / 0.000142 on FineWeb,
0.00922 / 0.02154 on WikiText, and 0.0000774 / 0.0000823 on TinyStories.
The 50:50 fit wins 24/32, 14/32, and 17/32 paired windows, respectively.
Both function fits beat the rank-256 usage-SVD control on most windows.
Thus the edited-pair gain does not yield a general downstream-KL gain; the
WikiText mean is particularly tail-sensitive. Full-model *interventions*
have not yet been evaluated.

A branch-only fit (100% QK2-constant training weight) does **not** improve
edited-response error over the 50:50 fit on any of the three natural-language
test corpora: 0.03386 versus 0.03384 on FineWeb, 0.04646 versus 0.04601 on
WikiText, and 0.02724 versus 0.02700 on TinyStories. It worsens the natural
response on all three. This small sweep suggests an interior compromise;
there is no evidence yet for a large intervention-faithfulness gain from
merely increasing the edit weight.

## Scope and next legibility step

This study chooses a *metric* for a structured low-rank cut. It does not
resolve factorization non-identifiability, exact QK branch swaps, or how to
rotate the retained subspace into human-readable components. A subsequent
legibility comparison should hold the selected subspace and approximation
budget fixed while trying sparsity in weights, token responses, group gates,
and causal intervention signatures. Compare component stability, causal
selectivity, held-out corpora, and task recovery separately. Toy multi-head
rules with known shared computations give recovery ground truth; the real
model remains the transfer test.

The fixed-projection orientation comparison is now reported in
[all-head gauge priors](ALL_HEAD_GAUGE_PRIOR_RESULTS.md). It tests weight and
token-response sparsity without spending reconstruction error; semantic and
selective causal validation remain distinct gates.

Artifacts: `full_width_results/all_head_function_aware_shared_key_s641_n8192_t160.*`,
`all_head_function_aware_shared_key_s642_n32768_t160.*`,
`all_head_shared_key_full_model_n16_s51-53_newdocs_function.json`,
`all_head_shared_key_full_model_n16_s61-67_newdocs_function_extra1.json`,
`all_head_shared_key_full_model_n16_s71-73_newdocs_function.json`,
`all_head_shared_key_full_model_n16_s81-83_newdocs_function_extra1.json`,
`all_head_function_aware_shared_key_qk2_constant_n8192.json`, plus the
intervention-aware fits `all_head_function_aware_shared_key_s644_n32768_t160_branch50.*`
and `all_head_function_aware_shared_key_s645_n32768_t160_branch100.*`.
