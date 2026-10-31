# Does the toy's key-side sparsity prior transfer to the 546M model?

The [nonlinear count-language toy](XOR_COUNT_CANONICAL_RESULTS.md) found that sparsity of a weight-derived key-token dictionary recovered two reusable variables, whereas sparsity of their activations or output factors mixed them. This experiment tests the analogous prior on the already validated rank-64 layer-0/head-0 QK1 cut of the 546M bilinear language model. It holds the bond matrix (X=AB^\mathsf{T}\) fixed and searches orthogonal component rotations (R\), so ((AR)(BR)^\mathsf{T}=X\). Every gauge represents the *same* attention function and full-model logits until a component is individually edited.

This is a different object from the toy's complete token-pair table: (B\) is the key side of one internal physical QK1 bond, not a factor of the entire attention function. Accordingly, a negative result could reflect the object chosen rather than failure of key sparsity on all possible function-level quotients.

Two L1 priors minimize mean \(\lVert v^\mathsf{T}BR\rVert_1\), where (v\) is the post-RoPE key vector. One uses uniformly sampled valid token/position pairs; it is a Monte Carlo approximation to a **weight-only domain measure** and does not use a text corpus. The other samples FineWeb training documents 0–14,999. Each uses 32,768 fit and 32,768 disjoint selection pairs, three optimizer seeds, and 200 CPU steps. FineWeb test documents 17,500–19,999 and independent WikiText-103/TinyStories samples supply 8,192 test pairs per corpus. Existing SVD, random, and pair-activation-sparse gauges are matched controls on the identical fixed cut. The uniform measure is sampled, not an exact full-vocabulary enumeration; positions and self-pair constraints remain part of the declared domain.

The maximum orthogonality and bond-reconstruction residuals are recorded in `head0_key_sparse_gauge.json` and checked below \(10^{-9}\). Therefore differences below concern component orientation, not compression fidelity or full-model next-token behavior.

| Gauge | FineWeb test key L1 | FineWeb effective active components | WikiText effective active | TinyStories effective active |
|---|---:|---:|---:|---:|
| SVD | .8945 | 23.13 | 23.01 | 22.91 |
| Random rotation | .8986 | 24.21 | 24.51 | 24.39 |
| Pair-activation sparse | .8905 | **19.36** | 20.71 | **17.93** |
| Uniform-domain key L1 | .8832 | 19.69 | **20.21** | 18.90 |
| FineWeb key L1 | **.7958** | 22.33 | 21.74 | 19.27 |

The prior does what it directly asks: FineWeb key L1 falls by about 11% versus SVD, and this transfers to WikiText and TinyStories. But the corpus-key gauge does **not** produce comparably sparse *pair contributions*: effective active components fall only from 23.13 to 22.33 on FineWeb. The uniform-domain key gauge happens to concentrate pair contributions more, while not minimizing FineWeb key L1 nearly as much. A one-sided key prior and a two-sided pair-contribution prior are genuinely different.

Across three fit seeds, mean optimal matched correlation of signed component-response vectors on the same 8,192 held-out FineWeb pairs is **.280** for uniform-key gauges and **.512** for corpus-key gauges. The latter is higher than the prior pair-activation-sparse gauge's **.236** in its separate matched-seed test, but these are different fitting objectives and draws, so this is a stability clue, not evidence of semantics. The strongest next criterion is whether stable components have selective *held-out downstream effects*, not just repeatable geometry.

Post-fit coarse token-form categories provide an initial legibility red team. The ten highest response-energy components per gauge receive labels on disjoint FineWeb selection documents; labels never affect fitting. Mean held-out log2 enrichment of those chosen labels:

| Gauge | FineWeb test | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| SVD | .033 | .785 | 1.215 |
| Random | .293 | .935 | 1.487 |
| Prior pair-activation sparse | .021 | .643 | .948 |
| Uniform-domain key L1 | .114 | .324 | .551 |
| FineWeb key L1 | -.076 | .545 | .791 |

Neither new key prior improves this coarse label criterion over the SVD or random basis. Broad token-form categories and corpus shifts make these labels weak proxies for semantics; the result is a failure of this *particular* legibility gate, not proof that the components have no meaning.

## Matched full-model component deletions

The ten candidate edits are the two highest pair-response-energy components of each of SVD, random, pair-activation-sparse, uniform-key, and corpus-key gauges. Each is partially deleted to target **.003 relative head-output error on 64 separate FineWeb calibration documents** from 15,000–17,499. The fractions are fixed before testing on two seeds × eight length-128 documents from each of FineWeb test, WikiText-103, and TinyStories. The native CE replay gate is zero. KL compares the original rank-64 cut's full-model distribution with the component-edited full model, so the baseline cut error is held fixed.

The effective affected-document fraction is \((\sum_j k_j)^2/(N\sum_jk_j^2)\) for per-document KL values \(k_j\). Lower means a more concentrated effect. This average is over two components × three corpora; the test head error need not equal its calibration target.

| Gauge | Mean test head error | Mean KL (×10⁻⁶) | Mean effective document fraction |
|---|---:|---:|---:|
| SVD | .00373 | 2.94 | .397 |
| Random | .00364 | 1.79 | .439 |
| Pair-activation sparse | .00415 | 1.95 | .681 |
| Uniform-key sparse | .00448 | 3.09 | .396 |
| Corpus-key sparse | .00343 | 5.08 | .395 |

The key-sparse components are more concentrated than the pair-activation-sparse controls on this small panel, but **neither key prior beats the SVD's mean localization**. The corpus-key gauge's mean KL is dominated by one FineWeb document: its first selected component reaches 386×10⁻⁶ KL and .040 head-output error on that document, far above the .003 calibration target. The corpus shift between calibration and that document matters. Across the other corpora its mean KL is much smaller. With only two selected components and 16 documents per corpus, this is evidence against a blanket legibility claim, not a stable population ranking of priors.

## An exploratory phrase-specific lead, with a transfer limit

The exceptional FineWeb window contains the bigram **“as much.”** Inspecting its exact first-layer terms shows a large QK1 component score for the “as” key and “much” query. Because this motif was selected *after seeing a test outlier*, the following follow-up is explicitly exploratory and cannot serve as unbiased validation of semantic discovery.

For a deterministic set of 32 other FineWeb documents with the same bigram, position the pair at key/query positions 7/8 and compare with a motif-free window from the same document. Under the already calibrated partial edit, median relative head-output error is **.0399** on motif windows versus **.00156** on controls for the corpus-key component. A matched SVD component also responds (.0176 versus .00256); the random component does not show this pattern. The original outlier is not among the 16 motif documents chosen for a further full-model check. On those 16, median **query-position** KL is **.00354** for the corpus-key edit versus **1.63×10⁻⁷** on their motif-free controls; SVD gives .00120 versus 5.55×10⁻⁸. These are per-position, motif-conditioned effects, not mean KL over arbitrary text.

The separate corpora contain only four WikiText and five TinyStories usable occurrences in the available tokenized samples. Their corpus-key median query KLs are **1.40×10⁻⁵** and **1.47×10⁻⁶**, respectively, much weaker and variable; the whole-window head-error medians do not separate motif from control there. External documents are only 129 tokens long, so their motif positions cannot be aligned to 7/8 as in FineWeb. This is a promising lead for a position/context-controlled causal test, not a verified transferable “as much” circuit. It also illustrates why rare, large component effects can be missed by broad label enrichment or average-support metrics.

Reproduce with `flagship_key_sparse_gauge.py`, `flagship_key_sparse_labels.py`, `flagship_component_ablation.py --gauge-panel key --n-docs 8 --seeds 23 29 --top-components 2 --target-head-error .003 --calibration-docs 64`, then `flagship_key_motif.py` and `flagship_key_motif_full_model.py`. The JSON and exact rotations are in `full_width_results/head0_key_sparse_*` and `head0_component_ablation_n8_matched_key_panel.json`. The external model and corpora are referenced by path rather than copied into this repository.
