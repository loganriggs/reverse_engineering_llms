# Do sparse attention components acquire transferable token-form labels?

The [exact-gauge sparsity test](EXACT_GAUGE_SPARSITY_RESULTS.md) found rotations of a fixed rank-64 QK1 cut that make individual pair responses more concentrated while preserving the complete attention function. That does not establish human legibility. This follow-up gives those components a deliberately simple, **post-fit** label test. The labels are never used to fit the bond or its component rotation.

The fixed cut is layer 0, head 0 of the 546M bilinear-attention model. For each of the SVD, random, factor-weight-L1, natural-pair activation-sparse, QK2-intervention-weighted activation-sparse, and rare-balanced activation-sparse gauges, choose the ten components with greatest response energy on a FineWeb **label-selection** sample. Give each component the query-token-form → key-token-form category with highest enrichment over the *original head's* pair-response energy, restricting candidates to categories with at least 2% baseline energy. The seven token-form classes are whitespace, digits, punctuation, space-prefixed alphabetic word, capitalized word, alphabetic word fragment, and mixed form. GPT-2 tokens are decoded directly from the cached vocabulary JSON; the decoder has checks for newline, space, punctuation, and `Hello` tokens. This is a coarse surface-form assay, not a semantic task ontology.

Selection pairs come from FineWeb documents 15,000–17,499. The selected label is tested on disjoint FineWeb documents 17,500–19,999 and on separate WikiText-103 and TinyStories corpora. Each corpus contributes **262,144 sampled length-128 causal pairs**. Labels are selected without looking at these test samples. The earlier 32,768-pair pilot was too noisy: its effective sample size for original squared response energy was only about 100 because high-energy pairs dominated. At 262,144 pairs, effective sample size is still only **647** on selection FineWeb, **762** on test FineWeb, **1,055** on WikiText, and **819** on TinyStories. The largest single pair accounts for about 1% of response energy. These diagnostics limit how precisely rare categories can be judged.

For each proposed label, test enrichment is the fraction of a component's weighted squared pair contribution in that category divided by the fraction of the *original head's* weighted squared response in the same category. Mean log2 enrichment across the ten selected components and three fitting seeds is:

| Gauge | FineWeb selection | FineWeb test | WikiText-103 | TinyStories |
|---|---:|---:|---:|---:|
| SVD (one basis) | .76 | 1.01 | .86 | 1.32 |
| Random orthogonal | .86 | **1.08** | .82 | **1.40** |
| Factor-weight L1 | .70 | 1.00 | .76 | 1.06 |
| Natural-pair activation sparsity | .60 | .79 | .44 | 1.15 |
| Natural + QK2-constant activation sparsity | .39 | .47 | .36 | .74 |
| Natural + rare-pair activation sparsity | .51 | .71 | .46 | 1.12 |

**This label assay does not favor the activation-sparse gauges.** On disjoint FineWeb, the SVD and random gauges have higher mean enrichment than any activation-sparse gauge. Results on external corpora differ because token-form base rates differ, but the activation-sparse gauges still do not show a consistent advantage. This does not negate their genuine component-concentration gain; it shows that concentration and this surface-form legibility score measure different properties.

The label space itself is a major limitation. About **84%** of original selection-pair response energy lies in the broad “space-prefixed word → space-prefixed word” class. All ten top SVD components and all ten top components of one random rotation receive the same “space-prefixed word → word fragment” label. Activation-sparse gauges distribute their proposed labels somewhat more, yet sampled high-energy exemplars often involve ordinary function words such as ` as`, ` so`, ` how`, and ` If`, not a clear named computation. A coarse word-form partition cannot distinguish lexical, syntactic, or algorithmic attention roles. The JSON retains per-component labels, enrichments, exemplars, and effective-sample-size diagnostics so these claims can be inspected.

No component was causally ablated in the **full model** in this label assay. A subsequent [full-model component-ablation test](COMPONENT_ABLATION_RESULTS.md) selects components without these labels and compares their downstream effect distribution at matched head-output error; it finds no localization advantage for the activation-sparse bases. A component that merely has a few recognizable top tokens is not yet an interpretable mechanism. The present result is a useful rejection of one weak proxy, not a proof that a more suitable prior or label scheme cannot work.

Run `flagship_component_labels.py --n-pairs 262144 --top-components 10` to reproduce. Results are in `full_width_results/head0_component_label_transfer.json`. The external checkpoint, corpora, and GPT-2 vocabulary remain referenced by path rather than copied into this repository.
