# Natural-text delimiter pilot: behavioral gate and first weight method

This is a **controlled rule in a natural-text carrier**, not a claim to know
FineWeb's complete generative process. For a pre-existing, nonnested prose
parenthetical, the generator resamples `(...)` or `[...]` while leaving its
words and all other text intact. The visible opener should determine the
selected closer. The `square_rate` parameter changes the rule's training
prevalence; 0.5 is deliberately enriched and not a calibrated natural rate.

The model question is a gate for interpretability: does the trained model
respond to this rule on documents it never saw? The main research question is
whether weight-based methods identify the structure that produces any such
response, across varied rules and generation settings.

## Data and behavioral measurement

The [builder](build_corpus_delimiter_pilot.py) streams FineWeb
`sample-10BT` source documents at offsets 44,000–51,999. Splits are by entire
source document (6,000 train, 1,000 validation, 1,000 test), with exact-hash
and 5-gram SimHash cross-split screening. The selected rule has 1,801
training passages and 300/334 validation/test **pairs** after excluding
unstable GPT-2 tokenizations. Every retained paired test prefix has exactly
one changed token, at the selected opener. A further 66 test pairs change an
earlier, already-closed parenthetical instead; their selected target token
does not change. The [manifest](results/corpus_delimiter_dataset_44000_52000_manifest.json)
contains only aggregates. The 14 MB token cache is local because it can
reconstruct source text. SimHash is a screening heuristic, not a guarantee of
no near duplicates.

Models use the repository's `DeepModel`: bilinear attention plus bilinear MLP
per block, width 64, four heads, 256-token windows, AdamW, 1,500 steps,
batch size 8. The gate compares the model's score for the actual round and
square next-token alternatives in both versions of a held-out passage.
This avoids mistaking an always-round shortcut for a learned pairing. Full
vocabulary top-1 is reported separately; GPT-2 sometimes merges a closer
with following punctuation. The source text is natural, but the small
training set does not support a capable general language model.

| Run | Paired test passages correct in both styles | Opener edit shifts score correctly | Pure-closer pairs top-1 in both styles |
| --- | ---: | ---: | ---: |
| Ordinary CE, RMSNorm, depth 2, 300 steps | 0.0% | 65.9% | 0.0% |
| 50× site CE, RMSNorm, depth 1, seed 71411 | 6.0% | 68.0% | 2.4% |
| 50× site CE, RMSNorm, depth 2, seed 71411 | 87.1% | 98.8% | 29.0% |
| Same, depth 2, seed 71412 | 81.1% | 98.8% | 13.0% |
| Same, depth 3 / 4, seed 71411 | 85.6% / 88.6% | 98.8% / 99.4% | 24.3% / 24.3% |
| 5% square training, RMSNorm, depth 2, seed 71411 | 84.1% | 99.7% | 19.5% |
| 50× site CE, **no norm**, depth 1, seed 71411 | 24.3% | 88.0% | 10.7% |
| 50× site CE, **no norm**, depth 2, seed 71411 | 88.3% | 97.0% | 39.6% |
| Same, seed 71412 | 88.6% | 99.1% | 40.8% |

All figures are single-setting pilot results, not estimates of population
performance. The 5% square arm has only 86 selected square training
passages, compared with 887 in the balanced arm; natural, unmodified square
brackets also occur in the source. It is striking that the paired behavior
largely survives, but this does not isolate why. Ordinary CE had only 300
steps, so its failure is a compute-budget baseline, not a proof that
unweighted training cannot learn the rule.

On the 66 nuisance controls, the no-norm depth-2 models changed their
round-versus-square margin by mean absolute 0.76/1.24 logits in the two
seeds, versus a mean 7.95/8.91-logit correct shift when changing the selected
opener on all 334 paired tests. The nuisance edit still flipped the two-way
decision in 4.5% of controls in both seeds. Thus the response is selective,
but not perfectly independent of other delimiter context.

## Weight-only screening versus causal necessity

For the **no-norm** depth-2 models, the
[probe](probe_delimiter_weight_vs_causal_heads.py) folds token embeddings
through the first bilinear MLP, approximating the first attention block by
its zero-output residual. It then computes a data-free vocabulary-wide RMS
difference in each second-attention head's bilinear QK score between ` (`
and ` [`, at relative distances 1, 8, 32, and 64. This is a weight-only
screening score, not a full circuit: it omits OV, the final MLP, downstream
readout, and the real query-token distribution.

Head and MLP output ablations on the paired test identify the causal path:

| Seed | First-attention head removals | Second-attention head removals (accuracy drop, percentage points) | First / second MLP removal |
| --- | --- | --- | --- |
| 71411 | each within 0.3 points | 32.6, 9.9, 0.9, −0.6 | 88.0 / 37.1 points |
| 71412 | each within 1.2 points | 13.5, 0.0, 25.1, 6.3 | 87.7 / 15.3 points |

The first attention block is negligible by this intervention, so an
embedding-to-first-QK decomposition would miss the learned delimiter
response. The first MLP and second attention matter, though whole-MLP
ablation also disrupts many computations. The folded QK opener score
distinguishes some second-layer heads from weak ones, but does **not** rank
their causal contribution reliably across seeds: in seed 71412, head 3 has
the largest raw score difference while head 2 causes the largest accuracy
drop. Head identities also vary across seeds. This is preliminary evidence
against treating a large QK contrast as a sufficient explanation. A complete
path score should include OV, downstream MLP/readout, and functional fidelity
under targeted removal.

**Architecture correction:** An early `--norm none` control accidentally
passed the truthy string `"none"` into attention, which retained RMSNorm
there while removing it from MLPs. Those mislabeled runs were archived
locally and excluded from this report. The trainer now passes `False` and
asserts that all blocks actually use `Identity` normalization. The no-norm
results above are the corrected, polynomial architecture.

## Reproduce and next use

The code requires the sibling `reverse_engineer/tensor_language` model
repository (tested at commit `61507e668a90e2a82e0ab003bff6645f66e1410d`),
a local GPT-2 tokenizer, PyTorch and Hugging Face Datasets.
The [trainer](train_corpus_delimiter_bilinear.py),
[evaluator](evaluate_corpus_delimiter_bilinear.py), and probe save aggregate
JSON in `results/`. One no-norm checkpoint is included at
[`checkpoints/corpus_delimiter_truenone_depth2_seed71411.pt`](checkpoints/corpus_delimiter_truenone_depth2_seed71411.pt)
for follow-up weight analysis. The second-seed checkpoint and 14 MB token
cache stay local; the latter is deliberately not pushed because it can
reconstruct source passages. The source-document offsets and seeds in the
manifest reproduce the data recipe.

This task has delivered one behaviorally eligible model family and a first
weight-method failure mode. Stop single-task work after a full-path
weight-only/causal comparison. Then return to the intended broader loop:
more realistic, variable known-rule generation; train across rule-mixture
hyperparameters; gate each rule by held-out behavior; compare canonical,
sparse, and intervention-based weight methods only on eligible rules. The
corpus transform remains a natural-form control alongside the symbolic
synthetic DGP, not a replacement for ground-truth entity, discourse,
bigram/skip, or induction rules.
