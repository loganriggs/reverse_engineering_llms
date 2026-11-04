# Natural repeated codes: causal source-dependence gate

The [opportunity screen](CORPUS_NATURAL_CODE_COPY_SCREEN.md) found naturally repeated three-letter codes in ordinary FineWeb document chunks. Their later occurrence could reflect copying, topical knowledge, or code-frequency priors. This experiment tests whether the existing bilinear nonce models actually use the earlier code when scoring that later site. It does not train a new model.

The [builder/evaluator](eval_corpus_natural_code_source_dependence.py) streams two further, source-disjoint 5,000-document ranges. For each accepted document it keeps the **unedited** source as the reference and creates one diagnostic arm by consistently changing all prior occurrences of one code to a different, one-token acronym-like code. Source and answer fit one ordinary nonoverlapping 256-token chunk; their BPE offsets match across arms. All and only previous occurrences of the selected code change in the visible input prefix. The later answer is scored as the original code in the natural arm and the substituted code in the diagnostic arm. The substitution is a controlled test of source use, not a claim that the edited document remains factually natural. Reconstructable token caches stay local.

| Fresh FineWeb offsets | Sources | Strict aligned pairs | Distinct source codes | Mean source–target gap |
|---:|---:|---:|---:|---:|
| 383k–388k | 5,000 | 317 | 172 | 55.0 words |
| 388k–393k | 5,000 | 334 | 177 | 57.3 words |

Two existing depth-4, width-128 bilinear nonce-focused models and two nonce+list+bracket models were loaded unchanged. The training nonce aliases were opaque **three-BPE-piece** strings; these test sites are naturally occurring **single-token** codes. Thus success would have been transfer to a different surface, whereas failure cannot identify whether the cause is tokenization, data, model capacity, or objective.

| Model family | Seed | Paired exact top-1, both panels | Paired two-choice, panels 1 / 2 | Positive source-edit response, panels 1 / 2 | Mean answer-margin response, panels 1 / 2 |
|---|---:|---:|---:|---:|---:|
| Nonce-focused | 72104 | 0 | .000 / .000 | .438 / .431 | -.0126 / -.0117 |
| Nonce-focused | 72105 | 0 | .003 / .006 | .495 / .527 | -.0093 / .0011 |
| Joint three-rule | 72104 | 0 | .003 / .003 | .502 / .488 | .0038 / .0034 |
| Joint three-rule | 72105 | 0 | .003 / .000 | .495 / .527 | .0035 / .0019 |

“Paired two-choice” requires the model to prefer the source-consistent code in **both** worlds. A predictor using only a fixed code prior cannot pass that strict pair by construction. “Positive source-edit response” requires the natural-minus-edited answer-logit margin to fall after the prefix edit; around .5 and near-zero mean shift show no consistent source-specific response. The input edit does change the model's output distribution at some sites (a spot check had a 0.69 maximum logit change), so this is not an identical-input evaluation bug. It simply fails to move the *answer-code comparison* in the required direction. Both panels agree. Full per-model scores, including background CE restricted to the selected natural chunks, are in the [first](results/corpus_natural_code_source_dependence_383000_5000.json) and [second](results/corpus_natural_code_source_dependence_388000_5000.json) aggregate results.

**Decision:** these checkpoints do not qualify for a natural single-token code-copy circuit claim. The new candidate is useful because it has authentic reference text and appreciable site yield, but a model must be trained on its actual code distribution and tested on held-out source documents and code combinations before weight decomposition. A useful control is a training arm with the earlier code masked or replaced while preserving target-code frequencies; the difference from full-context training would estimate how much predictive information the source contributes. Keep diagnostic edits separate from natural training prevalence. This result does not undermine the earlier three-piece fictional-alias findings on their own distribution; it limits transfer to a different, more natural surface.
