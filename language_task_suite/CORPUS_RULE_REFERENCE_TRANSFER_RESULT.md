# List and bracket transfer to ordinary FineWeb document order

The [unconditioned rule-stream pilot](CORPUS_RULE_REFERENCE_DGP_PILOT.md) showed that task-centered training windows oversample numbered lists and brackets. This test asks whether the saved joint-supervised bilinear models nevertheless use the two rules on fresh, ordinary document chunks. It evaluates behavior first; it makes no weight-circuit claim.

The [evaluator](eval_corpus_rule_reference_transfer.py) streams two independent, previously unused 5,000-source FineWeb ranges. For each eligible source, it pairs the unchanged document with a same-source counterfactual: a valid numbered-list `1,2,3 → 2,3,4` shift or a selected matched `(...) → [...]` change. It cuts both documents into ordinary nonoverlapping 256-BPE-token chunks, without task centering, and keeps only sites whose evidence and one-token answer fit a single chunk in both arms. The same source supplies the pair; the changed answer cannot be explained by a different topic or surrounding document. Source offsets 363k–373k are disjoint from the earlier model-training ranges. The token caches are local because they reconstruct source text; only aggregate scores are committed.

| Source range | Paired list sites | Paired bracket sites |
|---|---:|---:|
| 363k–368k | 50 | 1,533 |
| 368k–373k | 61 | 1,532 |

The models are the two depth-2, width-64, four-head bilinear checkpoints continued with explicit joint list+bracket supervision. They were trained on selected rule windows, including counterfactual list and bracket arms, rather than on this unconditioned source stream. Thus this is unseen-source and ordinary-order transfer, not zero-shot rule acquisition.

| Seed | Source range | List pair exact top-1 | Bracket pair exact top-1 | Bracket pair closing-character | Bracket pair two-choice | Correct causal-response direction |
|---:|---:|---:|---:|---:|---:|---:|
| 71503 | 363k–368k | .940 | .326 | .868 | .967 | list 1.000, bracket .998 |
| 71503 | 368k–373k | .984 | .300 | .856 | .963 | list 1.000, bracket .997 |
| 71504 | 363k–368k | .940 | .305 | .802 | .943 | list 1.000, bracket .995 |
| 71504 | 368k–373k | .902 | .307 | .810 | .950 | list 1.000, bracket .997 |

“Pair exact top-1” requires the model's highest-logit token to match the correct answer in **both** natural and edited arms. Bracket “closing-character” accepts a top token containing the right closer, even if it also contains adjacent punctuation; it is a deliberately looser metric. “Two-choice” asks only whether the correct arm's answer beats the other arm's answer. Causal response is positive when the natural-minus-edited answer-logit margin falls after the edit. The latter is nearly saturated and should not be used alone to rank interpretability methods.

The bracket gap is mostly at the full-vocabulary token boundary: pair exact is only about .30–.33, whereas the closing character is right in about .80–.87 of pairs. The list result is stronger but based on only 111 total sites across panels, so it does not establish performance across every list style. Natural-arm background cross-entropy, measured only in the selected rule-bearing chunks with the answer site excluded, ranges from 6.80 to 8.61 for these models. That is not a whole-stream language-modeling score and should not be compared directly with a corpus perplexity.

The intervention is supported by the selected source rule, but its shifted list or square-bracket prose can be less natural. It is a causal diagnostic, not a claim that all edited text is in-distribution web prose. There is no cross-split near-duplicate screen beyond disjoint source offsets, no exact oracle for ordinary prose, and no evidence here that other proposed language rules were learned. Aggregate results: [panel one](results/corpus_rule_reference_transfer_start363000_n5000.json), [panel two](results/corpus_rule_reference_transfer_start368000_n5000.json).

For weight-based interpretation, list and bracket are now justified local behavioral targets in these two models. The next method comparison should preserve exact answer behavior and response magnitude on both panels while reporting background loss and circuit sparsity separately. The broader DGP objective still requires additional naturally occurring, independently verified rules; an unconditioned stream is far too sparse in list examples to replace the enriched training stream by itself.
