# Aligning the language benchmark with the original proposal

The [original project proposal](https://www.lesswrong.com/posts/GncmyuKmp526RJpmj/ambitious-mech-interp-w-tensor-transformers-on-toy-languages) makes the controlled **joint** data-generating process (DGP) the central asset: specify bigram statistics, skip rules, induction, nesting and their interference, train a tensor-transformer on the resulting language, then use known ground truth to study suppression, error correction, compositional circuits and interpretability tools. The desired loop is to turn any new discovered computation into a new generator rule and repeat.

## What current benchmarks do and do not test

| Benchmark | Ground-truth mechanism | Main use | Limit |
| --- | --- | --- | --- |
| Original `generator.py` plus `mixed_rule_oracle.py` | Competing bigram/trigram/skip/induction and brackets | Primary interference and circuit-recovery test | 26 artificial tokens; not natural-text evidence |
| `surface_mixture_language.py` | Four competing verb proposals in readable event sentences | Check whether interference results survive a more linguistic surface | Rigid frames, 141 tokens, no realistic lexical diversity |
| `multi_genre_language.py` | Reports, lists, FAQ copying/recall, conditional policies | Transfer across visibly distinct dependency types | Genre header largely separates tasks; weak cross-task interference; 173 tokens |
| FineWeb panel | Observed web text, unknown latent rule labels | Calibrate surface and plausible dependency prevalence | Cannot itself verify synthetic latent mechanisms |

The new multi-genre audit replayed every prefix of 4,000 72-token streams and checked oracle normalization, sampled-token support, list numbering, FAQ object copying and place recall, and policy syntax. The tests passed. It found 6,835 report mixed-verb slots, 4,100 policy mixed-action slots, 3,316 FAQ answer-copy slots, 1,091 FAQ place-recall slots and 4,704 list-number slots. These counts establish coverage, not model mastery.

The realism test is negative. On **480 windows of 72 tokens each**, synthetic text has 157 distinct alphabetic words, mean sentence length 10.5 words and p90 16; the first 72 GPT-2 tokens of each frozen FineWeb window have 6,305 distinct words, mean 16.3 and p90 29. The synthetic and GPT-2 tokenizers differ, so this is still an approximate form comparison, but the mean words per window are close (58.5 versus 54.3). The diversity and sentence-structure gaps rule out a FineWeb-like claim. The added genres are worth retaining as controlled transfer tasks, but broadening the headers does not solve the original DGP problem by itself.

## Next experiment that bears on the thesis

Keep the original mixed-rule and readable report streams as the primary **collision** benchmarks. Train a model on a joint stream where the relative strength of a bigram and induction rule is specified in the visible context. Hold out entire collision patterns or lexical combinations, while maintaining a matched marginal-token control. On fixed prefixes, change only the rule-strength cue and measure the oracle and model shift in next-token log odds. This gives a causal target that ordinary reconstruction or next-token accuracy cannot provide.

Compare raw attention coordinates, a weight-only bond gauge, activation PCA, sparse/intervention-aware bases and a shared-dictionary featurizer at matched dimension and baseline-model fidelity. Evaluate four separate quantities: next-token oracle KL, fidelity of the rule-strength *effect*, selective mediation through the proposed component, and transfer to held-out conflict patterns. A low-rank bond that preserves logits but cannot explain the controlled shift is compression, not recovery of the circuit. Conversely, a component that follows the shift only on seen lexical items is not a reusable rule.

The multi-genre stream should then test whether any recovered component or learning-dynamics claim transfers to FAQ recall, list progression and policy action conflicts. If it does not, identify which dependency changes the circuit. Expand those mechanisms into the joint DGP only when their interactions and observable-prefix oracle are specified; avoid treating separate labeled prompt families as a substitute for interference.

Finally, use FineWeb to estimate the frequency and forms of *actual* dependencies rather than surface cue frequencies alone. Sample matched natural-text spans, manually annotate whether they exhibit copying, induction, list continuation, conditional action, nested closure, etc., and use that to choose mixture rates and distractors. The synthetic oracle remains the ground truth for mechanism tests; the natural-text audit constrains which toy regimes are worth studying.

The proposal's weight-only tensor-circuit ambition remains a hypothesis to test. Exact gauge transformations and useful SVD compression have been shown locally, but neither proves unique functional canonicalization or human-legible components. The success criterion is a compact description that predicts causal behavior on held-out combinations, ideally from weights alone, with data-aware methods as measured comparators.
