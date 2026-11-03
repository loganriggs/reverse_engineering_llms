# Natural-prose spelling-convention pilot

This is a deliberately narrow probe of one known document-level rule on
realistic prose. It is **not** the multi-rule synthetic language DGP. The
background text comes from FineWeb; only the spelling convention is sampled
and known. The purpose is to test whether natural variation plus a sparse
known rule is already enough for behavioral and weight-based analysis.

## Construction and audit

The transformation draws a US- or UK-associated spelling convention per
document and consistently rewrites lowercase whole words from 31 lexical
families, such as `color/colour` and `center/centre`. It does not rewrite
capitalized names. Some `-ize` spellings are also valid in British English,
so the latent variable is a sampled *convention*, not an assertion about a
document's true national dialect. [Cambridge's spelling guide](https://dictionary.cambridge.org/grammar/british-grammar/spelling-and-verb-forms)
motivates the variant bank.

For the scored site, the preceding style word belongs to a different lexical
family and is 20–300 words earlier. The model window is cropped after any
earlier style word. The GPT-2 tokenizer check retains only examples where
both target spellings are single tokens and paired prefixes differ at exactly
one cue-token position. Thus the paired US and UK variants have the same
source prose and target location, but opposite correct answers. This is a
supported intervention on the selected rule, not an exact oracle for all
natural next-token predictions.

The full build uses 24,000/3,000/3,000 whole source documents for
train/validation/test from FineWeb `sample-10BT` offsets 53,500–83,499,
cropped to 8,000 characters and 256-token model windows. A 64-bit five-gram
SimHash screen rejects cross-split near duplicates at Hamming distance ≤3.
It is a screen, not a guarantee of no semantic overlap. The local tensor
cache reconstructs text and is excluded from Git; only the aggregate
[manifest](results/corpus_spelling_style_dataset_53500_83500_manifest.json)
is published. The build yields 1,970 training windows (998 UK, 972 US),
245 paired validation windows, and 215 paired test windows. Only 8.2% of
training source documents yielded an aligned window, a substantial selection
filter. The naturally observed style words in the development screen were
US-skewed; the training prior was deliberately balanced.

The transformation unit check covers reversibility, capitalization, lexical
family separation, and the exact one-cue-token paired-prefix property.

## Behavioral gate

Depth-1 and depth-2 width-64, four-head bilinear LMs were trained from
scratch on next-token cross-entropy with the selected target position
upweighted. The measured gate is whether *both* paired variants prefer their
own target spelling over the alternative, and whether changing the cue moves
the US-vs-UK logit margin in the correct direction. Held-out documents were
never used in training.

| Training condition | Depth | Validation both-correct | Test both-correct | Test positive cue shift | Test US/UK top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| No norm, site weight 50, 1,500 steps | 1 | 0.4% | 0.5% | 54.0% | 9.3% / 3.3% |
| No norm, site weight 50, 1,500 steps | 2 | 0.0% | 0.9% | 54.0% | 11.2% / 2.3% |
| RMS norm, site weight 1,000, 2,000 steps | 1 | 0.8% | 0.5% | 43.7% | 10.2% / 9.8% |
| RMS norm, site weight 1,000, 2,000 steps | 2 | 0.4% | 0.9% | 51.2% | 7.4% / 7.9% |
| Paired training augmentation, RMS norm, site weight 1,000, 2,500 steps | 2 | 0.8% | 0.5% | 44.2% | 9.8% / 4.7% |

The strong-site depth-2 model reaches 85.5% top-1 and 0.677 nats at its
training targets, versus 7.8% and 6.64 nats at the held-out US validation
targets. Its final weighted training loss averages 2.07 nats over the last
50 updates. This is target memorization without useful transfer of the
spelling convention. Exact cue interventions have roughly chance-direction
effects on held-out documents. We therefore **do not apply weight-based
interpretability methods to claim a learned spelling rule** for these models.

The paired diagnostic adds the opposite-style version of every training
window, producing 3,940 training rows while keeping held-out documents
fixed. The US/UK prefixes differ at the cue token, and the target label
flips. This control is intentionally non-i.i.d. and not a realistic corpus.
It still does not produce reliable held-out cue response. Thus merely adding
the matched style counterfactual to the same 1,970 documents is insufficient
under this architecture/objective/budget. The failure does not isolate
architecture from sample size, optimization, lexical diversity, or
natural-language difficulty; it does rule out interpreting these fitted
weights as a learned spelling circuit.

## Research decision

Whether the paired diagnostic succeeds or fails, stop treating this
single-rule probe as the main generator. The main task remains a variable,
jointly sampled, multi-rule language distribution whose document and clause
structure are plausible, with measured gaps against independent natural
text. Each learned rule gets its own behavioral gate. Weight-method scoring
then asks which method finds that rule or its reusable computation, holding
faithfulness and analysis budget explicit. Capability is a gate, not the
research objective.

Scripts: `corpus_transformed_spelling_style.py`,
`corpus_spelling_tokenization.py`, `audit_corpus_spelling_style.py`,
`build_corpus_spelling_style_pilot.py`,
`check_corpus_transformed_spelling_style.py`,
`train_corpus_spelling_style_bilinear.py`, and
`augment_corpus_spelling_train_pairs.py`.
