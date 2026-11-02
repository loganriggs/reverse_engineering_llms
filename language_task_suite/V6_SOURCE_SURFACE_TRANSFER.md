# Source binding across quoted and unquoted claims

The [initial v6 source model](V6_SOURCE_CURRICULUM_TRANSFER.md) answers
`As Bob said, "the coat was missing". ... Who said the coat was missing?`
nearly exactly. Its training statements all use this fixed quoted template.
The [FineWeb cue inspection](FINEWEB_DEPENDENCY_PANEL.md) found both direct
speech and reported claims, so we added a second rendering of the **same
speaker–claim relation**:

`As Bob said, the coat was missing.`

The v6 generator now samples quoted versus unquoted statements with a
configurable `unquoted_source_rate`, defaulting to zero. The existing
vocabulary, question form, latent object→speaker map and exact oracle are
unchanged. A default-rate regression reproduced the first six saved v6
streams exactly. An independent audit of 1,000 mixed-form 128-token streams
replayed every prefix, checked sampled-token support and the original
narrative verb/object rules, and independently reconstructed source answers.
It found 996 quoted and 994 unquoted complete statements and 514 source
answers, with zero oracle failures. It retained the speaker–claim pair
holdout. A quoted-only control with the same seed had 1,905 quoted statements
and 488 source answers.

This is surface diversification, not realistic-language validation. On the
same frozen 480-window FineWeb comparison, quote-containing 128-token
windows fall from **87.3%** in the quoted-only control to **66.3%** in the
mixed-form generator, versus **34.4%** in FineWeb. Question-mark windows are
41.0%/41.9% versus 21.9%. Repeated claim wording and the small vocabulary
remain obvious gaps. The different statement lengths also change how many
claims and questions fit into a 128-token stream; the window percentages are
form cues, not a distribution-matching score.

## Zero-shot form transfer fails

The selected 2,000-joint-update **quoted-only** source model was evaluated on
2,000 fresh streams from the mixed-form generator. Its accuracy is **100%**
when the queried statement is quoted (`n=526`, including 155 pair-holdout
questions), but only **33.9%** when the queried statement is unquoted
(`n=607`), and **10.2%** on the latter's 157 pair-holdout questions. On an
all-unquoted 2,000-stream panel it reaches 37.4% answer accuracy, below the
query-blind Bayes reference of 46.6%; withheld-pair accuracy is 5.0%.

The strongest diagnostic is a grammar-valid edit of a **completed queried
statement** from quoted to unquoted or vice versa while leaving its speaker,
claim object and question unchanged. On all 1,133 mixed-form source questions,
the oracle answer distribution has TV **0** under this edit, but the model's
answer distribution changes by mean TV **.712** (effect L1 **1.425**).
This edit also shifts later token positions by two, so it establishes
surface-and-position fragility, not which of those two cues is responsible.
Ordinary query-object edits in the mixed stream have effect L1 **1.372**;
quoted-query accuracy alone hides that many alternative claims are unquoted.
Changing an unrelated speaker has model TV .119 rather than the oracle's 0.

The model has learned useful binding for the training format, but that
mechanism does not directly transfer to this closely related reported-speech
rendering. A 3,000-update mixed-form adaptation from the quoted model's
selected checkpoint is live. Its decision gate is simultaneous source
accuracy on both forms, held-out speaker–claim pairs, low form-edit TV,
correct query and relevant-speaker effects, and low irrelevant-speaker TV.
Verb, object and place losses must be reported separately. A success would
show acquired form invariance after training, not zero-shot semantic
generalization or a natural-text circuit. A mixed-form scratch control and
additional, less rigid forms would still be needed before stronger claims.

Code: `surface_mixture_v6.py`, `audit_surface_mixture_v6.py`,
`eval_surface_mixture_v6_sources.py`, and `train_surface_mixture.py`.
Records: `results/v6_source_{quoted_control,mixed_quoted_unquoted}_audit.json`,
`results/v6_source_step2000_mixed_forms_n2000.json`, and
`results/v6_source_step2000_unquoted_n2000.json`.
