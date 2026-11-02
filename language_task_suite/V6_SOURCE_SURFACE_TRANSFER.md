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

A same-seed rate grid quantifies the supervision-density tradeoff. Each row
uses 1,000 streams; FineWeb supplies the same 480 frozen comparison windows.

| Quote/source start rates; unquoted share | Quote windows | Question windows | Source answers | Repeated-word fraction (40 words) |
|---|---:|---:|---:|---:|
| .30/.30; 0 (quoted control) | 87.3% | 41.0% | 488 | 27.6% |
| .30/.30; .5 | 66.2% | 41.9% | 514 | 27.6% |
| .15/.25; .5 | 37.9% | 30.2% | 127 | 27.2% |
| .10/.25; .5 | 26.5% | 20.6% | 76 | 27.1% |
| Frozen FineWeb panel | 34.4% | 21.9% | — | 16.7% |

The lower-rate settings bring two punctuation cues near the FineWeb panel,
but yield far fewer source-answer targets and do not reduce lexical
repetition. Matching a few surface frequencies is not evidence that the
underlying dependency distribution matches web text.

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

To test whether the **absolute shift of the later question** explains the
failure, a third diagnostic rendering keeps the statement at 11 tokens:
`As Bob said, the coat was missing near home.` This is a controlled
same-length form, not a claim that such wording matches FineWeb. The
independent 1,000-stream audit passed with 967 quoted and 938 padded
unquoted statements. On a fresh
2,000-stream panel, the quoted-only model answers 100% of 490 questions
whose queried statement is quoted but only 34.2% of 480 whose queried
statement uses this same-length unquoted form; the latter's held-out-pair
accuracy is 4.2%. Replacing the complete queried statement with its
equal-length alternate changes the model answer by TV **.706** even though
the oracle answer and all later absolute positions are unchanged. Thus
shifting the question's absolute position is **not sufficient** to explain
the surface failure. The object position *within* the statement and its
wording still differ, so these remain competing explanations.

The model has learned useful binding for the training format, but that
mechanism does not directly transfer to this closely related reported-speech
rendering. A 3,000-update mixed-form adaptation started from the quoted
model's selected 2,000-update checkpoint. We selected its saved checkpoint
by the **larger** of quoted and unquoted answer CE on a restricted mixed-form
development panel, before opening fresh unrestricted panels. This criterion
is .0396/.0232/.0371 at adaptation steps 1,000/2,000/3,000, so the selected
checkpoint is **step 2,000**. At that point the restricted development panel
has quoted/unquoted accuracy **100%/99.6%** (`n=239/263`) and quote-toggle
TV **.012**. On the identical panel *before* adaptation, quoted/unquoted
accuracy was **100%/41.8%** and quote-toggle TV **.607**.

The selected model passes fresh unrestricted panels with speaker–claim pairs
withheld from both training stages. The table separates the form of the
**queried statement**; the query itself has the same wording in every row.

| Model and fresh panel | Quoted accuracy | Unquoted accuracy | Unquoted withheld-pair accuracy | Quote-toggle TV (oracle 0) |
|---|---:|---:|---:|---:|
| Quoted-only, seed 25313 | 100% | 29.8% | 3.4% | .693 |
| Mixed-form adapted, seed 25313 | 100% | **98.3%** | **97.2%** | **.025** |
| Quoted-only, seed 25314 | 100% | 34.1% | 2.5% | .690 |
| Mixed-form adapted, seed 25314 | 99.8% | **98.4%** | **98.1%** | **.016** |

The adapted model's overall source accuracy is .991/.990 on these panels;
query-object effect L1 error is .057/.065, and an irrelevant-speaker edit
moves answers by TV .012/.011. On a separate **all-unquoted** 2,000-stream
panel, answer accuracy is 97.8% (95.5% on withheld pairings). This is a
substantial acquired invariance for the two trained forms.

It does not generalize perfectly to a **third, untrained same-length form**.
On a fresh panel with the `near home` rendering, its unquoted-question
accuracy is 84.7%, withheld-pair accuracy 69.1%, and equal-length form-edit
TV .194. These improve on the quoted-only model's 34.4%, 4.1%, and .705,
but leave a material gap from the oracle. The original query-position
confound is removed by this comparison, yet within-statement position and
wording still differ.

On a separate unrestricted 1,024-stream mixed-form oracle-KL panel,
adaptation improves verb/object/place/source KL from
**.258/.708/4.005/1.440** to **.248/.654/1.998/.031**. It has not caught
up to the earlier source-blind model on verb and object rules, so it is not
a strong all-rule teacher. This is one adapted training trajectory. A
mixed-form scratch control and less rigid naturalistic surfaces would be
needed to assess how much of the result depends on the curriculum and how
far it extends toward real text. The [quoted-fit, unquoted-test bond-basis
comparison](V6_SOURCE_BOND_BASIS_RESULTS.md) tests local featurizer transfer
on this competent adapted model. Stop this source-specific sweep here and
return to multi-rule causal path objectives and broader task coverage.

Code: `surface_mixture_v6.py`, `audit_surface_mixture_v6.py`,
`eval_surface_mixture_v6_sources.py`, and `train_surface_mixture.py`.
Records: `results/v6_source_{quoted_control,mixed_quoted_unquoted}_audit.json`,
`results/v6_source_mixed_forms_{lowrate,lowquote}_audit.json`,
`results/v6_source_step2000_mixed_forms_n2000.json`, and
`results/v6_source_step2000_unquoted_n2000.json`. The same-length control is
`results/v6_source_equal_length_forms_audit.json` and
`results/v6_source_step2000_equal_length_forms_n2000.json`. The restricted
development comparison is in
`results/v6_source_{quoted_step2000_restricted_mixed_dev,mixed_adapt_step1000_restricted_dev}_n1000.json`.
Selected-model results are
`results/v6_source_mixed_adapt_step2000_{mixed_seed25313,mixed_seed25314,unquoted_seed25313,padded_seed25315}_n2000.json`
and `results/v6_source_mixed_adapt_step2000_mixed_rule_kl_n1024.json`.
