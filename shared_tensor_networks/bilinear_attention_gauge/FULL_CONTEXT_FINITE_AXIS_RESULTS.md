# Finite QK1-key edits reveal a punctuation-associated axis, not a unique mechanism

The [full-context sensitivity gauge](FULL_CONTEXT_SENSITIVITY_GAUGE_RESULTS.md)
made next-token gradients sparser without changing the rank-256 QK1-key
projector. This experiment asks whether its individual axes survive a finite,
held-out causal check. Axis selection uses **only FineWeb training documents**:
compute the first-order effect of deleting each key coordinate on next-token
log probability, choose the two stable axes with the highest row-normalized
punctuation-key enrichment and at least 0.3% of training deletion-effect
energy. They are full-context axes 7 and 2, each matched across two
independent varimax starts with absolute cosine above 0.997. The same-gauge
control is axis 4, selected for similar training effect energy but a
space-word label. A random-gauge axis is matched on that training energy.

Each finite intervention removes one coordinate of **one** pre-RoPE
first-layer QK1 key inside the fixed rank-256 reduced model, then runs all
18 layers. The controls and selected axes are measured against their own
unedited gauge baseline, avoiding float32 roundoff from comparing equivalent
rotations. Across the controlled panel, an identity projection changes logits
by at most `1.8e-5`, a gauge rotation by at most `3.0e-5`, and a zero-strength
deletion exactly reproduces its baseline. The gauge-only change in target
log probability is at most `6.7e-6` on the natural panel.

## Natural and same-position tests

The first panel uses 24 new documents per corpus (FineWeb test,
WikiText-103, TinyStories) and pairs a naturally occurring punctuation key
with a nearby space-prefixed word key. It shows a strong punctuation
preference in next-token log-probability effect for both selected axes versus
the controls. **But 20/24, 21/24, and 23/24 punctuation keys are one position
later** than their paired word keys across the three corpora. This panel is
therefore a useful screen, not a causal token-form test.

The decisive red team uses a second, disjoint set of 24 documents per corpus.
At the *same randomly chosen key position* in each document, substitute
either punctuation (comma, period, colon, semicolon) or a common
space-prefixed word (` the`, ` and`, ` of`, ` a`), cycling through the four
pairs. All other input tokens and the next-token target stay fixed within
each pair. For every variant, compare the final-token full-vocabulary logit
change caused by coordinate deletion. The table shows the fraction of the
two variants' pooled squared logit-change energy carried by punctuation;
`0.5` is parity, not a statistical null that accounts for token frequency.

| Axis | FineWeb | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Full-context punctuation axis 7 | **.968** | .940 | .938 |
| Full-context punctuation axis 2 | .406 | .440 | .768 |
| Same-gauge space-word axis 4 | .003 | .003 | .002 |
| Training-energy-matched random axis | .026 | .028 | .029 |
| Earlier pair-response punctuation axis 16, partial deletion | .977 | **.958** | **.986** |

For axis 7 minus the same-gauge control, paired document-bootstrap 95%
intervals for the punctuation-fraction difference are `[.902,.982]`,
`[.879,.969]`, and `[.845,.966]` in corpus order; versus the random control
they are `[.878,.961]`, `[.853,.947]`, and `[.805,.942]`. The result is not
solely the period token: excluding period while retaining all four word
variants leaves axis 7 punctuation fractions around `.86/.78/.77` across
the three corpora. Nevertheless, the four fixed token pairs are a narrow
slice of punctuation and words, not a universal punctuation class.

The *next-token target log-probability* is less uniform: axis 7's
punctuation-effect fractions are `.964/.687/.598` across the same corpora.
Axis 2 falls to `.138` on WikiText. Thus full-vocabulary response selectivity
does not imply a consistent effect on the specific next-token target.

The earlier pair-response-varimax axis 16 was selected before these windows
and is included as a stronger prior comparison. Its partial deletion strength
is `0.167`, chosen so its **predicted training** deletion-effect energy matches
full-context axis 7. It is at least as punctuation-selective in the
full-vocabulary logit readout. Axis 7 minus this control has 95% intervals
`[−.072,.013]` on FineWeb, `[−.077,.024]` on WikiText, and
`[−.138,−.018]` on TinyStories. Full-context gradient sparsity therefore
does **not** establish a better punctuation component than the prior
pair-response gauge. The partial control's actual held-out head-output dose
is smaller than axis 7's, so training energy matching is not exact
test-domain dose matching.

## Where the first-layer effect goes

For axis 7, punctuation carries only about **6%** of the deletion-induced
first-layer attention-output energy at the *final query position*, but
**75–80%** when measured across the *whole 64-position sequence*. Final
full-vocabulary logit effects are 94–97% punctuation. This is consistent
with earlier-position effects propagating through later layers. It does not
identify the relay path; measuring only the first layer's final position
would substantially misstate the intervention dose. The pair-response axis
has about 98% punctuation energy over the whole first-layer sequence as
well as in final logits.

All claims concern interventions on a **chosen reduced explanation**.
Rotation leaves that reduced function unchanged, but deletion of a gauge
coordinate is not deletion of a unique physical neuron or head. The native
versus reduced model's mean absolute target log-probability difference on
the controlled variants is `.0179/.0319/.0044` across corpora, so the
rank-256 model is close on average but not exact on every individual target.
Matched full-model responses to these specific deletions cannot be checked
against a unique physical deletion in the native model.

**Decision.** A full-context sensitivity prior exposes a reproducible,
finite punctuation-associated direction, but it is not clearly better than
the older pair-response prior and is not yet a task-level explanation. The
next informative experiment should identify specific output-token or
behavioral consequences of this direction, hold out *token IDs* as well as
documents, and trace which earlier-position signals later layers actually
use. Further sparsity optimization without those tests has reached its
stopping criterion.

The [held-out-key-ID follow-up](FULL_CONTEXT_UNSEEN_TOKEN_RESULTS.md)
tests eight frequency-matched punctuation/word pairs absent from the
gauge-fitting key positions. Both selected axes retain broad logit-response
selectivity across all eight pairs, while the earlier pair-response axis
remains at least as selective. The rank-256 reduction closely preserves
the native model's input-edit response on those same variants.

Reproduce with `flagship_full_context_finite_axis_test.py --n-docs 24
--seed 1131`, `flagship_full_context_same_position_edit.py --n-docs 24
--seed 1231`, `augment_full_context_pair_control.py`, and
`analyze_full_context_finite_axis.py` on each saved panel. The JSON files
and per-document bootstrap summaries are under `full_width_results/` with
stems `all_head_full_context_finite_axis_n24_s1131_l64_f100` and
`all_head_full_context_same_position_n24_s1231_l64_f100_with_pair_control`.
