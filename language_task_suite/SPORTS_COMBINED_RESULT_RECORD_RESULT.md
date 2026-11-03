# A score-and-record lead can be grounded, but does not fix the document

The previous sports act-plan V1 appended one of two record sentences. Those
sentences were true but increased phrase repetition. This follow-up asks
whether a *human-derived result lead* can combine the final score and
postgame records without that cost.

[`sports_combined_result_record_renderer.py`](sports_combined_result_record_renderer.py)
extracts RotoWire training-game sentences only when they contain both complete
team names, exactly one aligned final-score pair, and one or two parenthetical
win–loss records immediately after the corresponding teams. Each W–L pair
must equal that source game's box-score value. After removing the record
parentheticals, the score clause must pass the existing result guard plus a
new strict word whitelist. That extra review excluded unsupported claims such
as a comeback, injuries, defensive quality, or whether the game was exciting.
The resulting bank contains **49 distinct templates from 344 training-game
occurrences**. The analogous conservative SBNation screen had no accepted
combined forms, so this is a RotoWire-specific arm.

The sampler applies a bank form to a new sampled game world. It draws
synthetic pregame record priors from the same source-game partition, computes
postgame records from the sampled winner, and renders those numbers inside
the result sentence. A visible-only parser matches the complete sentence
against the guarded bank after accounting for the weekday and infers the
winner and margin. It reads inline W–L numbers independently. This is a
selected-site oracle, not an exact all-position next-token distribution.

The [semantic audit](audit_sports_combined_result_record.py) checked 100
sampled worlds each from train and held-out source-game partitions. Every
marked answer and inline record matched the visible text and sampled world.
Team swaps, +1 score edits, player-point leader changes when that rule was
present, and record-prior +1 edits passed their selective checks. The train
and held-out panels emitted 199 and 198 inline records respectively.

On 64 jointly eligible first-55-word crops, the form result is:

| RotoWire source games | Fixed empirical-stat document | Variable act plan, no records | Combined score-and-record lead | Human report |
| --- | ---: | ---: | ---: | ---: |
| Train: repeated-trigram share | .551 | .509 | .509 | .168 |
| Held out: repeated-trigram share | .568 | .524 | .531 | .173 |
| Train: top-ten word share | .390 | .404 | .415 | .269 |
| Held out: top-ten word share | .393 | .404 | .417 | .272 |

Compared with separate record sentences, integration avoids the clear
repetition penalty. The act-selection change, not the record lead itself,
accounts for most trigram improvement. Top-ten word concentration worsens.
Task-window repetition remains mixed: winner and top-player windows improve
somewhat, margin windows worsen on both panels. Generated documents still
read as a result plus separate player-stat statements and explicit recaps.

**Decision:** retain this renderer as a semantically guarded RotoWire
result/record fixture. Stop mining tiny variants of the same result lead as a
realism strategy. It can be a DGP surface control or a compositional rule
test, but it is not the accepted better-language training distribution. The
next substantial improvement requires a broader sampled event/discourse
world and ways to combine facts across clauses, with full-document and
task-window checks. Only behaviorally learned rules should enter later
weight-based canonicalization or sparsity comparisons.

The [aggregate](results/sports_combined_result_record_n100_form64.json)
contains generated examples and form statistics, not human source passages.
