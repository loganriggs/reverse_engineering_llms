# V14: conditional actions and earlier-fact binding in one sampled stream

The [V14 generator](surface_mixture_v14_conditional.py) extends V13's visible
item–attribute facts with a conditional continuation. For example:

> The package is clean, so Isaac uses it.

The action follows an exact attribute→action rule. The completed clause also
enters the visible item–attribute fact state, so later ordinary narrative
descriptions can reuse it. V14 retains the same sampled price, source,
section, mixed verb/object, and bracket families. The local action should be
easier than cross-sentence item binding; a model can learn one without the
other. The two must be scored separately.

## Oracle and counterfactual audit

The [V14 audit](audit_v14_conditional.py) replays the full V13 exact-prefix,
fact-start mixture, named object-component, and earlier-fact edit checks on
the V14 stream. It also checks every conditional action against the explicit
mapping and tests a supported local edit of the attribute before the action.
For a **later** description, editing an earlier conditional fact requires
repairing its already written action; otherwise the edited document is
outside the generator's support. The audit records these linked two-token
repairs separately from single-token conditional edits.

| Per 100 streams/sites | Reference 99641 / 99642 | Diagnostic 99651 / 99652 |
|---|---:|---:|
| Streams with conditional action | 46 / 35 | 64 / 60 |
| Conditional action sites | 64 / 43 | 107 / 100 |
| Supported local action edits | 64 / 42 | 105 / 96 |
| Conditional action and attribute-object reuse in one stream | 41 / 30 | 64 / 57 |
| Attribute-object sites with multiple pending facts | 40 / 32 | 119 / 107 |
| Linked conditional repairs in inherited later-fact edits | 160 / 113 | 207 / 195 |

The [reference](results/v14_conditional_reference_n100_l256_seed99641.json)
and [diagnostic](results/v14_conditional_diagnostic_n100_l256_seed99651.json)
records include all normalization, replay, fact-kind, collision, and edit
counts. The diagnostic rates intentionally oversample relation sites.
These are synthetic choices. A [fresh form screen](compare_v13_fineweb_fresh.py)
used FineWeb `sample-10BT` offset 17,000, disjoint from earlier panels. In
58 matched 100-word windows, colons occurred in 56.9% of V14 crops versus
25.9% of FineWeb, and repeated-word fraction in the first 40 words was
.273 versus .166. The [aggregate record](results/v14_conditional_fineweb_skip17000_n64.json)
contains hashes, not source prose. The V13 32-crop review motivates
conditional actions as a broad relation (10/32 crops) but does not validate
this exact sentence or rate. **V14 is still not naturalistic.**

An [exploratory layout sweep](sweep_v14_form_coverage.py) then reduced
numbered-line, quote, question, and section rates while leaving the new
attribute/conditional rule unchanged. On the same offset-17,000 development
documents, this reduced toy colon presence from .569 to .241, near the
FineWeb .259. But the repeated-word fraction remained .284 versus .166,
and source-answer sites in 64 full toy streams fell from 20 to 5; scope
answers fell from 1 to 0. The candidate was fixed before scoring a disjoint
offset-18,000 FineWeb panel with an independent toy seed:

| Offset 18,000, 58 matched windows | FineWeb | V14 reference | Prose-heavy candidate |
|---|---:|---:|---:|
| Colon presence | .431 | .500 | .103 |
| Repeated-word fraction, first 40 | .169 | .272 | .272 |
| Source-answer sites in 64 toy streams | — | 16 | 1 |
| Conditional-action sites in 64 toy streams | — | 29 | 33 |

See the [development](results/v14_layout_tradeoff_dev_skip17000_n64.json)
and [held-out](results/v14_layout_tradeoff_holdout_skip18000_n64.json)
records. This is a form diagnostic, not a statistical corpus-matching test:
the two natural panels differ, and 64 toy streams give noisy rare-task
counts. The reduced-layout candidate is **not adopted**. It removes task
coverage without repairing repetition or robustly matching punctuation.

## Model comparison and gates

The [small-model trainer](train_v13_attributes.py) accepts `--generator v14`
and trains a two-layer/four-head/width-64 RMSNorm bilinear LM with ordinary
next-token CE on the V14 reference stream. On two untouched reference panels,
the **local action** gate is conditional-action CE below 0.5 nats and mean
new-action probability response above 0.5 under supported local attribute
edits, with at least 50 edit cases per panel. The **cross-sentence binding**
gate remains attribute-object CE at most oracle entropy+0.5 and earlier-fact
object-edit mean absolute effect error at most 0.25, with at least 50
multi-pending edit cases per panel. These gates are set before inspecting
the trained V14 model.

Passing only the local gate would establish competence on a condition→action
mapping inside the shared stream, not on the earlier-fact binding. Only
competent rules should be used for sparse/canonical circuit comparison.
