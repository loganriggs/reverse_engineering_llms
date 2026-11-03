# Five-frame event grammar: exact oracle, still a form failure

[`event_frame_language_v2.py`](event_frame_language_v2.py) tests whether a single latent event law can support several word orders with an exact next-token oracle. The same adjective, object, verb, destination, and person are realized as active, object-topical, passive, *it* cleft, or *what* cleft sentences. The active and cleft forms expose the verb before adjective and object, so the oracle sums over those unseen fields rather than using the generator's private event draw. The truncated Zipf priors and competing lexical, adjacent, prior-event induction, and prior-place proposals remain explicit.

The [`audit`](audit_event_frame_language_v2.py) checks all five complete-sentence probability ratios against their frame priors, independently enumerates selected semantic marginals, checks supported prior-event edits, and replays every token in two independent 128×256-token samples. It also checks the irregular `took`/`taken` active/passive mapping. Both samples pass; they contain 3,135/3,133 object sites and 3,153/3,136 verb sites, with 3,073/3,053 complete events. These are rule-site opportunities, not evidence that a model learned them. Every non-padding next token would be a training target if this fixture were used for language-model training; no model was trained on V2.

The form comparison remains negative on first-100-word windows, using the same frozen FineWeb and V14 reference panel as V1:

| 128 windows | FineWeb | V14, two seeds | V1, two seeds | V2, two seeds |
|---|---:|---:|---:|---:|
| Word types | 4,047 | 326 / 332 | 39 / 39 | **91 / 91** |
| Distinct sentence starts | 528 | 59 / 60 | 2 / 2 | **24 / 24** |
| Top-ten word share | .209 | .365 / .369 | .628 / .633 | **.547 / .550** |
| Cross-document repeated-trigram occurrence share | .016 | .634 / .632 | .988 / .988 | **.878 / .869** |

Five frames and a larger lexicon improve every displayed measure relative to V1, but still trail V14 substantially and remain far from the human reference. The 24 sentence starts mostly come from the 20 names that can begin an active sentence; the other four frames still have fixed starts. The generator repeats one event proposition per sentence and lacks V14's source, bracket, attribute, price, and document-layout tasks. It is a tested semantic-order/oracle fixture, **not** the replacement multi-rule training distribution.

The next experiment should change the unit of generation: sample a typed world and several discourse acts per document, then realize each act through a broad, semantically validated grammar or typed corpus-derived fragment bank. Preserve exact visible-prefix marginalization where tractable, and separately audit any rule-site oracle where it is not. Measure complete-document and rule-window form, Zipf head/tail and burstiness, rule-site prevalence and causal edit validity under whole-combination holdouts. Train small bilinear models only after this form and task-coverage gate; use behavioral/causal competence to decide which rules enter the weight-based method benchmark. Stop adding isolated event frames if they do not close both the form and task-diversity gaps.

The [aggregate JSON](results/event_frame_language_v2_n128.json) contains generated examples and counts, no natural source text.
