# Focused proposition plans: coherence control, realism still failed

The earlier proposition sampler chose each document move independently. It
could attribute the same update repeatedly, compare an unchanged entity
pair on the same attribute several times, or list several unrelated counts
for one object. `planning_style='threaded'` is an experimental DGP setting
that adds a focal entity, a document-wide count unit and preferred record
key, a preference for unmentioned events, and caps on move kinds. It still
samples all seven page forms. These constraints are **design controls**, not
rates learned from FineWeb.

The [plan audit](results/document_proposition_threading_plan_audit.json)
sampled 500 eleven-move plans per form and planning style. The constructed
duplicate diagnostics changed as intended:

| Across 3,500 plans per style | Independent | Threaded |
|---|---:|---:|
| Repeat a source attribution for the same update | 1,397 | 0 |
| Repeat an entity/unit count | 727 | 0 |
| Repeat an unchanged comparison | 736 | 0 |
| Repeat a scoped entity/key record | 333 | 0 |
| Repeat a recall target | 678 | 0 |
| Recall within first six moves | 2,372 | 0 |
| Adjacent identical move kind, excluding steps | 8,222 | 2,257 |
| Distinct move sequences, summed across forms | 3,478 | 3,007 |

The duplicate-zero results follow from the sampler constraints; they are
not independent evidence of naturalness. A 500-plan-per-setting focus sweep
shows that `focus_rate` changes the topical share for reports from .794
at rate 0 to .959 at rate 1, and procedures from .748 to .894. Commercial
listings move only from .372 to .440 because unique counts and records
must spread across objects. Move-sequence diversity also falls, especially
for collections. This is an explicit tradeoff to address in the next plan
design.

Both eleven-move settings passed visible-text and causal audits with V2
syntax: [independent](results/document_proposition_independent_11_audit.json)
and [threaded](results/document_proposition_threaded_v1_audit.json).
Threaded passed 2,800 complete documents and 9,432 answer prefixes,
including update/source/comparison/count/step/record counterfactuals.
The independent setting passed 2,800 documents and 10,174 answer prefixes.
Their site counts differ because the planning policies create different
opportunities; these numbers do not compare model ability.

On two separately sampled, fixed-eleven-move 64-document, first-100-word
form panels, threaded did **not** improve the key naturalness proxy:

| Typed lexical stream | Independent | Threaded | FineWeb development reference |
|---|---:|---:|---:|
| Panel 94121: cross-document repeated trigram share | .453 | .449 | .0067 |
| Panel 94122: cross-document repeated trigram share | .418 | .451 | .0104 |
| Panel 94121: word types | 363 | 361 | 2,532 |
| Panel 94122: word types | 385 | 363 | 2,258 |

The synthetic panels share settings and seeds but not identical sampled
proposition plans; the natural panels are prior development references.
Manual samples remain formulaic. A document can enumerate object records
and immediately ask `Answer:` questions without a reason to do so.
`code` records can name an alias belonging to a different object because
the record semantics do not yet specify the relation. Focusing and
deduplicating moves is useful but does not supply a coherent document
purpose or realistic rendering.

**Decision:** keep the threaded option as a plan-level control and stress
test; do not use it as the main improved language distribution. The next
design needs a sampled document purpose and linked proposition chains
(what changed, why it mattered, who reported it, what followed), with
scored rules embedded in ordinary statements. Preserve exact visible
answers and causal edits. Evaluate structure and language form on fresh
held-out natural panels before training bilinear models and attempting
weight-based circuit recovery.

Reproduce:

```bash
python language_task_suite/audit_proposition_threading.py \
  --output language_task_suite/results/document_proposition_threading_plan_audit.json
python language_task_suite/audit_document_proposition_surface.py \
  --surface-version 2 --planning-style threaded --fixed-moves 11 \
  --output language_task_suite/results/document_proposition_threaded_v1_audit.json
```
