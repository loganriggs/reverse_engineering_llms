# Composed event clauses: a semantic pass and a form null result

The previous purpose pilot failed because the same small set of clauses
dominated report, correspondence, and register documents. This ablation asks
whether **putting two true propositions in the same sentence**, rather than
adding more separate sentences, reduces repeated scaffolding.

`document_world_composed_events.py` samples a second fact about another
object from the **pre-update** state. On 80% of update opportunities it
combines that fact with the attributed event, before or after the event, in
one of four clause positions. The visible-prefix parser checks the side fact
against reconstructed prior state, strips the composition, then checks the
scored answer. Typed lexical variation is available. This is a narrow
compositional-syntax test, not a full proposition/discourse planner.

The 300-world semantic audit, in canonical and typed lexical forms, passed
2,492 original sites and 2,492 edited sites. It made 600 random event-value
edits, 92 targeted value flips, and 285 targeted source flips; all targeted
answers changed correctly. It also rejected 211 deliberately false side
claims. The random edits changed only 32 of 2,492 scored answers because most
selected events were irrelevant to the sampled sites; the targeted checks
are the causal evidence.

For form, two paired 64-document panels used the same worlds and first
100 words, requiring both arms to have at least 100 alphabetic words. Both
arms had typed lexical variation and full V8 syntax entropy.

| Seed | Arm | Word types | Cross-document trigram reuse | Sentence-start types |
|---|---|---:|---:|---:|
| 94121 | V8 + lexicon | 469 | .3863 | 109 |
| 94121 | Composed events + lexicon | 456 | .3855 | 97 |
| 94122 | V8 + lexicon | 476 | .4069 | 103 |
| 94122 | Composed events + lexicon | 455 | .4144 | 95 |

Thus even a frequent true side clause produces **no meaningful form gain**.
On a separate paired 64-document panel requiring 120 words, first-60-word
trigram reuse shifted .406 → .393 and last-60-word reuse .362 → .369;
the effect is not merely hidden beyond the first window. Frequently reused
trigrams include “and stored it,” “several visitors discussed,” “and carried
the,” and “a short note.” The opening and incidental clause inventory
dominates; the new side proposition also creates its own repeated frame
(“as its owner/place/color”).

Do not train models on this candidate as a realism improvement. The next
generator must vary **which propositions and rhetorical moves form a
paragraph**, and how they combine, including introductions and incidental
content. More fixed clause templates around the V8 event list are unlikely
to close the gap. Retain the visible-only truth checker and targeted edits
when replacing the renderer. Behaviorally gate each learned rule before
testing canonicalization, sparsity, or intervention-oriented weight methods.

The [audit script](audit_composed_events.py) and
[full aggregate result](results/composed_events_semantic_form_300x2.json)
reproduce the semantic and paired first-100-word figures. These form metrics
are descriptive and cannot by themselves establish natural-language
realism. No model was trained in this ablation.
