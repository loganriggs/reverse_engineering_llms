# Proposition surface V2: paraphrases preserve rules but do not close the language gap

V2 is an experiment on the [V1 semantic fixture](DOCUMENT_PROPOSITION_SURFACE_V1.md),
not an approved model-training distribution. It adds alternative syntax for
introductions, updates, source attributions, counts, records, sections, and
fact questions/revisits. A separate visible-only parser normalizes these
clauses and checks the same underlying relations. `surface_version=1` still
reproduces the original fixture.

The [V2 audit](results/document_proposition_surface_v2_audit.json) passed
2,800 full documents and 10,214 scored prefixes over canonical and typed
lexical realizations. Targeted edits again flipped 656 update answers, 336
source answers, 870 comparison answers, 304 count answers, 88 step-action
answers, and 136 scoped-record answers; 454 deliberately false comparisons
were rejected. Counts include two lexical realizations of each plan. The
V1 regression audit reproduced its previous totals and form values.

| 64 documents × first 100 words | Typed V1 | Typed V2 | FineWeb reference |
|---|---:|---:|---:|
| Panel 94121: word types | 368 | 377 | 2,532 |
| Panel 94121: top-ten word share | .345 | .326 | .206 |
| Panel 94121: repeated trigram share | .455 | .423 | .0067 |
| Panel 94122: word types | 371 | 385 | 2,258 |
| Panel 94122: top-ten word share | .353 | .331 | .214 |
| Panel 94122: repeated trigram share | .474 | .418 | .0104 |

The FineWeb figures come from the previous frozen `skip40000` and
`skip41000` development panels at the same crop length. They are not
estimates of natural rule rates, and they have been used for development;
later acceptance needs a fresh holdout. Even after paraphrasing, recurring
typed V2 trigrams include `its name was` in 50–51/64 documents,
`update was reported` in 23/64, and `the owner of` in 21–23/64. The
full ranked lists are in the audit JSON.

Manual inspection finds a more fundamental issue than clause wording.
The independent move sampler can produce a paragraph with eight or nine
unrelated record counts, or compare the same two entities' place multiple
times without any intervening change. These are legal rule traces but not
plausible discourse. Some generated clauses are also awkward after typed
lexical substitution. More fixed paraphrases are unlikely to solve this.

**Decision:** retain V2 only as a controlled surface-variation stress test.
Do not train interpretability models on it as the realistic language DGP.
The next generator iteration should first impose a *document-level semantic
plan*: topic/purpose, related propositions, causal or explanatory links,
and bounded repetition of the same relation. It should realize the plan
through several compatible clause structures and a larger type-checked
lexicon, with rates exposed as DGP hyperparameters. Then rerun semantic,
counterfactual, form, and held-out generalization audits. Only after a
candidate passes those gates should bilinear-model behavioral competence
and weight-method recovery be evaluated.

Reproduce with:

```bash
python language_task_suite/audit_document_proposition_surface.py \
  --surface-version 2 \
  --output language_task_suite/results/document_proposition_surface_v2_audit.json
```
