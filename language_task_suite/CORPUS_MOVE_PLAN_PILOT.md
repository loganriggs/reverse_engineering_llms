# From real-document move inventory to an executable proposition plan

The old V8-centered renderer repeatedly produced reused phrase scaffolds,
even after lexical variation, truth-conditional extra sentences, purpose
wrappers, and composed clauses. The research objective needs a sampled
language distribution whose **underlying rule graph** is known and varied,
so that weight-based methods can be judged against rules models actually
learn.

I coded the opening 500 normalized characters of 32 distinct FineWeb
sample-10BT documents at offsets 42,000–42,031. The fixed-offset
[annotation audit](audit_fineweb_opening_moves_42000.py) records hashes and
labels, with [aggregate results](results/fineweb_opening_moves_42000_42031.json)
but no source prose. The panel contains 11 reports, 8 personal/opinion
pages, 4 reference/explainer pages, 3 quoted features, and 2 each of
procedures, collections, and commercial/review pages. The order of
headlines, claims, definitions, attributions, steps, lists, questions, and
comparisons differs across forms. One reader and 32 contiguous documents
do not estimate web-wide rates; the panel is a design inventory.

`document_proposition_plan.py` is the first executable layer of a new DGP.
It samples a form and an opening role, then a graph of typed entities and
facts, state updates, source claims, comparisons, quantities, ordered steps,
and scoped records. Moves reveal or revisit graph nodes only when their
dependencies are available. Form priors, document length, and a coupling
rate controlling reuse of earlier propositions are explicit knobs. The
seven form-specific required moves are coverage constraints, **not**
estimated natural prevalence.

The [plan audit](audit_document_proposition_plan.py) generated 2,800 plans
across seven forms and two coupling settings. All passed dependency and
truth replay. It found 2,784 distinct `(form, opening, move-sequence)`
triples; this demonstrates the sampler is not copying one fixed plan,
**not** that its language is realistic. Mutating a comparison's asserted
same/different value was rejected. A form-prior override produced only
the requested form in 50 trials. In 200 report plans per setting,
coupling 0 versus 1 changed the number of source-attribution moves from
0 to 485 and revisits from 0 to 199. For quoted features, the required
source move remains present even at coupling 0 (200 moves), so the knob
means *additional* reuse beyond form obligations. The
[aggregate JSON](results/document_proposition_plan_audit_2800.json) records
each form.

This is **not yet a toy language**. It has no surface renderer, visible-text
parser, full-token oracle, model training, or weight-method comparison.
The next required step is to realize these graph moves with a corpus-informed
semantic grammar and a substantially broader typed lexicon, then verify
that a separate parser reconstructs the emitted claims and that supported
edits produce the expected text and answer changes. The sampled graph
cannot be used as a hidden-state label for facts the text has not revealed.
Natural-form screens must assess the rendered text on independent development
and held-out panels; plan diversity alone is insufficient.

An initial Brown/FineWeb/WordNet lexical probe showed why a large automatic
noun list is not enough: a permissive physical-object sense filter admitted
words whose common use is abstract or a person name. The lexical bank needs
type and selectional checks, not only a frequency or dictionary lookup.
After semantic and form gates, the intended 1–4-layer bilinear grid should
test each rule and held-out rule/form combination behaviorally and causally.
Only learned rules should enter the canonicalization, sparsity, and
intervention-oriented weight-method comparison.
