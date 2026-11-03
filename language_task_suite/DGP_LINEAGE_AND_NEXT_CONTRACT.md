# What is actually sampled in the toy-language DGP?

This note corrects a conflation in earlier summaries. The separate 12-family
`grammar.py` suite generates individually labeled answer examples. It is **not**
part of the continuously sampled mixed-rule language. The original
`toy_models_of_tensor_networks/attn_circuits/generator.py` and the later
`surface_mixture_v*.py` stream are both autoregressive generators, but they
make rules interact in materially different ways. The user's original
embedding-to-unembedding QK/OV inspection and later sparse-weight runs also
used different task setups and analysis targets; the latter does not establish
that the former result failed to replicate.

## Two sampled processes, different kinds of competition

The original `LanguageGenerator` has token bigrams, token trigrams,
skip-bigrams, and induction. At an ordinary decision, all active proposals
compete with configured weights, normalized over active candidates. Shared
output symbols can be proposed by several mechanisms; a rule label records
which proposal generated the sampled token. Bracket spans are sampled into
the **same** sequence, although they use a special multi-token insertion path
rather than the ordinary weighted candidate mixer. Its exact observable
prefix oracle is in `mixed_rule_oracle.py` and the bridge report explains its
hidden bracket-state caveat.

The later v8 generator samples a full stream too. Its ordinary narrative has
four competing verb proposals (lexical, induction, frame cue, previous place)
and four competing object proposals. At clause starts it additionally chooses
separate `Where`, `As`, `Who`, `Section`, `Detail`, and `What` forms. Their
answers are exact functions of visible histories, but the source-claim and
section-fact records are excluded from narrative-history updates. Thus these
tasks share model parameters, vocabulary, and sampled context while mostly
remaining **separate serialized clause families**, not four-way competition
for the same answer token. V8 has **no bracket tokens or closing task**; the
nested-bracket experiment is a separate versioned branch. The earlier answer
implying that brackets were in the v8 stream was wrong.

The [fresh sampling audit](audit_dgp_joint_sampling.py) used seed 93001 and
300 streams of 256 tokens per setting. It checks that every sampled token has
positive exact-prefix probability and reports site and within-stream counts.

| Sampled process | Verb sites | Object sites | Source answers | Section answers | Streams with both source and section answers | Bracket streams |
|---|---:|---:|---:|---:|---:|---:|
| Original symbolic mixture | n/a | n/a | n/a | n/a | n/a | 283/300 |
| V8 section-training rates | 3,120 | 3,090 | 39 | 468 | 3/300 | 0/300 |
| V8 lower section rates | 3,656 | 3,617 | 69 | 3 | 0/300 | 0/300 |
| V8 source-heavy curriculum rates | 2,242 | 2,233 | 673 | 499 | 47/300 | 0/300 |

All five original symbolic families—bigram, trigram, skip-bigram, induction,
and bracket—occurred in most or all streams on this panel (bracket in 283;
the other four in 300 each). Original label frequencies and v8 counts are
not directly comparable because vocabularies, phrase lengths, and grammar
differ. The low-section setting is a demonstration of the task-density
problem, not a realistic prevalence estimate. Source-heavy curriculum
oversampling improves task coverage but changes the training DGP markedly.

The v8 section-training setting also produced 2,980 complete narrative
clauses, 274 source statements, 39 source questions, 374 section headings,
354 details, and 465 section questions across those 300 streams. The
section-fact and source systems are mostly independent in code. The sampled
text can still be useful for studying either learned task, but a claim of
**shared computation between them** needs much stronger joint coverage and
causal evidence. A section-only competent model is legitimate to interpret
as a section model.

Existing [FineWeb form checks](surface_mixture_v8_fineweb_comparison.json)
are a separate limitation: on 192 matched windows, the task-rich v8 had
colons in 75.0% versus 32.8% for FineWeb, parentheses in 0% versus 36.5%,
and repeated-word fraction .268 versus .164 over the first 40 words. At
lower section rates the colon frequency fell to 47.9%, but section-answer
coverage nearly vanished. These are surface-window diagnostics; they do not
measure the prevalence or structure of the underlying natural-language
dependencies.

## Contract for the next generator version

1. Preserve a **single sampled autoregressive stream** and an exact
   visible-prefix next-token oracle. Report the probability mass of every
   active rule at collision sites, not only a label for the sampled rule.
2. Preserve the original collision primitives—bigram, trigram, skip,
   induction, and bracket closure—inside the stream. Avoid claiming a
   bracket task is integrated when it lives in a different branch.
3. Add source, scope, correction, and similar tasks through a shared latent
   document/event state whose facts also influence ordinary narrative
   predictions. A question form can remain, but it should query records that
   participate elsewhere in the language rather than only a separate fact
   table. Specify which cross-family interactions are intended and which are
   deliberately absent.
4. Separate a **reference DGP** from conditional/oversampled training. The
   latter may be necessary for rare tasks; always report task-site density,
   joint within-stream coverage, and evaluation under the untouched reference
   DGP. Do not call an oversampled stream more naturalistic.
5. Audit normalization, sampled-token support, exact answers, grammar-valid
   counterfactuals, and at least two independent samples of collision and
   task co-occurrence. Hold out combinations and whole featurizer task
   families separately; a model-training task holdout is a stronger claim.
6. Calibrate surface forms and dependency distances against **disjoint**
   FineWeb samples, including human review of whether a detected cue really
   instantiates the proposed dependency. Do not tune to punctuation counts
   alone. Keep the symbolic collision generator as a mechanistic control even
   if a richer English-like realization is built.

The immediate research action is to design that versioned shared-state DGP
and validate it before further claims about sparse/canonical circuits on a
supposedly unified language. The existing competent mixed-verb and section
models remain useful baselines, but they answer narrower questions.

Raw [sampling counts and examples](results/dgp_joint_sampling_n300_l256_seed93001.json).

## V9 partial implementation (2026-11-02)

The [V9 shared-stream pilot](V9_SHARED_STREAM_PILOT.md) now lets source claims
participate in ordinary narrative object prediction and puts variable-length
bracket spans into the same prose stream. It passes exact-prefix support,
source-mixture arithmetic, and opener-swap audits on two independent panels.
The pilot is **not** the completed DGP contract: section facts still do not
influence prose, the provisional lower-rate setting is not FineWeb-calibrated,
and only 0/100 and 2/100 streams in two panels combine a source answer,
source-influenced prose object, and bracket. Next build linked document events
that create such interactions at an auditable reference rate; then train a
small bilinear model and compare learned circuits to the planted mechanisms.

## Skip-rule identifiability correction (2026-11-02)

The [sparse v3 rule graphs](ORIGINAL_V3_SHARED_BRACKET_SUCCESSOR.md) uncovered
an additional requirement for any skip-task DGP. The original v3 skip table
uses four distinct adjacent triggers (`big`, `small`, `old`, `new`), each paired
with exactly one earlier place and one output. A model can therefore predict
the labeled skip event from the adjacent trigger alone. In a 60%-zero attention
model, the trigger-only embedding→unembedding path visibly favors all four
outputs; removing the selected earlier-place attention edge has a large effect
for only one of the four mappings, replicated on two fresh panels. High
accuracy on the other three does **not** show that the model learned a skip
dependency.

For the next generator, cross anchors and triggers: each trigger should occur
under multiple anchors with *different* correct outputs, and each output
should occur under multiple triggers. Sample the combinations in the same
autoregressive stream, report the full anchor×trigger frequency table, and
check that an adjacent-token-only oracle cannot solve the task. Evaluate
support-preserving anchor edits while holding the trigger and intervening
tokens fixed. Graphs should show the planted conditional table, full-model
probabilities, weight paths, and per-row causal effects. Preserve bracket
closure separately: the legacy `skip_trigram` label means opener-dependent
closing, not a generic skip-trigram relation.

The [crossed-skip joint control](CROSSED_SKIP_JOINT_DGP_AND_CIRCUITS.md) now
implements and passes that mechanistic screen: 16 crossed pairs, exact
short-distance anchor edits, a competent two-layer model, and retained
two-cue behavior plus causal anchor paths after 60% attention pruning.
Its FineWeb comparison is still strongly negative. The next **realism** step
must use more plausible document relations, not just add crossed symbolic
rules. The reviewed FineWeb panel contains recurring field-value/numeric
bindings, conditional actions, attribution, and list or FAQ scope; these are
better grounded candidates for a shared sampled document state than V11's
literal heading→place mixture. Keep the crossed model as a control for
whether canonicalization or sparse featurizers recover known reuse, while
building and auditing the richer DGP separately.

## V12 item-price/FAQ pilot (2026-11-02)

The [V12 report](V12_PRICE_FAQ_DGP_AND_MODEL.md) adds item–price facts that
can boost the same item's probability in ordinary narrative object slots and
later determine a cost-question answer. Two independent reference-rate and
enriched-rate panels pass sampled-token/oracle replay, named-component
arithmetic, and support-preserving fact-price and query-key edits. The
reference candidate yields only 4 complete price answers per 100×256-token
panel; the enriched training regime yields 28/40 answer sites and must be
reported as a changed distribution. On a fresh disjoint FineWeb offset,
broader field/numeric binding is visible in a 16-document manual review, but
none of those short crops contains V12's exact fact→prose→FAQ triplet. Form
statistics also show excess colons and repeated words. V12 is a causal
synthetic pilot, **not yet a realistic reference language**. The small
bilinear training gate was item-price prediction and both supported edit
responses on untouched reference panels. Matched ordinary-CE and 100×
answer-weighted two-layer runs both failed: answer CE stayed near the
unconditional six-price baseline, worse than a visible-price-set guess,
and both exact edit responses were essentially zero on two fresh panels.
Stop this price-only circuit/loss-weight detour. Next diversify the sampled
document relations and surfaces; use already competent tasks for
featurizer/canonicalization comparisons while this DGP improves.

## V13 item-attribute reuse pilot (2026-11-02 continuation)

The [V13 report](V13_ATTRIBUTE_BINDING_DGP_AND_MODEL.md) adds a completed
`The item is attribute.` fact that affects later **ordinary narrative**
adjective and object slots. All pending facts contribute to the adjective
mixture; the observed adjective selects which item's object proposal gets
boosted. The initial latest-fact-only version was removed before training
because it created a recency shortcut. Two independent 100×256 reference
panels pass exact-prefix, named-component, and supported earlier-fact edit
audits, with 51/30 multi-pending object sites. The two diagnostic panels
have 142/117 such sites. On a fresh FineWeb offset, broad item-attribute
descriptions occur in 8/16 reviewed short crops, but only one clearly
restates the same named item's attribute; none contains the full V13
multi-pending pattern. Form mismatch remains large (colons 63.6% toy vs
32.7% FineWeb; repeated-word fraction .257 vs .164). V13 is an improved
synthetic binding stress test, **not a validated realistic distribution**.
The reference-stream bilinear baseline must pass held-out attribute
prediction and earlier-fact edit gates before using its learned paths for
interpretability. The realism task now needs diverse document surfaces and
semantic dependencies, not further price-only loss tuning.
