# A semantic/syntax tree verifies rules but does not produce natural form

This vertical slice addresses a concrete failure of the natural-corpus
controls: task sites were rarely jointly available in a 256-token context.
The [sampler](semantic_syntax_document_v1.py) draws 2–4 objects, initial
colors, 4–8 state changes, and speakers who report those changes. It also
draws a *syntax tree*: change clauses can be wrapped in attribution,
earlier-state relative clauses, temporal adjuncts, and coordination. The
document introduces objects before use and may place recaps after an
object's last change but before unrelated later events. The two selected
sites are ordinary final-color and last-source assertions, not separate
question/answer blocks. Source reuse, lexical frequency bias, syntactic
operator rates, and discourse interleaving are independent DGP controls.

The visible-text parser replays the emitted introductions and attributed
changes without reading the sampled world. It computes each answer from
the prefix before the target token. A 5,000-seed generator check passed.
The [formal audit](audit_semantic_syntax_document_v1.py) then sampled 512
documents in each of three settings on two independent natural-reference
panels: **3,072 documents and 6,144 selected sites**. Every document fitted
inside 256 GPT-2 tokens and passed visible replay. For each document it
rerendered a supported last-color edit and a last-source edit, checked that
the targeted answer changed while the other answer stayed fixed, and
rejected two deliberately false recap tokens. These are semantic integrity
checks, not model competence or linguistic realism.

| First 60 words, 512 draws | Natural 130k / 140k | Simple grammar | Rich, independent sources | Rich, coupled sources |
| --- | ---: | ---: | ---: | ---: |
| Cross-document repeated-trigram share | .032 / .032 | .847 / .847 | .830 / .830 | .835 / .834 |
| Top-ten word share | .207 / .210 | .553 / .553 | .494 / .494 | .499 / .499 |
| Distinct dependency signatures | 880 / 815 | 34 / 34 | 98 / 98 | 100 / 100 |

The rich independent arm used 424 coordination nodes in the 130k panel
and 442 in the 140k panel. Adding genuine tree composition and more varied
discourse barely changed the dominant repetition gap. Examples still read
like an inventory of color changes and attributed recaps. The natural
reference is a random 60-word crop, not mechanism-matched, but the gap is
too large to interpret this arm as realistic language. The two independent
panels give similar results.

The site-local check reaches the same conclusion. In 512 41-word windows
centered on each rule family, the rich independent arm has repeated-trigram
share .802/.802 for current color and .801/.801 for latest source across the
two panels. Random natural 41-word windows have .024/.022. Thus whole-document
metrics are not concealing good task-bearing prose.

**Decision:** Retain V1 as a correctness fixture for jointly sampled
semantics and syntax, including causal edits. Do not train it as the main
improved-language distribution or interpret its weights as recovering
natural-language rules. The limiting resource is the variety of *base
realizations* and discourse purposes, not just the number of operators
that combine them. More hand-added sentence frames are unlikely to close
a repeated-trigram gap of this size.

The next surface experiment should use a substantially larger proposal
distribution, but keep the symbolic event graph and visible-text validator
as the authority for facts. A local stronger language model can propose
entire utterance plans or realizations; proposals must be rejected if they
omit a required relation, assert an unsupported one, or fail a supported
counterfactual edit. This is a new test of whether a neural surface prior
can broaden the grammar safely, not permission to treat an LM's prose as
ground-truth semantics. If the acceptance rate or form audit fails, stop
that proposal route and return to corpus-annotated meaning/text pairs.

Aggregate outputs: [130k panel](results/semantic_syntax_document_v1_130000_n512.json)
and [140k panel](results/semantic_syntax_document_v1_140000_n512.json).
