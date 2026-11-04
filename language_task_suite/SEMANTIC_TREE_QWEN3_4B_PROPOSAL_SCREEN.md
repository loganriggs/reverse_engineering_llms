# A stronger surface proposer is not yet a safe language renderer

The [semantic/syntax-tree V1 audit](SEMANTIC_SYNTAX_TREE_V1_RESULT.md)
established exact visible-rule replay but rejected the language form: the
richest arm repeated cross-document trigrams in about 83% of its first-60-word
occurrences, versus 3.2% in two disjoint FineWeb preview panels. This screen
tests whether a stronger local language model can supply diverse realizations
for the same symbolic facts. It does **not** promote generated sentences into
the training distribution.

[The proposer](propose_semantic_tree_utterances.py) sampled 30 one-sentence
realizations per fact kind from Qwen3-4B, varying a style and a syntactic
device. It kept bracketed semantic slots and applied a deliberately narrow
mechanical filter for required slots, site ordering, an attribution verb, a
change verb where needed, and some obvious unsupported claims. The sampled
world and visible-text replay remain the authority for meaning.

| Fact kind | Mechanical accepts / 30 | Distinct accepted strings |
| --- | ---: | ---: |
| Attributed color change | 21 | 12 |
| Final-color recap | 15 | 8 |
| Latest-source recap | 27 | 12 |

Manual inspection of the accepted strings found that slot correctness is
insufficient. For example, “as recorded in the change log” introduces a log
absent from the world; “was determined to be” suggests a different epistemic
event; and “with [SOURCE] being the one who reported it” can refer to the
final state rather than the required *last change*. Several forms omit a
determiner before `[OBJECT]`. Many accepted strings are near-duplicates with
the same “final color” and “By the end” openings. These are illustrative
semantic and form failures, not a quantified human-acceptance rate.

**Decision:** Keep this as a reproducible negative screen. Do not run a
larger slot-template sweep or train on its mechanically accepted bank. The
next renderer should pair actual discourse acts with corpus-attested
realizations and independently verify what each utterance asserts, including
its response to controlled world edits. A prospective small pilot must first
pass both semantic interventions and heldout natural-form comparisons; only
then train a model and behaviorally gate weight analysis. The source-grounded
quote control already failed the binding behavioral gate, so it supplies no
binding circuit claim.

Output: [90 raw proposals and filter decisions](results/semantic_tree_qwen3_4b_proposals_seed74251_n30.json).
The model weights and any natural source prose remain local.
