# Why a larger word list cannot repair V3 by itself

The [hidden-genre V3 pilot](GENRE_DOCUMENT_V3_PILOT.md) fails a natural-text
frequency screen. This audit asks whether adding names, nouns, and verbs to
the *existing sampled sentence slots* could close that gap. The answer for
the fixed V3 samples is no: the repeated sentence frames already force too
much mass onto a few words.

[`audit_genre_v3_lexical_ceiling.py`](audit_genre_v3_lexical_ceiling.py)
replays independently seeded 256-token documents per regime, keeps the first
49 with at least 100 alphabetic words, and takes those first 100 words,
matching the existing FineWeb development screen. It separates sampled
occurrences in open lexical slots (names,
objects, verbs, places, events, roles, attributes, asides, step actions)
from fixed frame words. The first `Document` title token is classified by
position as a frame word even though `document` is also in the object lexicon.

| First 49 × 100 alphabetic words | Task-rich V3 | Reference-rate V3 | FineWeb development screen |
| --- | ---: | ---: | ---: |
| Distinct types | 368 | 380 | 1,993 |
| Singletons | 63 | 56 | 1,336 |
| Top-ten frequency share | 35.8% | 36.0% | 21.4% |
| Open-slot occurrences | 2,508 | 2,588 | — |
| Frame-slot occurrences | 2,392 | 2,312 | — |

In **both** V3 regimes and a second independent seed panel, all ten most
frequent words come from frame slots. The second panel yields top-ten shares
of 35.8% task-rich and 36.2% reference-rate; it has 377 and 373 types,
respectively. Some 256-token documents have fewer than 100 alphabetic words
and are excluded by the same eligibility rule.
These frame words include `the`, `is`, `a`, `answer`, and repeated lead-ins.
Holding the
sampled sentence forms and frame tokens fixed, replacing every open-slot
word with a different novel word could increase the number of distinct types,
but cannot lower the top-ten share below 35.8% task-rich or 36.0% reference.
Those lower bounds are already far above the FineWeb screen's 21.4%.

This is a counterfactual **for fixed sampled slot placements**, not a theorem
about every possible grammar or a different resampling process. A new lexicon
could alter article choices; a changed sentence mix could alter everything.
It does show that lexicon-only expansion is insufficient for this pilot and
that matching a Zipf exponent in open slots would be an uninformative fix.

The next generator should vary complete utterance shapes and communicative
functions: direct and indirect attribution, ordinary background prose,
pronouns and aliases, optional clauses, list/layout forms, and exogenous new
facts. Its hidden document genre should change probabilities of these forms
without announcing a deterministic label. Each new form must have a
visible-prefix next-token oracle and a semantic-state replay; source, bracket,
number, and local-rule edits must retain independent competence and causal
checks. The task-rich regime can oversample rare semantic dependencies, but
the reference regime needs calibration against a disjoint natural-text panel
rather than this small development screen.

Numeric outputs: [task-rich](results/genre_document_v3_lexical_ceiling_task_rich_n49.json),
[reference-rate](results/genre_document_v3_lexical_ceiling_reference_n49.json),
[second task-rich seed](results/genre_document_v3_lexical_ceiling_task_rich_seed48001_n49.json),
[second reference-rate seed](results/genre_document_v3_lexical_ceiling_reference_seed48001_n49.json).
