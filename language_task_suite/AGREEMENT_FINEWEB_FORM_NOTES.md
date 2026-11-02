# Agreement toy task: FineWeb surface check

The version-3 agreement generator is an exact-rule probe, not a natural-text match. `audit_fineweb_agreement_forms.py` scanned three frozen, nonoverlapping cached FineWeb panels totaling **864 windows × 513 GPT-2 tokens**. It round-tripped representative decoded windows through the cached tokenizer and counted narrow case-insensitive regex surface forms. The [JSON](results/fineweb_agreement_form_audit.json) records panel hashes, counts, and short examples.

| Surface proxy | 480-window panel | 192-window panel A | 192-window panel B |
| --- | ---: | ---: | ---: |
| `the N near/beside the N is/are/was/were` | 0 | 0 | 0 |
| `the N PREP the N is/are/was/were`, broader PREP set | 12 | 6 | 7 |
| `the N that the N VERB is/are/was/were` | 0 | 0 | 0 |
| Same relative form with `who/which` | 0 | 0 | 0 |

Among the 25 broad prepositional hits, 21 use **of**, two **on**, one **with**, and one **by**. Examples include “The missions of the CEA are,” “the nature of the complaint was,” and “The point of the material is.” These counts are weak evidence about surface form only: the regex can include false positives and misses longer modifiers, punctuation, different determiners, and nonadjacent finite verbs. It does not establish that the closest noun is a syntactic attractor or estimate the prevalence of subject–verb agreement in FineWeb.

This is enough to reject a specific realism claim: the v3 generator's high frequency of rigid `near/beside` and `that the ... helps` frames is unsupported by these panels. A better version should include more natural prepositional and relative-clause surfaces, varied subject and predicate lexemes, and longer distractors. The revised generator must keep an exact visible-prefix oracle and grammar-valid causal edits. Validate it against additional disjoint text panels and human-inspected examples; hold out full lexical combinations or syntax forms at model training time to test rule transfer. Do not tune a decomposition on the same examples used to select the revised grammar.
