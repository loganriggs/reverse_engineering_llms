# A broader proposition bank preserves tasks but worsens local form

V15 showed that two alternate color-update frames preserved answers without improving language variety. V16 tests a different hypothesis: perhaps **more kinds of facts**, rather than more phrasings of one fact, would reduce repeated prose. At each sampled V10 fact-act slot it substitutes one of six truth-conditional discourse statements: persistence of an unaffected attribute, a conjunctive owner/color/place fact, an inter-entity equality comparison, an ordinal revision count, an alias identity, or the latest named source. The underlying event graph, act slots, and scored query sites are held fixed. These acts do not change world state.

Two independent 128-world panels sampled 841 and 821 V16 acts. Across both panels, all 1,552 original scored answers remained identical to the paired V10 renderings and passed the visible-prefix checker. One supported upstream edit per world was rerendered and passed the answer check. These checks establish that the new acts did not corrupt the *scored* rules; they do not prove every redundant sentence would be interpreted correctly by a human. The generator still lacks an exact next-token oracle for arbitrary prefixes.

| Form screen | V10 baseline | V16 bank |
| --- | ---: | ---: |
| Seed 56701 first-100 word types | 542 | 524 |
| Seed 56701 first-100 top-ten word share | .317 | .327 |
| Seed 56701 first-100 repeated trigrams | .515 | .551 |
| Seed 56702 first-100 word types | 539 | 525 |
| Seed 56702 first-100 top-ten word share | .314 | .320 |
| Seed 56702 first-100 repeated trigrams | .500 | .535 |
| Seed 56701 source-site 41-word repeated trigrams | .411 | .479 |
| Seed 56702 source-site 41-word repeated trigrams | .408 | .470 |

The source-site windows matter: the new clauses make the local text around an actual rule-bearing answer more repetitive, not just the document opening. The number of scored sites whose rendered prefix falls under a 256-BPE-token context is essentially unchanged: 378→377 and 373→377 in the two panels. Thus this is a form failure, not primarily a loss of task opportunities. The prefix count can differ by one token at the final boundary.

The six new semantic operations all reused a very small object, verb, and clause inventory. They add valid world facts but another family of stock phrases. This rejects the immediate strategy of scaling up hand-written clauses in this tiny ontology as the realism fix. The next generator route should draw more surface variety from a substantially broader corpus or learned language prior, while keeping typed, counterfactually checked rule-bearing spans and whole-combination holdouts. Corpus-grounded source snippets must be screened for unsupported or contradictory claims after edits; the scored answers still need visible-text validation.

Reproduce with `document_world_v16_proposition_bank.py` and `audit_v16_proposition_bank.py`. Aggregate results and generated examples are in `results/v16_proposition_bank_seed56701_n128.json`.
