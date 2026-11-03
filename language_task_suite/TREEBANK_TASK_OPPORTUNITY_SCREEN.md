# How often do rule opportunities co-occur in natural parsed sentences?

The whole-subject agreement pilot intentionally oversamples sentences with agreement attractors. Before using parsed BookCorpus sentences as a *joint* syntax-task carrier, I counted several opportunities in the first 100,000 parse trees of the same external treebank. The [aggregate count](results/treebank_task_opportunities_100k.json) contains no source sentences.

| Strict opportunity proxy | Sentences / 100,000 |
|---|---:|
| Paired opening and closing quote marks | 14,896 |
| Eligible direct NP–VP agreement clause | 1,041 |
| Eligible agreement clause with opposite-number PP attractor | at most 27 |
| Eligible agreement and paired quote in same sentence | 152 |
| Repeated alphabetic word of length ≥4 | 6,263 |

There are 1,058 eligible agreement *sites* and 27 attractor sites. The eligibility filter requires an `S` node with adjacent NP and VP, a short unambiguous subject with an allowed determiner, and a VP beginning with tagged `is/are`, `has/have`, or `does/do` and a 2–22-token continuation. These counts are deliberately narrow; they are **not estimates of all agreement or all attractor constructions in English**. The repetition proxy includes common words and is not a coreference label. Quote pairing is a visible punctuation relation, but the count alone does not certify a learnable long-range rule. This source has parsed sentences, not persistent document state.

The screen explains a design constraint. A sampler drawing independent natural sentences would rarely show an eligible agreement attractor, and it would almost never combine one with quote closure in the same sentence. The pilot's 50% attractor training rate is a diagnostic curriculum, far from the observed frequency under this strict corpus filter. For broad, realistic multi-rule training, use naturally co-occurring rules in document sources (the sports alias×bracket bridge is one existing example), or construct a document state process that can produce varied rule sites without repetitive templates. Report both the natural-rate arm and any deliberately enriched task arm, then check form at the rule-bearing windows. Do not infer general natural-language rates from these parse-tree counts.

Script: `audit_treebank_task_opportunities.py`. Raw treebank data remains external and local.
