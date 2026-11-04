# Enriched WebNLG: annotated surface forms, with limits on reuse

The previous DART/ToTTo screens found that human prose is varied but naive slot swaps change the facts. The [enriched WebNLG corpus](https://github.com/ThiagoCF05/webnlg) directly annotates each lexicalization with a modified RDF triple graph, discourse ordering, a delexicalized template, entity-reference tags, and a lexicalization template. We cloned the official v1.6 release locally and audited its **English training split**. Source XML and text remain outside Git; the [auditor](audit_enriched_webnlg_reuse.py) exports aggregates and file hashes only. The corpus license is CC BY-NC-SA 4.0.

| Coverage measure | Training count |
| --- | ---: |
| Entries / lexicalizations | 6,940 / 18,102 |
| Lexicalizations marked `good` | 17,979 |
| Templates with every tag in the entry's entity map | 18,081 |
| Templates mentioning every mapped graph entity | 16,975 |
| Distinct category + role-labeled predicate graphs | 3,257 |
| Graph shapes shared by at least two entries | 973 |
| Entries in those shared shapes | 4,656 |
| Shared graph shapes with at least two good forms | 970 |
| Distinct `AGENT-1` root entities | 441 |
| Roots with two distinct / two disjoint fact graphs | 335 / 332 |

This is more promising than automatic entity detection: the references are explicitly aligned, and the same semantic graph shape often has multiple human realizations. An example category is an airport with `runwayLength` and `operatingOrganisation` facts; a cross-entry form swap in that family can remain readable and factually aligned. But a ten-case manual cross-entry screen also found awkward or false combinations: a target reference may be a pronoun where the source form needs a name; descriptions such as “a major region” can be unsupported for the target; and some `good` source forms themselves appear to assign the wrong relation to a role tag. The `good` flag and role-graph equality are useful filters, **not** a semantic validation certificate.

Sampling separate, unmodified human lexicalizations for the same root is safer than relexicalizing across graphs, but a small manual pair screen found duplicate facts stated differently and occasional questionable source claims. Only 332 root entities have two disjoint fact graphs in this train split, so this route has limited document diversity. Do not promote simple same-root concatenation as the desired realistic document DGP.

The next concrete use of this corpus should be two-level. First, make a **finite-support reference DGP** from original, accepted triple/text pairs with explicit category and graph sampling probabilities; its exact token-level next-token oracle can be computed by a prefix trie over the finite text support. This is a reproducible baseline for model learning and interpretability, not compositional generalization. Second, build a smaller *reviewed* relation ontology and template bank for relexicalization and linked document acts, with source-disjoint entity/graph-combination holdouts and an independent relation checker. Only the second can support claims that a weight method recovers reusable rules rather than memorized strings. Both arms need matched form and task-window comparisons to natural text, behavioral rule gates, and then weight-method tests.

Audit output: [aggregate and source hashes](results/enriched_webnlg_v16_reuse_train.json). No model was trained in this screen.
