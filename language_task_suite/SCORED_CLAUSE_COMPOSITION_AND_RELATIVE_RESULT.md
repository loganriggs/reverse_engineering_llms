# Composition reached the rule sites, but the language-form gap remained

The [earlier syntax audit](SEMANTIC_SYNTAX_OPPORTUNITY_AUDIT.md) showed that joining only unscored background assertions improved document-level form while barely changing task-bearing windows. Two follow-ups kept the same sampled proposition plans and the same independent visible-answer tests:

1. **Scored coordination** lets adjacent declarative statements containing answer sites join with “, and”. Questions and numbered steps remain separate. The operator expands the joined sentence from visible tokens alone before the existing semantic parser; marked site indices are remapped.
2. **Truth-checked relative clauses** attach a claim about another already-visible attribute of the queried entity *inside* a scored fact/update sentence. For example, a color query may mention who owns the object. The checker removes the clause only after independently recovering that owner/color/place from the preceding visible text. The added clause never supplies the queried attribute itself.

Both are *surface operators*, not new semantic world tasks. They exercise a rule-bearing context while preserving the existing sampled fact, update, source, comparison, count, and record rules. They do not supply a full next-token oracle. The two source-disjoint FineWeb development panels are previously monitored ranges, not a final untouched naturalness test.

The scored-coordination implementation passed reversible expansion, independent parsing, and all 1,295/1,276 marked answer sites in two 512-document panels. It joined 1,009/1,039 sentence pairs; 388/343 answer sites fell inside a coordinated sentence. The relative operator likewise passed all documents and sites, inserting 246 relative clauses per panel, each inside a scored sentence. A separate seven-form integrity audit checked 1,400 canonical/typed-lexical documents and 3,442 answer sites; it rejected 692 deliberately false relative claims and observed 105 supported initial-owner edits that changed a later owner relative clause. The owner-edit test excludes documents where another owner update could shadow the initial fact. These are semantic-integrity checks, not evidence of natural prose.

| 512 first-100-word documents | FineWeb 130k / 140k | Baseline V3 | Scored coordination | Scored relative clauses |
|---|---:|---:|---:|---:|
| Accepted sentences per 100 input words | 5.24 / 5.21 | 10.35 / 10.19 | 8.19 / 8.02 | 10.20 / 10.05 |
| Distinct root-child dependency signatures | 1,227 / 1,210 | 123 / 113 | 212 / 185 | 129 / 122 |
| Top-ten signature share | .179 / .171 | .614 / .603 | .500 / .485 | .611 / .600 |
| Parsed relative-clause sentences | 457 / 483 | 3 / 2 | 5 / 2 | 189 / 189 |

The relative construction changes the specific structural deficit it targets, but remains a small fixed family of clause shapes. It does not approach the natural distribution of sentence lengths or dependency patterns. Scored coordination improves document form more broadly, yet the site-local test remains decisive:

| 41-word windows centered on one sampled rule site per eligible document | FineWeb random windows | Baseline V3 | Unscored coordination | Scored coordination | Scored relative clauses |
|---|---:|---:|---:|---:|---:|
| 130k, 430 windows: repeated cross-document trigram share | .023 | .645 | .642 | .644 | .641 |
| 140k, 431 windows: repeated cross-document trigram share | .022 | .644 | .642 | .645 | .644 |

Natural windows are random positions, not matched to equivalent semantic tasks. Still, after composition actually reached hundreds of scored clauses, their local wording remained nearly as repetitive as the baseline. A simple increase in the number of sentence-shape operators is insufficient when every operator is attached to a small, fixed set of proposition phrasings. These arms should remain semantic and form controls; **do not train or interpret a new bilinear model on them as if realism had been achieved**.

The next generator change should separate a typed semantic clause AST from its probabilistic linguistic realization. It needs multiple constructions for introducing, revising, attributing, comparing, and querying the *same* proposition, with relative/coordinate/negative operators allowed within those constructions. The realization prior should be calibrated on a broader corpus-derived syntax inventory, while every assertion remains independently parseable to the same world graph. Use development panels to adjust rates and an untouched source panel to confirm full-document and rule-window form. Then train 1–4-layer bilinear pilots, gate each rule and held-out combination behaviorally, and compare the weight-only, data-aware, sparse, and causal objectives only on learned behavior.

Reproduction code: [coordination](document_proposition_coordination.py), [relative clauses](document_proposition_relative.py), [form audit](audit_semantic_syntax_opportunities.py), and [relative integrity audit](audit_proposition_relative_integrity.py). Aggregate, text-free outputs: [scored coordination 130k](results/semantic_syntax_scored_coord_fineweb130000_n512.json), [scored coordination 140k](results/semantic_syntax_scored_coord_fineweb140000_n512.json), [relative 130k](results/semantic_syntax_relative_fineweb130000_n512.json), [relative 140k](results/semantic_syntax_relative_fineweb140000_n512.json), and [relative integrity](results/proposition_relative_integrity_700plans.json).
