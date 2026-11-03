# Natural fact combinations do not justify canned coordinated templates

The sports DGP's player facts and result facts are mostly rendered as separate sentences. A new [source-corpus screen](audit_sports_fact_fusion.py) asks which combinations actually occur in the first twelve sentences of RotoWire and SBNation reports. It uses the existing whole-game train/heldout partition, an exact final-score pair, and a player's exact box-score point total near `N points`. Outcome words are lexical opportunities, not certified claims. No human source sentences are saved.

| Source / partition | Reports with final score + player points in one sentence | Reports with two player point mentions in one sentence | Reports with player points + outcome word in one sentence |
|---|---:|---:|---:|
| RotoWire train | .0033 | .1911 | .0996 |
| RotoWire heldout | .0042 | .2062 | .0791 |
| SBNation train | .0125 | .0485 | .0188 |
| SBNation heldout | .0096 | .0536 | .0223 |

The two-player metric may overcount when a shared points phrase sits near more than one name; it is an opportunity screen. Even so, the contrast is clear: forcing score and player points into the same sentence would copy a rare source form. Coordinating player statistics is common enough in RotoWire to test, and outcome wording is another candidate. The [aggregate](results/sports_fact_fusion_first12.json) includes the numerator counts.

The existing base sports renderer already has four fixed coordinated-stat forms, but the variable act-plan had used only separate empirical stat clauses. I exposed its pair-rate control in the [act-plan renderer](sports_document_actplan_v1.py) and ran a [matched-world ablation](sweep_sports_stat_coordination.py) on 64 jointly eligible first-55-word crops per source and partition. Every answer site was independently checked from visible text. The table gives repeated-across-document trigram occurrence share; lower is better.

| Source / partition | Corpus-derived singleton stats | Half empirical, half eligible for fixed pairs | Fixed coordinated pairs |
|---|---:|---:|---:|
| RotoWire train | .519 | .561 | .607 |
| RotoWire heldout | .505 | .549 | .585 |
| SBNation train | .561 | .609 | .644 |
| SBNation heldout | .579 | .619 | .664 |

Top-ten word concentration also worsens in all four panels; for example RotoWire train rises .402 → .434 → .471. The fixed-pair arm has fewer sentence-start types, 144 versus 186 for empirical singletons, despite packing more facts per sentence. This is a negative *form* result, not a test of model learnability. The [aggregate](results/sports_stat_coordination_n64.json) contains the paired structural metrics and 372 checked answer sites per method across all panels.

The practical direction is to mine and verify a larger set of actual coordinated player clauses, then place them under an act/evidence plan that varies what the document says and why. A corpus-derived clause bank still needs semantic guards for extra numbers, player roles, team affiliation, and unsupported evaluations. Do not train the current fixed-pair variant as a realism improvement. Keep the no-record empirical act-selection arm as the stronger control, and require heldout full-document **and task-window** form improvement before a new primary bilinear model/method comparison. This work supplies a concrete target for the broader sampled language DGP; it does not make the current sports pilot realistic.
