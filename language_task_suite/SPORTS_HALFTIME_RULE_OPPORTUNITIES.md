# A document-level halftime-to-final rule candidate

To broaden the sampled sports world beyond final-score recaps and player statistics, I screened RotoWire and SBNation reports against structured quarter scores. A report is eligible only when its four recorded quarters sum to each team's final score. The [source audit](audit_halftime_rule_opportunities.py) then looks for the exact **first-half** pair, in either order, within 20 source tokens of *half*, *halftime*, or *intermission*. This is a **surface opportunity**, not a certified claim: the sentence's team reference, score orientation, and extra propositions still need semantic validation before any template or task site is used. No source prose is saved in the [aggregate](results/sports_halftime_rule_opportunities.json).

| Source / game split | Consistent quarter records | Near-halftime score pair | Also has final-score pair | Also has named-player-stat proxy | Halftime pair + alias + bracket |
|---|---:|---:|---:|---:|---:|
| RotoWire train | 2,535 | 165 | 147 | 165 | 60 |
| RotoWire heldout | 665 | 36 | 31 | 36 | 12 |
| SBNation train | 5,686 | 761 | 532 | 754 | — |
| SBNation heldout | 1,469 | 172 | 117 | 171 | — |

Among near-halftime-pair reports, 42/165 RotoWire training games and 191/761 SBNation training games have a halftime leader different from the final winner; corresponding heldout counts are 13/36 and 40/172. This makes the rule meaningfully different from merely copying the final winner. A source-only screen of short sentences with just the two score numerals yielded only 10 RotoWire and 119 SBNation training candidates; many still contain unsupported narrative claims, pronouns, or ambiguous score orientation. Therefore the larger counts above must **not** be treated as a safe reslottable sentence bank.

The decision is to explore **halftime state → final outcome** alongside final-score and player facts, preferably using SBNation's larger naturally co-occurring pool. The next implementation must parse and validate the complete halftime sentence against team identities and quarter records; reject any extra unsupported numeric or narrative relation; then preserve both its visible source and its answer under selective score edits. Keep whole-game training holdouts. Measure the resulting task-bearing sentence forms separately from the rest of the report. If the safe sentence bank is too small, retain the natural examples as an evaluation/control corpus rather than inflating them into a repetitive primary generator. Requiring alias and bracket in the same document is too restrictive at present, so the broad DGP should sample rule prevalence and combinations rather than force every task into every document.

These counts establish task opportunity and an information-gaining next filter, not model competence, causal fidelity, or an interpretable circuit.
