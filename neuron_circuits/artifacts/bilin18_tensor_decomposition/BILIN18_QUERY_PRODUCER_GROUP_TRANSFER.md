## H574 — Producer-pair transfer depends on template and punctuation

Freeze H573's selected pairs before evaluating new-template interventions: binding attention9+MLP10; copying attention7+attention4. Test own-task pair, opposite-task pair, their four-write union and self. Each changed write is clamped only at the final destination to that prompt's base counterpart. All other computation remains live. This is 512 interventions on 128 edited new-template cases. Prompts and native competence were previously characterized in H572; the producer-pair interventions are new, not a fully independent lexical validation.

| Condition | Own pair residual ratio | Other pair | Four-write union |
|---|---:|---:|---:|
|binding_new_paren|0.706|0.900|0.594|
|binding_new_comma|0.754|0.859|0.604|
|copying_new_paren|0.475|0.867|0.412|
|copying_new_comma|1.138|0.780|0.825|

Binding's pair reduces the gate discrepancy by about 25–29%; the union reduces it about 40%. Copying's pair reduces it about 52% under parentheses, but increases it about 14% under commas. The union also helps parentheses much more than commas. This is not a stable two-component circuit across formats, even though native task competence remains high.

Whole-write interventions can affect many routes besides the selected L14 reader. Their output-margin changes are recorded but must not be attributed solely to the measured gate; direct path-specific mediation has not been established. Likewise, the joint effects need not sum because subsequent computation changes. The result identifies conditional upstream contributors and a clear transfer failure, not necessary/sufficient semantic feature units.

Next H575 should resolve attention9 and attention4/7 into individual head writes with exact reconstruction controls, keeping MLP10 as a whole-write reference. Test which heads drive or oppose the query adjustment before selecting small circuits. Use separate discovery and evaluation conditions, preserve punctuation reversals, and compare local gate effects with full output effects rather than conflating them. The remaining MLP contribution and source-reader-downstream composition remain open.

Audits: residual reconstruction relative error <=1.72e−6; raw-query reconstruction <=1.98e−6; self margin error <=7.63e−6 and coefficient error <=9.98e−7; source unchanged, finite outputs. H573–H574 total 4,224 interventions, no new prompts or FineWeb documents; tracked total 26,112. Broad goal active and incomplete.
