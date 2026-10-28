## H577 — The producer-head reversal localizes mainly to suffix reads

Reuse both competent copying templates and punctuation forms from H572. For L4H0 and L7H8, reconstruct the final-position head write as the sum over all causal source positions of the signed bilinear attention pattern times the mixed value, followed by the output projection. Mixed values include both the context-dependent local value and persistent token-identity bus. Partition positions into six exhaustive, disjoint groups: selected source name, earlier candidate-name tokens, other prefix tokens, inserted span, suffix excluding final token, and final token itself.

Test each group's deletion and replacement by its base-prompt contribution, plus whole-head deletion, whole-head base replacement and self. Base replacement removes the edit-induced group change; deletion removes the entire native contribution. They answer different questions. This yields 3,840 interventions across 128 edited prompts and two producer heads. All other producer reads and later computation remain live.

| Condition/head | Whole-head replacement ratio | Suffix replacement | Inserted-span replacement | Selected-name replacement |
|---|---:|---:|---:|---:|
|copying_old_paren_L4H0|0.893|0.903|1.013|1.000|
|copying_old_paren_L7H8|0.752|0.847|1.001|0.998|
|copying_old_comma_L4H0|0.829|0.858|1.013|1.000|
|copying_old_comma_L7H8|0.725|0.806|0.964|0.996|
|copying_new_paren_L4H0|0.675|0.681|1.024|1.004|
|copying_new_paren_L7H8|0.796|0.950|1.001|1.000|
|copying_new_comma_L4H0|1.260|1.262|0.968|1.000|
|copying_new_comma_L7H8|0.990|0.936|1.008|1.000|

L4H0's new-template punctuation reversal is largely reproduced by replacing only the suffix contribution: residual ratio 0.681 for parentheses versus 1.262 for commas, almost the same as whole-head replacement (0.675/1.260). Replacing the inserted span or selected name does little in comparison. This localizes the relevant edit-induced change to reading later words, rather than a direct read from the insertion itself. It does not show that inserted tokens are globally unimportant: earlier layers may have propagated their information into those later words.

L7H8 also depends on suffix and final-token contributions, with more distributed effects across conditions. Group effects need not sum in the live model. Norm and projection summaries in JSON describe the writes, not unique additive causal mediation. Output-margin changes remain separate from the target L14H4 gate statistic.

Audits: full source-group reconstruction max relative error 5.79e−7; complete group replacement replays prior head clamps within 1.91e−5 margin; self replay <=2.01e−5; finite outputs and source unchanged. Saved grouped writes. No new prompts or corpus documents. Next H578 crosses suffix routing and values separately at token-aligned positions to distinguish changed attention weights from contextual information in values.
