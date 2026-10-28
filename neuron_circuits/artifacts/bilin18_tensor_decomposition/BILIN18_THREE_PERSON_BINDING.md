## H546 — An early irrelevant person exposes a localized harmful identity path

After H545's weak arbitrary-mapping competence, test natural three-person binding. Eight triples are rotated through active pair/distractor roles; either active person can be the giver. Only the active pair go to the market; the third person stays home. Place the distractor before the active-pair sentence, between that sentence and the giving clause, or inside a parenthetical after the repeated giver. There are 144 new prompts, with every name and giver role counterbalanced.

These are declared natural-language expectations, not a logical guarantee that an absent person could never receive a parcel. Score all three candidates, full-vocabulary top-1 and recipient-versus-distractor discrimination excluding the giver. Keep the four heads fixed and neutralize initial-giver, recipient, distractor and repeated-giver bus sources separately using the three-name mean. No successful subset is selected.

| Distractor placement | Three-way accuracy | Full-vocabulary top-1 | Recipient vs distractor | Both-giver blocks correct | Strong gate |
|---|---:|---:|---:|---:|---|
|distractor_first|33.3%|33.3%|33.3%|20.8%|fail|
|distractor_middle|93.8%|91.7%|93.8%|87.5%|pass|
|distractor_parenthetical|100.0%|89.6%|100.0%|100.0%|pass|

The later-distractor conditions pass competence, including the parenthetical case where the irrelevant name is more recent than the intended recipient. The early-distractor condition fails, with only 16/48 correct answers. A secondary earliest-mentioned-nongiver heuristic agrees with 32/48 early-distractor, 45/48 middle and 48/48 parenthetical model choices (125/144 overall). This post hoc observation motivates a primacy hypothesis but is not an independently confirmed algorithm.

### Causal localization

| Placement | L14H4 source neutralized | Correct candidate answers after intervention | Native correct answers |
|---|---|---:|---:|
|distractor_first|recipient|0/48|16/48|
|distractor_first|distractor|47/48|16/48|
|distractor_first|repeated_giver|16/48|16/48|
|distractor_middle|recipient|11/48|45/48|
|distractor_middle|distractor|48/48|45/48|
|distractor_middle|repeated_giver|44/48|45/48|
|distractor_parenthetical|recipient|38/48|48/48|
|distractor_parenthetical|distractor|48/48|48/48|
|distractor_parenthetical|repeated_giver|48/48|48/48|

In the failing early-distractor condition, neutralizing the distractor's value at L14H4 improves both candidate and full-vocabulary accuracy from 16/48 to 47/48. Neutralizing the recipient instead reduces both to 0/48. Neutralizing the repeated giver leaves candidate accuracy at 16/48 and barely changes recipient-versus-distractor evidence. L8H3 effects on that nongiver distinction are much smaller.

Thus the recipient and distractor value paths through L14H4 are behaviorally consequential, including a path favoring the wrong person. The result goes beyond an arbitrary fitted steering direction: it identifies a named source, a fixed head, a destination and a local value contribution affecting a specific failure. It still does not show that L14H4 alone performs association selection or that the whole model uses a single primacy rule.

The three-name-mean replacement is not pure deletion. It can introduce recipient-name evidence along with reducing distractor evidence. The source to alter is also identified from the experiment's declared distractor role, not by an automatic model repair system. Before claiming that removal of a misleading signal causes the rescue, test deletion, graded attenuation and replacement with a mean of unrelated names that excludes all candidates.

Audits: 144 native prompts, 2,304 geometry records and 2,448 interventions; source roles/token positions asserted, self margins reproduce exactly, all outputs finite, source/manifest hashes unchanged. Structured templates and role rotations are correlated; population-general performance is not established. No new FineWeb documents; tracked total 26,112.
