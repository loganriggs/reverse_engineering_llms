## H548 — Earlier cue availability does not resolve harmful distractor copying

Test 192 new prompts: eight name triples, three role rotations, two givers, home/work absence cues, and cue-before versus cue-after the distractor name. The 96 pairs match the distractor token and position, total length, and final token and position. Wording and syntax differ, so this is not a pure semantic-feature manipulation. Head L14H4, distractor source role and five interventions were fixed before evaluation.

| Cue placement | Native accuracy | Distractor-edge deletion accuracy | Deletion margin gain |
|---|---:|---:|---:|
|home_before|31.2%|75.0%|+0.989|
|home_after|41.7%|85.4%|+0.917|
|work_before|20.8%|75.0%|+1.061|
|work_after|29.2%|81.2%|+1.037|

Candidate accuracy and full-vocabulary top-1 coincide in every condition above. Putting the cue before the name does not improve either cue family: home is 15/48 before versus 20/48 after, work 10/48 versus 14/48. Deletion transfers to the new prompts, reaching 36–41/48, but does not reproduce the earlier 47/48 rescue. This is evidence for a harmful contribution, not a universal repair.

Actual paired source-key, destination-query and joint swaps change mean correct-answer margins by less than 0.080 nats in absolute value; deletion changes them by 0.917–1.061 nats. These swaps affect other edges of the head too, whereas deletion alters only the distractor persistent-bus edge at the final destination. The simple hypothesis that the source key fails because the absence cue comes too late is weakened. The experiment does not prove the key lacks an absence representation, or identify the full routing algorithm.

Next H549 separates source-key content, destination-query content and assigned rotary key position in the earlier first/middle distractor pairs. All eight combinations will be tested as local bus-edge coefficient substitutions, with other edges and downstream computation live. This distinguishes candidate explanations of the selected pathway; it does not move physical tokens or establish a complete circuit.

Audits: 192 native prompts, 960 interventions; self margin error zero; all 96 matching assertions pass; finite results and unchanged source/manifest. No new FineWeb documents; tracked total remains 26,112. Broad goal active and incomplete.
