# Two alternate fact frames preserve the rules but do not improve form

The V14 variable-graph document generator has several rule families, but its prose repeats too many stock clauses. V15 tests a narrow change: for each sampled color update, choose between a source-first sentence (a source describes the item's new color after an actor changes it) and an action-first sentence (the actor's change precedes the source's report). The propositions, source identities, event order, and answer sites stay fixed. A visible-text transducer maps either completed frame to the canonical event sentence before the existing answer parser runs; it cannot read the sampled world state.

On two independent panels of 128 **paired** worlds, V15 sampled 344 and 367 new color frames, split roughly evenly between the two word orders. All 1,552 scored sites across the two panels passed the visible-prefix answer check, with the same site families and answers as the baseline renderings. One supported upstream edit per world also changed the intended answer and passed the visible-prefix check. This validates these two forms for the checked rule sites; it is not a full next-token oracle or a human assessment of every sentence.

| First 100 words, 128 documents | Baseline V10 renderer | V15 color frames |
| --- | ---: | ---: |
| Seed 56601 word types | 544 | 525 |
| Seed 56601 top-ten word share | .318 | .318 |
| Seed 56601 cross-document repeated-trigram share | .505 | .513 |
| Seed 56602 word types | 538 | 527 |
| Seed 56602 top-ten word share | .312 | .314 |
| Seed 56602 repeated-trigram share | .510 | .521 |

The new frames are an example of **valid structural variation without a realism gain**. They recycle the same nouns, verbs, and discourse relationship, so adding two word orders for one fact type does not address the V14 frame bottleneck. The current FineWeb first-100-word screen has roughly .016 repeated trigrams and .209 top-ten word share on a different, non-topic-matched panel; V15 remains far away. Do not train a new model on V15 merely because its semantic checks pass.

The next DGP experiment should broaden the typed proposition and discourse-operation inventory, sampling several different relationships among events rather than expanding one clause template. The form audit must cover both full document starts and task-bearing windows, and rule-site coverage must be measured before training. The RotoWire/SBNation source check reinforces the need for a new source or generative route: the conservative 100-word score-and-bracket eligibility filter yields 185/39 train/heldout RotoWire reports but only 9/1 SBNation reports.

Code: `document_world_v15_color_frames.py` and `audit_v15_color_frames.py`. Aggregate data: `results/v15_color_frames_seed56601_n128.json`.
