## H1039 — Diverse task screen adds five eligible training cohorts; reserved panels need repair

Previous turn was progress: H1038 established conditional within-family causal transfer and ended the basis-diagnostic series. This turn returns to the task-supervision/featurizer loop with a fixed20-family native-competence and token-alignment screen.

Before any forwards, saved a manifest of80pairs across20candidate task families:12training,4development,4evaluation. All family assignments fixed before outcomes. Domain-related families cross splits (e.g. comparative/superlative, addition/multiplication); these are whole named-task exclusions, not unseen-primitive exclusions. Original sequence_copy and spatial_inverse remain excluded. Number-sequence continuation is related to copying, so future transfer to copying cannot be called a wholly novel computational domain. Some candidate tasks and lexical items were investigated historically; these are additions to the current five-family training setup, not discoveries of previously unknown capabilities.

Tokenization requires disjoint single-token continuation answer groups (original, capitalized, uppercase spellings), exact prefix preservation, and equal-length counterfactual prompts for positional intervention eligibility. Six pairs fail answer validity, leaving74pairs/148endpoints captured at the current pre-attention11 boundary with native logits. Equal length is necessary here but not sufficient to establish semantic position alignment. The weekday pair with identical Monday answers was rejected before inference rather than treated as a binary contrast.

Fixed cohort gate: at least4valid equal-length pairs, >=75% both-endpoint forced-choice accuracy, >=50% intended full-vocabulary top-answer accuracy. A passing train cohort retains all four pairs, including any model errors; no per-example success filtering.

| Eligible training family | Both-endpoint accuracy | Native top-answer accuracy |
|---|---:|---:|
| Antonym |100%|75%|
| Noun plural |75%|50%|
| Past tense |100%|100%|
| Country-language recall |100%|87.5%|
| Number successor sequences |100%|100%|

Deliverable:20eligible training pairs/40endpoints with native boundary states, output targets, and an explicit eligibility manifest. Added to the existing60discovery pairs, these could yield80pairs across10named training families, after causal-port and semantic-alignment checks. This is a candidate expansion, not yet a fitted featurizer or verified causal-supervision bank.

No reserved development/evaluation family passed the complete gate. The failures are informative but must not be overinterpreted. Animal category is perfect on its two aligned pairs but lacks four pairs. Object color has only three aligned pairs and does not meet the binary threshold. Synonym and past-participle probes often favor the intended answer over the contrast but predict an intervening article or other word in unrestricted generation. Comparative, calendar, ordinal, and superlative prompts often continue with 'the'; arithmetic prompts also show incorrect numerical continuations. This screen does not establish inability at these tasks under other natural formats. Answer groups are deliberately narrow and may undercount valid paraphrases.

All reserved prompts were read for native competence, so they are not untouched evaluation in an absolute sense. They have not been used to fit or select representations. Repairing prompt format or alignment after this screen creates adaptive development work; a separate frozen confirmation panel will be needed before claiming transfer. Do not move passing development/evaluation families into training to make the benchmark easier.

Verification: manifest hash frozen before forwards and unchanged; all148token-valid prompt strings unique across splits.148prefix captures and148suffix passes, plus20independent direct full forwards (one per family), replay error0. Independently recomputed all74pair metrics and20family summaries from saved logits; exact agreement. Zero feature-fitting rows, no original copying/spatial training rows, no new natural documents. Native capability and causal reliance remain separate claims.

Next: repair reserved cohort construction before fitting the expanded model: enforce token-valid distinct contrasts and aligned lengths before execution, use natural completion formats rather than metalinguistic prompts where appropriate, and preserve the original family assignments. Record every attempted template and its competence outcome. In parallel within this workstream, run residual-versus-other-port causal controls on the20eligible training pairs, so added tasks supply actual causal targets rather than only labels. The resulting bank must feed the scalar/block/shared dictionary comparison at matched capacity; no standalone task mechanism detour is needed.
