## H843 — Task-state transfer depends strongly on depth and token position

**Main result.** Weak H841–842 final-position swaps were not sufficient grounds to reject early-layer representations. Early changed-token residual states transfer much of the answer contrast for agreement and sentiment, while final-position donor transfer becomes substantial only later. For copying, capital recall and entity binding, early residual-only sequence swaps remain weak, but complete-state swaps reproduce donor outputs. This motivates separating persistent architectural routes.

**Terms.** Transfer fraction is donor-directed answer logit-margin change divided by the native donor-recipient margin gap, aggregated across paired directions. It is not accuracy, information content or a unique mediation fraction. `Changed` means positions with unequal token IDs under indexwise alignment; `unchanged` means equal IDs. `Sequence` swaps all residual positions but preserves recipient normalized embeddings x0 and persistent layer0 attention values. `Complete` swaps all three and should recreate the donor computation.

**Design.** All80 competent H840 pairs, five families, both directions; final-position block output swapped at each of18 layers. Layers1/2 additionally receive changed/unchanged/all residual positions or complete dynamic state, only for76 exactly equal-length pairs. Three capital pairs and one sentiment pair are excluded only from sequence interventions. Equal length is not proof of semantic token alignment. Candidate featurizers were not refit or selected. All panels are reused exploratory tasks.

|Task|First layer with >=50% final-position transfer|L1 changed|L2 changed|L1 sequence|L2 sequence|
|---|---:|---:|---:|---:|---:|
|agreement|11|0.716|0.640|0.749|0.778|
|capital_recall|13|0.005|0.005|0.004|0.005|
|entity_binding|14|0.025|0.025|0.025|0.026|
|sentiment|9|0.588|0.591|0.583|0.617|
|sequence_copy|8|0.078|0.069|0.072|0.070|

Layer indices are zero-based. The >=50% threshold is a descriptive summary of this sweep, not a preregistered discovery criterion or mechanistic onset. Final-position agreement jumps from about0.35 atL10 to0.94 atL11; capital recall from0.25 atL12 to0.89 atL13; entity binding from0.35 atL13 to0.91 atL14. Sentiment grows more gradually; copying crosses0.5 atL8. These are state-transfer observations, not proof that those blocks individually compute the entire operation.

For early agreement/sentiment, changed-token patches dominate unchanged-token patches. This could reflect ordinary lexical input differences rather than a discovered context manifold: the tokens themselves change, and a featurizer that subtracts a token mean may intentionally omit much of that signal. Before assigning semantic structure to residual components, separate token-mean, context-residual and architectural persistent routes.

**Persistent-state ambiguity.** Complete early-state swaps reproduce donor predictions for every family, but residual-only swaps are near zero for capital recall and small for entity binding/copying. The model reinjects both original normalized embeddings and layer0 values into later blocks. H843 changes those together in its complete control, so it cannot distinguish their contributions, redundancy or interactions. The next experiment is the full residual/value/embedding factorial on the same equal-length pairs.

**Audits.**4,560 whole-model forwards. Native runner matches established implementation exactly, identities exact; final-block donor swap maximum logit mismatch3.82e-5 and complete early-state mismatch4.49e-5, within1e-4 tolerance. H842 repeated final-position margins match exactly. All hashes unchanged and metrics finite; no reserved-test evaluation; natural-document pool unchanged at26,112. Paired bootstrap intervals for early controls retained in analysis JSON.

Figure: task_state_location.png. Depth panel uses all80 pairs; early-position panel compares identical equal-length subsets within each task. Complete controls at1.0 are architectural validation, not discoveries.
