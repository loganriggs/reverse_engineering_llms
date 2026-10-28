## H549 — Rotary position dominates the selected distractor-edge difference

Reuse 48 matched first/middle distractor pairs from H546, with the same final query position and distractor token. Cross three factors: destination-query content, source-key content, and the position assigned to the key's rotary transform. Rotary position is the model's position-dependent rotation of query/key vectors. Both bilinear attention factors are recomputed. Apply each hybrid coefficient only to the distractor's persistent-bus contribution through L14H4 at the final destination, leaving all other edges and downstream computation native. Test both recipient contexts and deletion: 864 interventions.

This virtual position intervention does not move a token, change the causal mask, or replace complete Q/K states. It isolates one pathway's coefficient. The eight factorial cells are fixed before evaluation and none is selected away.

| Query / key / rotary-position donor | Early recipient accuracy | Early margin change | Middle recipient accuracy | Middle margin change |
|---|---:|---:|---:|---:|
|first/first/first|33.3%|+0.000|66.7%|-0.522|
|first/first/middle|56.2%|+0.402|89.6%|-0.118|
|first/middle/first|35.4%|-0.012|64.6%|-0.536|
|first/middle/middle|66.7%|+0.474|93.8%|-0.047|
|middle/first/first|37.5%|+0.026|62.5%|-0.497|
|middle/first/middle|62.5%|+0.454|93.8%|-0.066|
|middle/middle/first|35.4%|+0.004|62.5%|-0.520|
|middle/middle/middle|72.9%|+0.522|93.8%|+0.000|
|zero|97.9%|+1.227|100.0%|+0.696|

For the early recipient, position alone raises accuracy from 16/48 to 27/48 and the correct-answer margin by 0.402 nats. Key content alone changes the margin by −0.012, query content alone by +0.026, and both contents with early position by +0.004. All three changes give 35/48 and +0.522 nats; deleting the edge gives 47/48 and +1.227. There are content–position interactions: small content effects at the early position do not imply content is globally irrelevant.

The reverse positional substitution with middle query/key contents lowers middle-recipient candidate accuracy from 45/48 to 30/48 and margin by 0.520 nats. Replacing all three factors lowers accuracy to 32/48. These results support a position-sensitive harmful copying contribution in this head. They do not establish that rotary position explains the entire native first/middle performance gap: the recipients still differ through many other pathways, and even the full middle coefficient gives only 35/48 on the early prompts.

Candidate and full-vocabulary accuracy are separately retained in JSON. Middle native is 45/48 candidate versus 44/48 full-vocabulary correct; deletion is 48/48 versus 46/48. Early accuracies coincide for these interventions.

The current leading explanation is a reusable token-identity reader whose routing depends strongly on position, with contextual interactions and additional computation elsewhere. The simple cue-arrives-too-late account was weakened by H548. A full abstract recipient-selection mechanism remains unsupported; the model also fails arbitrary three-name mappings.

Next: freeze a prospective positional-response prediction and test several source–destination distances across competent binding and sequence-copy tasks, with agreement and sentiment as task-diverse controls. Separate virtual phase changes from physical padding/clause movement, report native competence and all failures, and test whether content explains prediction residuals. Avoid fitting a universal distance curve to this diagnostic panel and calling it generalization. Twenty-hypothesis tournament remains the broader research framework; this experiment only updates the position/content branch.

Audits: native coefficient replay max error 3.78e−7; identical source bus vectors across paired contexts; self margin error zero; finite results; unchanged source and cache hashes. Reused 96 prompts, no new corpus documents. H548–H549 together add 192 new prompts and 1,824 interventions. Tracked FineWeb total 26,112. Broad goal active and incomplete.
