## H809 — Standard indirect-object completion passes the counterbalanced competence gate

After H808 failed, test a separately specified familiar indirect-object task: “John and Mary went to the market. John gave a parcel to” should complete with Mary; changing the giver to Mary changes the answer to John. A second surface uses store/book. Include sixteen planned name pairs, both initial name orders, and both giver choices. Source positions for the fixed L2H5 test are the two initial names; later repeated giver mentions are not substituted for those sources.

No intervention outcomes are used for selection. Complete four-cell items must have single-token source/answer validity and matched length/positions across the two giver choices. The same candidate, full-vocabulary, paired-query, and minimum-item competence thresholds as H808 apply.

Both surfaces retain eight items and pass. Candidate accuracy is 100% for each expected recipient position; both giver choices are correct for every retained order/item pair. Full-vocabulary accuracy is 96.875% for market/parcel and 100% for store/book. This defeats a fixed first-name/last-name answer shortcut on the retained panel, though it does not establish broad semantic understanding. The two surfaces share names and syntax and are correlated tests, not independent tasks.

Items 5,8,9,10,11,12,14,15 are excluded in each surface because of source/answer token validity or incomplete/aligned paired queries. Per-context and item-level reasons are saved; no incorrect examples were discarded. Native evaluation covers 64 contexts. Tracked hashes unchanged, all margins finite, and the first source's mixed L2H5 value vector is exactly invariant across the later giver choice. The reserved test remains untouched and the document pool stays 26,112.

### Locked mechanistic follow-up

Keep L2H5 and L6 query MLP, perturbation coefficients 0/.25/.50, and the four response-construction arms own/input_swap/baseline_swap/both. Pair contexts that differ only in the later giver. Project the first initial name's mixed value vector and inject it at the second initial name. Native source-local states occur before the changed giver and should be query-invariant; audit this directly and avoid artificially duplicating them as separate conditions.

The primary readout must hold token identity fixed across contexts: use first-initial-name log probability and first-minus-second-name logit margin. Report their response difference between contexts where the first name is versus is not the correct recipient. Correct-answer log probability, KL, and accuracy are secondary. This prevents a changing target label from masquerading as a changed effect of the same direction. Cross the L6 raw pre-MLP baseline and input response, apply RMS explicitly, and transplant the computed MLP response onto each recipient's own baseline. Do not reselect a head/layer if transfer fails.

The task is eligible for testing the frozen mechanism; competence and invariant input direction alone do not validate that mechanism. Distinct-task transfer, useful manifold structure, and featurizer fidelity remain open.
