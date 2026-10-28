## H691 — Training-only discrete swaps repair learned shared groups

H690 established capacity deficits and instability in learned memberships. Here we enumerate all60 single-group swaps (remove one of3current atoms, add one of5others) plus the incumbent membership. For each candidate, independently refit continuous dictionary/encoder weights for500Adam steps from the same incumbent using identical minibatches. Choose by full-training reconstruction only, retaining the untouched incumbent unless a candidate strictly improves it. Repeat three rounds. True group gates remain supplied. Heldout triples/all-group combinations are evaluated only after all training-driven choices.

| Seed | Initial train relative MSE | Final train relative MSE | Final held-combination relative MSE | Accepted membership swaps |
|---|---:|---:|---:|---:|
|11|2.17173e-07|2.17173e-07|8.23922e-09|0|
|12|0.0878277|6.33887e-10|8.10067e-10|1|
|13|0.187633|1.7614e-08|2.22601e-08|2|

One swap repairs seed12's support structure, and two swaps repair seed13's. A further continuous refit reaches nearly exact reconstruction. Seed11's already successful incumbent is preserved: even refitting its unchanged membership at this schedule slightly increases training error, so the guard rejects it. All three recover the teacher's binary membership pattern after atom matching. Mean matched atom cosines are0.992,0.932,0.948; exact membership and reconstruction do not establish unique atom directions.

This strengthens the optimization explanation: the failed structures are locally repairable using training data, and their repairs generalize to heldout higher-order compositions. It does not establish robust end-to-end discovery: gates are supplied, the search warm-starts from prior learned fits, and there are only three seeds on one teacher. Enumeration is costly:183candidate refits×500steps per seed, versus6000steps in H690. This is not an equal-compute superiority benchmark. The search guarantees training-error nonincrease through an explicit incumbent fallback, not generalization improvement in general.

Implementation verification: batched forward outputs exactly match the reference SharedGroups implementation at the initial checkpoints. A separate double-precision check compares actual runner forward code and all continuous parameter gradients against two independent reference models; maximum gradient discrepancy6.94e-18. Every candidate retains exactly3memberships per group. All source hashes unchanged, finite results, no original-model forwards/new documents, reserved LLM test split untouched.

Next move beyond this conditional toy rather than over-optimize it: test genuinely learned gates with a sign-symmetric detector appropriate for Gaussian latent directions, keeping teacher-gate controls; test trained hierarchy and selective shared-computation reuse; compare reconstruction/active-dimension/parameter budgets. Then apply verified candidates to early LLM activations and the original stability, manifold and causal-computation investigation. Broad goal active.
