# Live-score ledger: visible arithmetic, but not yet a learned rule

The retrospective sports-article pilots had no reliable visible score history from which to compute an in-game run. This separate **live-feed genre** prints a score ledger after each scoring play. It makes the rule unambiguous: for the queried team, `Run` is the final printed score minus the first printed score. `Latest` copies the final printed home score. It is a controlled task surface, **not** a claim that this format resembles a sports article or FineWeb prose.

Each sampled document uses a 4–7-play within-period window from a real 2016/17 NBA play-by-play game. The `described` arm retains the source play descriptions; the `bare` arm renders each action as `Play: TEAM scored N`. Both print the same score lines and answer sites. All tokens except padding contribute to training cross-entropy; the answer sites are evaluation probes, not supervised loss masks. Whole games are partitioned between train and heldout pools.

The [generator](sports_live_ledger_dgp.py), [audit](audit_sports_live_ledger_dgp.py), [trainer](train_sports_live_ledger.py), and [causal edit evaluator](eval_sports_live_ledger_edits.py) make the pilot reproducible with the local NBA event archive. The 200-world-per-source/split [audit](results/sports_live_ledger_audit_n200.json) checked 4,800 visible answer sites across the paired arms with no failures. The described arm has more varied play text, but the repeated `Score`/`Play` wrapper remains conspicuous: repeated cross-document trigrams in its first 80 words range roughly .64–.78 across the four source/split panels. It has not passed the natural-prose form audit.

The [training results](results/sports_live_ledger_copy_control_bare_1000.json) and [described-arm results](results/sports_live_ledger_copy_control_described_1000.json) use 5,000 sampled training documents, 400 heldout documents, 1,000 AdamW steps, width 48, four heads, and bilinear depths 2 and 4. The RotoWire arms drew from 739 distinct training games and 186 heldout games; heldout OOV token share was negligible. The table gives heldout **number-candidate** accuracy (copy candidates 0–200; run candidates 0–40), not full next-token accuracy.

| Surface / depth | Latest copy | Run first | Run second |
|---|---:|---:|---:|
| Most-common training answer | .020 | .280 | .348 |
| Bare / 2 | .658 | .413 | .428 |
| Bare / 4 | .845 | .480 | .583 |
| Described / 2 | .148 | .340 | .395 |
| Described / 4 | .713 | .458 | .430 |

These results establish that depth 4 learns some heldout score-site behavior, especially copying, but answer accuracy alone cannot distinguish subtraction from distributional shortcuts. The [edit panel](results/sports_live_ledger_edits_n100.json) uses 100 additional heldout-game samples per arm and model:

* **Score offset:** add one to every printed home score. The `Latest` answer rises by one; both `Run` answers stay fixed. Bare depth 4 predicts the copy increase in 78/100 cases and preserves the two run predictions in 89/100 and 87/100. Described depth 4 gives 68/100, 86/100, and 86/100. This supports some sensitivity to score values and some offset invariance.
* **Coherent play edit, bare arm:** increase one `Play: TEAM scored N` amount by one and increase every subsequent printed score for that team. Exactly one run target rises; the other stays fixed. The bare depth-4 model raises the affected run prediction in only **44/200 (22%)** cases; depth 2 does so in **14/200 (7%)**. On unaffected run targets, depth 4 preserves its prediction in **123/200 (61.5%)** cases. For depth 4, edited accuracy on affected targets drops from 55% to 24%. The edited point values remain within 2–4, but this intervention changes a sampled empirical event and its wrapper, so it is a controlled generalization test rather than an in-distribution event.

The earlier unstratified edit average was misleading because it pooled targets that should increase with targets that should remain unchanged. The stratified result is the one to use. The model has **not** passed a causal arithmetic gate. It is premature to interpret weight factors as an implementation of the run subtraction rule. `Latest` copy is a better current candidate for weight-method benchmarking, with the same behavioral and causal caveats.

Next: vary the score and action sampling so affected-run edits lie in a better covered part of the training distribution, then repeat the stratified intervention on disjoint games. If the run response remains weak, treat that rule as not learned and use the copy site plus independently learned tasks for initial canonicalization/sparsity comparisons. Separately improve the document/genre renderer and re-run the held-out natural-form audit before calling this a realistic language distribution.
