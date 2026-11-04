# Joint code, list, and bracket training: partial coexistence

The [code-only continuation](CORPUS_NATURAL_CODE_TRAINING_RESULT.md) made naturally repeated codes causally useful but damaged the same model's list and bracket behavior. This follow-up tests whether rehearsing all three rule families in a joint stream prevents that interference. It is a behavioral gate before weight-based analysis, not an interpretation result.

The [joint trainer](train_corpus_natural_code_joint.py) continues the same two depth-4, width-128 bilinear starting checkpoints for 4,000 AdamW updates. Every batch of eight has two **unedited natural-code** chunks, two consistently edited code chunks, two list chunks, and two bracket chunks. All rows get full-token next-token cross-entropy, with marked-site weights 200 for codes and 500 for lists/brackets. The code sources are the same 1,194 distinct FineWeb documents from offsets 393k–413k as the earlier matched training run. List and bracket training rows come from their existing source-disjoint caches. The code edit arm is explicitly alternate-world augmentation; its factual semantics are not claimed to be natural FineWeb.

Fresh ordinary-document-order evaluation uses two code-copy panels (317/334 sources) and two list/bracket panels (list 50/61, bracket 1,533/1,532 sources). No evaluation source enters this continuation. The trainer saves resumable states and evaluates at 0, 2,000, and 4,000 updates. At 4,000 it has drawn 16,000 code rows, matching the *code-row exposure* of the earlier code-only 2,000-step, batch-eight run; the joint run also draws 8,000 list and 8,000 bracket rows, so total optimization steps and non-code exposure differ.

The figures below show the two fresh panels as `panel 1 / panel 2`. Code pair two-choice requires the model to prefer the matching code in **both** natural and edited worlds. Code pair exact requires the correct full-vocabulary top token in both. List pair exact and bracket paired closing-character are likewise both-world measures.

| Seed | Updates | Code pair two-choice | Code pair exact | List pair exact | Bracket paired closing-character |
|---:|---:|---:|---:|---:|---:|
| 72104 | Start | .003 / .003 | .000 / .000 | .960 / .918 | .837 / .843 |
| 72104 | 2,000 | .189 / .165 | .028 / .015 | .980 / .951 | .875 / .877 |
| 72104 | 4,000 | .467 / .434 | .076 / .063 | .940 / .934 | .697 / .691 |
| 72105 | Start | .003 / .000 | .000 / .000 | .980 / .984 | .821 / .807 |
| 72105 | 2,000 | .366 / .329 | .088 / .090 | .980 / .967 | .774 / .771 |
| 72105 | 4,000 | .445 / .383 | .110 / .108 | .980 / .967 | .747 / .747 |

The joint allocation largely protects list behavior while code copying improves, a meaningful contrast with code-only paired continuation, where list exact fell to .20–.54. Bracket closing-character behavior is more fragile. Two-choice bracket scores remain around .93–.96 at 4,000 steps, but that easier measure conceals a 6–15-point drop in the more meaningful correct-character pair score. Selected bracket-chunk background CE is around 7.04–7.08 at 4,000 versus 7.18–7.23 initially; background prediction and bracket character behavior can move in opposite directions.

Code behavior is still incomplete. At 4,000 updates, strict two-choice on **new source–edit code combinations** is .398–.412 for seed 72104 and .350–.412 for 72105 across panels. On entirely unseen source codes it is at most .098. Full-vocabulary code pair exact is only .063–.110. Causal edit-response direction is positive on roughly .90–.94 of code pairs, again much easier than exact generation. The paired code-only condition at matched code-row exposure reaches about .47–.56 strict two-choice but destroys list behavior. The joint condition improves task coexistence at a cost to code accuracy and partial bracket retention.

**Decision:** these are useful partial three-rule models, but no checkpoint passes a strong all-three-rule gate. The 2,000-step checkpoints retain list/bracket well but have weak code copying; the 4,000-step checkpoints gain code copying while bracket character behavior degrades. Do not claim a recovered natural-code circuit or use code accuracy alone to select a canonicalization prior. The next broad experiment should vary DGP task prevalence and model capacity jointly, keeping an unedited source-level reference arm, source/code-combination holdouts, and full-vocabulary plus causal metrics. For immediate weight-method tests, list and bracket remain the behaviorally qualified rules; copying is a partial-learning/transfer challenge. Results are in [seed 72104](results/corpus_natural_code_joint_seed72104_s4000.json) and [seed 72105](results/corpus_natural_code_joint_seed72105_s4000.json). Reconstructable token caches and model checkpoints remain local.
