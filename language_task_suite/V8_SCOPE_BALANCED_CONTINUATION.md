# Balanced continuation of the section-lookup curriculum

The [short-to-long v8 curriculum](V8_SCOPE_CURRICULUM_TRANSFER.md) taught a
three-layer bilinear model to use the requested section key, but its joint
checkpoint regressed on verb, object and source rules. This follow-up tests
whether task-balanced supervision restores those rules while preserving the
section mechanism. It is still **one training seed** and a continuation,
not an independent replication or matched-compute architecture comparison.

We continued the 1,200-joint-step curriculum checkpoint for another 1,200
updates at learning rate .001, batch 32, length 256. The vocabulary,
generator parameters and width-64, four-head, RMSNorm, three-layer bilinear
model were unchanged. Ordinary next-token CE remained in the objective;
exact-oracle soft targets received weight 2 at each verb, object, object-place
answer, source-answer and section-answer site, with weight 1 for supported
counterfactual question, source and section answers. The development panel
had 64 streams. Checkpoints and full command arguments are in
[`results/v8_scope_balanced_continue_seed101/results.json`](results/v8_scope_balanced_continue_seed101/results.json).

All three checkpoints were then evaluated on the *same* independent v8
streams. Lower exact-oracle next-token KL is better. A/B are two fresh
256-stream panels (seeds 39101/39102), separate from development.

| Rule site | Scratch KL A/B | Curriculum KL A/B | Balanced KL A/B |
|---|---:|---:|---:|
| Verb | .578 / .593 | .928 / .926 | **.547 / .562** |
| Object | **1.115 / 1.126** | 1.769 / 1.782 | 1.117 / **1.122** |
| Object-place answer | 4.146 / 4.512 | 4.828 / 5.131 | **3.947 / 4.033** |
| Source answer | 2.653 / 2.638 | 5.014 / 4.767 | **2.104 / 2.288** |
| Section answer | .970 / .904 | .491 / .487 | **.268 / .246** |

The joint-rule regression is repaired in this setting: object is approximately
back to scratch, while the other listed rules improve. This comparison
includes **2,700 prior curriculum/joint updates plus 1,200 balancing updates**
for the balanced model, versus 1,200 joint updates for scratch. It does not
isolate the balancing loss from additional training time.

Section-question causal tests use exact full-prefix-supported edits. Each
200-question panel is shared across checkpoints (seeds 39001/39002), with
key-blind Bayes accuracy .456/.454. TV is total variation between the model's
answer distributions before and after an edit. Oracle TV is 1 for changing
the requested key or its color, and 0 for changing an irrelevant section's
color.

| Checkpoint | Accuracy A/B | Key-edit TV A/B | Relevant-color TV A/B | Irrelevant-color TV A/B |
|---|---:|---:|---:|---:|
| Scratch | .475 / .460 | .001 / .001 | .528 / .512 | .509 / .501 |
| Curriculum | .740 / .770 | .606 / .604 | .743 / .718 | .369 / .329 |
| Balanced | **.885 / .905** | **.810 / .845** | **.847 / .875** | **.189 / .181** |

The strong key effect and reduced irrelevant-color response support better
scoped retrieval, beyond answer accuracy alone. The mechanism is still
imperfect: key-edit effect L1 error is .590/.460 from the oracle, and the
irrelevant edit should ideally have zero effect. Two- and three-section cases
also improve on a separate 300-question panel: balanced accuracy is .950
(`n=238`) and .800 (`n=50`), with key-edit TV .892 and .716. Four-section
accuracy is .500 (`n=12`), so there is not yet strong evidence of scaling to
more sections.

This is a useful **candidate joint teacher** for the next featurizer test,
not a finished explanation. Before treating it as reliable, replicate the
training with an independent seed and compare against a scratch model with
comparable joint-stage compute. The short curriculum did not withhold entire
section–color combinations; its synthetic surface and answer rates remain
unlike natural web-text prevalence. For featurizer work, preserve whole-rule
training holdouts and report reconstruction, teacher KL, exact-oracle
prediction, causal edit fidelity and new-task discovery separately. In
particular, a section-aware feature should mediate requested-key and relevant
color edits while suppressing irrelevant-color effects.

Raw [joint KL](results/v8_scope_balanced_continue_seed101/joint_rule_fresh39101.json),
[causal panels](results/v8_scope_balanced_continue_seed101/scope_fresh39001.json),
and [section-cardinality results](results/v8_scope_balanced_continue_seed101/cardinality_fresh39003.json)
are saved with the [final model](results/v8_scope_balanced_continue_seed101/model_depth3.pt).
