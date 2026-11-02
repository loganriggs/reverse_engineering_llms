# Source-attribution curriculum in the joint toy language

The v6 generator mixes ordinary narrative rules with source claims such as
`As Bob said, "the coat was missing".` and later asks
`Who said the coat was missing? Answer: Bob.` The original two-layer joint
model largely ignored the queried claim: on a common unrestricted 1,000-stream
panel, its source-answer accuracy was .442 versus a .465 query-blind Bayes
baseline, and changing the queried object moved its answer distribution by
only .0026 total variation (oracle: 1). An unrelated speaker edit moved it by
.503 (oracle: 0). Source-answer loss alone would hide this failure.

## Controlled source curriculum

The source lookup pretrainer uses the **same v6 vocabulary and two-layer,
width-64, four-head RMSNorm bilinear architecture** as the joint model. Its
short, 56-token prefixes contain two source claims after a fixed valid
narrative clause. It trains on the correct next answer token for 3,000
updates. Both the pretrainer and the subsequent joint generator exclude
speaker–claim-object pairings where `(speaker_index + object_index) % 4 == 0`.
This is a compositional pair holdout, not a whole-task holdout. An exact-oracle
audit replayed 128,000 restricted generator prefixes, including 482 source
answers; no excluded statement pair appeared. The pretrainer reached .994
answer accuracy on 2,048 short held-out-pair examples, with query-edit TV
.997 (oracle 1) and irrelevant-speaker TV .008 (oracle 0).

The short skill did **not** work by itself on full, varied joint streams. On
the common unrestricted panel its answer accuracy was .444 versus the .465
key-blind baseline; its .550 query-edit TV often pointed in the wrong
direction (effect L1 error 2.164). This makes joint transfer a genuine test,
not a claim that the short pretraining task already solves v6.

The restricted joint fine-tune uses ordinary next-token CE plus the v6 verb,
object and source soft targets and source counterfactual loss. At 1,000 joint
updates, its restricted development source-answer KL is .0481. The more
important unrestricted, full-stream causal panels are:

| 1,000 joint updates | Source accuracy | Key-blind accuracy | Query-edit TV / oracle | Query effect L1 | Irrelevant-speaker TV / oracle |
|---|---:|---:|---:|---:|---:|
| Stream seed 23001; all 545 questions | .974 | .465 | .949 / 1 | .163 | .042 / 0 |
| Stream seed 23001; 140 withheld-pair questions | .921 | .468 | .887 / 1 | .323 | .097 / 0 |
| Stream seed 23002; all 450 questions | .980 | .478 | .964 / 1 | .142 | .046 / 0 |
| Stream seed 23002; 111 withheld-pair questions | .955 | .489 | .954 / 1 | .228 | .090 / 0 |

The held-out-pair subgroup is selected by the **original queried claim** in
the unrestricted evaluation stream. The query edit can switch to another
claim; the relevant and irrelevant speaker edits preserve oracle validity.
This is one model seed with two independent evaluation streams, not evidence
of training-seed reliability. It establishes a competent joint source task
at the first saved checkpoint, including compositional pair transfer and
substantially better causal specificity than the original v6 model.

At **2,000 joint updates**, the same model improves further. On the two
unrestricted panels, source accuracy is .996/.998; query-edit effect L1 error
is .019/.019, and irrelevant-speaker TV is .0039/.0056. The withheld-pair
subgroups have .986/1.000 accuracy, query-effect L1 .050/.041, and
irrelevant-speaker TV .011/.012. In contrast, the matched restricted scratch
model's first 1,000-update checkpoint has .417 source accuracy and query
effect L1 **2.253** on the common panel; its withheld-pair accuracy is only
.043. At 2,000 scratch updates, its common-panel source accuracy remains
.413, with query effect L1 **2.210** and withheld-pair accuracy .036. This is
an early-learning advantage for the curriculum, not a final comparison:
scratch has better restricted-development verb and object KL at 2,000 steps
(.232/.442 versus curriculum .263/.727), and its 6,000-step trajectory is
still running. A [local bond-basis comparison](V6_SOURCE_BOND_BASIS_RESULTS.md)
finds that task-conditioned eight-dimensional directions preserve the
2,000-step source effect, while task-blind stream PCA does not.

The 3,000-step curriculum model remains good but is weaker on source edits
than its 2,000-step checkpoint: full-panel source accuracy is .989/.989 and
query-effect L1 .066/.052. Its restricted development source KL also rises
from .0012 to .0663, so selecting step 2,000 by that development metric does
not require peeking at the unrestricted test panels.

A separate **unrestricted** 1,024-stream oracle-KL panel shows the multi-rule
tradeoff. Lower is better:

| Model | Verb KL | Object KL | Place-answer KL | Source-answer KL |
|---|---:|---:|---:|---:|
| Original v6 scratch baseline, 3,000 updates | .199 | .391 | 2.010 | .979 |
| Curriculum joint, 2,000 updates | .260 | .696 | 3.939 | **.013** |
| Curriculum joint, 3,000 updates | .258 | .635 | 2.133 | .028 |
| Restricted scratch, 1,000 updates | .292 | .558 | 2.865 | 1.468 |

The curriculum solves a previously failed source rule, but the other rules
are worse than in the old checkpoint. No current v6 checkpoint is a strong
teacher for *all* four families. A lower-learning-rate, rebalanced second
stage starts from the selected 2,000-step checkpoint and tests whether source
binding can be retained while verb, object and place prediction improve.

## Next decision gates

The 3,000-update curriculum run is complete; a restricted 6,000-update
scratch control and the rebalanced second stage are running on CPU. Compare
their full saved learning curves on the *same*
unrestricted causal panels before attributing the gain to pretraining. Then
check whether the improved source rule coexists with competent verb and
object rules; high source accuracy alone is not a competent joint teacher.
Only after those gates should a canonical basis or sparse circuit be judged
on source, verb and object effects, including irrelevant-edit controls and
whole-rule-family featurizer holdouts. Keep prediction, teacher fidelity,
activation reconstruction and causal fidelity separate. This source surface
is a controlled exact-oracle benchmark, not a validated model of FineWeb.

Reproduce with `pretrain_joint_source_lookup.py`, the
`--exclude-source-pairs` and `--save-snapshots` flags in
`train_surface_mixture.py`, and `eval_surface_mixture_v6_sources.py`.
Current records are in `results/v6_source_lookup_pretrain_pairholdout_seed0/`,
`results/v6_source_pair_holdout_audit.json`,
`results/v6_source_curriculum_common_panel_n1000.json`, and
`results/v6_source_joint_step{1000,2000,3000}_*_n1000.json`. The shared
multi-rule oracle panel is `results/v6_source_joint_rule_kl_n1024_seed23201.json`.
