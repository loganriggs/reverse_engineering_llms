# Local OV canonical form: reconstruction improves without reliable rule transfer

The [multi-rule carrier pilot](SPORTS_REPORT_MULTIRULE_PILOT.md) produced two behaviorally competent copy rules in each of two bilinear-model seeds, with separate causally important second-layer heads. This experiment asks whether a **data-free canonical form** or an activation-weighted form can compress those heads while preserving both learned rules. It tests one exact local object: the head's linear value-output map $W_h=O_hV_h\in\mathbb R^{64\times64}$, whose rank is at most 16. QK routing, RoPE, the other heads and layers, and the readout stay native. This is invariant to an internal invertible change of basis between $V_h$ and $O_h$, but it is **not** a canonical decomposition of the complete attention function.

The [comparison script](compare_multirule_ov_rank_bases.py) uses three rank-$r$ maps:

- **Weight SVD:** the best Frobenius approximation of $W_h$ under the ordinary Euclidean metric, requiring no examples.
- **Score-only data SVD:** let $C=\mathbb E[xx^\top]$ at the normalized second-attention input, estimated on 256 train-group score-only documents. Truncate the SVD of $W_h(C+\epsilon I)^{1/2}$ and map back. This minimizes a regularized local expected output error on those inputs. One fit arm retains the human carrier (~36,561 token positions); a stricter arm uses only score result and score query sentences (~5,944 positions). Neither contains the *inserted player-copy question*, but the full-carrier arm may contain incidental human player-stat prose.
- **Random input subspace:** project $W_h$ onto a seeded random rank-$r$ input subspace, a capacity-matched null.

Each approximated map is refactored into the head's $O$ and $V$ matrices; all other weights remain fixed. The evaluation draws 64 new, unique held-out game keys per seed, with original, score-edited and player-edited documents. We report exact correctness on **both** original and edited answers, native-to-reduced output KL at the two answer positions, targeted/nuisance output TV, and the fit-map error. Score and player head reductions are tested separately and jointly. The score-only fit has no access to those held-out games. A rank-16 weight or data SVD reconstructs the map to relative Frobenius error below $5\times10^{-8}$ and the **joint** model to answer-site KL below $7\times10^{-9}$; this verifies the refactor and evaluation path.

## Joint reduction results

Each cell below gives **winning-score / player-points paired exact accuracy** when *both* critical heads are reduced to the indicated rank. Native accuracy on this evaluation panel is .906/.859 in seed 96701 and .938/.766 in seed 96702. The 64-game panel makes one case worth about .016 accuracy; small differences need caution.

| Seed | Method | Rank 8 | Rank 12 |
| --- | --- | --- | --- |
| 96701 | Weight SVD | .609 / .750 | .906 / .844 |
| 96701 | Full score-only data SVD | .531 / .750 | .906 / .812 |
| 96701 | Strict score-sentences-only SVD | .531 / .781 | .891 / .844 |
| 96701 | Random input | .031 / .047 | .062 / .031 |
| 96702 | Weight SVD | .641 / .781 | .906 / .781 |
| 96702 | Full score-only data SVD | .594 / .734 | .922 / .781 |
| 96702 | Strict score-sentences-only SVD | .719 / .750 | .922 / .781 |
| 96702 | Random input | .016 / .141 | .031 / .266 |

At rank 12, weight-only and both data-aware choices mostly retain both rules; rank 8 often loses many exact score answers. The data-aware SVD always lowers its **own fitted expected OV output error** relative to the ordinary SVD, as its objective predicts, but it does not consistently improve held-out behavioral fidelity. For example, seed 96701's rank-8 full-data score-head fit error falls from .423 to .383, yet joint score accuracy falls from .609 to .531 and score-site KL rises from .567 to .751. Conversely, in seed 96702 the strict score-only rank-8 arm raises score accuracy from .641 to .719 while lowering player accuracy from .781 to .750. At rank 12 the two seeds also disagree on the small KL/accuracy tradeoffs. This is evidence against treating local reconstruction quality as a reliable proxy for causal rule preservation, not evidence that distribution weighting is never useful.

Both heads can be reduced jointly without a large additional penalty beyond their individual losses on this panel. The other rule is nearly unchanged when only one critical head is reduced, consistent with the earlier head-level separation. These results do not establish that the head contains a unique algorithm or that the rank-$r$ singular vectors are individually human-legible. The [seed-1 full-carrier aggregate](results/sports_report_multirule_ov_rank_seed96701_fit256_eval64.json), [seed-2 full-carrier aggregate](results/sports_report_multirule_ov_rank_seed96702_fit256_eval64.json), [seed-1 strict-fit aggregate](results/sports_report_multirule_ov_rank_seed96701_strict_fit256_eval64.json), and [seed-2 strict-fit aggregate](results/sports_report_multirule_ov_rank_seed96702_strict_fit256_eval64.json) contain every rank, individual head, joint case, KL, edit TV, and reconstruction error. Model tensors and source prose remain local.

**Decision:** the useful invariant object here is the head's composed $OV$ map, and SVD supplies a verified exact full-rank gauge for it. A score-only distribution prior changes which rank-$r$ directions survive but has no replicated advantage over weight-only SVD in held-out two-rule behavior. Stop this local-OV detour at the two-seed comparison. The next interpretability test should include the **routing and readout path** (or an intervention-aware objective), with a predeclared fidelity budget and a rule-family fit holdout; meanwhile the primary DGP work must reduce the large task-window form gap before calling this a realistic language laboratory.
