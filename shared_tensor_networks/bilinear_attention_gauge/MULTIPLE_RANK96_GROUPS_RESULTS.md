# One robust attention group; later energy groups are only partial candidates

The [first rank-96 group test](RANK96_GROUP_STABILITY_RESULTS.md) found a stable, high-energy four-axis subspace in the 546M model's layer-0/head-0 QK1 bond, even though its individual sparse axes varied across fit seeds. This follow-up asks whether that is one of **several** reproducible groups or merely the dominant mode. It tests two label-free rules on both the uniform weight-only cut and the FineWeb-weighted cut. Neither rule is allowed task labels or held-out corpus results when constructing groups.

## Disjoint energy-ranked quartets

Each exact-gauge basis is sorted by QK2-and-OV-weighted squared pair contribution on 16,384 FineWeb selection pairs. We form 12 disjoint four-axis quartets from the first 48 axes, then use a Hungarian assignment to match quartets between each identity-start and random-start fit by **projector overlap**, with the same assignment used to measure held-out function agreement. Six independently random orthogonal gauges undergo identical energy ranking and assignment as a chance-match control. The quartet width is supplied; the assay asks how many of the candidate groups survive rather than assuming an answer.

Mean cross-start projector overlap and held-out FineWeb QK1 group-response cosine, averaged over nine fit pairs:

| Cut / group rank | Sparse overlap | Random-gauge overlap | Sparse response cosine | Random response cosine |
|---|---:|---:|---:|---:|
| Uniform / first | **.657** | .067 | **.924** | .228 |
| Uniform / second | .369 | .067 | .489 | .140 |
| Uniform / third | .234 | .072 | .407 | .166 |
| Uniform / fourth | .149 | .068 | .255 | .175 |
| Corpus / first | **.664** | .069 | **.914** | .223 |
| Corpus / second | .252 | .073 | .449 | .175 |
| Corpus / third | .192 | .065 | .256 | .163 |
| Corpus / fourth | .123 | .056 | .230 | .153 |

The second group retains some function transfer: its sparse response cosine is .662/.468 on WikiText-103 and .486/.398 on TinyStories (uniform/corpus), above the corresponding random controls. The third group's transfer is weaker, and by later quartets its agreement approaches the assignment-enhanced random baseline. The Hungarian match is computed using all 12 candidate quartets and can inflate overlap, which is why the controls use the same assignment. It does **not** imply an unknown group count has been recovered.

## Full-model check for the second and third quartets

For each fitted and random basis, we partially delete the second or third quartet, calibrating every deletion to `.001` aggregate relative layer-0 head-output error on 64 separate FineWeb documents. We then measure per-document KL from the unedited rank-96 cut after all 18 layers on two seeds × eight documents per corpus. These are the same held-out document windows used for the first-group assay. The identity-bond logit gate is zero. Deletion fractions range .164–.307. The resulting mean KL effects are roughly (0.15–1.94) × 10⁻⁶ nats/token; the assay tests **effect-profile reproducibility**, not a large behavioral change.

Centered cross-start correlation of per-document KL effect vectors:

| Cut / group | FineWeb test | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Uniform / first, from previous panel | .896 | .806 | .884 |
| Uniform / second, same energy rank | .610 | .650 | .949 |
| Uniform / second, projector-matched rank | .576 | .732 | .921 |
| Corpus / first, from previous panel | .792 | .811 | .905 |
| Corpus / second, same energy rank | .223 | .388 | .047 |
| Corpus / second, projector-matched rank | .563 | .387 | .530 |

The second quartet has **partial** causal-profile transfer. For the corpus cut, matching its rank can improve FineWeb and TinyStories correlations, but they remain well below the first group. For the third quartet, cross-start matching is unavailable in 6/9 uniform and 3/9 corpus fit pairs because its best match falls outside the three quartets ablated. Among the available matches, the corpus cut's WikiText correlation is .084, below the random third-group control's .454. It would be misleading to count that quartet as recovered. The sample is small, and correlation can be driven by document-level sensitivity shared across many edits.

## Co-activation graph red team

As a different grouping rule, connect axes when the absolute Pearson correlation of their absolute pair contributions exceeds a threshold, then take connected components. A null threshold set to the **99.9th percentile of all off-diagonal correlations from three random gauges on the selection pairs** is .0944 for the uniform cut and .1051 for the corpus cut. At those thresholds, the sparse fits produce one component of **20–21 axes** (uniform) or **26–31 axes** (corpus), plus at most a two-axis component. This is not a clean multi-group partition. The random gauges produce only 2–7 or 3–6 edges respectively, so the sparse gauges do have more correlated activity. But a small threshold change alters the graph sharply: at .12 the sparse uniform graph's largest groups shrink to 4–5 axes, while the corpus graph varies from 9 to 18, and WikiText retains only about 42–62% of those corpus-fit sparse edges depending on cut. Threshold tuning could manufacture apparent groups without reproducible functional identities.

## Decision

The **first high-energy group** passes projector, held-out pair-function, and calibrated full-model effect-profile controls across starts, priors, and corpora. A second disjoint group is a candidate, especially under the uniform cut, but does not meet the same robustness standard. The third and later groups do not establish a repeatable decomposition. These findings narrow the next optimization problem: choose group structure using functional or downstream causal signatures directly, rather than only pair energy or co-activation. In a toy with known multiple computations, that method must recover complete groups and held-out combinations before it is trusted on the 546M model. On the real head, test group selection on one set of contexts and prediction of selective effects on entirely held-out contexts or tasks; preserve the natural and QK2-intervened fidelity gates for the underlying cut.

Reproduce with `flagship_rank96_multiple_groups.py`, `flagship_rank96_group_ablation.py --n-docs 8 --groups 1 2`, and `flagship_rank96_coactivity_graph.py` after the random-start gauge runs in the previous report. Outputs are `full_width_results/head0_rank96_multiple_groups.json`, `head0_rank96_group_ablation_n8_groups1_2.json`, and `head0_rank96_coactivity_graph.json`. The checkpoint and corpora are referenced externally.
