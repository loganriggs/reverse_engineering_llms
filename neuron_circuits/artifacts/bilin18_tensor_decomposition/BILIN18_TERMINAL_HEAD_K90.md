# Joint Layer-17 head-coalition k90

All nine complete post-boundary head changes define a terminal branch. Heads are added jointly in either output-conditioned tensor-score order or routing-norm order, with the final bilinear MLP and normalization recomputed after every prefix. Fidelity is the original-distribution Fisher reconstruction of the complete branch's logit change.

## Locked results

- **K0_all_head_joint_edit_closes: not supported.**
- **K1_spelling_terminal_branch_is_very_sparse: supported.**
- **K2_unrelated_terminal_branch_is_small_coalition: not supported.**
- **K3_routing_only_order_is_near_output_order: supported.**
- **K4_coherent_spelling_is_strictly_sparser: supported.**
- **K5_terminal_branch_aligns_with_natural_edit: supported.**

| Family/split | N | quotient k90 median/p90 | routing k90 median/p90 | branch→natural alignment/cosine |
|---|---:|---:|---:|---:|
| spelling/0 | 80 | 2.0/6.1 | 2.0/4.0 | 0.077/0.568 |
| spelling/1 | 80 | 2.0/6.1 | 2.0/5.0 | 0.071/0.569 |
| unrelated/0 | 80 | 5.0/8.0 | 5.0/6.1 | 0.020/0.226 |
| unrelated/1 | 80 | 5.0/8.0 | 5.0/6.0 | 0.025/0.310 |

## Numerical closure red team

The locked minimum-relative-fidelity closure gate fails. The failure is concentrated in tiny-denominator examples rather than a missing head: cumulative versus direct head summation differs by at most `4.6e-5` in attention state and `4.6e-5` in logits, while minimum branch Fisher energy reaches `4.5e-12`. The first-percentile all-head fidelity is `0.990--1.000` and mean all-head fidelity is `0.9995--1.0000`. The preregistered gate remains recorded as failed; absolute diagnostics support floating-point accumulation order as the cause.

## Interpretation boundary

This is sufficiency for the complete Layer-17 post-boundary attention branch, not for the entire natural token edit. The branch-to-natural alignment states how that local object relates to whole-model behavior. A low k90 is a behavioral quotient result: omitted heads may still alter logits in directions orthogonal to this branch metric.
