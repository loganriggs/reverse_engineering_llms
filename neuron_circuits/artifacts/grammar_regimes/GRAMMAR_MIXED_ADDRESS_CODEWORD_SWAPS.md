# Valid mixed-address codeword swaps

Complete normalized states from real evaluation examples are swapped before the original final bilinear tensor. `answer_accuracy` scores the recipient's original answer; `donor_answer_accuracy` scores the answer associated with the donor state.

## Locked hypotheses

- **Q0_native_tensor_is_exact: supported.**
- **Q1_same_cell_codewords_are_functionally_equivalent: not supported.**
- **Q2_answer_equivalence_crosses_relations: not supported.**
- **Q3_wrong_answer_codeword_routes_its_donor_answer: not supported.**
- **Q4_answer_routing_crosses_relations: not supported.**
- **Q5_wrong_answer_codeword_overrides_target: supported.**
- **Q6_relation_is_not_a_required_final_controller: supported.**
- **Q7_rule_and_arbitrary_codewords_share_the_quotient: not supported.**
- **Q8_template_transfer_is_stable: supported.**
- **Q9_signal_gate_passes: supported.**

| Seed | Status | same-cell fid | same-answer/new-relation fid | next-answer/same-relation target/donor acc | next-answer/new-relation target/donor acc |
|---:|---|---:|---:|---:|---:|
| 0 | unselected | 0.665 | 0.580 | 0.166/0.682 | 0.160/0.675 |
| 0 | selected | 0.495 | 0.496 | 0.055/0.828 | 0.070/0.811 |
| 101 | unselected | 0.666 | 0.626 | 0.132/0.729 | 0.102/0.745 |
| 101 | selected | 0.409 | 0.431 | 0.052/0.822 | 0.043/0.835 |
| 104 | unselected | 0.578 | 0.501 | 0.230/0.623 | 0.230/0.617 |
| 104 | selected | 0.435 | 0.428 | 0.121/0.699 | 0.106/0.696 |

These are quotient tests: high same-answer fidelity means many geometrically different states implement the same downstream operation. It does not imply that the upstream model computes those states in the same way.
