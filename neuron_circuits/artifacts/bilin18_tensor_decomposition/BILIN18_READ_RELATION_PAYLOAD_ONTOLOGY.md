# Payload ontology of validated read relations

This separates the validated address variable from its payload function. For each candidate edge, the exact native edge write and the finite final-logit difference caused by removing that edge are collected. Subspaces are learned independently on two corpus splits. L7H8 and L9H7 are evaluated on the same adjacent-function-word edges.

## Locked results

- **O1_payload_write_subspaces_replicate: supported.**
- **O2_output_effect_subspaces_replicate: not supported.**
- **O3_shared_read_has_distinct_functions: supported.**
- **O4_shared_read_has_same_function: not supported.**
- **O5_candidate_payload_is_more_compressible: not supported.**
- **O6_boundary_payload_is_stable: supported.**

## Program summaries

| Program | write cross-capture | output cross-capture | mean-output split cosine | candidate/control output effective rank |
|---|---:|---:|---:|---:|
| function_L7H8 | 0.765 | 0.213 | 0.672 | 42.8/40.6/53.1/41.4 |
| function_L9H7 | 0.659 | 0.230 | 0.663 | 42.2/35.9/39.7/40.4 |
| boundary_L9H8 | 0.907 | 0.279 | 0.584 | 37.4/43.4/38.8/35.8 |

## Same address variable, two functions

Exact L7H8/L9H7 OV cosine: `-0.231`.
- Split 0: write-subspace overlap `0.142`, output-subspace overlap `0.298`, paired write cosine `0.513`, paired output cosine `0.136`.
- Split 1: write-subspace overlap `0.156`, output-subspace overlap `0.283`, paired write cosine `0.563`, paired output cosine `0.133`.

## Empirical output poles

- **function_L7H8** promotes `['agonist', 'igmatic', 'oppable', 'cedented', 'acter', 'ileged', 'urger', 'bial']` and suppresses `['ates', ' titles', ' versions', ' sentences', ' combinations', ' notebooks', ' paths', 'angers']` in its split-0 mean finite effect.
- **function_L9H7** promotes `['ownt', 'ungle', ' remembered', 'idden', 'iscopal', 'ynes', 'urbed', 'cyclopedia']` and suppresses `[' similar', ' example', 'wise', ' accordingly', ' identical', ' equivalent', ' sizes', ' name']` in its split-0 mean finite effect.
- **boundary_L9H8** promotes `['it', 'way', 'he', 'its', 'w', 'co', ' Users', 'sex']` and suppresses `['═', '--', 'Uncommon', 'aeda', 'itiveness', 'Palestinian', 'eanor', 'interrupted']` in its split-0 mean finite effect.

## Guardrails

A low-dimensional effect subspace is a compact behavior family, not necessarily a human concept. Mean logit poles can cancel context-dependent roles, while singular subspaces omit signs and may combine distinct uses. Exact OV cosine is weight-only but ignores QK addressing, normalization, and downstream nonlinear computation. Cross-split capture tests reusable geometry; it does not establish that the decoded token poles are a complete semantic explanation.
