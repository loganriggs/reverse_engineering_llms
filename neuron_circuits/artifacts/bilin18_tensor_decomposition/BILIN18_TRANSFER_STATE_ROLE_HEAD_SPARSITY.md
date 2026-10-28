# Native-head sparsity inside the four-layer Q+V program

The 36 Layer-7--10 head invocations in the supported query+value recurrence are ranked only by leave-one-head-out weighted suffix-adjoint error on two discovery contexts. Frozen top-k and random-k sets are evaluated on two disjoint contexts. Each program must generate both its read sites and local rank-16 payload bases before held-out candidate-operator interventions.

## Locked hypotheses

- **H0_all_head_program_recomposes_near4_qv: supported.**
- **H1_head_importance_is_context_stable: supported.**
- **H2_top8_heads_are_behaviorally_sufficient: supported.**
- **H3_top8_heads_beat_random: not supported.**
- **H4_top16_heads_are_behaviorally_sufficient: supported.**
- **H5_selected_heads_span_multiple_layers: supported.**
- **H6_native_head_sparsity_is_supported: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Selected top eight

L9H7, L7H8, L7H0, L9H8, L8H1, L8H2, L9H1, L8H8

## Held-out causal fidelity

| Target | top4 | top8 | top16 | top24 | top36 | random4 | random8 | random16 | random24 | random36 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0H7 | 0.489 | 0.498 | 0.546 | 0.584 | 0.593 | 0.479 | 0.494 | 0.510 | 0.556 | 0.593 |
| L4H2 | 0.453 | 0.462 | 0.475 | 0.480 | 0.490 | 0.431 | 0.457 | 0.478 | 0.493 | 0.490 |
| L5H1 | 0.335 | 0.361 | 0.358 | 0.380 | 0.400 | 0.348 | 0.358 | 0.398 | 0.412 | 0.400 |

## Head importance

| Rank | Invocation | Discovery LOO error | Evaluation LOO error |
|---:|---|---:|---:|
| 1 | L9H7 | 0.13629 | 0.13871 |
| 2 | L7H8 | 0.13622 | 0.16128 |
| 3 | L7H0 | 0.12503 | 0.11123 |
| 4 | L9H8 | 0.08642 | 0.05999 |
| 5 | L8H1 | 0.07934 | 0.09491 |
| 6 | L8H2 | 0.06324 | 0.03688 |
| 7 | L9H1 | 0.06193 | 0.10669 |
| 8 | L8H8 | 0.06151 | 0.04793 |
| 9 | L10H3 | 0.05241 | 0.03354 |
| 10 | L7H1 | 0.05152 | 0.05582 |
| 11 | L8H3 | 0.05104 | 0.03763 |
| 12 | L7H3 | 0.05042 | 0.03349 |
| 13 | L10H4 | 0.04893 | 0.03379 |
| 14 | L8H7 | 0.04830 | 0.02213 |
| 15 | L9H6 | 0.04634 | 0.11102 |
| 16 | L7H2 | 0.04320 | 0.06436 |
| 17 | L10H5 | 0.04222 | 0.09240 |
| 18 | L9H0 | 0.04114 | 0.04675 |
| 19 | L7H5 | 0.04030 | 0.06956 |
| 20 | L7H7 | 0.04003 | 0.05281 |
| 21 | L9H5 | 0.03826 | 0.05045 |
| 22 | L10H2 | 0.03750 | 0.05208 |
| 23 | L7H4 | 0.03685 | 0.03608 |
| 24 | L9H4 | 0.03682 | 0.04062 |
| 25 | L10H1 | 0.03264 | 0.03489 |
| 26 | L9H3 | 0.03125 | 0.05844 |
| 27 | L8H4 | 0.02601 | 0.03352 |
| 28 | L7H6 | 0.02442 | 0.04427 |
| 29 | L8H6 | 0.02428 | 0.02371 |
| 30 | L9H2 | 0.02148 | 0.03798 |
| 31 | L10H8 | 0.02113 | 0.02417 |
| 32 | L10H0 | 0.01936 | 0.01643 |
| 33 | L8H5 | 0.01519 | 0.04664 |
| 34 | L10H6 | 0.01342 | 0.02004 |
| 35 | L10H7 | 0.00709 | 0.01043 |
| 36 | L8H0 | 0.00464 | 0.00635 |

## Guardrails

Leave-one-out error is conditional on the other 35 invocations and is not additive importance. A sparse top-k win supports a reusable native execution subset for this Q+V program; failure can mean context-dependent routing or distributed redundancy rather than absence of head-level structure. Only two contexts are used per split, so ranking stability and matched random controls are required before interpreting individual heads.
