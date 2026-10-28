# Polynomial degree of Layer-0/1 token-edit trajectories

Vector-valued paths are sampled at five fit nodes and four held-out interleaved nodes along the straight source-to-target embedding interpolation. Per-example polynomial coefficients are fit jointly across all vector coordinates. Fidelity is normalized by held-out delta energy from the unedited state, so the large unchanged baseline cannot make the score look artificially good.

## Locked hypotheses

- **P0_product_path_is_quadratic: supported.**
- **P1_quadratic_materially_beats_linear: not supported.**
- **P2_cubic_adds_little: supported.**
- **P3_product_path_is_nearly_planar: supported.**
- **P4_midpoint_derivative_is_a_secant_program: not supported.**
- **P5_quadratic_write_description_transfers_ood: supported.**
- **P6_curvature_is_material: supported.**

| Layer | Corpus | Product polynomial d1/d2/d3/d4 | Pre d2 | Write d2 | Product rank-2 energy | Quadratic coefficient share | Midpoint secant error |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.967/0.997/1.000/1.000 | 0.996 | 0.997 | 0.9993 | 0.440 | 0.386 |
| 0 | tinystories | 0.977/0.997/1.000/1.000 | 0.997 | 0.997 | 0.9995 | 0.431 | 0.376 |
| 1 | fineweb | 0.972/0.997/0.999/1.000 | 0.997 | 0.997 | 0.9966 | 0.445 | 0.338 |
| 1 | tinystories | 0.963/0.997/0.999/1.000 | 0.997 | 0.996 | 0.9950 | 0.461 | 0.356 |

A low-degree path is an edit-local description, not a global semantic coordinate system. Its use is to compile a specified finite edit into native execution coordinates; it does not imply that unrelated edits share neuron identities.
