# Numerical red-team of the midpoint suffix JVP

The router's batched forward-mode derivative is compared with a single-example JVP, reverse-mode VJP duality under random output cotangents, and a finite-radius sweep. AD agreement tests implementation correctness without assuming that a finite 1% step is locally linear.

## Locked hypotheses

- **D0_batched_and_single_jvps_match: supported.**
- **D1_forward_reverse_ad_duality_holds: not supported.**
- **D2_a_smaller_finite_step_validates_the_derivative: not supported.**
- **D3_the_original_step_measures_finite_curvature: not supported.**
- **D4_suffix_curvature_increases_with_depth: not supported.**

| Layer | Corpus | Batch min cosine / max error | Duality max error | epsilon: min cosine / median error / max error |
|---:|---|---:|---:|---|
| 0 | fineweb | 0.9999999/3.13e-06 | 1.27e-04 | 0.01: 0.995530/0.0242/0.0954; 0.003: 0.918120/0.0787/0.3966; 0.001: 0.751301/0.2427/0.6602; 0.0003: 0.360258/0.5964/0.9329 |
| 0 | tinystories | 0.9999999/3.34e-06 | 4.13e-05 | 0.01: 0.987410/0.0223/0.1588; 0.003: 0.889531/0.0744/0.4572; 0.001: 0.361197/0.2676/0.9488; 0.0003: 0.146380/0.5546/1.0455 |
| 1 | fineweb | 0.9999999/3.22e-06 | 3.05e-05 | 0.01: 0.997688/0.0203/0.0682; 0.003: 0.982052/0.0778/0.1890; 0.001: 0.863382/0.2086/0.5051; 0.0003: 0.455161/0.5385/0.8908 |
| 1 | tinystories | 0.9999999/3.39e-06 | 1.24e-03 | 0.01: 0.997812/0.0179/0.0661; 0.003: 0.976729/0.0686/0.2167; 0.001: 0.726718/0.1699/0.6873; 0.0003: 0.473592/0.4405/0.8812 |

If batch/single equality and JVP/VJP duality pass while finite steps fail, the analytic derivative is valid and the failure measures suffix curvature at that intervention radius. The original finite-difference gate remains failed.
