# Large-edit stress test of the analytic Jacobian

The weight-derived Layer-6-to-final MLP Jacobian was tested without refitting from 0.08 to 1.0 coordinate units. The median natural standard deviation of these coordinates is 0.536; thus the largest intervention is 1.86 median standard deviations. The primary metric scores positive and negative edits separately. An earlier draft used their central difference, which cancels even-order nonlinear terms; that implementation was caught during red-teaming and is retained only as a diagnostic.

## Locked hypotheses

- **S0_small_finite_extension_holds: supported.**
- **S1_quarter_unit_extension_holds: supported.**
- **S2_half_unit_extension_holds: supported.**
- **S3_unit_scale_retains_useful_prediction: supported.**
- **S4_breakdown_is_monotonic: supported.**
- **S5_scale_profile_transfers_ood: supported.**

| Corpus | One-sided 0.08 | 0.25 | 0.50 | 1.00 | Native at 1.00 | Central diagnostic at 1.00 |
|---|---:|---:|---:|---:|---:|---:|
| fineweb | 1.000 | 0.999 | 0.996 | 0.983 | 0.960 | 1.000 |
| wikitext | 1.000 | 0.999 | 0.997 | 0.987 | 0.957 | 1.000 |
| tinystories | 1.000 | 0.999 | 0.997 | 0.986 | 0.969 | 1.000 |

Failure at large scale would not refute the local mechanistic account; it would bound its chart size. Success far beyond a typical coordinate standard deviation would support treating one Jacobian chart as a behaviorally useful regime rather than an infinitesimal description.
