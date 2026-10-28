# Input-decoder transfer under incoming-head removal

Remove each whole attention-head write at each selected destination, recompute RMSNorm and local MLP, and predict changes in four frozen output coordinates. Upstream state is fixed. This tests local finite response, not downstream behavioral effect or individual source-edge identity.

Locked gates: `{'exact': True, 'intervention_transfer': True, 'beats_affine': True}`. Head-write summation relative RMS error: `2.53e-07`.

| Layer | Affine finite-change fidelity | Quadratic16 | Quadratic64 |
|---|---:|---:|---:|
| 1 | 0.650 | 0.905 | 0.968 |
| 8 | 0.112 | 0.274 | 0.518 |
| 12 | 0.183 | 0.648 | 0.815 |
| 17 | 0.311 | 0.998 | 1.000 |

These fidelities predict intervention-induced changes, with no mean coefficient contributing to the denominator. All heads are evaluated, with natural effect-energy weighting. This neither identifies a unique semantic basis nor explains the upstream production of head values and routing.
