# Magnitude-equalized semantic-motif causal red team

Each semantic edge was only partially removed, so the norm of its removed local attention write equaled the fully removed nonsemantic control edge at the same head and destination.

## Outcomes

- **E1_recurrence_survives_equalization: not supported.**
- **E2_both_local_composition_motifs_survive: not supported.**
- **E3_boundary_survives_equalization: not supported.**

| family | sites | semantic fraction removed | equalized semantic KL | control KL | paired ratio | win fraction |
|---|---:|---:|---:|---:|---:|---:|
| recurrence_reference | 18 | 0.207 | 0.000181 | 0.000203 | 1.07 | 61.1% |
| local_composition_a | 18 | 0.168 | 0.000199 | 0.000208 | 1.03 | 50.0% |
| local_composition_b | 12 | 0.297 | 0.000337 | 0.000269 | 1.88 | 66.7% |
| boundary_multiscale | 30 | 0.354 | 0.000169 | 0.0002 | 0.88 | 40.0% |
