# Assay-benchmark design changes

## Smoke-test corrections before the five-seed run

1. The first local-atlas generator had a dominant common rank-four base, so the supposedly invalid global rank-four explanation achieved 0.884 fidelity. The common base was reduced to 10% scale so the planted chart genuinely rotates with state.
2. The taut constraint null was anticorrelated at -0.139 because wrong clusters have disjoint support. The gate had incorrectly used absolute correlation. It now asks whether a wrong-source control has positive predictive correlation, which is the intended null test.

Both corrections were made after a one-seed, 1,024-example smoke run and before the first five-seed, 4,096-example run.

## Injection correction after the first five-seed run

The pure-mechanism gates all passed, but the native-axis detection curve remained positive at a nominal 6.25% signal. Inspection found that decoder rows outside the 32 planted channels were zero. The “dense” activation background therefore could not produce output through non-planted channels and was not a genuine dense causal background.

The decoder now has random normalized writes on the remaining 96 channels. These rows are inactive in the pure planted mechanism but carry the dense injection. The original pure thresholds are unchanged. Signal fractions 0.9 and 0.75 were added between 1.0 and 0.5 to resolve the detection boundary exposed by the first run. The corrected rerun overwrites the main result; the invalid first-run values remain described here rather than being silently forgotten.
