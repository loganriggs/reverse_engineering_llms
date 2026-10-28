# Preregistration: sources of the Layer-17 context register

The embedding-router red team found genuine lexical generalization. The model
also exposes an architectural explanation: before the final attention/MLP,
Layer 17 forms

```text
pre = lambda_carried * x_16 + lambda_embedding * x0 + attention(...)
```

with learned coefficients approximately 1.02 and 8.0. This experiment freezes
the rank-16 tensor and splits the exact pre-normalization state into carried,
direct-embedding, and final-attention sources.

Position-conditioned source means are fit on the discovery half. Every one of
the eight source subsets is then reconstructed as the mean pre-state plus the
native variation of included sources, passed through the real RMS normalization,
projected into the context register, and contracted into the local operator.
Subset Shapley allocations and single-source mean replacements use this same
finite nonlinear execution.

This is an attribution of variation around a declared discovery baseline, not
a claim that information is stored in mutually exclusive percentages. Earlier
layers have already transformed and repeatedly reinjected `x0`, so the carried
source can contain lexical information too. The test localizes the final
operator-control boundary only.
