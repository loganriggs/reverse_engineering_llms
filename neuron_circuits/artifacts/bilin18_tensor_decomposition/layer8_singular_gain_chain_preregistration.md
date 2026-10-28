# Preregistration: gain chain of the singular ordered path

For the representative `[11,13,14]`, head-6, QK2/value-bus circuit, propagate
the ordered derivative path directly and compare the failure with its matched
query-1 control.

Measure four stages:

1. the pairwise interaction tangent from attention layers 11 and 13 at the
   normalized layer-14 attention input;
2. normalized head-6 QK2 query derivative per unit input tangent;
3. projected QK2/value-bus attention output per unit QK2 derivative;
4. final Fisher response per unit layer-14 post-MLP path tangent.

Each amplification hypothesis requires a failure/control gain ratio >=5. The
direct ordered path must match the prior subtraction-derived factor response
within 1e-3 relative error before stage attribution is accepted. Ratios can
multiply; more than one stage may contribute.
