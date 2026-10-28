# Preregistration: head decomposition of singular third-order paths

The pairwise Mobius audit finds thirteen below-0.8 causal failures among 2,688
triples. Cost-whitened Gram error, not greedy margin, identifies the exceptional
regime. This experiment decomposes each failed triple's exact third-order
response by attention head at the last selected layer.

For ordered layers `i < j < k`, the third-order response is

`T_ijk = R_ijk - R_ij - R_ik - R_jk + R_i + R_j + R_k - R_empty`.

At layer `k`, retain one attention-JVP head at a time in every term containing
`k`. Fixed-state linearity implies that the head-specific residuals sum exactly
to `T_ijk`. Every failure is paired with the same query and triple at the other
selected site from the same corpus.

The head identity is not chosen from results. Localization passes if the median
largest absolute projection share is at least 0.50 and effective absolute-share
support is at most 3. Amplification passes if failure relative residual norm is
at least five times its matched control. Recomposition error must be <=1e-4 for
every row before mechanistic conclusions are accepted.

This audit localizes the final attention factor of an ordered third-order path;
it does not by itself localize the earlier two factors or explain why the state
activates the path.
