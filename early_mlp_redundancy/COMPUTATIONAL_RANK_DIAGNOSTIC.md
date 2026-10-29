# Output dimension, computation, and redundancy are different quantities

M89 completed the frozen toy suite and native algebra diagnostic. All 16 decoder spans pass on all 4,092 unique training positions. No model was fitted and no evaluation data was accessed.

## A one-dimensional output can require many interactions

Consider f(x,y)=sum_i x_i y_i for independent k-dimensional x and y. Its output dimension is one. Its cross-coefficient matrix is the k-by-k identity, with rank k. Consequently, exact representation by a sum of separable products (a_j dot x)(b_j dot y) requires at least k products: each term contributes a rank-one matrix. This statement concerns that architecture and equality on an open input support, not all possible programs or data distributions.

Coordinate probes verify the identity matrix for k=1,2,4,8. On restricted data x=t*u, however, the same target becomes t*(u dot y): one product suffices. Numerical errors are below 1.5e-16 relative. Ambient weight complexity therefore need not equal on-data complexity.

## Shared output directions do not imply removable computation

Compare k channels computing distinct products x_i*y_i into the same scalar output with k copies of x_1*y_1/k. Both have one output direction. With independent standard normal inputs, deleting one channel gives population relative squared error 1/k for distinct products and 1/k² for copies. Neither is exactly invariant to raw deletion.

Rescaling survivors by k/(k-1) restores the copies exactly; it leaves error 1/(k-1) for the distinct-product case. For k=8, the 8,192-sample diagnostic measures raw deletion errors 0.12595 versus 0.015625. Compensation gives 0.14454 versus numerical zero. This distinguishes distributed contributions, duplicate computations, and compensated substitutability. It does not explain why a trained optimizer chose one implementation.

## Native projected computation is executable, but retains all products

For the native product write F(z)=W[(Lz)⊙(Rz)], an orthonormal basis Q of a learned decoder span, and its center c:

    projected(z) = (I-QQᵀ)c + Q (QᵀW) [(Lz)⊙(Rz)].

This factors computation into an input-product dictionary, a product-to-output-coordinate map, and output directions. It never needs to form the 1,152-dimensional native write first. The diagnostic compares this QR-based computation with an independently formed native write projected through an SVD basis. Across all 16 spans, maximum relative discrepancy is 1.92e-15 and maximum absolute discrepancy is 9.10e-12, below the frozen relative tolerance 1e-5.

The factorization retains **all 4,608 native products** and maps them to 128 output coordinates. It provides an exact implementation of the M88 projection function in real arithmetic, not a faithful implementation of the full native MLP. M88 already measured the projection's residual behavioral errors. Down_bias is excluded from both product-write formulas and remains an unchanged external addition.

The dense linear-map operation count falls from 15,925,248 to 11,354,112 multiply-accumulates per position because the output map factors through 128 dimensions. These counts exclude elementwise products, offsets, normalization, and memory traffic; they are not measured speedups. Input-product count is unchanged, and no minimal circuit or sparse causal unit has been found. Algebraic equality also does not guarantee identical float32 downstream execution; this stage checks float64 algebra, not another transformer replay.

## Consequence for the research direction

The M88 failure does not justify rejecting compact output features. Our one-product-per-coordinate predictors coupled two different choices: how many directions describe the output and how many interactions compute their coefficients. M89 proves those choices can be separated and verifies an exact native-derived interface. It does not establish whether capacity, optimization, or data coverage caused the learned predictors' failures.

Before fitting a larger encoder, the next useful question is whether native products feeding these output spaces are actually removable on data. A fixed deletion test can distinguish broadly distributed required contributions from concentration in a smaller subset, and report signed cross terms to reveal cancellation. Failure of one selection rule would not prove irreducibility. Raw deletion and compensated substitution must remain separate tests.

Stop this algebra diagnostic here. Next freeze a training-only product-contribution inventory and a bounded subset-deletion protocol, including native full-output and all 16 projected-output controls, before executing it. Rank by a prespecified training contribution-energy score, retain deterministic random controls and failures, and report both local geometry and the limits of its behavioral interpretation. No optimizer tuning or new training is authorized by this diagnostic alone. Subsequent causal validation and task transfer remain necessary before calling a component a reusable unit or manifold.

Evidence: [frozen protocol](validation/computational_rank_diagnostic_protocol.json), [results](validation/computational_rank_diagnostic.json), and [reproducible runner](diagnose_computational_rank.py). The native inputs and frozen states are hash-bound local artifacts; large tensors are not bundled in this packet.
