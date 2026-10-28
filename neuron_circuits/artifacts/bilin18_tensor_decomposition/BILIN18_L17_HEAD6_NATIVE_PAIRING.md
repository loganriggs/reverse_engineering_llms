

## H354 — Native MLP validation of context pairing

Reuse H346 head6 payloads and H353 confidence-matched donor contexts. Compute full native MLP response B(z,z)−B(x,x), without projecting it into the compiler span. This excludes the direct head residual path. Evaluate exact directional Fisher energy at each own or matched intact final state, with RMS derivative, unembedding, cap derivative and probability covariance. Compute compiler responses through the same directional pipeline as a replay control.

Divide by Euclidean response energy and the same compiler-output-span metric trace used in H353, then average equally across sites and realizations. The output response is now native, but this scale convention is not full residual-space trace normalization. It preserves comparability with the previous experiment and remains a dependency to test.

|Response|Metric|Moment control/actual [95% interval]|Donor control/actual [95% interval]|
|---|---|---:|---:|
|compiled|own|1.878 [1.654, 2.152]|1.788 [1.581, 2.024]|
|compiled|matched|1.269 [1.177, 1.376]|1.211 [1.117, 1.311]|
|native_mlp|own|1.817 [1.632, 2.032]|1.666 [1.493, 1.857]|
|native_mlp|matched|1.295 [1.203, 1.393]|1.211 [1.125, 1.298]|

Native own/matched ratios are1.404 [1.260,1.578] for moment controls and1.376 [1.233,1.545] for donors. Both frozen gates pass: native own/matched lower95%>1.2 and native own control/actual lower95%>1.2, both families. The pairing effect survives removal of the compiler restriction on output responses, though it is somewhat smaller. This weakens H352 candidate16 as a complete explanation, without proving semantic coordination or eliminating the normalization dependency.

Uncertainty:4000 joint document resamples plus independently resampled global control realizations and matched-neighbor ranks. Pointwise intervals conditional on fixed neighbor selection and donor assignments, whose dependencies are not reconstructed. Reused panel, not a new independent confirmation. Native MLP response is distinct from full native source intervention, which includes the direct path.

Audit: compiler directional-metric results replay H353 within relative error 2.34e-05; native computation’s compiler response agrees with cached coefficient output within 7.73e-07. All full native and compiler response vectors, donor indices, reference traces and per-site Rayleigh values saved. Native weights unchanged.

Next: remove the shared compiler-span scale convention by using unnormalized metric energy and scale-invariant paired site comparisons. If pairing survives, freeze a native statistic for independent confirmation before extending to other heads or semantic tasks. Broad research goal remains active.
