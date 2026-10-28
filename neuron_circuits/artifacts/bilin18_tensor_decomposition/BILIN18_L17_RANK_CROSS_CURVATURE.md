

## H367 — Probability-metric cancellation and finite curvature both matter

Reuse H366’s 128-document panel, selected heads 0/1/6, and 13 native MLP response conditions. Keep its post-final-RMS delta Δ fixed and split it into u=P₂Δ and v=(I−P₂)Δ using the frozen weight-defined top-2 axes. Evaluate targets a+tΔ and patches a, a+tu, a+tv at t=1, 1/4 and 1/16, directly through unembedding and cap. Amplitude scales the normalized-activation path, not a source-ablation strength. No second RMS normalization.

At baseline, let F be the probability-weighted Fisher metric including the logit-cap derivative. Compute Eᵤ=uᵀFu, Eᵥ=vᵀFv, C=2uᵀFv, and E=Eᵤ+Eᵥ+C. Local Keep2 recovery is 1−Eᵥ/E; local Remove2 recovery is 1−Eᵤ/E. The Euclidean projection components are orthogonal, but that does not force C=0 under F. This is signed algebraic accounting, not an independently manipulated causal mediator.

|Head|Family|Eᵤ/E|Eᵥ/E|C/E (joint 95% interval)|Keep2 local|Keep2 full amplitude|
|---|---|---:|---:|---|---:|---:|
|0|actual|1.014|0.748|-0.762 [-1.187, -0.337]|0.252|0.266|
|0|moment|0.500|1.098|-0.598 [-1.023, -0.173]|-0.098|-0.097|
|0|donor|0.447|1.081|-0.528 [-0.953, -0.103]|-0.081|-0.080|
|1|actual|0.775|1.091|-0.866 [-1.291, -0.442]|-0.091|-0.064|
|1|moment|0.711|1.083|-0.793 [-1.218, -0.368]|-0.083|-0.069|
|1|donor|0.717|1.105|-0.822 [-1.247, -0.397]|-0.105|-0.085|
|6|actual|1.041|1.181|-1.221 [-1.646, -0.796]|-0.181|-0.142|
|6|moment|1.114|0.809|-0.923 [-1.348, -0.498]|0.191|-0.008|
|6|donor|1.091|0.796|-0.887 [-1.312, -0.462]|0.204|0.019|

All nine simultaneous cross-term intervals are negative. The self terms sum to 1.53–2.22 times total energy, with a negative cross term bringing their sum back to one. This explains why isolated components can be poor approximations despite carrying substantial local energy. It does not establish semantic antagonism or that the model explicitly implements cancellation.

Maximum absolute deviation between finite and local recovery across all 18 head/family/patch combinations is 22.37 percentage points at amplitude 1, 6.24 percentage points at amplitude 0.25, 1.60 percentage points at amplitude 0.0625. The strongest finite correction occurs in head 6’s controls: Keep2 local recovery of 19–20% becomes approximately −1% to 2% at full amplitude; Remove2 worsens from roughly −9% to −11% locally to −31% to −34% at full amplitude. Cancellation persists in the small-amplitude limit, while curvature changes its behavioral consequences.

Frozen gates: {'small_amplitude_local': True, 'full_amplitude_local': False, 'head6_control_cancellation': True}. The small-amplitude <3pp agreement criterion passes; full-amplitude <5pp criterion fails; both head6 control cross-term joint upper bounds are below −0.2. These are tests on a reused panel, not independent confirmation. Cross intervals use 4,000 shared document/control bootstrap resamples, simultaneous over nine groups; fixed donor assignments remain conditional. Amplitude-difference gates use point statistics.

Audit: {'energy_identity_max_relative': 2.3961737993457124e-14, 'euclidean_cosine_max': 5.943226375854783e-16, 'h366_mean_kl_replay_max_relative': 1.2421340731385874e-05}. Float64 unembedding evaluation replays H366’s per-condition mean KL to 0.00125% relative error. Energy identity and Euclidean orthogonality pass near machine precision; finite KL values pass positivity/finite assertions; input hashes verified. Per-site energies, finite KLs, exact code/config and summary saved. No native model weights modified.

Hypothesis update: raise probability-metric component cancellation; retain finite curvature as an additional head-dependent effect; reject curvature as the sole explanation of failed sufficiency. Keep the earlier successful rank-2 pairing predictor scoped to a relative statistic, not a two-unit model of native computation.

Next discriminating experiment: separate probability weighting from the logit-cap derivative in the cross term using a 2×2 metric factorial. Under uniform vocabulary covariance without the cap, these weight eigenvectors should make the cross term approximately zero; that is a strong algebraic control. This can identify why weight-orthogonal components interfere in the actual probability metric before searching for alternative subspaces or semantic labels. Broader goal remains active.
