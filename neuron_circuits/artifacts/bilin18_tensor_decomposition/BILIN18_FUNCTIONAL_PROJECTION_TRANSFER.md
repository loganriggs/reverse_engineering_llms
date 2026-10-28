## H1016 — Four functional read directions preserve finite number effects across intervention constructions

Previous turn was progress: H1015 identified aligned number-sensitive gradients and a recipient-gain limitation. This turn tests those coordinates as a causal projection, with no evaluation gradients, rather than treating favorable gradient geometry as validation.

Freeze one shared basis from SVD of48discovery is/are gradients: rank1andrank4. Compare against rank1/4uncentered PCAof discovery source directions and seeded random orthonormal rank1/4bases. All are fixed1152-dimensional directions; rank4storage4608numbers. Rank is matched, projected norm is not forced to match: this tests what each subspace retains, not equal-norm steering potency. Basis hashes saved before model evaluation.

Evaluate48discovery and32historical-test recipient states. Two input families: (a)original five-source changes from the rank-one number-decoder intervention; (b)natural opposite-number subject-token replacement, propagated through the native prefix, taking the resulting whole subject-position preL11residual difference. Both normalized to the same fixedrho1803.343; other recipient positions and buses stay fixed. Thus family(b)is derived from natural token changes, but the final edit is a rescaled single-position patch, not an unmodified end-to-end token substitution. It is independent construction within grammatical number, not independent task evidence.

| Projection | Decoder-family is/are error | Natural-token-family is/are error |
|---|---:|---:|
| Gradient1 |19.7%|20.1%|
| Gradient4 |7.7%|7.2%|
| SourcePCA1 |18.8%|12.5%|
| SourcePCA4 |14.6%|11.3%|
| Random1 |100.0%|100.0%|
| Random4 |99.4%|99.5%|

Errors are relative L2between projected-edit and full-edit finite is/are damage on32historical-test states. Full input effect RMS2.561/2.624; Gradient4effectRMS2.414/2.472. Rank4gradient projection retains only23.4/21.6%of edit squared norm on average. Its orthogonal remainder preserves most input-vector energy but has only5.42/4.70%of full number-effect RMS. Projection and remainder effects do not add perfectly:3.7–4.1%relative nonclosure, consistent with finite nonlinear interactions. Do not translate these results into a percent of all model computation understood.

Finite output specificity: for Gradient4, can/will and period/comma effectRMS0.051/0.073(decoder),0.044/0.098(natural), versus target2.414/2.472. Related was/wereRMS2.155/2.178. These are four fixed contrasts, not a whole-vocabulary collateral audit or a complete unrelated-task battery. Gradient projection need not preserve the full edit's small unrelated effects, and those small-effect relative errors are reported in the machine-readable results rather than hidden by target performance.

Post-result noun-cluster bootstrap resamples8test noun pairs, retaining both numbers and both templates together (2000draws,seed1016). SourcePCA4error minus Gradient4error95%interval:+3.64to+11.56percentage points for decoder family; -0.76to+9.12for natural family. The natural-family point advantage is not resolved by this small sample. Both families share nouns/contexts and are not independent replications. Test noun states were already opened in H1012/H1014; projected effects are newly measured, not fresh lexical authority. Gradient4is not chosen by tuning an unreported rank sweep; both registered ranks and all controls are reported.

Interpretation: causal supervision provides a useful low-dimensional linear subspace for this number-edit channel. Functional relevance differs from source-activation energy: preserving~22%of edit energy can preserve most target effect. This is a stronger candidate unit than unvalidated activation clusters, but it is not a learned nonlinear manifold or a validated BSF. It also does not solve H1014recipient-gain prediction: native suffix computes effects during validation, and no standalone output predictor is established here.

Audits: no evaluation gradients; native margin discrepancy1.67e-5absolute from position-only matrix operations; orthonormal bases within1.56e-15; unchanged basis hashes; independent Torch metrics exact;projection/remainder orthogonality1.58e-12absolute; finite.160prefix sequences,80native sequences,1520suffix sequences. All source data/selected indices/projection effects saved. No new reservedBSFtest data or natural documents; pool26112unchanged. No remaining process;H961pending;broad goal active.

Next return this positive causal-coordinate result to the multitask featurizer question: build analogous task-specific read subspaces for at least one genuinely different computation, using discovery-only causal/gradient supervision, then test shared versus separate coordinates under matched reconstruction and active-dimension budgets. Preserve the validated number subspace as an anchor rather than declare it a universal feature. Quantify finite collateral and transfer to a fresh template/noun authority after architecture freeze. Compact causal projection is one necessary piece; executable gain prediction, task diversity and trustworthy group/manifold discovery remain open.
