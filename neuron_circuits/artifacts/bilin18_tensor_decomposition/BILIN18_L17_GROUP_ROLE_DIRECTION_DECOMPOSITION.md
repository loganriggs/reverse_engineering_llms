

## H419 — Parallel, mixed and orthogonal source contributions

Reuse H413 documents, H416 shuffles and H410 groups. Project each recipient residual into head6's native output subspace and normalize it to an ambient unit axis v. Decompose each source h=(h·v)v+h_perp. In the fixed-input-scale bilinear derivative, split each group response into its parallel and orthogonal parts. Their core/add cross is PP=2<Cp,Ap>, mixed=2(<Cp,Ao>+<Co,Ap>), and OO=2<Co,Ao>. It reconstructs H418's fixed–fixed term. Double arithmetic is used, with the unchanged native output metric.

For angle changes, normalize components by the original full-response energy within each condition, as in H418. For both-matched versus actual sources, use the SAME pooled actual full-response energy denominator, because parallel amplitudes match and differing denominators could otherwise create a spurious parallel difference. Component fractions divide signed component changes by their total; values outside[0,1] remain possible. This is directional accounting, not a semantic or unique mediation decomposition.

Predeclare4000 paired document/mapping bootstrap samples. A joint6 band covers PP/mixed/OO fractions in angle−raw and both−norm-only. A separate joint3 band covers those fractions for both−actual with the common denominator. Gates test whether one component exceeds80% at the lower bound in both angle settings, and whether mixed or OO exceeds80% for the actual-source gap. Replay threshold is1e-9 full-energy-normalized; actual/both parallel-amplitude matching threshold is1e-5 source-norm-scaled. No reselection or fresh confirmation.

|Condition|PP|Mixed|OO|Fixed-scale sum|
|---|---:|---:|---:|---:|
|actual|0.403605|-1.247522|0.356861|-0.487056|
|raw|0.354270|-1.000654|0.222522|-0.423862|
|norm_only|0.467601|-1.330927|0.350948|-0.512378|
|angle_only|0.148304|-0.151417|0.093351|0.090237|
|both|0.159418|-0.137831|0.117078|0.138665|

|Norm setting|Component|Angle-change fraction|Joint6 interval|
|---|---|---:|---|
|donor|Parallel–parallel|-0.400634|[-0.5066486901044329, -0.2946196698881268]|
|donor|Mixed|1.651891|[1.5458769845426952, 1.7579060047590014]|
|donor|Orthogonal–orthogonal|-0.251257|[-0.3572718247627217, -0.1452428045464155]|
|recipient|Parallel–parallel|-0.473368|[-0.5793823126739303, -0.36735329245762416]|
|recipient|Mixed|1.832591|[1.7265769770579553, 1.9386059972742615]|
|recipient|Orthogonal–orthogonal|-0.359224|[-0.4652381947084845, -0.2532091744921783]|

|Component|Both−actual contribution/common denominator|Fraction of gap|Joint3 interval|
|---|---:|---:|---|
|Parallel–parallel|-0.000000|-0.000000|[-0.015372080292637409, 0.015372073523599991]|
|Mixed|0.898569|1.072125|[1.056752527911358, 1.0874966817275955]|
|Orthogonal–orthogonal|-0.060449|-0.072125|[-0.08749667834307683, -0.05675252452683942]|

Gates: {'replay': True, 'parallel_match': True, 'angle_parallel80': False, 'angle_mixed80': True, 'angle_orthogonal80': False, 'actual_mixed80': True, 'actual_orthogonal80': False}. Audits: {'orthogonality_scaled_error': 6.090753518266329e-16, 'actual_both_parallel_amplitude_norm_scaled_error': 4.239478798105117e-07, 'FF_reconstruction_full_energy_normalized_max_error': 1.6973995802314074e-12}. Independent array checks: {'FF_replay_max_normalized_error': 1.6973995802314074e-12, 'all_component_fraction_replay': True, 'common_denominator_gap_replay': True}.

Outcome: replay and parallel-amplitude matching pass. Mixed directional terms dominate both angle changes and the actual-versus-matched fixed-scale difference. PP-only and OO-only dominance gates fail. For angle−raw at donor norms, component changes are PP−0.205966, mixed+0.849237, OO−0.129171, summing to H418's+0.514099 fixed-scale change. Mixed fraction is1.651891,[1.545877,1.757906] in the joint6 band. At recipient norms the changes are−0.308183,+1.193096,−0.233870, summing to+0.651043; mixed fraction1.832591,[1.726577,1.938606]. Fractions above one mean other signed components offset the mixed contribution, not that a causal mediation percentage exceeds100%.

With the common actual full-response denominator, both−actual contributions are PP−2.84e-9, mixed+0.898569 and OO−0.060449. Mixed fraction is1.072125,[1.056753,1.087497]; OO fraction is−0.072125,[−0.087497,−0.056753]; PP fraction is numerically zero, interval[−0.015372,0.015372]. Thus matching does not simply add a larger positive parallel-only numerator. Actual and fully matched parallel amplitudes agree within4.24e-7 of source norm, and their PP numerators agree correspondingly. Using different condition denominators would have obscured this: normalized PP values are0.403605 actual and0.159418 both, although the common-denominator numerator difference is zero.

The negative fixed-scale alignment is chiefly a mixed-term balance. Actual PP and OO terms are positive (0.403605 and0.356861), while mixed is−1.247522, leaving total−0.487056. Raw donors likewise have positive PP/OO (0.354270/0.222522) and strongly negative mixed−1.000654. Fully matched donors have much weaker negative mixed−0.137831 against positive PP/OO0.159418/0.117078, leaving+0.138665. These levels use each condition's original full-response denominator and are descriptive; the fraction comparisons use their specified simultaneous families.

Numerical audits: source parallel/orthogonal dot error6.09e-16 after norm scaling; H418 fixed-scale cross reconstruction error1.70e-12 relative to original full Fisher energy; independent saved-array replay confirms all fractions and the common-denominator contrast. No new factor ranks, source maps or native output vectors were fitted. No fresh documents were used.

Interpretation: the operative quantity is a relationship between the two groups' responses to different directional components, not either group's isolated response to the parallel axis. However, the mixed terms appear in a quadratic output-energy statistic; they are not a newly identified multiplicative interaction in the forward model. A derivative and a quadratic readout can produce such terms without a semantic feature pair. At a fixed recipient, the source-to-group derivative is linear, so donor-averaged cross energies depend on source first/second moments. It would be premature to attribute the pattern to higher-order or semantic source structure merely because Gaussian controls failed after an additional matching transformation.

Next discriminating intervention: preserve a natural raw donor's norm and recipient-relative angle, but take its orthogonal direction from a DIFFERENT donor. Pair the same-donor reconstruction as a numerical/control baseline. This changes the association between parallel amplitude and orthogonal direction without changing those scalar geometry constraints, unlike directly imposing the actual recipient source's angle. On a fresh panel, freeze six donor-pair mappings and compare raw, same-donor reconstructed, and cross-donor recombined inputs. Test both the predicted mixed-cross change and native finite group prediction/clamping, since an energy decomposition alone is not enough to identify a computational unit. If recombination changes the finite role as predicted, it supports a functionally relevant joint directional structure; if only the local cross statistic changes, the interpretation must remain local/metric-specific. Raw Gaussian controls without post-sampling geometry matching are also a useful future check of the second-moment explanation, but should not be conflated with H414's transformed moment controls.

### Updated twenty-hypothesis priorities

Numbers retain the earlier tournament mapping. These are qualitative assessments, not posterior probabilities.

|#|Hypothesis|Current status / discriminating next evidence|
|---|---|---|
|1|Learned source orientations align with high-sensitivity MLP directions.|Earlier geometry-controlled evidence remains; now test which directional associations control group response.|
|2|Low coupling is solely head6 dominating a summed denominator.|Weakened by earlier separate-source normalization; not an explanation of the present mixed decomposition.|
|3|Small native groups capture much of a source computation.|Supported locally at partial fidelity; no universal90% or semantic-unit result.|
|4|Remaining behavior needs broadly distributed factors.|Still viable; failures of current selection rules do not prove irreducibility.|
|5|Compensating combinations matter more than individual rankings.|Supported by rescue and mixed-response accounting; finite recombination is the next causal test.|
|6|Input-RMS radial correction alone produces the angle pattern.|Contradicted for this selected contrast: it offsets the fixed-scale effect.|
|7|Norm and angle alone determine group roles.|Insufficient: actual and fully matched share them but have different mixed contributions.|
|8|Generic high-dimensional geometry explains ordinary additivity.|Earlier support remains; this alone does not explain learned group directional covariance.|
|9|Denominator choices create the apparent role differences.|Important confound controlled here with a common denominator; the mixed numerator difference persists.|
|10|Results depend on one document panel.|Fresh H413 roles weaken this for selected contrasts; H414–419 mechanisms still need fresh confirmation.|
|11|Head6 has special response support.|Scope remains restricted; head4's amplitude sensitivity and head8 transfer differ.|
|12|Output/operator alignment explains grouping without unique feature identities.|Leading; mixed directional relationships, not isolated parallel responses, dominate.|
|13|Specific natural source-recipient pairing is necessary.|Weakened by raw shuffles; conditional directional association remains a sharper hypothesis.|
|14|One compact group is reusable across sources.|Partial transfer exists but uniformly high fidelity failed; test joint directional structure rather than universal labels.|
|15|Compact local formulas reflect architecture rather than recovered features.|Strong caution: exact decomposition identities are not themselves discoveries of reusable semantic units.|
|16|Native output geometry determines which errors matter.|Supported; all current mixed terms remain metric-dependent.|
|17|Local mechanisms survive finite interventions.|Unresolved for the new mixed-direction claim; test finite recombination outcomes.|
|18|Single/pair-source results extend to many simultaneous sources.|Still untested by this decomposition.|
|19|The groups correspond to semantic features.|Unsupported; no controlled linguistic evidence has been added.|
|20|Useful units are context-conditioned response operators.|Leading descriptive model, but compact shared parameters and predictive causal transfer remain open.|

The broad goal remains open. The next fresh panel must exclude8896 completed-panel documents. No experiment processes remain running.
