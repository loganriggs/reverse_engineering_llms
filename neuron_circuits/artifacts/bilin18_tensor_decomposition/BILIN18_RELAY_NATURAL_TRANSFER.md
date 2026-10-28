## H853 — Task-blind correct-site dictionaries recover part of the causal contrast, but no isolated causal atom

**Main result.** Natural-text featurizers trained at the actual preMLP4 site preserve substantially more of the relay effect than fixedPCA, but remain less faithful than the supervised one-dimensional contrast benchmark. Scalars and blocks preserve a substantial fraction; shared-group extensions are weaker in this implementation/training regime. No single atom closely aligns with either frozen task direction. Partial causal reconstruction is not yet recovery of an interpretable fundamental unit.

**Design.** All ten H852 models were frozen before task evaluation, with training/selection based only on H851 natural data. Evaluate all78 H849 pairs, both directions and all gap/format variants. All original native competence/alignment rules retained. Compare global and token-residualPCA48/64 fit only on natural training data. These tasks have been used in earlier research stages, so this is task-blind fitting but not an untouched research confirmation panel.

At the token-identical relay, add decoded donor-minus-recipient normalizedshape residual to the native recipient shape, renormalize and restore recipient rawRMS. Keep recipient reconstruction error and persistent inputs. Exact donor shape with recipient radius is the causal reference; full rawdonor swap separately verifies the small radius effect. The model is not replaced by its full featurizer reconstruction. Token means cancel in donor differences at the identical token, though nonlinear encoder support can still depend on its centering.

**Adjacent-prompt results.** Values list both seeds separately; PCA has one fit. Preservation is relative to exact-shape relay intervention, not the whole native task or accuracy. Effect error compares individual directed answer-margin changes with the exact-shape reference.

|Task|Representation|Mean relay effect preserved|Effect relative RMSE|
|---|---|---|---|
|agreement|scalar48|57.6%, 73.5%|0.447, 0.286|
|agreement|scalar64|56.8%, 72.7%|0.461, 0.304|
|agreement|block8|56.9%, 72.1%|0.481, 0.310|
|agreement|shared_fixed|19.5%, 31.1%|0.814, 0.727|
|agreement|shared_learned|21.5%, 17.0%|0.798, 0.843|
|agreement|residual_pca64|39.8%|0.613|
|agreement|global_pca64|24.4%|0.770|
|sentiment|scalar48|74.4%, 55.1%|0.293, 0.512|
|sentiment|scalar64|57.1%, 56.0%|0.460, 0.461|
|sentiment|block8|66.4%, 53.1%|0.377, 0.517|
|sentiment|shared_fixed|51.8%, 42.0%|0.528, 0.618|
|sentiment|shared_learned|16.9%, 5.7%|0.835, 0.941|
|sentiment|residual_pca64|28.6%|0.716|
|sentiment|global_pca64|15.5%|0.839|

Raw-versus-shape reference effect error is only0.005 agreement and0.002 sentiment on the adjacent panel. Thus the main representation errors are not explained by recipient-radius restoration.

Across sentiment gap variants, scalar48 preserves roughly51–75%, block8 51–75%, fixedshared31–52%, learnedshared6–18%, while residualPCA64 preserves27–29%. Scalar64, despite the best natural reconstruction, is not consistently the best causal representation. Agreement gaps weaken the reference route itself; per-gap and per-format results remain saved rather than pooling them into a universal ranking. Seed variation is material, and these are only8 agreement/5 sentiment lexical pairs repeated across variants.

**Decoder alignment versus actual behavior.** Best atom squared cosine with the supervised direction is only0.108–0.240 for scalar models (varying task/seed). Best block8 subspace contains0.169–0.372 of direction energy; fixedshared best group0.223–0.350, learnedshared0.127–0.206. These are postfit maxima over many atoms/groups, not independently discovered semantic labels. A group has8 dimensions, so its overlap cannot be compared to a scalar's as equal capacity. Some decoder subspaces overlap the direction while actual gated reconstructions remain poor. This makes encoder/support choice a live bottleneck alongside dictionary coverage.

H852's learnedshared dictionaries cover only101 of512 atoms through their memberships, despite balanced initial coverage. Fixedshared retains512 but still underperforms scalar/block models. Thus coverage collapse alone is unlikely to explain every gap, and a more expressive group structure does not guarantee an effective learner. Training budget, STE optimization, gate parameterization and support constraints remain alternatives to architectural unsuitability.

**Coverage caveat.**48 relay-token occurrences lack a training-specific mean and use the global mean; token identity remains equal within each patched pair. This removes donor lexical-mean substitution but does not guarantee in-distribution nonlinear encoding. Agreement FormatA/FormatB results are separately retained. Both show substantial scalar/block transfer and weaker shared-group fidelity; do not dismiss all errors as missing means, nor assume centering cannot contribute.

**Next discriminating experiment.** Separate representational capacity from inference: project the true donor contrast into (a) the entire learned decoder span and (b) the union of the recipient/donor active decoder atoms, comparing these oracle coefficients with the learned encoder difference at the same site. These are diagnostic least-squares oracles, not unsupervised discoveries or deployed encoders. If the active span preserves the effect but learned coefficients do not, improve coordinate inference/training. If only the full dictionary works, examine gating/support. If even the full span fails, the dictionary or training distribution misses the relevant direction. Compare effective rank and conditioning, retain both seeds and all architectures, and include exact references. This gives a more informative target for improving featurizers than adding hierarchy without knowing the bottleneck.

**Audits.**2,652 full-model forwards. Identity logits exact; raw-reference margins replayH849 within1.44e-5. Frozen task sources, checkpoints and bases unchanged; all effects finite; reserved test untouched; natural-document count26,112 unchanged. H851–853 used2,560 prefix forwards plus2,652 full forwards and ten task-blind training fits. This turn made concrete progress on the user's featurizer comparison at a validated causal site. Broad representation/computation goal remains active.
