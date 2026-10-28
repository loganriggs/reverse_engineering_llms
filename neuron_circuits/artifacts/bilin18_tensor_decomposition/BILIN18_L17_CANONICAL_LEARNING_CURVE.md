## H478 — Fixed-representation learning curve on fresh documents

Freeze the existing rank128 kernel coordinates, their768 support inputs, and the16-dimensional output basis. Test whether more receiver examples improve probability-weighted coefficient learning. Allocate1792 new complete-document-hash-disjoint windows: first1024 for training, next256 for validation, final512 for confirmation. Training sizes256/512/1024 are nested prefixes. All training sizes use the same32 donor vectors cyclically; validation and confirmation use64 donor vectors. Thus receiver coverage changes, not donor-family breadth. Both signed unit interventions are evaluated. The data remain FineWeb from the existing new cache; this is freshness within the tracked research continuation, not a model-training holdout.

Independent audit rehashed all13504 previously tracked documents and all1792 new documents against their own original token caches, verified the129-token windows, and found zero complete-document-hash overlap. Total tracked documents is now15296. The builder, manifest, split, fitting procedure, ridge grid and four primary comparisons were fixed before outcomes. Save all candidate maps and validation selections before collecting confirmation model states or intervention outcomes.

Each fit solves the full2048-dimensional convex Fisher-weighted normal equations, with a penalty anchored to the original map. Normalize Fisher Gram matrices by their mean diagonal; penalty is ridge times mean Hessian diagonal. Grid: [1e−6,1e−4,.01,.1,1,10,100]. Select separately for each size by pooled finite validation KL, including the original predictor as fallback. All three sizes select ridge.1. The original representation and output basis carry earlier training/validation history, but none of these1792 documents was used to select them.

| Predictor | Fresh confirmation KL recovery −/+ | Raw-response recovery −/+ |
|---|---|---|
|original|80.15% / 77.07%|93.70% / 93.78%|
|n256|80.56% / 77.86%|90.50% / 90.63%|
|n512|81.33% / 78.72%|90.59% / 90.65%|
|n1024|81.58% / 79.10%|90.00% / 90.09%|

Recovery is one minus pooled prediction error divided by pooled native intervention effect. It is not accuracy or the mean of per-example ratios. Raw recovery uses squared layer-write errors; KL recovery uses final probabilities. Native intact state is retained to isolate response prediction; no full-layer replacement, intact-state CE or loss-preservation claim is tested.

| Predeclared primary comparison | Gain | Joint95% interval | Lower>0 |
|---|---:|---|---|
|n1024_minus_original_negative|+1.44pp|[-1.81, +4.69]pp|False|
|n1024_minus_n256_negative|+1.02pp|[-0.31, +2.35]pp|False|
|n1024_minus_original_positive|+2.03pp|[-0.57, +4.63]pp|False|
|n1024_minus_n256_positive|+1.24pp|[-0.62, +3.09]pp|False|

All four gates fail. Point estimates improve monotonically with training size on validation and confirmation, but the joint intervals do not establish gains either over the original predictor or from256 to1024 examples. This is not proof of no benefit, a plateau, or a deficient input representation. It also does not confirm that increasing receiver coverage resolves the oracle gap. The learned probability objective trades some internal squared-error fidelity for probability fidelity: raw recovery drops from~93.7% to~90.0% at1024.

Bootstrap uses4000 paired samples, resampling eight receivers within each of64 fixed donor strata, with the same indices across models and signs. Four simultaneous standardized bands condition on the fitted maps, donor pool, calibration and validation selection. They do not estimate variability from drawing another training set or source pool. Training diagnostics evaluate each selected map on all1024 training receivers; for the smaller fits that includes receivers outside their fitting prefix, so those rows are not pure in-sample training scores.

Effects are concentrated. Negative effects have squared-weight effective count11.49 out of512; the largest case supplies23.78% of total effect and the largest26 supply71.33%. Positive effects have effective count27.60, largest13.59%, and largest26 share61.42%. These are concentration descriptions, not replacement bootstrap sample sizes. The wide intervals make the direction of the learning curve more tentative than its smooth point estimates suggest.

Audits: all regularized normal-equation relative residuals below1e−7; output-basis projection error below1e−6; all arrays finite. Complete-document exclusion/window checks passed independently. Frozen selection was saved before confirmation collection. Per-case outcomes, baseline inputs, candidate coefficient maps, validation scores and bootstrap specifications are archived. No experiment process remains active.

Next discriminate input representation from remaining estimation error: compare rank128 with an expanded rank256 empirical input basis using the same training receivers, fixed16 output directions, source assignments and Fisher objective. Select regularization on validation and reserve a new confirmation panel before selection. The expanded basis must be constructed only from the original training support or current training inputs, never confirmation inputs. This is a controlled test of a competing explanation, not a conclusion that the learning curve plateaued. Broader CE-objective comparisons, rank1 source/tail tests and transfer across heads/layers remain open. The broad feature/computation objective remains active.
