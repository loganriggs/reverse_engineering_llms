## H479 — Expanded input coordinates with fixed output directions

Test whether the input representation limits coefficient prediction. Keep H478's1024 training receivers,256 validation receivers,32 training donors/64 evaluation donors,16 output directions and Fisher regression objective. Compare128 versus256 input coordinates. Retain the original128 coordinates exactly; append centered kernel eigenmodes129–256 from the original768 support inputs. No new validation or confirmation inputs enter the coordinate construction. The expanded feature Gram matrix on the support is audited as whitened. Padding the original coefficient matrix with zeros preserves its function.

This doubles learned coefficient count from2048 to4096, while holding output rank and kernel support fixed. It tests a larger predictor class, not equal-parameter efficiency. Each width uses its own mean-Hessian-diagonal-normalized anchored ridge grid [1e−6,1e−4,.01,.1,1,10,100], selected by finite validation KL including the original map as fallback. Equal ridge multipliers do not imply equal functional regularization across widths. Save selections and maps before collecting outcomes for a new512-receiver confirmation panel.

Independent complete-document audit rehashed15296 excluded documents and512 new documents against their original caches, verified all129-token windows and found no overlap. Total tracked documents15808. Confirmation is fresh for this continuation and selected models, not a model-training holdout or new source-family test. Validation has been reused adaptively across this research, so the new confirmation is the evidence for transfer.

Selected candidates: {'r128': 'r128_0.1', 'r256': 'r256_0.1'}.

| Predictor | Fresh confirmation KL recovery −/+ | Raw recovery −/+ |
|---|---|---|
|original|73.93% / 68.82%|92.35% / 91.53%|
|r128|76.21% / 71.64%|89.68% / 89.12%|
|r256|77.31% / 72.83%|89.81% / 89.24%|

Recovery is one minus pooled response prediction error divided by pooled native effect. It is not accuracy or mean per-case recovery. Native intact state is retained as a response-isolation baseline. No intact-state or observed-token CE claim is made.

| Predeclared primary comparison | Gain | Joint95% interval | Lower>0 |
|---|---:|---|---|
|r256_minus_original_negative|+3.39pp|[+1.43, +5.34]pp|True|
|r256_minus_r128_negative|+1.10pp|[-1.13, +3.34]pp|False|
|r256_minus_original_positive|+4.01pp|[+1.49, +6.53]pp|True|
|r256_minus_r128_positive|+1.20pp|[-1.27, +3.67]pp|False|

2 of four predeclared gates pass. Failed superiority gates are not proof of equivalence or absence of benefit.

Negative effects: effective count25.08/512, largest-case share14.17%, largest26 share61.31%. These describe concentration, not replacement statistical sample sizes.

Positive effects: effective count35.19/512, largest-case share11.49%, largest26 share55.64%. These describe concentration, not replacement statistical sample sizes.

Bootstrap uses4000 paired within-donor resamples (eight receivers for each of64 fixed donors), shared across both signs/models. Four simultaneous standardized bands condition on fitted models, donor pool, calibration and selection. They do not propagate training-sample variability or uncertainty in the learned feature basis.

Audits: positive256th kernel eigenvalue; support feature whitening error below1e−6; original map projection error below1e−6; all regularized normal-equation residuals below1e−7; all outputs finite; rank128 validation scores independently replay H478 within1e−7 absolute KL. Full candidate maps, appended coordinate encoder, selections, baseline inputs and per-case results are archived. Document exclusions and window contents pass independent audit. No process remains running.

The wider coordinate model can still miss functions outside its fixed empirical quadratic feature span. This experiment does not establish fundamental units or unique canonical coordinates; signs and degenerate eigenspaces retain ordinary basis ambiguity. Any effect of extra coordinates is evidence about this representation and intervention distribution. The broad objective, observed-token CE comparison, source rank1/tail decomposition and transfer across heads/layers remain open.

Interpretation and hypothesis update: r256 improves over the original predictor by3.39/4.01pp with joint95% intervals[1.43,5.34]/[1.49,6.53]pp, passing both gates. The isolated width increment r256−r128 is1.10/1.20pp with intervals[−1.13,3.34]/[−1.27,3.67]pp, failing both gates. Thus the combined learned refit and expansion transfers on this panel, while a contribution from the extra coordinates is not established. Neither an input-capacity bottleneck nor its absence follows. Fresh-panel recovery levels differ from earlier cohorts; that alone does not diagnose distribution shift.

The leading unresolved explanations remain: (1) errors in predicting coefficients within the fixed output space; (2) receiver-dependent sensitivity makes a globally linear coefficient map inadequate; (3) missing input coordinates; (4) output-span capacity varies by receiver; (5) concentrated examples and training-sample variation obscure modest effects; (6) a local Fisher fitting objective differs from finite KL. H479 supplies stronger evidence for a transferable refit but weak discrimination among these explanations. The earlier H476 oracle93–95% was measured on another receiver panel and cannot be treated as this panel's ceiling.

Next run a matched oracle diagnosis on this exact confirmation panel using the fixed16 output directions and the actual selected r256 predictions. Measure raw projection, scalar Fisher correction and full-span Fisher correction, evaluating finite KL as well as local quadratic error. This determines whether the large remaining gap is again coefficient prediction or whether the output space itself covers less of these responses. This follow-up is explicitly post-hoc, not another fresh confirmation. It also checks the earlier capacity finding across receiving contexts before spending more experiments on increasingly wide input maps.
