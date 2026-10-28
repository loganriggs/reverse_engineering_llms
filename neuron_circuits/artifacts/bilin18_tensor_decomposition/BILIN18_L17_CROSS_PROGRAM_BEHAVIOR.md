

## H275 — Explicit cross-core programs improve conditional behavior, at substantial cost

H274 found an energy tradeoff between actual-only and mixture partner bases. H275 materializes both frozen rules and evaluates their downstream next-token effects. Compilation happens once; correction evaluation needs no native tensors. The exported programs contain the frozen mean, core/partner reads, exact mean-linear map,528 within-core product coefficients and1024 cross-core product coefficients, all with full1152-dimensional outputs.

For normalized MLP input x, a=x−mu, u=aC, v=aP. The centered correction is K(x)=aJ+sum_{i<=j}u_i u_j W_ij+sum_ij u_i v_j T_ij, where J=J_E(mu), W includes doubled off-diagonal B_E(c_i,c_j), and T_ij=2B_E(c_i,p_j). E is native minus original compact compiler. The constant E(mu) is omitted because this experiment uses differences K(z)−K(x). It is NOT an intact residual replacement. Both a standalone forward evaluator and a stable polynomial-difference evaluator are exported.

**Conditional mediation.** For each changed pre-MLP state, the predicted final state is changed + native_intact_write + original_compiler_change + K(z)−K(x) + Down_bias. Compare with changed + native_changed_write + Down_bias. The native intact write is a common oracle baseline. Thus we test preservation of the intervention response, not replacement of the whole native MLP. Controls: original compiler response alone, common exact mean-linear+within-core response, actual-trained cross program, mixture-trained cross program and exact residual-change oracle. The final native RMS/head/softcap are unchanged.

Fresh128 documents exclude all1728 unique document hashes from seven recent panels; file hash and exclusions verified. Four sites, nine source heads, scales0.1/1, actual and norm/state-angle-matched source perturbations. No fitting or model selection. Not a model-training holdout or globally unused text. Float32 logits feed float64 KL arithmetic.

| Full edit / correction | Residual response fidelity | Mean conditional KL (nats/site) | KL reduction vs compiled-only | Top1 agreement |
|---|---:|---:|---:|---:|
|actual / compiled|0.00%|0.00020018045|0.00%|99.197%|
|actual / common|17.41%|0.00019610427|2.04%|99.284%|
|actual / actual|39.65%|0.00013151147|34.30%|99.327%|
|actual / mixture|38.17%|0.00013404112|33.04%|99.349%|
|actual / oracle|100.00%|2.0133028e-13|100.00%|100.000%|
|angle / compiled|0.00%|0.00028683013|0.00%|99.240%|
|angle / common|19.33%|0.00026303739|8.30%|99.284%|
|angle / actual|29.05%|0.00022228692|22.50%|99.349%|
|angle / mixture|30.82%|0.0002149804|25.05%|99.392%|
|angle / oracle|100.00%|2.0013751e-13|100.00%|100.000%|

Locked gates: cross_behavior: PASS, mixture_angle_advantage: FAIL, mixture_actual_retention: FAIL, half_kl: FAIL.

- actual scale0.1, cross_behavior: 35.28% relative KL reduction [31.01, 39.54], paired document-bootstrap95%.
- actual scale0.1, mixture_advantage: -2.09% relative KL reduction [-3.64, -0.53], paired document-bootstrap95%.
- actual scale1.0, cross_behavior: 32.94% relative KL reduction [29.06, 36.87], paired document-bootstrap95%.
- actual scale1.0, mixture_advantage: -1.92% relative KL reduction [-3.62, -0.21], paired document-bootstrap95%.
- angle scale0.1, cross_behavior: 15.37% relative KL reduction [11.73, 18.87], paired document-bootstrap95%.
- angle scale0.1, mixture_advantage: 3.63% relative KL reduction [0.86, 6.27], paired document-bootstrap95%.
- angle scale1.0, cross_behavior: 15.49% relative KL reduction [11.97, 19.07], paired document-bootstrap95%.
- angle scale1.0, mixture_advantage: 3.29% relative KL reduction [0.41, 5.83], paired document-bootstrap95%.

Cross-behavior passes: actual-trained cross terms reduce the common-only KL by>=10% at both actual scales. Mixture angle advantage is positive but about3–4%, below5%; mixture actual-retention fails the−3% lower-confidence-bound criterion; actual program fails50% KL recovery versus compiled-only. No thresholds relaxed. These relative reductions use different explicitly named baselines and are not percentage points of residual-energy fidelity.

**Interpretation.** The cross-core rule is not merely an algebraic artifact: it preserves additional conditional next-token behavior on new documents. The energy-objective tradeoff also appears downstream, but absolute KL errors are already small (compiled-only about0.00020 nats/site for full actual removals). Top1 agreement is already99.2% before these corrections and changes little. Neither a34% relative KL reduction nor the high top1 agreement establishes large practical task gains or intact-model fidelity. This is reproducible conditional evidence, not semantic feature identification.

**Costs.** Each correction stores3,189,888 float32 scalars (12,759,552 uncompressed bytes): mean1,152; core36,864; partner36,864; dense linear1,327,104; within-core608,256; cross-core1,179,648. This is20.0% of the native MLP's15,925,248 weight scalars (bias excluded). Adding the original504,224-scalar compiler brings the stored correction-plus-compiler to3,694,112 scalars,23.2% of native weights. The dense linear term accesses all1152 inputs; this is not a64-read implementation. The two alternative exported models duplicate common arrays; production packaging could share those, but each standalone count remains explicit. No speedup is claimed without a benchmark. Native baseline/model execution is an additional cost of conditional evaluation.

**Numerical audit.** The initial evaluator subtracted two large float32 outputs, giving maximum relative difference error4.53e-4. Its JSON/log/row arrays are preserved with the subtracted_forward suffix. The final evaluator computes the exact polynomial difference from sums and differences of input coordinates, reducing the maximum error to6.05e-7; all gates remain unchanged. Float64 compiled response agrees with direct native-tensor decomposition within7.60e-13 relative error. The arbitrary-input test covers nonzero mean, compiler offsets and linear terms, within/cross coefficients, and both forward subtraction and stable differences. Three relevant tests passed; the changed difference evaluator's test was rerun successfully. Full project suite not rerun.

**Next discriminator: remove the conditional native baseline.** Evaluate the frozen program as an intact correction to the original compiler by adding the exact constant E(mu), without refitting anything. Test actual and mixture programs against native intact outputs/logits on independent documents, with total storage and natural-state residual/behavior errors reported. This directly asks whether the candidate circuit describes the computation itself, rather than only changes around a supplied native answer. Keep the original compact compiler and conditional metrics as baselines; failure of intact substitution must not be obscured by successful response mediation.

Exports: residual_cross_program.py provides load, forward and change; residual_cross_actual_program.npz and residual_cross_mixture_program.npz hold standalone arrays. Inputs must already be RMS-normalized. The omitted E(mu) constant is deliberate for response use and must be accounted for before any intact substitution claim. All jobs complete; native weights/original compiler unchanged, figures inspected. Broad research goal remains active.
