## H784 — Finite L1 value edits require the bilinear interaction term

Previous turn was progress: H783 established non-additive joint restorations and passed exact graph-cut controls. H784 moves inside the implicated L1/L4 MLPs, testing exact algebra before causal interpretation.

For normalized MLP input u, let L=Left(u), R=Right(u), and D=Down. The module is D(L*R)+bias, with elementwise product and no activation gate in this checkpoint. A receiver-to-donor change obeys:

    delta_output = D(delta_L * R) + D(L * delta_R) + D(delta_L * delta_R)

The first two terms sum to the first-order approximation at the receiver normalized input. The third is the finite bilinear interaction. The common output bias cancels. We capture the actual native factors, compute the decomposition in float64, and test each term, their first-order sum, their full sum, and the native output difference at both copied-value positions. Each first-pass intervention adds the selected term to a receiver MLP output. A second pristine receiver run reads only its resulting L5H5 local-value change, with native receiver routing. This preserves the previous isolated-pathway definition.

### Results

All numbers below are answer-margin RMS across aligned edits. Error is the RMS discrepancy from the native single-module donor-output substitution, not from the full donor prompt or the earlier joint restorations.

| Edit | Site | Native site-swap effect RMS | First-order-only effect RMS | First-order error | Interaction-only effect RMS |
|---|---|---:|---:|---:|---:|
| Copied-value change | L1 | 0.2458 | 0.4857 | 0.3562 | 0.1882 |
| Copied-value change | L4 | 0.1500 | 0.1612 | 0.0484 | 0.0472 |
| Key rename | L1 | 0.1029 | 0.0989 | 0.0295 | 0.0273 |
| Key rename | L4 | 0.1176 | 0.1238 | 0.0247 | 0.0244 |

For these finite L1 value changes, the first-order approximation is poor, and the interaction term is necessary to reproduce the actual module-output change and its isolated causal readout. The L4 value-change discrepancy is smaller. This does not prove a special L1 computation: differences in input displacement, representation scale, and subsequent transformations could explain the contrast. Separately intervened term effects need not add after downstream nonlinearities.

Neither individual factor change reproduces the full L1 task site swap: left-only and right-only margin errors are 0.1646 and 0.1638. Their similar errors do not establish symmetric semantic roles, nor does naming factors Left/Right assign either one a meaningful gate/content interpretation.

At the module-output level, the interaction norm relative to the full change is 0.686 for L1 task edits, 0.425 for L4 task edits, 0.243 for L1 key renames, and 0.313 for L4 key renames. These are vector norm ratios, not causal shares. First-order terms and interaction can cancel. This receiver-anchored expansion is not a unique semantic decomposition: for a quadratic function, expanding about the midpoint can absorb the finite interaction into the midpoint derivative.

### Normalization diagnostic

We also captured the raw residual immediately before MLP RMS normalization. With r(x)=sqrt(mean(x^2)+epsilon), the normalized-input change decomposes exactly as:

    u_d - u_r = (x_d-x_r)/r_r + x_d*(1/r_d-1/r_r)

The second term is the normalization-scale change. Its norm relative to the total normalized-input change is 0.278 (L1 task), 0.102 (L4 task), 0.486 (L1 key rename), and 0.097 (L4 key rename). It partly cancels the fixed-scale raw-change term in every pooled case. These are geometric diagnostics, not causal tests of normalization, and the fixed-scale raw-change term need not be a pure angular change.

### Verification and limitations

The initial absolute output-decomposition threshold of 1e-4 failed: maximum float32-versus-float64 discrepancy was 0.001263. We did not treat that as evidence against the algebra or silently loosen the absolute threshold. The float64 algebra identity holds to 5.61e-16 relative L2; the maximum relative L2 discrepancy from actual float32 module changes is 2.46e-6. Full reconstructed versus native site-swap readouts differ by at most 9.54e-6 in answer margin, with pooled RMS differences about 2.5e-6. Native site-swap readouts replay H781 exactly. These checks support a numerical-rounding explanation for the absolute discrepancy at the scale of these activations.

Normalization reconstruction error is at most 7.48e-7. Native logits/local values, identity controls, requested output replacements, and the H780 target readout replay exactly. 12,864 model forwards; 480 aligned edits (256 task, 96 nuisance, 128 identity); the same 32 nuisance alignment exclusions. Model/source/input hashes unchanged and metrics finite. Original reserved test untouched; natural-document pool remains 26,112. The analysis runner additionally asserts native and analytic swap agreement.

These are reused exploratory copying cases. They demonstrate finite nonlinear structure along a selected pathway, not verified manifolds, semantic labels, a complete model explanation, or a causal explanation of all H783 joint-restoration effects. No new featurizer has been validated here.

Next: test whether the apparent L1/L4 difference persists under controlled perturbation size and fresh contexts, then across other already-established task families. Freeze the prediction before inspecting the new outcomes. Use transferable local transformations and finite-intervention accuracy as criteria when revisiting scalar, block, overlap, and DAG featurizers; reconstruction alone remains insufficient.
