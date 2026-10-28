## H806 — Input response identity has the larger effect at L6 MLP, with a smaller local-baseline contribution

H805 established that L6 query MLP response transfer strongly changes the context gap. H806 separates the raw pre-MLP baseline b from the finite input response d. For each paired key context, cross the origin of b and d, explicitly compute RMS(b+d) and RMS(b), and evaluate their MLP-output difference. Transplant only that difference onto the recipient's original zero-perturbation MLP output. The recipient incoming residual skip response and surrounding computation remain intact. Thus baseline_swap changes the baseline used to compute the transferred module response; it does not replace the recipient's whole local state.

### Results

The context gap is different-key minus same-key intended-token log-probability response to the +.50 L2 value intervention. Both local L2 source-state choices and queries are averaged within lexical item.

| L6 response construction | Vertical-bar gap | Colon/newline gap |
|---|---:|---:|
| Own baseline, own input response | +.3862 | +.2172 |
| Own baseline, swapped input response | -.1494 | +.0663 |
| Swapped baseline, own input response | +.3113 | +.1740 |
| Both swapped | -.2206 | +.0193 |

Input-response swaps change the gap by -.5355 [-.7131,-.3626] and -.1509 [-.2139,-.0932]. Baseline-only swaps change it by -.0748 [-.1083,-.0430] and -.0432 [-.0666,-.0230]. These are descriptive paired item-bootstrap intervals. The input-response manipulation has the larger mean effect in both formats, but local-baseline conditioning is measurable. Adding the baseline swap to the input swap changes the gap by a further -.0712 and -.0470, respectively.

The factorial interaction of these four gap measurements is small compared with the main shifts: +.00360 [-.01639,.02205] and -.00378 [-.00629,-.00147]. This is not a claim of exact additivity, unique causal shares, or a complete mechanism. The MLP response and downstream behavioral effect remain functions of the chosen recipient background.

The evidence favors a substantial context difference already present in the arriving pre-MLP response, rather than the L6 local baseline being its dominant origin under this experiment. The MLP still transforms that response, and the local baseline contributes. Next freeze this distinction for fresh vocabulary and a smaller original perturbation instead of continuing to select explanations on the same panel.

### Exact arithmetic and nonlinearity

With normalized endpoints u and u+du and ungated bilinear MLP B, the finite response decomposes exactly as D[(Lu)(Rdu)+(Ldu)(Ru)] + D[(Ldu)(Rdu)]. The bias cancels. RMS normalization is applied to the raw endpoints before this expansion; it is not linearized or omitted.

The quadratic term's pooled RMS norm divided by the full module-response RMS norm is .173/.167 in native pairings and .180/.169 in crossed pairings. These norm ratios are not output-effect shares. Nor do they measure all nonlinearity in raw coordinates: RMS normalization and the baseline-dependent linear term also matter. This calculation audits the algebra and documents a non-negligible finite quadratic component without attributing behavioral importance to it.

### Verification and limitations

2184 model forwards completed: 104 native local-state captures, 416 hybrid input captures, and 1664 response-replacement runs. Recomputed RMS inputs match the captured MLP inputs exactly. The bilinear expansion matches direct module responses within 1.20e-5 relative L2; re-evaluated own/donor module responses match captured differences within 1.93e-5 relative L2. Every zero-delta transfer reproduces full-vocabulary logits exactly. Own and both-swapped responses reproduce the corresponding H805 none/MLP arms within 4.88e-6 log probability. Local L2 clamp relative error is 1.40e-8, all tracked hashes unchanged, and all responses finite. Accuracy remains 98.4%/100% across arms.

The same 16/10 aligned lexical items are reused. Bootstrap intervals average state and query variants within item, are descriptive, and have no multiplicity correction. Hybrids may leave native activation manifolds. Reserved test untouched; natural-document pool remains 26,112. No featurizer/manifold/DAG is newly validated.

### Frozen next validation

Saved configs/bilin18_mlp_input_baseline_fresh_locked.json before fresh runs. Use 16 bird-value words with the existing two key-context formats, L2H5 perturbation, and L6 query MLP four-arm crossing. Test +.50 and +.25 coefficients without reselecting layers or heads. Exclude only token-alignment failures and report all exclusions. Require at least eight aligned items per format and competence gates of .75 candidate/.50 full-vocabulary accuracy; otherwise report limited interpretability without replacing the panel based on outcomes. Primary comparison is input-swap versus baseline-swap change in the context gap. This is vocabulary transfer; distinct competent-task validation is still outstanding.
