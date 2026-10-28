## H811 — Copying response transfer persists at small displacements overlapping IOI

H810 did not reproduce the strong copying-task response mechanism on competent indirect-object completion (IOI), but the original .50 coefficients produced very different downstream displacements. H811 prespecifies lower copying coefficients .025/.05/.10, keeps L2H5 and L6 query MLP fixed, and repeats the four baseline/input-response crossing arms. Compare with already measured H807 copying .25/.50 and H810 IOI .25/.50. No per-example adaptive dose matching or head/layer reselection is performed.

The eligible primary panel remains the nine-item bar-separated bird-copying panel. The six-item newline panel remains exploratory because it failed the earlier frozen item-count gate. The dose test uses float64 bilinear differences and the scale-aware audits established in H810.

### Primary copying dose curve

Context gap = different-key minus same-key intended-word log-probability response, averaging query and local-source-state choices within lexical item.

| Source coefficient | Mean relative L6 input displacement | Own gap | Input swap | Baseline swap | Both swaps |
|---|---:|---:|---:|---:|---:|
| .025 | .00200 | +.00566 | -.00848 | +.00596 | -.00859 |
| .050 | .00411 | +.01242 | -.01827 | +.01267 | -.01891 |
| .100 | .00870 | +.03000 | -.04183 | +.02894 | -.04504 |

Input-response swaps reverse the gap at every tested dose; baseline-only swaps have small effects with intervals spanning zero. Input-minus-baseline gap contrasts are -.01445 [-.02071,-.00868], -.03094 [-.04443,-.01841], and -.07077 [-.10042,-.04239]. These descriptive item-bootstrap intervals do not imply population-wide certainty or unique causal shares.

At the smallest dose, the input-swapped gap is -.00848 [-.01307,-.00451] while the own gap is +.00566 [.00042,.01129]. Thus the signature is present at mean relative L6 displacement .2%, rather than appearing only in the previous several-percent regime. Absolute effects are correspondingly small, and positive doses alone do not establish a derivative or perfect local linearity.

### Comparison with IOI

Market/parcel IOI at .25/.50 has mean relative L6 input displacements .00326/.00718. Copying .05/.10 yields .00411/.00870, with overlapping per-example ranges. Earlier copying .25/.50 yields .02531/.05459 on this surface. These are descriptive realized displacements, not an exactly matched randomized intervention.

IOI's L6 input-versus-baseline log-probability contrast at .25/.50 was +.00303/+.01388; its corresponding margin contrast was -.00010/+.01541. In contrast, copying retains the strong relative reversal signature throughout the smaller-dose curve. This weakens the claim that a large copying dose alone caused the cross-task difference. Task, token position, lexical content, and context definition still differ, so it does not uniquely isolate a task-specific circuit.

The H810 failed strong-transfer result stands. IOI still has a clear context-dependent two-name margin response largely untouched by L6 MLP swapping, while copying's context effect is strongly altered at that same module. The two tasks therefore provide useful contrasting causal-response benchmarks. They should not be collapsed into one universal feature label merely because a dictionary reconstructs their activations.

### Verification and behavioral limits

2460 model forwards completed. All tracked hashes unchanged. RMS inputs reconstruct exactly. Float64 bilinear expansion relative error is at most 1.03e-13. Difference from captured float32 module responses is at most .000336 relative to the small response but only 1.32e-6 relative to the endpoint, passing the 1e-5 endpoint-scaled gate. Zero-delta full-logit replay is exact; own-response logits match within 1.53e-5. Source clamp relative error is 1.63e-8, and all responses finite.

Primary copying full-vocabulary accuracy across the small-dose arms is 94.4–95.8%, compared with 97.2% native. These runs include crossed L2 local-source states, so native-to-hybrid errors cannot be assigned solely to the small perturbation or the later MLP swap. Each response is still compared with its own hybrid zero baseline. Neither small norm displacement nor accurate algebra proves on-manifold validity.

The panel is reused; bootstrap averages duplicated/query and local-state variants within item, and intervals are descriptive without multiplicity correction. The original reserved test remains untouched; document pool stays 26,112.

### Return to representation validation

Use the verified copying and IOI response families as positive and contrasting controls for compact representations. Next prespecify a lexical split and compare a shared low-rank input-response dictionary against task-specific dictionaries, evaluating both held-out reconstruction and downstream effects after reconstructing the MLP input response. Keep local baselines explicit and apply the exact RMS/bilinear transformation. These are internal held-out fits on previously examined panels, not fresh untouched data. A compact representation only becomes useful evidence if it preserves the distinct causal response patterns on held-out items; reconstruction alone will not count as validation of a manifold or reusable computational unit.
