## H810 — Frozen indirect-object transfer does not reproduce the strong copying-task response mechanism

Run the H809 locked L2H5-to-L6 query MLP test on the competent indirect-object panel, with no head/layer reselection. Inject the projected first initial name's mixed value at the second initial name, at coefficients 0/.25/.50. Pair prompts differing only in the later giver. Cross the L6 raw pre-MLP baseline and finite input response; apply RMS explicitly and add the computed MLP response to each recipient's original module baseline.

Both projected direction and L2 source state are exactly invariant across giver changes, including at nonzero coefficients. No artificial local-state duplication is used. The readout holds identity fixed: first initial name's log probability and first-minus-second-name logit margin. Context gap means response when the first name is correct minus response when the second name is correct. This differs from simply subtracting correct-answer responses with changing labels.

### Primary results at coefficient .50

| Surface | MLP response arm | Fixed-name log-probability gap | First-minus-second margin gap |
|---|---|---:|---:|
| Market/parcel | Own | +.00545 | +.27928 |
| Market/parcel | Input swap | +.01869 | +.29266 |
| Market/parcel | Baseline swap | +.00481 | +.27724 |
| Market/parcel | Both | +.01737 | +.28849 |
| Store/book | Own | -.01401 | +.29023 |
| Store/book | Input swap | -.00289 | +.29973 |
| Store/book | Baseline swap | -.01498 | +.28786 |
| Store/book | Both | -.00445 | +.29520 |

The fixed-name log-probability gaps have broad intervals spanning zero. The native margin-response gap is clear: descriptive name-pair bootstrap intervals [.23925,.32864] and [.24385,.34495]. Yet transferring the entire L6 MLP response barely changes that margin gap: +.00922 [.00282,.01617] and +.00497 [-.00183,.01085]. It neither removes nor reverses the gap as the earlier copying-task transfers did.

Input-swap versus baseline-swap margin contrasts at .50 are small and positive (+.01541 [.00865,.02239], +.01187 [.00574,.01757]); at .25 they are near zero (-.00010 [-.00353,.00358], -.00258 [-.00567,.00056]). This does not support a strong task-general L6 transferable-response mechanism. Smaller effects do exist, so the result is not a claim of zero contribution.

The discrepancy between log probability and two-name margin matters: competition with the rest of the vocabulary can mask a context-dependent relative preference between the two names. We report both predeclared readouts rather than selecting the more favorable one. Mean correct-answer log probability decreases by .022/.030 in the .50 own arm while top-1 accuracy is 100% in all .50 arms; mean KL is approximately .0044/.0058. Confidence changes and accuracy are distinct measures.

### Dose comparison and interpretation

A post hoc comparison of saved raw pre-MLP inputs shows that coefficient .50 yields mean relative L6 input displacement .00706 (range .00344–.01147) on indirect-object prompts versus .05349 (.03428–.07646) on H807 bird copying. Thus the same additive source coefficient does not match downstream perturbation size. The two tests also change the contrast from repeated/different keys to giver identity and use different source positions/content. These differences limit a claim of universal failure.

The frozen test nevertheless failed to reproduce the strong copying result under its specified intervention. Do not relabel this as successful transfer merely because some small input-swap contrasts are detectable. Plausible explanations now include task-specific readers, weaker propagation of this source direction, or finite-dose nonlinearities in copying. Next use a prespecified small-dose response curve on both tasks with the same fixed head/layer, including lower copying coefficients to overlap the IOI input-displacement range. Report coefficient and realized displacement separately; matching a post-treatment displacement alone is not a causal equivalence guarantee. No new head/layer selection should be hidden inside that test.

### Numerical audit and repair

The first completed run and diagnostic repeat failed the original relative module-response check: subtracting nearly equal float32 module outputs produced relative errors up to .000321. Source invariance, zero-delta logits, and own-response final logits already passed. No behavioral result from those failed runs was reported as verified.

The final implementation evaluates the bilinear difference in float64, with du computed by subtracting float64 endpoints, and audits the exact expansion. Maximum expansion relative error is now 5.60e-14. Difference from captured float32 module responses is .000270 relative to the small response, but only 1.58e-6 relative to the module endpoint norm. The endpoint-scaled gate is 1e-5; the stricter downstream full-logit gate remains 1e-4. This is an explicit scale-aware correction, not an unreported relaxation of the failed threshold.

All zero-delta full logits replay exactly. Own-response full logits replay within 9.54e-6, native H809 logits within 1.05e-5, and RMS input reconstruction is exact. Tracked hashes are unchanged in the accepted run, all response metrics finite, and source direction/state invariance exact. Accepted run: 960 forwards. Including the failed and diagnostic runs: 2880 forwards this experiment. The analysis retains only the verified final run; logs preserve the failures.

Eight name pairs and both initial orders are included in each of two correlated narrative surfaces. Bootstrap averages order variants within pair, is descriptive, and has no multiplicity correction. Hybrid responses are not proven on-manifold. Reserved test untouched; document pool 26,112. General featurizer fidelity, reusable manifold structure, and a complete computational account remain unresolved.
