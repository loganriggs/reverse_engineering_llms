

## H293 — Small response-direction gain confirms, but intact accuracy is worse

Freeze H292 conditional96 winner (beta0, output ridge.1), H291/H290 natural128 and ridge64 winners, and H287 state+response joint64. No fitting or selection on the new panel. All share the frozen original compiler and refined residual base. The response-only maps use training natural-state mean centering with no newly fitted intercept; this convention is fixed before outcomes. Joint64 keeps its original feature offset. Ordinary intact KL is measured separately from conditional response KL, which uses a common native intact-output anchor and only replaces MLP changes.

Fresh128 complete documents exclude3392 prior unique hashes. Full source-document/window/file hashes and exact window contents independently pass. Four positions, all nine heads, full actual removals and norm/pre-dot-matched angle controls. Bootstrap clusters all positions/heads/payloads within documents,2000 draws. All saved metrics finite; oracle conditional KL<1e-11 and angle matching discrepancies<3.2e-7. Prior algebra tests cover maps and conditional statistics; no full-suite rerun. Existing-discovery and repeated-validation selection history is retained, but this panel has not selected any candidate.

| Comparator | Conditional-basis pooled response MSE reduction | Paired95% | Conditional downstream KL reduction |
|---|---:|---:|---:|
|natural|0.70%|[0.45,0.98]%|1.45%|
|ridge64|0.80%|[0.53,1.13]%|1.56%|
|joint|2.16%|[1.82,2.55]%|3.73%|

| Model | Fresh intact KL |
|---|---:|
|base|0.053750|
|joint|0.048890|
|ridge64|0.053919|
|natural|0.054374|
|conditional|0.055362|

- Conditional versus ridge64, intact KL reduction: -2.68%, paired95% [-5.76,0.31]%.
- Conditional versus natural, intact KL reduction: -1.82%, paired95% [-5.11,1.35]%.
- Conditional versus joint, intact KL reduction: -13.24%, paired95% [-17.02,-9.98]%.

All four locked gates FAIL. Response MSE gains versus natural and ridge64 are positive but below2%. Intact retention requires lower95% reduction>=−2%; it fails versus both joint and ridge64. The intact loss versus joint is clear; versus ridge64 the interval includes no difference but is too broad to establish retention. No threshold relaxed.

**Evidence update.** Covariance conditioning identifies some outside-span response information: the small validation advantage survives fresh testing, so the entire effect is not merely validation over-selection. But its magnitude is limited, and response-only fitting worsens ordinary predictions. The tested linear readout is not a generally better computational surrogate. Conditional downstream KL improves modestly too, supporting a response effect beyond output-energy bookkeeping, while retaining the oracle-anchor limitation.

Together with H285–287, the leading explanations are now: (1) objective/distribution coverage matters—response supervision repairs the directions it sees; (2) static linear readouts from these selected spans capture only a limited additional component; (3) nonlinear/context-dependent interactions or unrepresented directions remain plausible sources of the much larger residual. Pure optimization failure within the tested linear MSE class is weakened by H288 exact solves. Raw covariance selection is insufficient, while conditional geometry contributes a small real gain. These findings do not identify semantic units or an intrinsic feature count.

**Next high-information diagnostic.** Account for the remaining response error algebraically using the native quadratic model and the frozen96-dimensional read span. Separate mean-linear, within-span quadratic, span/complement cross, and complement-only terms, comparing their exact response contributions on these already-used documents. This is a labeled mechanism diagnostic, not a deployable predictor or new holdout claim. It can distinguish missing interactions from missing directions without another fit/validation search. Preserve term cross-correlations: energy is not additive, and a term's isolated norm is not its causal contribution. Use stable polynomial differences and exact replay checks. A large within/cross interaction residual would motivate nonlinear features; dominant complement-only response would motivate span expansion by a different criterion. Do not call successful accounting semantic identification.

All jobs terminal. Candidates remain frozen and none is promoted as a complete replacement. Original native weights/compiler/base unchanged. Goal remains active.
