

## H286 — Reversing response supervision tests head-specific coverage

H285 supervised response losses on heads0–5 and excluded6–8. H286 reverses that split while retaining the same64 input features, normalization, H280 initialization, discovery/validation documents, actual/angle payload seed, intact minibatches, eight loss-strength/rate combinations and400 updates. The response objective is normalized by base error in the supervised group. Three versus six supervised heads is an intentional remaining asymmetry; this is not an equal-head-count randomization. Heads excluded from the added response loss were still used by the earlier base program.

Both splits select lambda10/rate.1/step300 by their original validation rule; the reverse split has eight eligible checkpoints. A secondary comparison fixes this same candidate in advance from H285, and therefore matches the selected comparison here. Zero-strength checkpoints across forward and reverse fits are bitwise identical, as are feature normalization arrays. This verifies that changes to the supervised response group do not change the extra-optimization control. The sufficient-statistic loss/gradient test passes after generalizing the runner to accept a config stem.

The fresh common128-document panel excludes3136 prior unique complete-document hashes; full-document/window/file hashes and exact source-window contents independently pass. Both frozen models and the declared zero-strength control are evaluated on this same panel. No fresh outcome selects weights, checkpoints or head groups. Bootstrap intervals cluster all sites/heads by document and condition on these fitted models. Conditional response KL continues to use the native intact-output anchor; intact KL remains a separate measurement.

| Payload | Head group | Reverse vs forward response MSE reduction | Paired95% interval |
|---|---|---:|---:|
|actual|heads6_8|11.78%|[10.70,12.95]%|
|actual|heads0_5|-8.36%|[-9.18,-7.55]%|
|angle|heads6_8|2.91%|[2.45,3.40]%|
|angle|heads0_5|-2.36%|[-2.59,-2.14]%|

Positive means the reversed fit is better; negative means the forward fit is better. Both use the forward-error denominator, fixed before evaluation. The crossover gate requires >=5% improvement on heads6–8 and <=−5% on heads0–5 with both intervals excluding zero. It is not a gate merely for either sign being correct.

| Model | Fresh intact KL |
|---|---:|
|base|0.055441|
|h280|0.051079|
|optimization_control|0.050217|
|forward|0.049798|
|selected|0.050438|

Locked gates: crossover_actual: PASS, crossover_angle: FAIL, state_retention: PASS. No thresholds relaxed.

The actual-response crossover passes: reversing supervision improves heads6–8 by11.78% [10.70,12.95]% and worsens heads0–5 by8.36% [7.55,9.18]%, relative to the forward model. Angle controls show the same signs (2.91% improvement and2.36% worsening), but fail the5% magnitude gate. State retention passes against the extra-optimization control (relative KL reduction−.44%, interval[−1.84,.90]%). Thus actual-response gains track supervision within this fixed readout family; intrinsically difficult heads alone do not explain H285. This supports incomplete supervision coverage, while leaving shared-readout interference and the unequal group sizes unresolved. The paired split comparison holds the input feature map and optimizer schedule fixed. Its scope is these head groups and these fixed-feature readouts. It cannot establish intrinsic computational units, exhaustive generalization, or a lower bound on representation size. State retention must be interpreted against the extra-optimization control, not only the older H280 adapter.

**Next discriminator.** Fit the same adapter with all nine response heads under the same objective grid, and compare it with both split fits on a new panel. Use validation constraints for each head group separately, rather than an aggregate that could hide damage. This asks whether one shared readout can combine the split-specific gains without sacrificing intact KL. Success would favor incomplete supervision coverage; failure would motivate a quantitative interference/capacity analysis before changing feature geometry. Keep actual and angle metrics separate and preserve the zero-strength control.

All training and evaluation jobs terminal. Native weights/compiler/base unchanged. Goal remains active.
