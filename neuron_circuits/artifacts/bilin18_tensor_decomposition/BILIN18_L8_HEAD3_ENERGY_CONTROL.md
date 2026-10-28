# Adaptive response-energy-weighting control

Six new sham fits plus four reused primary controls frozen before a second fresh test; selected on natural plus even-head validation only. All response fits receive the same number of gradient draws.

| Fit | Selected step | Natural R2 | Even | Actual head3 | Other odd | Shuffled3 | Gaussian3 | Isotropic3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.831 | 0.505 | 0.216 | 0.746 | 0.417 | 0.319 | 0.553 |
| even_only_rep0 | 900 | 0.878 | 0.611 | 0.303 | 0.780 | 0.421 | 0.354 | 0.566 |
| actual_rep0 | 1400 | 0.873 | 0.620 | 0.538 | 0.791 | 0.388 | 0.397 | 0.561 |
| shuffled_rep0 | 1600 | 0.871 | 0.604 | 0.269 | 0.778 | 0.517 | 0.386 | 0.585 |
| gaussian_rep0 | 1600 | 0.872 | 0.617 | 0.322 | 0.782 | 0.457 | 0.397 | 0.551 |
| isotropic_rep0 | 1600 | 0.870 | 0.611 | 0.274 | 0.774 | 0.432 | 0.347 | 0.558 |
| even_only_rep1 | 1100 | 0.879 | 0.611 | 0.310 | 0.783 | 0.423 | 0.357 | 0.564 |
| actual_rep1 | 1400 | 0.874 | 0.618 | 0.539 | 0.792 | 0.389 | 0.398 | 0.561 |
| shuffled_rep1 | 1600 | 0.873 | 0.601 | 0.272 | 0.784 | 0.541 | 0.427 | 0.559 |
| gaussian_rep1 | 1600 | 0.872 | 0.610 | 0.342 | 0.785 | 0.456 | 0.401 | 0.565 |
| isotropic_rep1 | 1600 | 0.870 | 0.606 | 0.271 | 0.771 | 0.418 | 0.346 | 0.551 |

Locked gates: {'generic_support': False, 'context_pairing': True, 'generic_regularization': False}.

All payloads are norm matched before RMS recomputation. Shuffled/Gaussian/isotropic test draws are independent from fixed training payloads. Two replicates vary SGD and training payload draws, not the text corpus. Full endpoint scores, histories and conditional row-bootstrap contrasts are in JSON. Four L8 coordinates remain a limited output population.


## H232 — Response-energy weighting does not rescue sham supervision

**Adaptive follow-up, locked before new fitting and new test.** Discovery-only
per-axis multipliers equate actual-to-sham quadratic target energy in supplemental
training loss. Multipliers range1.06–3.11 for shuffled/Gaussian and1.23–7.86 for
isotropic. Six new sham fits keep the same payloads, SGD seeds, rank, source quotas
and update count. Four original actual/even-only fits are reused unchanged.
All ten evaluated models froze before new test19432–19559. This controls target
energy scaling, not every gradient-conditioning or optimization difference.

| Supplement | Actual head3 fidelity, replicate0 /1 |
|---|---|
| Even-only control | .303 / .310 |
| Actual head3 | .538 / .539 |
| Energy-weighted shuffled | .269 / .272 |
| Energy-weighted Gaussian span | .322 / .342 |
| Energy-weighted isotropic | .274 / .271 |

Actual pairing still beats shuffled by .269/.267 and Gaussian by .216/.197.
Conditional row-bootstrap intervals for actual-minus-Gaussian are [.143,.310]
and [.142,.243]. Generic-support and generic-regularization gates fail again;
context-pairing gate passes. All six sham fits select the step1600 endpoint;
optimization is not exhaustively ruled out. Correcting a simple energy mismatch
nevertheless does not close the gap. Native response replay relative RMS6.04e-6.

**Synthesis and boundary.** H228 showed that an explicitly source-corrected
quadratic chart works even on meaningless payloads in the source span. H231–232
show something different: distilling a fixed global rank64 decoder from examples
requires the natural source–state pairing to reproduce actual head3 responses
well in this setup. These are compatible statements about analytic support
preservation versus distribution-dependent fitting. Neither proves semantic
primitives, a unique feature basis, or that conditional programs are necessary.
The strongest current picture is a program whose important bilinear interactions
depend on the source and execution distribution; ordinary value compression can
miss those interactions. This remains a four-coordinate observation.

**Next priority: population and downstream coverage, not more four-axis tuning.**
Measure how much of the full1152-dimensional L8 MLP write and head-induced write
change the four studied axes actually cover, across all nine heads. Compare
frozen natural-write, residual, response-derived and weight-derived output
subspaces at matched ranks. Fit response geometry on existing discovery text;
use fresh test19560–19687. Separate natural-state variance, raw source-response
energy, centered response variance and per-head coverage. Test whether head3's
special role survives the full output population or is selection bias (T12).
Then run nonlinear suffix interventions at fresh rows19688–19719, comparing
source-removal runs with native MLP response, no MLP response, and projected MLP
responses. Keep the upstream head removal identical across those comparisons.
This isolates how much downstream behavior the local response descriptions
preserve (T13), instead of equating coefficient fidelity with understanding.

**Record.** BILIN18_L8_HEAD3_PAYLOAD_TRAINING.md and
BILIN18_L8_HEAD3_ENERGY_CONTROL.md; JSON/NPZ/logs under matching stems; bias audit
script and JSON; paired PNG/SVG figure. The energy-control report is explicitly
separate from the primary report. Overall goal remains open; all current fits
and evaluations completed.
