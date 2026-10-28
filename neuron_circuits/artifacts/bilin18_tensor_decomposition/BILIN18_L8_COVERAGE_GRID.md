# Text and source coverage information grid

Rank64 per scalar; common validation selection; all checkpoints frozen before fresh test. Odd heads are unseen sources only in even-head fits.

| Fit | Step | Natural R2 | Even | Odd | Half | Pairs | Gaussian | Isotropic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.861 | 0.644 | 0.561 | 0.560 | 0.583 | 0.368 | 0.501 |
| rows128_even | 700 | 0.899 | 0.655 | 0.613 | 0.606 | 0.633 | 0.354 | 0.470 |
| rows128_all | 1100 | 0.901 | 0.655 | 0.668 | 0.650 | 0.675 | 0.378 | 0.457 |
| rows512_even | 1200 | 0.901 | 0.670 | 0.625 | 0.621 | 0.643 | 0.364 | 0.459 |
| rows512_all | 1500 | 0.901 | 0.673 | 0.734 | 0.711 | 0.731 | 0.431 | 0.461 |
| value_only | 300 | 0.900 | 0.642 | 0.601 | 0.592 | 0.616 | 0.354 | 0.478 |
| rows128_even_step400 | 400 | 0.896 | 0.663 | 0.599 | 0.596 | 0.622 | 0.360 | 0.485 |
| rows128_even_step1600 | 1600 | 0.898 | 0.639 | 0.625 | 0.614 | 0.641 | 0.344 | 0.447 |
| rows128_all_step400 | 400 | 0.892 | 0.664 | 0.630 | 0.621 | 0.645 | 0.392 | 0.485 |
| rows128_all_step1600 | 1600 | 0.900 | 0.646 | 0.675 | 0.655 | 0.680 | 0.373 | 0.444 |
| rows512_even_step400 | 400 | 0.891 | 0.679 | 0.595 | 0.598 | 0.621 | 0.368 | 0.490 |
| rows512_even_step1600 | 1600 | 0.900 | 0.664 | 0.632 | 0.626 | 0.647 | 0.362 | 0.447 |
| rows512_all_step400 | 400 | 0.884 | 0.671 | 0.648 | 0.638 | 0.660 | 0.427 | 0.491 |
| rows512_all_step1600 | 1600 | 0.901 | 0.672 | 0.736 | 0.713 | 0.733 | 0.431 | 0.459 |
| value_only_step400 | 400 | 0.900 | 0.641 | 0.605 | 0.595 | 0.619 | 0.352 | 0.474 |
| value_only_step1600 | 1600 | 0.898 | 0.638 | 0.613 | 0.602 | 0.626 | 0.348 | 0.461 |

All conditions use the same natural-value data. Counts refer to source-response text rows, with four sites per row. Five versus nine sources changes support and unique response count, while optimizer updates/draws remain fixed. Both payload families are held out. Paired row-bootstrap intervals condition on one training realization and fixed payload draws. Four selected L8 outputs do not establish whole-model coverage.


## H230 — More examples help chiefly when the missing source is represented

**Question and design.** Does the H229 transfer gap come from too few text
examples or too narrow a source set? A locked 2×2 grid used128/512 nested training
rows and five-even/all-nine heads. All four fits use rank64 per scalar, the same
2048-site natural fitting panel, spectral initialization, response weight1,
256-example minibatches and1600 updates. A value-only control uses the same
natural objective. Common checkpoint selection uses natural plus even-head
validation error, with per-axis scales fixed from the first128 even-head training
rows. Step400/1600 snapshots were frozen alongside selected factors before test.

**Fresh test results.**

| Source-response training rows | Trained heads | Natural R2 | Even response | Odd response | Gaussian odd-span | Isotropic |
|---|---|---:|---:|---:|---:|---:|
| 128 | Even five | .899 | .655 | .613 | .354 | .470 |
| 128 | All nine | .901 | .655 | .668 | .378 | .457 |
| 512 | Even five | .901 | .670 | .625 | .364 | .459 |
| 512 | All nine | .901 | .673 | .734 | .431 | .461 |

Odd heads are unseen sources only for the even-five fits. The all-nine results
measure new-text generalization for familiar sources. Baseline spectral64 has
natural R2 .861, even .644, odd .561, Gaussian .368, isotropic .501. Value-only
fitting gives .900/.642/.601/.354/.478 respectively.

Quadrupling rows with even-only sources improves odd response by only .011
(conditional paired row-bootstrap95% [.004,.019]); with all sources, the gain is
.066 [.044,.088]. Adding sources improves odd response by .054 [.043,.065] at128
rows and .109 [.081,.136] at512 rows. The post-result difference-in-differences is
.055 [.035,.076]. Example quantity and source coverage interact in this panel.
Even-head gains from more rows are only .015–.018. Source-count changes also
change the number of unique response examples, but total gradient draws remain
409600 per response-trained fit. This is not a matched-unique-example experiment.

**Important heterogeneity.** A post-result head audit shows that head3 contributes
86.3% of the pooled .109 source-coverage gain at512 rows. It carries43.2% of odd
response energy, and its fidelity changes .419→.636. Head1 changes .814→.833,
head5 .681→.684, head7 .736→.780. Head0 worsens versus baseline (.507→.456).
Thus the effect is not uniform across sources. All four scalar output axes
improve under the512-row source-coverage contrast: [.601,.623,.824,.417] becomes
[.722,.719,.858,.606]. The heterogeneity is across heads, not confined to one
selected output coordinate. These diagnostics do not turn four axes into
population-wide coverage.

**Optimization and payloads.** Step400→1600 raises odd fidelity by .026–.087
across factorial fits, with the largest gain in512/all. Longer fitting also lowers
isotropic fidelity by .032–.043. Selected steps are700/1100/1200/1500; the value-only
control selects300. Thus duration does matter and the H229400-step limit was not
innocuous, but it does not uniformly recover a general response operator. The
512/all Gaussian gain over512/even is .067 [.008,.123], whereas isotropic gain is
.001 [-.013,.018] and isotropic fidelity remains below spectral. Gains specialize
to sources and observed/natural payload structure rather than all input changes.

**Updated explanation.** Source coverage and statistical generalization are real
limitations of this fitted decoder; a stronger global decoder can be learned when
sources are represented. This weakens any claim that conditional programs are
intrinsically necessary. It does not establish that all source relations need
separate features, that head3 is semantically special, or that rank64 is optimal.
The evidence favors T01/T02/T03 as sensitivity/support accounts, keeps capacity
and objective tradeoffs live, and localizes the next diagnostic to head3. T07's
original covariance-estimation claim is NOT directly tested: natural covariance
was held fixed; only response training examples changed.

**Next high-information discriminator.** At fixed512 rows, rank64 and1600 updates,
compare even-head training supplemented with head3 actual payloads, shuffled
same-head payloads, Gaussian payloads in head3's weight span, or isotropic
norm-matched payloads. Keep the extra-payload draw fraction fixed and compare an
even-only control. Freeze selection using the same even-head validation rule;
use fresh test rows19304–19431, including head3, other heads, and payload controls.
If shuffles and Gaussian payloads match actual head3 training, generic support
coverage suffices. If shuffles work but Gaussian does not, natural within-span
anisotropy matters without correct binding. If only actual payloads work,
source-state correlations matter, but semantics would still be unproven. If
isotropic supervision works broadly, generic derivative regularization becomes
more plausible. After this focused discriminator, prioritize output-population
coverage and downstream behavior (T12/T13), rather than indefinitely optimizing
these four coordinates.

**Audit and artifacts.** Head-write reconstruction errors <4e-7 and native
finite-response replay relative RMS8.97e-6. Twelve reused algebra/control tests
pass. Fresh training18536–19047, validation19048–19175, test19176–19303. All fits
completed; no process left running. Report BILIN18_L8_COVERAGE_GRID.md, PNG/SVG,
full JSON, frozen factors, training/validation/test caches, row-error arrays and
post-result headwise/axiswise diagnostic JSON. Intervals condition on one training
realization and fixed payload draws; no multiplicity-corrected probabilities.
