## H772 — Additional documents do not establish a general sensitivity-weighting benefit

Previous goal turn was progress: H771 completed matched weighted/unweighted fits, actual KL evaluation and uncertainty checks, then froze all parameters. H772 evaluates every frozen fit on additional documents without refitting or selecting configurations.

The panel contains 64 predetermined validation documents at lengths 32 and 120, selected by a fixed hash ordering and excluding all H765 and H769 validation documents. This quadruples the previous document count. These documents were used in earlier natural reconstruction evaluation; this is additional research-validation evidence, not a pristine reserved-test result. The original reserved test remains untouched.

All eight H771 fits are retained: the large-data reference and the matched unweighted, measured-weight and shuffled-weight fits in each of write and Fisher geometry. Fixed and adaptive coordinate selection, budgets8,32,64,128, replacement site and KL definition are unchanged. The saved fit hash is checked unchanged after evaluation.

| Geometry | Selection | Budget | Unweighted mean KL | Measured-weight mean KL | Shuffled-weight mean KL |
|---|---|---:|---:|---:|---:|
| write | fixed | 8 | 0.003001 | 0.00286503 | 0.00318997 |
| write | adaptive | 8 | 0.00135623 | 0.00153143 | 0.00163911 |
| write | fixed | 32 | 0.000897386 | 0.000827801 | 0.00100377 |
| write | adaptive | 32 | 0.000275902 | 0.00037256 | 0.000258211 |
| write | fixed | 64 | 0.000290244 | 0.000281646 | 0.000266206 |
| write | adaptive | 64 | 5.65283e-05 | 7.44041e-05 | 5.30929e-05 |
| fisher | fixed | 8 | 0.00303845 | 0.00270491 | 0.00313368 |
| fisher | adaptive | 8 | 0.00132918 | 0.00144233 | 0.00159159 |
| fisher | fixed | 32 | 0.00116274 | 0.00118079 | 0.00130173 |
| fisher | adaptive | 32 | 0.000234508 | 0.000375118 | 0.000319503 |
| fisher | fixed | 64 | 0.00046742 | 0.000409537 | 0.000429287 |
| fisher | adaptive | 64 | 6.59457e-05 | 5.8037e-05 | 5.91268e-05 |

Measured weighting lowers mean KL in 6 of12 nontrivial comparisons, down from9 of12 on H771. Both adaptive budget8 gains reverse: write geometry changes from .001356 unweighted to .001531 weighted, and Fisher geometry from .001329 to .001442. Adaptive write geometry is worse with measured weights at every tested nontrivial budget on this panel. Fixed write geometry retains lower means at all three budgets, but its measured-minus-unweighted descriptive intervals all include zero.

Two of twelve paired measured-minus-unweighted bootstrap intervals are wholly negative, for fixed Fisher budgets8 and64; ten include zero. The earlier fixed-write rank32 interval that barely excluded zero now crosses zero. Some measured-versus-shuffled fixed-subspace comparisons favor measured weights, but this does not establish a general sparse-featurizer benefit. Intervals use10,000 paired document bootstrap draws, average the two lengths within each document, omit training-fit uncertainty, and are not corrected for multiple comparisons. Do not promote isolated intervals as confirmed discoveries.

Panel mean KL differs substantially from H771 even with unchanged fits, underlining context sensitivity and the danger of choosing methods using small panels. Both the mean comparisons and their uncertainty should be retained; neither a simple win count nor a pooled average erases method/budget dependence.

Interpretation: the current scheme is not verified as a better featurizer and should not be advanced into manifold interpretation on that basis. Context sensitivity is real (H768/H769), but learning a global basis with scalar-weighted second moments does not automatically yield better adaptive sparse codes. Small training effective sample size, fixed global geometry, and the difference between a covariance objective and TopK KL remain unresolved. This is not evidence against all output-aware feature learning.

Audits:8,320 whole-model forwards; additional-panel exclusions pass; frozen-fit/model/input hashes unchanged; full-rank replay<=3.25e-5; cached-head replay<=4.30e-5; inverse error<=2.00e-15; all outputs finite. Training weight effective sample sizes remain35.3contexts and23.2documents because nothing was refit. No causal transfer was performed for these fits. Unique natural document pool remains26,112.

Next research decision: stop treating this weighting scheme as a candidate established improvement. A more direct control is output-KL training versus reconstruction training with identical parameterization, initialization, training contexts and optimization budget. Begin with a small fixed-rank decoder before adding sparse gates, so changes in architecture do not obscure whether the objective helps. Audit teacher/native replay, gradient correctness, zero/full-rank controls and held-out actual KL; then test causal transfer before advancing into manifold claims. Keep the existing scalar, block and shared-group baselines as references. This follows the user's goal of verified computational structure rather than optimizing ever more elaborate sensitivity proxies.
