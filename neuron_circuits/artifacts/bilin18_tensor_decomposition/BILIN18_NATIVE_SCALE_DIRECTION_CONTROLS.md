## H788 — Native-scale tests weaken the broad L1 story and retain a modest L4 copying lead

Previous turn was progress: H787 separated odd/even responses and discovered that fixed radius .30 extrapolated beyond most native donor displacements. H788 tests the predeclared alternative at fractions .25/.5/1 of each pair's actual donor angle, with both perturbation signs and explicit native input/output references.

For each receiver normalized MLP input, rotate toward or opposite the true donor tangent direction by a fraction of its native angular distance. Apply the same pair-specific angle to within-family shuffled and three seeded random tangent directions. Input norm remains fixed. Radius is matched across controls for each base/edit/layer, not across layers: this measures changes at native scales rather than fixed-size layer geometry. Negative and random directions are synthetic states even when their displacement magnitude matches a native edit; bounded distance does not prove manifold membership.

The same eight families and final-position sites L1/L4 are retained without outcome-based filtering. Native donor MLP input substitution and native donor MLP output substitution are evaluated independently. Positive true-direction fraction1 reaches the norm-scaled donor input. No new lexical/template data or fitting is introduced in this control experiment.

### Main findings

Native-scale L1 first-order module errors are much smaller than in the fixed-.30 tests. At the true positive endpoint:

| Competence-passing family | L1 relative module error | L4 relative module error | L1 native swap margin RMS | L4 native swap margin RMS |
|---|---:|---:|---:|---:|
| Agreement | .142 | .191 | .0366 | .0424 |
| Capital recall | .143 | .210 | .0266 | .1262 |
| Entity binding | .080 | .064 | .0239 | .0493 |
| Copying | .214 | .224 | .1184 | .1248 |

Thus L1 does not generally have the larger nonlinear error at native task scales: L4 is larger in three of the four competent families. This is compatible with the prior matched-radius result, because native L1 displacements are usually smaller. It corrects any interpretation that generic fixed-radius L1 curvature necessarily dominates natural task edits. Relative module error is interaction-vector norm divided by full-change norm, pooled in squared norm; it is not a causal share.

The clearest remaining directional lead is L4 copying. Its paired odd donor-directed mean grows smoothly with native-angle fraction:

| Fraction of native donor angle | True direction | Shuffled task direction | Random pooled |
|---|---:|---:|---:|
| .25 | +.01268 | +.00293 | -.00183 |
| .50 | +.02486 | +.00586 | -.00356 |
| 1.00 | +.04576 | +.01176 | -.00642 |

At fraction1, true-minus-random is +.05218 with descriptive item-bootstrap interval [.01904,.08521]. True-minus-shuffled is +.03401 with interval [-.00252,.06464]. The same qualitative pattern holds at smaller fractions: positive true-versus-zero and true-versus-random intervals, but true-versus-shuffled intervals include zero. These use only eight reused items, cluster reciprocal edits and seeds within item, and have no multiplicity adjustment. They prioritize a test; they do not establish independent confirmation, a semantic feature, or a useful compression basis.

The odd effect is distinct from the native positive swap. L4 copying's actual donor-output swap has mean donor-directed effect only +.00274, despite odd mean +.04576: the even component contributes about -.04303 to this signed mean. Consequently the odd statistic cannot be reported as the native swap's net benefit. At L1 copying the odd mean is +.00972 with uncertainty spanning zero, whereas the native swap mean is -.03575. Agreement and capital recall show no clear positive true-direction odd effect on this panel. L4 entity binding has a small positive odd mean (+.01191), but does not clearly exceed shuffled or random controls.

The stronger L1 entity-binding response in the fixed-radius stress test collapses at native scale: native swap margin RMS is .0239 and mean donor-directed effect .00287. This cautions against building semantic feature claims from large off-native perturbations.

### Implementation checks

12,544 model forwards: 11,520 signed/control/fraction interventions, 768 native input/output reference runs, and 256 zero controls. No new natural documents or reserved-test evaluations.

Constant norm error <=1.34e-15; chord-radius error <=4.45e-16. True endpoint versus norm-scaled donor input error <=4.64e-14 relative L2. Actual donor/receiver normalized-input norms differ by at most 1.60e-7 relatively. Native input versus native output substitutions agree within 6.68e-6 answer margin. Positive true endpoint versus actual native output swap agrees within 8.11e-6, so the small norm discrepancy does not materially change this readout. Identity logits differ by at most 2.17e-5 across differently batched runs. All effects finite; source/model/input hashes unchanged.

Competence passes remain agreement, capital recall, entity binding and copying. Other families remain reported but cannot support successful-task interpretation. Original reserved test untouched; natural-document pool remains 26,112. Full records include nuisance edits and all fractions, not just the selected copying lead.

### Decision and next test

The broad hypothesis that early-layer finite nonlinearity directly reveals semantic units is weakened. General layer geometry, native displacement scale and even/odd downstream cancellation jointly explain why the stress tests looked stronger than natural swaps. We should stop expanding this reused panel merely to make a favorable contrast clearer.

Freeze L4 final-position copying as the primary next replication target, with L4 entity binding as a task comparator and L1 as a layer comparator. Use fresh vocabulary and wording, native-angle fractions fixed in advance, opposite signs, shuffled and random controls, actual donor-output endpoint references, and native competence gates. Evaluate odd direction sensitivity and net native-swap effects separately. A reproducible true-versus-shuffled advantage would motivate item-sensitive representations; transfer of shuffled directions would motivate shared family-level operators. Only if that causal structure transfers should a shared-dictionary/manifold featurizer be asked to compress it. No featurizer or manifold is newly validated by H788.
