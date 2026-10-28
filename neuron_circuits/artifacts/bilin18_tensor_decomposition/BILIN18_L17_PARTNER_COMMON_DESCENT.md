

## H299 — Repeated common descent improves both families, without compact coverage

Fixed core96 and partner rank32, same 768 discovery training documents and 256 validation documents as H296–298. All nine heads, actual removals and angle-matched controls, full removal. Native cross coefficients remain fixed. At each of eight iterations compute full separately normalized training gradients, project into the horizontal tangent space, and use the negative sum of unit family gradients. Choose the first QR-retracted step in [1,.5,.25,.125,.0625,.03125] satisfying Armijo decrease for BOTH training losses (coefficient .0001). Validation never chooses step sizes; afterward select maximum worst-family validation fidelity, including step zero.

Eight steps accepted: 1,1,1,.5,.25,.25,.25,.25. Nine larger trial steps rejected. The selected checkpoint is step8. The run stopped at its iteration budget, not convergence or line-search exhaustion.

| Family | Initial train | Selected train | Initial validation | Selected validation | Validation gain |
|---|---:|---:|---:|---:|---:|
| actual | 15.377% | 18.137% | 16.595% | 17.394% | 0.799pp |
| angle | 14.172% | 19.224% | 15.315% | 18.763% | 3.448pp |

Locked gates: gain of at least2pp in BOTH validation families FAILS; both validation fidelities at least80% FAILS; every accepted step improves both training objectives PASSES. The last is an algorithm audit, not evidence of generalization. All recorded Armijo decisions independently replay correctly. Saved partner orthogonality and core exclusion have maximum absolute errors below1e-5.

**What changed.** The previous Adam step-zero outcome did not establish an unavoidable tradeoff: repeated controlled steps improve both families on training and on reused validation. Gradient cosine becomes negative after iteration4 (about−.30 by iteration8), yet common descent continues. Negative cosine alone does not imply opposing gradients or Pareto stationarity. This supports optimization limitations as part of the explanation, while the poor absolute fidelity keeps distributed interactions, insufficient partner rank, and limitations of the fixed native-coefficient projection plausible. These explanations can coexist.

**Limits.** This is a local trajectory from one initialization, on heavily reused validation. It neither estimates a global rank32 ceiling nor establishes transferable gains. No fresh document, intact-model or downstream-KL test was run. The fixed native coefficients remain an oracle diagnostic; compiling the rank32 cross term densely still requires3,538,944 output coefficients. No semantic units have been identified. The serialized configuration includes inherited unused fields (Adam rates/checkpoints, old fit-head split); the executed algorithm uses cfg.heads=all9 and cfg.iterations=8, as recorded above.

**Next information-gain comparison.** Separate optimization from representation capacity with a small locked factorial: rank32 versus64, proxy initialization versus its common-descent refinement, same documents and equal gradient-evaluation budget. Include training and validation curves and per-family error, not only a selected scalar. Large training gains with little validation gain favor estimation limits; substantially better rank64 training/validation favors a rank limitation; little improvement at either rank leaves optimization and projection restrictions unresolved. Before treating a selected improvement as real, freeze the comparison and evaluate on previously unused documents. Do not merely extend the same trajectory indefinitely: it cannot by itself distinguish those explanations.

All jobs terminal. Saved trajectory and frozen bases: bilin18_l17_partner_common_descent*. Native model, compiler, and original base remain unchanged. Broad research goal remains active.
