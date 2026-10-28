## H756 — Reconstruction and causal preservation diverge across heads and seeds

Frozen H755 scalar/block featurizers and H754 fixed/adaptive/random baselines are tested by adding F(donor)-F(receiver) at both selected L5 head positions. The native baseline is retained; this is not replacement by F(donor), and nonlinear differences are never computed as F(donor-receiver). Each edit is run with downstream L13Q2/K2 states free, fully clamped to native, or operation-coordinate-clamped as before. No task fitting occurs.

Three targets are compared with the full raw head swap: total intended-margin change; the free-minus-full-node-clamped effect; and the free-minus-operation-coordinate-clamped effect. All reported errors are relative RMS, not H755's relative squared reconstruction error. The table uses120 native-competent narrative contexts; failed list tasks remain separately recorded.

| Head / representation, budget8 | Total-effect error | Full-node-dependence error | Operation-dependence error |
|---|---:|---:|---:|
| H3 fixed write PCA | 85.9% | 82.7% | 91.4% |
| H3 adaptive write PCA | 60.8% | 49.9% | 54.3% |
| H3 scalar, mean3seeds | 47.7% | 36.1% | 44.0% |
| H3 block4×2, mean3seeds | 57.1% | 61.0% | 61.7% |
| H3 random adaptive, mean3seeds | 75.7% | 69.1% | 63.4% |
| H5 fixed write PCA | 84.3% | 93.3% | 83.4% |
| H5 adaptive write PCA | 68.2% | 79.5% | 67.7% |
| H5 scalar, mean3seeds | 80.0% | 68.3% | 52.8% |
| H5 block4×2, mean3seeds | 90.6% | 91.5% | 74.1% |
| H5 random adaptive, mean3seeds | 77.7% | 76.8% | 68.9% |

Reference RMS magnitudes are0.3572/0.0234/0.0195 logits for H3 total/full-node/operation targets, and0.3790/0.0390/0.0369 for H5. The dependence targets are small subcomponents of the full head-swap effect; errors must be interpreted alongside these scales. Averaging seed-level errors does not create independent statistical replications of the shared templates.

H3 scalar total-effect errors span40.9–53.9%, and operation errors40.1–48.7%. This is a reproducible advantage over the tested block baseline on this panel, but still substantial lost effect. H5 scalar total-effect errors span61.5–103.2%, while operation errors span38.3–61.6%. H755 natural reconstruction MSE at H5 varied only38.76–39.15% across these seeds. Almost identical reconstruction can therefore coexist with materially different functional preservation.

The preferred representation also depends on the target: H5 adaptive PCA preserves the total effect better on average than scalar features, while scalar features better preserve the operation-dependent part. Random adaptive coordinates are competitive with some learned fits on total effect. Block grouping does not show a consistent benefit. This weakens any claim that a reconstruction winner automatically identifies the meaningful computational units.

Limitations: finite-training fits, reused task/validation panels, task-informed head choice, and no fresh diverse-task confirmation yet. The block baseline is signed unit-decoder block TopK, not Grassmannian BSF. Coordinate-restoration interventions retain their documented spherical complement correction. No architecture is declared an ideal featurizer or a verified semantic manifold.

Audits:18,072 model forwards =17,280 representation/head/clamp interventions +720 no-swap controls +72 self-swaps. Full-head H753 margin replay error<=8.58e-6; no-swap/self-swap logit error<=1.24e-5; raw full clamps exact; compact coordinate error<=7.07e-7; normalized norm error<=1.62e-6; complement scaling0.922–1.040. All frozen hashes unchanged and outputs finite. Every seed is retained; zero original reserved-test evaluations, natural document pool26,112 unchanged.

Next: test learned overlapping groups on shared dictionaries with exact union decoding and matched active-coordinate controls, then evaluate both total and component-specific effects. Treat H5 seed instability as a diagnostic target rather than selecting the best causal seed. Continue diverse-task confirmation before interpreting groups as manifolds or assigning unsupervised task labels.
