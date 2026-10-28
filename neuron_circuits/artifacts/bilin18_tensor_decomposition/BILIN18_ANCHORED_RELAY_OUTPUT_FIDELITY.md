## H947 — Supervised anchors transfer substantially better downstream; endpoint and effect objectives remain close

Run4992 whole-model evaluations of all six frozenH946 students on all312 H944 jobs. Keep native recipient MLP baseline; replace only the finite MLP change, with and without the exact residual change. Compare full-vocabulary distributions to exact matching-path interventions, and add zero-MLP-change KL baselines (identity for MLP-only, residual-only for joint). This is intervention fidelity, NOT native replacement by the compact student.

Representative seed0 values below; seed1 is retained and closely similar, with no seed selection:

| Family/arm | Continued joint-effect error | Endpoint anchors | Effect anchors |
|---|---:|---:|---:|
| Agreement full |73.7%|24.1%|24.0%|
| Agreement rank1 |71.1%|12.9%|12.2%|
| Sentiment full |85.9%|26.4%|27.0%|
| Sentiment rank1 |83.0%|17.3%|17.3%|

The corresponding joint KL ratios to zero MLP change are agreement full .122/.118 (endpoint/effect), agreement rank1 .050/.044, sentiment full .096/.099, sentiment rank1 .047/.047. These are substantial improvements over dropping the MLP change, while preserving native reconstruction error in the intervention setup. All vectors/full distributions matter: improved joint behavior does not imply accurate entire local updates, which still have54–73% error.

Isolated MLP margin errors remain45–46% agreementfull,29–31% agreementrank1,32–33% sentimentfull,21% sentimentrank1. Exact residual-path information and downstream interaction therefore help the joint result; it cannot be credited entirely to an independently accurate MLP model. Both anchors improve behavior considerably over continued random-probe training, but no universal effect-loss advantage emerges. The main evidence is that discovery mechanism coverage supplies useful structure that random interventions missed.

All input/output points are reused research panels, with exact train/evaluation prompt overlap removed beforeH946 optimization. These are supplied mechanisms and task-informed sites, so success is supervised transfer validation, not new semantic-manifold discovery. General natural responses still fail and native surrogate replacement remains unverified. Neither the64 products nor their partitions have yet earned an interpretable-unit label.

Verification:4992 expected/actual forwards, identity-logit max3.43e-5, historical joint-margin replay1.79e-5, all hashes unchanged/finite, no fitting or reserved test. Residual/MLP/joint component summaries exactly replayH945. Full perexample metrics and zero-change KL retained. Natural document pool unchanged.

Next assess the learned computation rather than only scores: use the bilinear product decomposition to measure whether task improvements concentrate in a small reproducible set of products or require distributed cancellations, accounting for factor swaps/scales and whole-product output equivalence. Then test selective product removal/reuse across tasks and fresh contexts, with discovery-only selection. Do not equate good supervised task reconstruction with recovered fundamental units. An independent task/mechanism or new combination is needed for a discovery claim.
