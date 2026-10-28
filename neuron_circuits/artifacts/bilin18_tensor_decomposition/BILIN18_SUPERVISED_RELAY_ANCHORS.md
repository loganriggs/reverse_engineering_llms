## H946 — Discovery mechanism anchors improve held-out local responses, without a special effect-loss advantage

Fine-tune both H943 effect-trained64-product students for2000 additional steps under three arms: continued natural native/random-effect training; same objective plus discovery task endpoints; same objective plus discovery task finite effects. Original H848 discovery rows supply full rawrelay contrasts. One exact evaluation prompt overlaps discovery sentiment-04; an input-identity audit rejects the first run before optimization. Remove that ENTIRE discovery pair, retain every evaluation case, and rerun. Final anchors:8 agreement and7 sentiment pairs,30 directed rows, no exact prompt overlap with H849.

All arms start from the same-seed checkpoint and use the same natural minibatches/fresh Adam schedule. Anchors have equal task weighting and fixed totalweight1. Endpoint and effect objectives use different declared normalizers; both receive the same endpoint information. No projected task axis enters training. All final checkpoints retained; this is explicitly supervised known-mechanism validation, not discovery or untouched research-wide testing.

| Arm | Held-out agreement full local error | Sentiment full local error |
|---|---:|---:|
| Continued |97.4–97.5%|97.5–97.6%|
| Anchor endpoints |72.2–72.3%|53.9–54.2%|
| Anchor effects |72.5–72.6%|56.3–56.4%|

Effect training fits discovery much more closely (agreement7.7–8.2%, sentiment11.5–11.8%) than endpoints (32–34%,16%), but does not improve held-out local errors. That gap exposes limited generalization rather than a successful unique feature prior. Natural development native error worsens from91.1% continued to92.3% endpoints/92.0% effects; independent random-effect errors remain102–103%. None is a general MLP surrogate.

Verification:6 fits,60 accepted local teacher evaluations, saved effects replay exactly, hashes unchanged, finite, exact prompt sets disjoint, discovery-only anchors. Rejected overlap log retained; no original whole-model or reserved-test forwards. Dataset source disjointness here concerns exact inputs; no claim all lexical tokens are novel. Natural document pool unchanged.

Next full-model output evaluation is required because local-vector improvements do not establish behavioral fidelity. Preserve recipient native MLP baseline and exact residual change as inH945, add explicit zero-MLP-change KL reference, and retain all seeds/arms. After that, test whether improved predictions arise from reusable/stable components rather than naming the supplied labels as discoveries.
