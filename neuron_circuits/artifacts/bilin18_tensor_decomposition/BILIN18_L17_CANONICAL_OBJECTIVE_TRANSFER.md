## H468 — Held-out objective transfer and intervention-delta faithfulness

Freeze H467's validation-selected single updates: KL fraction.01, CE fraction.03, and the original H466 write map. Evaluate the same128 H466 test contexts without refitting or selecting steps. These test documents were historically used and baseline results were already viewed; the two alternatives were frozen before their test scores. No new corpus documents. All maps retain the same64 empirical bilinear input coordinates and rank16 output budget.

| Intact test map | Normalized write MSE | Native-output KL | CE change versus native model | Top1 agreement |
|---|---:|---:|---:|---:|
|write|0.175631|0.125486|+0.032109|83.594%|
|kl|0.175693|0.123870|+0.040913|82.812%|
|ce|0.176494|0.130306|+0.044512|82.031%|

| Frozen contrast | Estimate | Joint95% interval | Directional gate |
|---|---:|---|---|
|CE_update_minus_write_CE|+0.012402|[-0.007894, +0.032699]|False|
|CE_update_minus_write_KL|+0.004819|[-0.003391, +0.013030]|False|
|KL_update_minus_write_KL|-0.001616|[-0.004476, +0.001244]|False|
|CE_update_minus_write_writeMSE|+0.000862|[+0.000544, +0.001181]|True|

The four prespecified directions are CE update improves CE, CE update worsens KL, KL update improves KL, and CE update worsens write accuracy.4000 paired test-document bootstrap replicates, joint4 standardized maximum-deviation intervals, conditional on frozen calibration/validation selection. Only write degradation passes. CE update minus write CE is+.012402 with interval[−.007894,+.032699], so the validation CE improvement does not establish a held-out benefit or a significant held-out worsening. Its KL change+.004819 has interval[−.003391,+.013030]. KL-directed update lowers test KL by.001616 on point estimate, interval[−.004476,+.001244], which also fails its directional test. CE update write MSE increases.000862, interval[.000544,.001181]. Thus this run does not establish a successful task-loss-versus-faithfulness tradeoff, despite showing one on validation.

To separate full-state error from response error, evaluate two predictions under each signed source intervention. Full-state prediction uses the approximated MLP write directly. Delta prediction adds only predicted_write(changed_input)−predicted_write(original_input) to the exact native intact state. Compare the latter to the actual intervened state. The exact baseline is an evaluation device for isolating response error, not part of the deployable full-write predictor. The direct residual input path stays fixed. Recovery is1−pooled prediction error/pooled intervention effect. A zero predicted change gives0 recovery; negative values mean larger error than that baseline on the point estimate.

| Map | Negative intervention KL / raw-write recovery | Positive intervention KL / raw-write recovery |
|---|---|---|
|write|-34.07% / 63.74%|-29.38% / 60.78%|
|kl|-32.33% / 63.66%|-27.31% / 60.71%|
|ce|-32.09% / 63.60%|-27.94% / 60.66%|

Mean native effect KL is.00203481 for negative and.00187658 for positive amplitude. Raw-write change recovery is about61–64%, but KL recovery point estimates are negative for every map. Uncertainty is broad: the write map's negative KL recovery marginal95% interval is[−1.212,+.196], positive[−.955,+.148]. Other maps have similar intervals. This does not establish that every predictor is significantly worse than zero change; it does show no strong response coverage on this panel. These descriptive intervals condition on the deterministic donor assignment (64 donors used twice), not independent donor-population sampling. Full-state fit and causal-response fidelity remain distinct requirements.

Implication for the data-canonical proposal: defining equivalence on ordinary states alone can leave intervention-important distinctions unresolved. The present approximation also truncates the empirical function family and uses limited training data, so the result cannot establish a general impossibility of natural-data recovery. Nor have these three maps been optimized globally for their objectives. The idea remains useful, but the empirical measure and behavior being preserved must be specified alongside the canonical coordinates.

Next information-gaining comparison: construct ordinary-state and intervention-augmented empirical bilinear metrics on training documents only, then fit equal feature/output ranks and compare intact reconstruction and intervention-delta prediction on frozen held-out cases. Count all reference-support storage, not just latent rank. Include a control that expands training examples without adding new perturbation directions, so better coverage is not automatically attributed to source geometry. Keep output objective fixed initially to distinguish the choice of data-supported input directions from optimization of the output loss. After separating these effects, return to KL/CE alternatives with a training protocol that demonstrably improves held-out objectives. Fresh-document confirmation remains necessary for selected conclusions.

Numerical audits pass: the write baseline replays H466 within1e−4 in stored per-example metrics; zero-amplitude predicted deltas vanish within1e−8; saved arrays are finite. Configs, maps, raw rows and paired bootstrap draws are preserved. Total12992 corpus documents. The research goal remains active.
