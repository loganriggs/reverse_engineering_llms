## H454 — Frozen compact operation on fresh documents

Freeze H453 joint rank128/256 encoders/maps, H452 weight rank128/256 maps and H451 simplified output rule. Sample512 new complete documents with no hash overlap against12480 prior documents; independently rehash all prior/new documents and verify sampled token windows. Randomly select128 receiving contexts before baseline collection, then use24 fresh source draws and eight fresh random output spans. All512 documents receive baseline collection;128 receive interventions. Total unique corpus documents12992. This is fresh receiving-document transfer conditional on the same model, frozen calibration and donor pool, not model-training holdout.

No refit or reselection. The simplified output rule uses top128 probabilities, shared tail mean, constant isotropic tail and receiving RMS transport with condition cap1000. Exact native retained response and radial correction remain. Dense/full-output and dense/simplified-output references distinguish input-map approximation from output-fit approximation.

| Predictor | Learned finite KL recovery | Random mean recovery | Linear-response energy coverage |
|---|---:|---:|---:|
|dense_full_output|97.549%|95.309%|reference|
|dense_simple_output|97.567%|94.942%|reference|
|weight_128_simple_output|95.813%|93.205%|91.410%|
|weight_256_simple_output|96.552%|93.898%|94.615%|
|joint_128_simple_output|95.919%|93.290%|94.057%|
|joint_256_simple_output|96.687%|94.036%|96.553%|

| Frozen contrast | Estimate | Joint95% interval | Lower > −.02 |
|---|---:|---|---|
|joint128_minus_dense_learned|-0.016478|[-0.025027, -0.007930]|False|
|joint128_minus_dense_random_mean|-0.016521|[-0.025069, -0.007972]|False|
|joint256_minus_dense_learned|-0.008802|[-0.017351, -0.000254]|True|
|joint256_minus_dense_random_mean|-0.009058|[-0.017607, -0.000510]|True|

Four frozen2pp noninferiority gates compare joint128/256 against dense/simple for learned/randommean spans. Joint95% maximum-absolute-deviation bands use4000 paired source/context bootstrap samples and resampling of eight spans. Weight-versus-joint and full-versus-simple contrasts are descriptive here. Frozen donor pool and calibration uncertainty are not resampled.

Audits: independently rehashed12480 excluded complete documents plus512 fresh, overlap0, all windows match source. Compiled-versus-expanded map response relative error4.7e-15; solve residual2.92e-15; maximum condition number1000.000000. All gates pass and arrays finite; no outcome exclusions.

Fresh-document transfer supports rank256 but not the frozen rank128 tolerance claim. Joint128 recovery95.919% learned/93.290% random versus97.567%/94.942% dense/simple gives losses1.648pp/1.652pp. Joint95% intervals are[−2.503,−0.793] and[−2.507,−0.797]pp, so both2pp noninferiority gates fail even though point losses are inside the margin. This does not prove rank128 cannot meet the tolerance; the earlier reused-panel pass did not confirm here.

Joint256 recovery96.687%/94.036% loses0.880pp/0.906pp, intervals[−1.735,−0.025] and[−1.761,−0.051]pp. Both frozen gates pass. Rank256 compiled maps+encoder use5,013,504 scalars,23.61% of dense map storage (about4.24x fewer), excluding other predictor components. Rank128 remains11.81% of dense storage but lacks this confirmation. There is no established sharp minimum rank or whole-model compression ratio.

Weight128 recovery95.813%/93.205% and weight25696.552%/93.898% remain close to their joint counterparts on point estimates; their advantage comparisons were not primary here. Full-output dense97.549%/95.309% versus simplified dense97.567%/94.942% suggests modest output-rule differences, but no new noninferiority claim is made for that contrast. All variants have controlled correction norm ratios, at most1.16, and numerical checks pass. Absolute recovery shifts versus the prior panel should not be attributed solely to contexts: fresh source draws and random spans changed too.

The scientific update is a frozen compact input-dependent operation with demonstrated practical transfer at rank256 for this late-layer/head intervention family. It remains a structured conditional predictor using native retained factors, exact radial information,16 source coordinates and context-sensitive output fitting.

Next separate failure of the first-order response law from input compression as intervention magnitude changes. Freeze dense/simple, joint128 and joint256; test source amplitude0.25/0.5/1/2 with native finite targets, identical receiving contexts and source draws, and scale the analytic linear/radial correction consistently. Include signed edits to expose odd/even curvature if feasible, and keep native retained responses exact at each amplitude. Compare compact-minus-dense losses separately from dense-versus-target errors. This targets the validity range of the putative computational unit rather than adding rank to chase a threshold. Follow source-family and cross-layer transfer after the amplitude mechanism is clear; semantic content remains unestablished.

Cost/scope: map-plus-encoder storage is2,506,752 scalars at rank128 and5,013,504 at256, versus21,233,664 dense maps. Baseline model, retained/radial terms, source coordinates and output fitting remain additional. This is conditional source-response prediction in one late-layer/head setting, not intact-model compression, a semantic ontology or measured runtime acceleration.
