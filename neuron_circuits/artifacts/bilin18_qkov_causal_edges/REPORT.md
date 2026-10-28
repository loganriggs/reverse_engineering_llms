# Bilin18 QK×OV causal edge experiment

## Verdict

The weight-and-activation QK×OV calculation identifies attention edges that causally control the selected native neuron on disjoint FineWeb examples. The result survives routing-only and random-edge controls. It does **not** support a one-edge→one-neuron→one-logit circuit: restoring the selected neuron removes only about one third of the edge's additive logit change, because the native attention edge writes a full residual vector read by many channels.

The all-one reference and custom forwards match exactly: max logit error 0.

The transfer result is not equally strong at every depth. Layers 0 and 2 retain
prediction correlations of 0.98 and 1.00 with at least 93.8% sign agreement;
layer 1 falls to correlation 0.72 and 68.8% sign agreement. The aggregate edge
ranking succeeds, but the layer-1 formula needs a richer normalization or
multi-edge treatment before it should be called a reliable local explanation.

## Selection panel

|layer|n|prediction r|sign agreement|median |Δ product||fraction of full local-attention effect|median KL|mean ΔNLL|mediated logit fraction|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|16|0.971|100.0%|496.00|46.1%|0.0009123|+0.0518|44.4%|
|1|16|0.898|100.0%|41.50|40.0%|0.001815|-0.0277|24.3%|
|2|16|0.985|100.0%|57.00|95.2%|0.002755|-0.0154|32.1%|

Paired target-edge controls:

|control|QK×OV wins|median |Δproduct| advantage|one-sided paired sign p|unit-cluster mean [95% CI]|
|---|---:|---:|---:|---:|
|route_only|81.2%|51.25|9.16e-07|239.86 [91.48, 430.52]|
|same_head_random_source|100.0%|67.25|3.55e-15|288.25 [130.02, 484.84]|
|same_source_random_head|97.9%|54.00|1.74e-13|284.21 [122.72, 488.14]|
|uniform_random|95.8%|66.75|3.41e-13|289.05 [127.65, 500.83]|

## Transfer panel

|layer|n|prediction r|sign agreement|median |Δ product||fraction of full local-attention effect|median KL|mean ΔNLL|mediated logit fraction|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|16|0.979|100.0%|642.00|51.4%|0.0006778|-0.0060|42.5%|
|1|16|0.722|68.8%|26.25|28.4%|0.003662|+0.1918|22.1%|
|2|16|0.996|93.8%|47.50|43.7%|0.00132|+0.0263|31.3%|

Paired target-edge controls:

|control|QK×OV wins|median |Δproduct| advantage|one-sided paired sign p|unit-cluster mean [95% CI]|
|---|---:|---:|---:|---:|
|route_only|81.2%|39.50|7.61e-06|275.07 [73.26, 561.93]|
|same_head_random_source|87.5%|59.00|2.55e-09|346.60 [109.67, 678.41]|
|same_source_random_head|93.8%|64.50|6.56e-11|333.61 [108.30, 655.51]|
|uniform_random|95.8%|62.50|3.41e-13|354.63 [118.09, 689.05]|

## Red-team limits

- Neurons were selected in the earlier semantic atlas. The transfer panel is fresh, but this is evidence about those twelve neurons, not an unbiased population rate.
- Edge ranking is recomputed from weights and activations for each input. Transfer therefore validates the explanatory formula, not one fixed absolute source position.
- The routing-only control isolates the extra value of OV/factor alignment. Random controls are weaker and sometimes quantize to zero product change in bfloat16.
- An attention edge writes a 1,152-dimensional vector. Its final loss effect can be large even when the selected neuron is not its principal mediator.
- Signed NLL changes can cancel: ablating a learned edge occasionally improves a particular next-token prediction. Product prediction and KL are cleaner mechanism tests.
- Sites are correlated within twelve neurons. The report therefore includes bootstrap intervals over neuron-level means rather than treating 48 sites as independent models.
