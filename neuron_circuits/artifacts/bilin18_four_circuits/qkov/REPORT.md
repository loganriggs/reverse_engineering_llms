# Bilin18 QK×OV causal edge experiment

## Verdict

The weight-and-activation QK×OV calculation identifies attention edges that causally control the selected native neuron on disjoint FineWeb examples. The result survives routing-only and random-edge controls. It does **not** support a one-edge→one-neuron→one-logit circuit: restoring the selected neuron removes only about one third of the edge's additive logit change, because the native attention edge writes a full residual vector read by many channels.

The all-one reference and custom forwards match exactly: max logit error 0.

## Selection panel

|layer|n|prediction r|sign agreement|median |Δ product||fraction of full local-attention effect|median KL|mean ΔNLL|mediated logit fraction|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|24|0.815|91.7%|28.00|24.1%|0.001394|+0.0336|25.4%|

Paired target-edge controls:

|control|QK×OV wins|median |Δproduct| advantage|one-sided paired sign p|unit-cluster mean [95% CI]|
|---|---:|---:|---:|---:|
|route_only|70.8%|21.00|0.032|66.62 [-74.65, 200.81]|
|same_head_random_source|91.7%|19.50|1.79e-05|113.13 [20.85, 211.67]|
|same_source_random_head|87.5%|27.50|3.3e-05|117.04 [21.95, 212.12]|
|uniform_random|83.3%|27.50|6.06e-05|119.38 [21.77, 217.00]|

## Transfer panel

|layer|n|prediction r|sign agreement|median |Δ product||fraction of full local-attention effect|median KL|mean ΔNLL|mediated logit fraction|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|24|0.466|75.0%|25.25|15.6%|0.002528|+0.1300|27.2%|

Paired target-edge controls:

|control|QK×OV wins|median |Δproduct| advantage|one-sided paired sign p|unit-cluster mean [95% CI]|
|---|---:|---:|---:|---:|
|route_only|62.5%|9.09|0.154|-8.99 [-126.49, 70.19]|
|same_head_random_source|91.7%|23.00|1.79e-05|66.65 [27.77, 118.97]|
|same_source_random_head|91.7%|23.50|1.79e-05|69.51 [24.53, 119.73]|
|uniform_random|95.8%|25.25|1.49e-06|73.35 [28.98, 124.74]|

## Red-team limits

- Neurons were selected in the earlier semantic atlas. The transfer panel is fresh, but this is evidence about those twelve neurons, not an unbiased population rate.
- Edge ranking is recomputed from weights and activations for each input. Transfer therefore validates the explanatory formula, not one fixed absolute source position.
- The routing-only control isolates the extra value of OV/factor alignment. Random controls are weaker and sometimes quantize to zero product change in bfloat16.
- An attention edge writes a 1,152-dimensional vector. Its final loss effect can be large even when the selected neuron is not its principal mediator.
- Signed NLL changes can cancel: ablating a learned edge occasionally improves a particular next-token prediction. Product prediction and KL are cleaner mechanism tests.
- Sites are correlated within twelve neurons. The report therefore includes bootstrap intervals over neuron-level means rather than treating 48 sites as independent models.
