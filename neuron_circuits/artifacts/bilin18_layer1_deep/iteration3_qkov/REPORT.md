# Bilin18 QK×OV causal edge experiment

## Verdict

The weight-and-activation QK×OV calculation identifies attention edges that causally control the selected native neuron on disjoint FineWeb examples. The result survives routing-only and random-edge controls. It does **not** support a one-edge→one-neuron→one-logit circuit: restoring the selected neuron removes only about one third of the edge's additive logit change, because the native attention edge writes a full residual vector read by many channels.

The all-one reference and custom forwards match exactly: max logit error 0.

## Selection panel

|layer|n|prediction r|sign agreement|median |Δ product||fraction of full local-attention effect|median KL|mean ΔNLL|mediated logit fraction|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|48|0.736|79.2%|26.50|56.3%|0.002391|-0.0191|26.1%|

Paired target-edge controls:

|control|QK×OV wins|median |Δproduct| advantage|one-sided paired sign p|unit-cluster mean [95% CI]|
|---|---:|---:|---:|---:|
|route_only|70.8%|15.62|0.00276|57.35 [13.52, 122.27]|
|same_head_random_source|95.8%|26.00|4.18e-12|80.48 [32.66, 144.77]|
|same_source_random_head|93.8%|24.75|8.02e-12|80.16 [33.43, 142.50]|
|uniform_random|95.8%|26.50|4.18e-12|80.66 [33.57, 143.15]|

## Transfer panel

|layer|n|prediction r|sign agreement|median |Δ product||fraction of full local-attention effect|median KL|mean ΔNLL|mediated logit fraction|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|48|0.642|70.8%|25.75|41.1%|0.004076|+0.0095|26.8%|

Paired target-edge controls:

|control|QK×OV wins|median |Δproduct| advantage|one-sided paired sign p|unit-cluster mean [95% CI]|
|---|---:|---:|---:|---:|
|route_only|70.8%|13.88|0.00276|37.61 [16.33, 59.70]|
|same_head_random_source|97.9%|23.12|1.74e-13|68.82 [36.16, 104.96]|
|same_source_random_head|91.7%|21.75|1.23e-10|66.33 [34.46, 103.66]|
|uniform_random|95.8%|23.50|3.41e-13|66.82 [34.10, 103.06]|

## Red-team limits

- Neurons were selected in the earlier semantic atlas. The transfer panel is fresh, but this is evidence about those twelve neurons, not an unbiased population rate.
- Edge ranking is recomputed from weights and activations for each input. Transfer therefore validates the explanatory formula, not one fixed absolute source position.
- The routing-only control isolates the extra value of OV/factor alignment. Random controls are weaker and sometimes quantize to zero product change in bfloat16.
- An attention edge writes a 1,152-dimensional vector. Its final loss effect can be large even when the selected neuron is not its principal mediator.
- Signed NLL changes can cancel: ablating a learned edge occasionally improves a particular next-token prediction. Product prediction and KL are cleaner mechanism tests.
- Sites are correlated within twelve neurons. The report therefore includes bootstrap intervals over neuron-level means rather than treating 48 sites as independent models.
