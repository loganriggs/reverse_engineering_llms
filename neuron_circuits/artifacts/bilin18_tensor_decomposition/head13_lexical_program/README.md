# Conditional head13.0 lexical write program

This package maps a recipient/donor pair of **raw float32 source-token embeddings** to two residual-write differences. Output shape is `[..., 2, 1152]`; the positions are the source token and the immediately following final token. It uses first-layer value weights, head13.0 output weights, native scalar mixing parameters, and two routing coefficients fitted on eight discovery examples.

```python
import torch
from execute import delta_writes
weights = torch.load('weights.pt', weights_only=True)
write_delta = delta_writes(weights, recipient_embeddings, donor_embeddings)
```

Keep weights and inputs on the same device. No model object, attention calculation, hidden-state prefix, or downstream evaluator is accessed by this executor. Raw embeddings and their token lookup remain external inputs. The package does not return logits or answer the language question independently.

The fixed routing was validated on short country-to-language prompts whose source is immediately before the final token. It is not validated for longer distances, competing source mentions, or arbitrary contexts. Native-prefix and QK removal applies only to generating this isolated write. Installing it in a model still needs the baseline state and native downstream computation.

`manifest.json` records hashes, tensor sizes and external dependencies; `independent_audit.json` records a separate NumPy and CPU replay. The package has 294,917 scalar coefficients (1,179,668 tensor bytes) and 442,368 matrix multiply-accumulates per embedding pair, excluding normalization, scalar operations and embedding lookup. It is an extraction/approximation of one causal path, not a parameter-compressed replacement of a full model component.
