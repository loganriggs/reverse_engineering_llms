## H560 — Response weighting within the same span barely changes transfer

Use exactly H558's 40 discovery contexts and 40 excluded lexical-block contexts. Compare ordinary context PCA with a response-weighted projection at identical ranks 4/8/16/32. Both lie in the same 39-dimensional centered discovery-context span. Construct the response metric from the 40 discovery normalized displacements at +1/16 amplitude, each normalized to unit length. It measures mean squared bilinear linear-response error over the cross-product of discovery contexts and displacement directions, not downstream KL. Whiten coordinates using that metric, perform PCA there, and unwhiten. Eigenvalues are floored at 1e−8 of the largest for numerical stability. No test context or output is used for fitting.

The weighted method is guaranteed no worse on its training quadratic objective within this span; checks confirm reductions of roughly 0.5–2.5% over context PCA. This small training improvement is not evidence of generalization. Evaluation uses exact/zero/global-mean controls, two amplitudes and eight rank/method combinations: 880 interventions. KL is exact-perturbed distribution versus approximate-perturbed distribution, normalized against omitting only L8's response.

| Family, +1 | PCA rank 32: response error / KL ratio | Weighted rank 32: response error / KL ratio |
|---|---|---|
|binding|19.1% / 0.036|19.0% / 0.036|
|color_prose|51.0% / 0.242|51.1% / 0.243|
|color_record|58.1% / 0.302|58.1% / 0.302|
|name_prose|39.9% / 0.281|39.9% / 0.280|
|name_record|57.2% / 0.520|57.1% / 0.516|

Held-out differences are tiny and not uniformly beneficial. This particular metric change does not remedy the copying fidelity gap. It does not rule out a different response metric, a KL-trained model, or directions outside the observed training span. These are retrospective test examples; H561 freezes the bases for prospective lexical examples.

Audits: weighted training objective no worse at each rank; disjoint fit/test IDs; source/bases unchanged; exact margin replay within 1.91e−5 and exact KL within 2.77e−11; finite nonnegative KL. No new prompts or corpus documents. Broad goal active.
