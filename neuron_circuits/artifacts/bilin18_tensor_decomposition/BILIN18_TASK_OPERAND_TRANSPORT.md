## H534 — Portable task writes coexist with different operand interactions

Following H533's eight-family screen, retain its five discovery-selected MLP layers: agreement L12, sentiment L12, capital recall L13, entity binding L10 and sequence copying L10 (zero-based). No layer is selected again. Reuse every eligible discovery/transfer pair. This is a mechanistic diagnostic on the H533 panel, not fresh confirmation on a new sample.

### What was tested

A bilinear MLP multiplies two learned projections of its normalized input, then maps their product back into the residual stream. Write these operands as L and R and the output map as D. Given a paired donor change (dL,dR), its exact output change decomposes as:

`D(dL * R) + D(L * dR) + D(dL * dR)`.

The first two terms change one operand at a time. The last is the finite cross term: information lost by adding the two separate effects. Evaluate each term, their sum without the cross term, and the complete donor write while holding the recipient's bypass and upstream computation native and recomputing downstream computation. These algebraic terms sum in MLP output space; their effects on answer probabilities need not add because downstream computation is nonlinear. A large cross-only effect is consequently not a unique fraction of the full effect.

For agreement and sentiment, separately average the discovery MLP-output changes from singular to plural or positive to negative. Freeze those two vectors. Add each to base transfer contexts and subtract it from counterfactual contexts. Compare against its reversed sign, a same-norm mean nuisance change, and 16 seeded isotropic Gaussian vectors of the same norm. No transfer example determines the vector. These controls match Euclidean norm, not residual covariance or answer-readout alignment.

### Results on transfer

Effects below are donor-directed changes in answer log-odds: how strongly the model favors the donor's answer relative to the recipient's answer. They are not changes in task accuracy.

| Task | Complete donor write | Without finite cross term | Increment from including cross term | Descriptive 95% pair-bootstrap interval for increment |
|---|---:|---:|---:|---|
|agreement|1.8437|1.5014|0.3423|[0.2616, 0.4249]|
|capital_recall|0.8757|0.6690|0.2067|[0.1510, 0.2613]|
|entity_binding|0.1454|0.0994|0.0460|[0.0374, 0.0522]|
|sentiment|0.8239|0.8249|-0.0010|[-0.0043, 0.0020]|
|sequence_copy|0.2886|0.2751|0.0135|[-0.0193, 0.0458]|

Agreement loses about 19% of this selected-layer donor effect when the cross term is omitted. Sentiment's effect is essentially unchanged. Capital recall and binding also have positive cross-term increments, though binding's total local MLP effect is small. Copying remains inconclusive. This distinguishes the selected local operations; it does not show that agreement is globally nonlinear while sentiment is globally linear. The normalized operands already include upstream processing, and downstream context determines how a write is used.

| Task | Shared discovery-mean change | Reversed shared change | Same-norm nuisance | Range of 16 random-direction means | Positive shared directions |
|---|---:|---:|---:|---:|---:|
|agreement|1.6320|-1.4891|0.0822|[-0.1767, 0.1260]|16/16|
|sentiment|0.9454|-0.9211|0.0855|[-0.1948, 0.2233]|16/16|

Both shared writes transport in the predicted direction across all eight transfer pairs, in both directions. Reversing the vector reverses the effect on every tested direction. Shared-minus-nuisance effects have descriptive pair-bootstrap intervals of [1.264, 1.868] for agreement and [0.833, 0.889] for sentiment. These intervals resample only eight deliberately constructed pairs, keeping the two directions together; they omit uncertainty from discovery, templates and model selection. They are not population confidence claims. Some random directions also move all answers consistently, which is why effect magnitude and control comparisons matter more than sign counts alone.

### Interpretation and remaining alternatives

The strongest supported statement is that a reusable *causal output change* exists within each of these two tested tasks. This is stronger evidence than low-rank reconstruction alone: the direction learned from one set of contexts changes behavior in other contexts, and its sign predicts the effect. It is not yet an identified semantic feature or a complete causal algorithm. A generic bias toward the declared output-token classes could explain this transport. New tests must vary the readout/output vocabulary while preserving the underlying number or polarity distinction, and compare against output-aligned control directions.

Reusable writes and conditional operations are compatible: a context-sensitive operation may produce a portable output. Here agreement's selected MLP operation uses a meaningful finite cross term, while sentiment's selected effect is approximately additive for these paired changes. The checkpoint's Left and Right coordinates are not claimed to be canonical semantic axes.

The next high-information tests are (1) task/readout-crossed transfer to distinguish a semantic feature from an answer-word bias, and (2) attention/residual/value-bus localization for binding and copying, whose single-MLP effects were small. Broader multi-layer hypotheses remain open. H533's twenty-hypothesis list remains the working alternatives; prioritize reusable features, conditional operations, routing, and source/reader dependence rather than another compression sweep.

### Verification and scope

The 960 operand interventions and 608 transport interventions are finite; complete-donor replay agrees with H533 to within 6.7e-6 in margin, and self swaps agree within 6.2e-6. The algebra was independently checked in float64 using the same rounded weights as inference: maximum algebra error 5.7e-14, maximum relative L2 discrepancy from the actual float32 donor-write difference 1.5e-6. Initial comparison against unrounded checkpoint weights was corrected to match the inference loader. Inference uses float32 computation after the shared loader converts checkpoint weights to bfloat16; it is not unrounded float32 checkpoint inference.

Both runs use a small synthetic prompt suite. Discovery and transfer vary templates and items together, and some vocabulary overlaps; a crossed design is still needed. The exploratory negation examples also have pragmatic ambiguity (for example, “not wonderful” need not mean wholly negative), and ambiguous-word probes do not specify uniquely required completions. Their gate failures are format-specific observations. No new FineWeb documents were consumed; tracked total remains 26,112. Source and manifest hashes, per-intervention results, operands, frozen transport vectors and the numerical audit are archived. The broad research goal remains active.
