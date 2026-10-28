# Testing native neuron circuits in language models

Research proposal · 16 October 2026

## Objective

For a single prompt, how many neuron activations must remain for a language model to compute its answer? How does that number change with the fidelity requirement, extraction method, and model architecture? Do the resulting circuits continue to compute the right answer when the prompt changes?

The proposed experiment extracts small sufficient circuits from frozen models, measures their fidelity as a function of size, and tests their transfer to controlled counterfactuals. Compare conventional transformers with models containing bilinear MLPs and polynomial attention. Within each model, compare activation-based circuit extraction against methods that exploit weight-derived interactions.

The central hypothesis is that explicit interaction structure helps recover compact, reusable causal explanations. A small circuit for one answer is an initial result; a circuit that predicts changed answers under changed inputs is the stronger target.

## Motivation and scope

Nick Cammarata’s essay *Neural Networks Are Already Clean* argues that native mechanisms and their connections are productive units of interpretation. It uses neuron examples, upstream and downstream relationships, and interventions to explain computations. The essay reports less progress on attention and emphasizes studying attention through the neurons it reads and affects.

Circuit extraction makes one part of this perspective operational: can native mechanisms support a compact executable explanation of an answer? It does not directly test whether every neuron has a coherent semantic interpretation, or settle superposition versus recursive generalization. An understandable computation may require many neurons; a small sufficient circuit may still be semantically opaque.

This proposal separates three empirical questions:

1. **Circuit size:** How much of the computation is sufficient for one answer?
2. **Extraction quality:** Which method finds the best fidelity–size tradeoff under a fixed search budget?
3. **Reusability:** Does a circuit preserve the computation across examples, including examples with different correct answers?

## Define the masked computation

Use one binary mask variable per **layer, token position, and MLP channel**. In a gated or bilinear MLP, mask the channel after its multiplicative activation and before the output projection. Do not count the two input branches as independent neurons in the initial comparison.

For a bilinear MLP:

\[
h=(Lx)\odot(Rx),\qquad f_m(x)=D(m\odot h).
\]

For a SwiGLU MLP, replace the first expression with its actual gated activation. Preserve output biases and disclose all unmasked pathways.

Every masked forward pass must **recompute downstream activations from the masked upstream state**. Cached intact activations may be used for ranking or as fixed intervention baselines, but never replayed as the retained computation.

Minimal PyTorch hook logic, with a fresh forward pass for every evaluation:

```python
# h is computed from the currently masked upstream residual stream.
# mask has shape [batch, position, channel] for this layer.
h = h * mask
update = output_projection(h)
```

Keep model weights frozen and gradients enabled only as needed for importance estimates or mask optimization. Recompute normalization statistics, attention scores, and all downstream nonlinearities. In autoregressive evaluation, rebuild caches under the mask; do not reuse the intact model's key/value cache.

Start with zero ablation. Repeat selected conclusions using a fixed mean or matched corrupted-input baseline:

\[
\tilde h=m\odot h+(1-m)\odot b.
\]

Report the baseline: different baselines define different intervention questions. Replacement activations are injected information, so baseline sensitivity belongs in the interpretation.

## Fidelity and circuit size

For a correctly answered single-token example, let the intact answer be \(y\), and define

\[
\Delta(x;m)=\ell_y(x;m)-\max_{v\ne y}\ell_v(x;m).
\]

The ideal objective is

\[
\min_{m\in\{0,1\}^N}\|m\|_0
\quad\text{subject to}\quad
\Delta(x;m)\ge\alpha\Delta(x;\mathbf1).
\]

Use several fidelity levels, for example \(\alpha\in\{0.5,0.9,1.0\}\), and separately measure simple answer preservation. For multiple-choice tasks, report both the candidate-answer margin and, where appropriate, the full-vocabulary margin.

Also measure \(D_{KL}(p_{full}\|p_m)\). Preserving the winning token is much weaker than preserving the distribution. For multi-token answers, apply masks consistently over the teacher-forced continuation and report per-position fidelity, answer log probability, and free-generation correctness separately. Teacher-forced fidelity alone does not establish free-generation fidelity.

Report:

- Retained activation count and fraction, broken down by layer and position.
- Answer accuracy, margin retention, and output KL versus circuit size.
- Search cost in forward/backward evaluations and elapsed time.
- Retained structural computation where definable, separately from measured runtime.

Dense masked execution does not automatically save FLOPs or wall-clock time. Include an all-zero-MLP mask to measure how much the residual and attention pathways can do on their own.

Methods find **small sufficient circuits**, not certified global minima. On a tiny model or restricted candidate set, enumerate masks to calibrate the heuristic optimality gap. Fidelity need not improve monotonically as units are added: cancellation makes this a genuine search problem.

## Extraction methods

### Thresholding baselines

Build retention curves using random ranking, absolute activation, absolute gradient × activation, and direct output contribution where available. Include layer-stratified ranking to diagnose whether a global scale difference causes one layer to dominate selection.

Raw activation magnitude depends on parameter scaling. Add a write-magnitude baseline such as \(|h_k|\|d_k\|\), where \(d_k\) is the output projection column. Under a compensating channel rescaling, this is more stable than \(|h_k|\) alone. Keep signed contributions for analysis even when ranking by magnitude.

### Iterative pruning

Start with all channels retained. Remove a batch of low-ranked activations, rerun the masked model, and recompute importance. Reduce batch size or restore units when fidelity fails. Save the best feasible mask at each budget; do not assume a single greedy trajectory finds the best answer.

### Learned masks

Optimize continuous gates with a sparsity penalty or a fixed gate budget. A possible objective is

\[
\mathcal L(g)=D_{KL}(p_{full}\|p_g)
+\mu\max(0,\gamma-\Delta(x;g))
+\lambda\sum_i g_i,\qquad g_i\in[0,1].
\]

Use a small number of restarts. Threshold or select top-k gates, then evaluate the resulting **binary** mask with a fresh forward pass. Fractional gates change activation strengths and are not themselves the requested retained-neuron circuit.

Compare methods at matched extraction budgets. Report per-example distributions, not only the best examples.

## Weight-informed extraction in bilinear models

For

\[
f(x)=\sum_k d_k(l_k^\top x)(r_k^\top x),
\]

expand the residual stream into its actual additive writes, including the embedding:

\[
x=\sum_i a_i v_i.
\]

Then channel \(k\) contains terms

\[
a_i a_j(l_k^\top v_i)(r_k^\top v_j).
\]

For distinct sources, combine both orderings into

\[
C_{kij}=(l_k^\top v_i)(r_k^\top v_j)
+(l_k^\top v_j)(r_k^\top v_i).
\]

These coefficients suggest **joint parent groups** for pruning or mask initialization. A useful computation can require both parents even when either looks weak under an individual ranking. Use the coefficients together with observed source amplitudes; weight magnitude alone does not measure usage.

For a direct output contrast \(u\), incorporate \(u^\top d_k\). For earlier layers, follow specified downstream branches rather than treating direct unembedding projection as the total causal effect. Preserve normalization as an explicit operation. Folding observed RMS denominators into coefficients describes the observed pass; intervention evaluation must recompute them.

The clean comparison uses these scores to select the **same neuron-mask intervention space** as the baselines. Directly deleting interaction terms is a separate experiment with a different, more expressive intervention space.

Primary test: at fixed retained-neuron count and search budget, does interaction-informed extraction improve fidelity and counterfactual transfer?

## Include attention in stages

| Stage | Masked units | Interpretation |
|---|---|---|
| A | MLP channel–position activations | MLP sufficiency conditional on intact attention |
| B | Stage A plus head output–position vectors | Joint MLP and attention-output sufficiency |
| C | Stage B plus source–destination head contributions | Finer routing and payload sufficiency |

For Stage C, define the intervention precisely. Multiplying \(A_{ts}W_OW_Vx_s\) by an edge mask after softmax deletes delivered contributions without redistributing the remaining attention mass. Masking logits before softmax instead reallocates attention and is a different experiment. Use the former first and label any latter experiment separately.

Head-output vectors and scalar MLP activations have different costs. Report separate counts and a stated cost convention rather than summing them as interchangeable nodes. Masking an output also leaves the computation producing that output intact unless it is separately masked.

Weight-informed attention scores can combine

\[
v_i^\top M_{ts}v_j
\quad\text{and}\quad
l_k^\top Ov_m,
\]

where \(M_{ts}\) is the QK map including positional effects and scaling, and \(O=W_OW_V\). These identify the query-side request, key-side match, transported value, and downstream reader. In unnormalized polynomial attention, their products yield explicit expansion coefficients. In softmax attention, QK terms explain scores but source competition still requires the full softmax calculation.

Do not enumerate every interaction tensor. Generate candidates locally around active sources and relevant downstream readers, then compare candidate-generation cost against the baseline search budget.

## Prevent answer selection from masquerading as computation

A per-example mask is selected after observing a target answer. Its pattern can itself encode a choice that the retained network merely expresses. Therefore use three evaluation levels:

1. **Per-example extraction:** optimize a separate mask for each prompt. This measures local compressibility under an answer-informed search procedure.
2. **Frozen-mask transfer:** reuse the mask on aligned paraphrases and counterfactuals without refitting. Change the correct answer as well as surface details.
3. **Task-family circuits:** learn one mask on training examples and evaluate held-out examples. Use fixed template positions or a preregistered role alignment; never select a different alignment using the test answer.

Include changed names, swapped relationships, changed retrieval keys, distractors, and changed payloads. Start with equal token lengths so position masks transfer unambiguously. Later test variable lengths with a separately specified alignment rule.

Controls should include random masks with matched layerwise counts, masks extracted for different answers, and the embedding/residual-only baseline where meaningful. An answer-preserving mask that fails whenever the correct answer changes is weak evidence for a reusable mechanism.

## Necessity and redundancy

Sufficiency asks what must remain; necessity asks what must be removed to disrupt the answer. Measure both, using separately optimized masks. The complement of one sufficient circuit is not automatically a minimal necessary deletion set.

Repeat extraction with different seeds and penalties discouraging reuse of previous circuits. Measure overlap and test joint deletions. Several distinct sufficient circuits can indicate redundancy, but also search instability or compensating mechanisms created by masking. Use transfer and intervention results to distinguish these possibilities.

## Architecture comparison

Start with a controlled factorial comparison where trained checkpoints are available:

| MLP | Attention | Main comparison |
|---|---|---|
| SwiGLU | Softmax | Conventional reference |
| Bilinear | Softmax | Effect of the MLP computation |
| SwiGLU | Polynomial | Effect of attention computation |
| Bilinear | Polynomial | Combined structure |

Match tokenizer, training data, task suite, and evaluation procedure. Record parameter count, width, depth, training compute, and full-model performance. Perfect matching on all axes is generally unavailable; present performance-conditioned comparisons and the paired set of prompts answered correctly by both models, alongside each model's full results.

Keep normalization fixed in the first comparison. Test alternative normalization separately so its effect is identifiable. Use multiple training seeds for controlled small-model experiments.

Native neurons versus learned feature units is a useful follow-up, but reconstructed feature models introduce approximation error. Include the unpruned reconstruction baseline, intervention location, decoder cost, and reconstruction fidelity before comparing circuit counts. Native-neuron and feature-unit counts are not automatically equivalent.

## Initial implementation plan

1. Build a single-token relational retrieval task with balanced answers, distractors, and controlled query/payload changes. Use approximately 100 extraction prompts and a separate transfer set for the pilot.
2. Implement Stage A masks for one conventional and one bilinear model. Check that all-one masks reproduce logits and that a masked run genuinely propagates upstream changes downstream.
3. Run random, activation, write-magnitude, and gradient × activation retention sweeps. Use a coarse grid, then refine around fidelity transitions.
4. Apply iterative pruning and learned masks to a smaller subset; compare equal search budgets.
5. Add bilinear interaction-informed ranking or initialization without changing the allowable intervention.
6. Freeze masks for counterfactual transfer, including different correct answers. Estimate task-family circuits separately.
7. Add Stage B attention masks. Proceed to Stage C only after the simpler masking protocol is validated.
8. Expand to additional task families, checkpoints, and model sizes once the pilot identifies a real method or architecture difference.

No model retraining is required for the first pilot if suitable checkpoints exist. Budget extraction by measured GPU time per prompt and total forward/backward passes; learned masks require backpropagation through a frozen model and can still have substantial activation-memory cost.

Store the model/checkpoint identifier, prompt and answer, intact logits, intervention baseline, mask location, extraction seed and budget, binary mask, retained counts, fidelity results, and transfer results. Save each successful mask with its evaluation configuration so results can be reproduced.

## Deliverables and interpretation

Produce fidelity–size curves, distributions of smallest-found feasible circuit sizes, transfer accuracy at matched budgets, necessity-versus-sufficiency comparisons, and extraction-cost curves. Show a few interpretable circuits only after presenting aggregate results.

Evidence for the weight-based approach would be better fidelity or transfer at the same intervention and search budgets. Evidence for reusable native-neuron circuits would be frozen circuits that correctly handle counterfactuals and novel combinations. Small answer-specific circuits without transfer establish local compressibility only. Large circuits establish limited sparsity at the chosen granularity, not that the underlying neurons are unintelligible.

The target result is an executable account of how an answer is computed that remains valid when the answer should change.

## Source and provenance

Motivation: Nick Cammarata, *Neural Networks Are Already Clean*, Goodfire, 2026. The discussion above draws on the supplied essay, particularly its native-neuron methodology and attention section. [Essay](https://already-clean.goodfire.com/).

The experiment design, equations, evaluation criteria, and proposed defaults here are a research proposal, not reported experimental findings. This brief is self-contained and does not require the preceding conversation.
