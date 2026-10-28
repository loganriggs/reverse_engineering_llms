# Native-neuron circuit pilot: arithmetic

## Main result

For 24 correctly answered prompts, gradient × activation found a **median 90%-margin circuit retaining 23.7%** of the 1,284,096 layer–position–neuron activations (IQR 17.0%–43.9%). Random ranking required a median 98.4%.

This is substantial compression, but not an ultra-small circuit. Circuit size varied sharply by prompt (2.0%–65.6%), so a single headline neuron count would be misleading.

The position-specific count hides another important fact: after collapsing token positions, the median circuit touched **97,828 of 116,736 unique layer–channel neurons (83.8%)** at least once. Most parameter neurons participate somewhere; sparsity is much stronger over *when* a neuron fires than over a globally reusable neuron subset.

## What happens when neurons are removed

- Errors degrade in a graded way: masks first produce nearby and far numeric answers; at severe ablation they increasingly produce non-numeric tokens.
- Frozen 90%-margin gradient × activation masks remained exactly correct on **79.2%** of changed arithmetic problems. Their failures were {'exact': 19, 'far': 2, 'nearby': 3}.
- A single task-family mask retained **93.3%** exact accuracy at approximately 20% retained activations on 30 held-out problems.
- Necessity is much sparser than sufficiency: after deleting roughly 0.10% of the highest-attribution activations, exact accuracy was only **8.3%** on the 12-prompt necessity subset.
- Baseline choice is consequential. At 20% retention, zero ablation reached 83.3% accuracy (median KL 0.249); the 24-prompt arithmetic mean reached 100.0% (KL 0.071); and the 550,000-token WikiText-103 mean with its correct gradient × (activation − mean) ranking reached 91.7% (KL 0.145).
- Reusing the zero-ablation ranking with the corpus mean reached only 25.0% at 20%. Mean replacement changes both the intervention and which units count as important.
- Iterative re-ranking did not improve the six-prompt 20% circuit subset over one-shot gradient × activation (both 100% exact with the same median margin retention), while requiring 8 extra backward evaluations. More search was not automatically better here.

## Interpretation

The model's arithmetic behavior is distributed across many MLP activation instances, but it is not uniform. Attribution-based selection compresses the computation far better than random selection, and a tiny targeted deletion can be disproportionately damaging. Per-input masks partly transfer when the operands and answer change, but they are not fully reusable.

Counts here are activation instances indexed by layer, prompt position, and MLP channel—not unique parameter neurons. Attention remains intact, so this is MLP sufficiency conditional on the original attention computation.

## Scope and cautions

- Model: `Qwen/Qwen2.5-0.5B`; 152/152 eligible prompts answered correctly before masking.
- Task: single-token, single-digit addition, subtraction, and multiplication; 24 extraction prompts and 30 held-out task-family prompts.
- Primary intervention: binary zero ablation after the gated MLP activation and before its output projection, with a fresh full forward pass every time.
- Large-corpus baseline: 50,000 fixed-length windows (550,000 tokens) streamed from WikiText-103 training text. Corpus-mean rankings use |gradient × (activation − mean)|.
- Semantic expected distance assigns absolute error to digits and cost 10 to non-numeric vocabulary mass. The reported nearby/far/non-numeric categories are less assumption-sensitive than that scalar.
- The extraction is heuristic and answer-informed. It finds small sufficient masks, not certified minima or proof that individual neurons have human-readable meanings.
- Dense masking measures causal sufficiency, not runtime or FLOP savings.
