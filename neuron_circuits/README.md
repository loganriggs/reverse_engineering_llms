# Native-neuron circuit pilot

Current high-level mechanism update: [CURRENT_COMPUTATION_REGIME_UPDATE.md](CURRENT_COMPUTATION_REGIME_UPDATE.md)

This repository implements the core experiment in
[`neuron_circuit_experiment_brief.md`](neuron_circuit_experiment_brief.md): how
many native MLP activation instances must remain for a language model to retain
an answer, and how does its behavior degrade as those activations are removed?

The current pilot uses `Qwen/Qwen2.5-0.5B` and 152 fixed-length, single-token
addition, subtraction, and multiplication prompts. It masks each gated-MLP
activation after the multiplicative nonlinearity and before the output
projection, indexed by layer, token position, and channel. Every evaluation is
a fresh forward pass with attention, normalization, and downstream MLPs
recomputed.

## Reproduce

```bash
uv sync --extra research --extra dev
uv run pytest -q
uv run python scripts/run_arithmetic_pilot.py \
  --examples 24 --family-eval 30 \
  --output artifacts/arithmetic_pilot
MPLCONFIGDIR=/tmp/neuron-circuits-mpl \
  uv run python scripts/analyze_arithmetic_pilot.py artifacts/arithmetic_pilot

# Optional large-corpus mean-ablation baseline
uv run python scripts/compute_corpus_mean.py \
  --windows 50000 --positions 11 --batch-size 64 \
  --output artifacts/arithmetic_pilot/wikitext103_mean.pt
uv run python scripts/evaluate_corpus_mean.py \
  --baseline artifacts/arithmetic_pilot/wikitext103_mean.pt \
  --root artifacts/arithmetic_pilot --examples 12
MPLCONFIGDIR=/tmp/neuron-circuits-mpl \
  uv run python scripts/analyze_arithmetic_pilot.py artifacts/arithmetic_pilot
```

The experiment runs four rankings (random, activation magnitude, write
magnitude, and gradient × activation), refines the 90%-margin transition,
freezes per-input masks onto changed problems, learns a shared task-family
ranking, reranks iteratively, performs a separately optimized necessity sweep,
and repeats the primary retention curve with a fixed mean baseline.

## Results

Open the single-file, phone-friendly
[`artifacts/arithmetic_pilot/report.html`](artifacts/arithmetic_pilot/report.html),
or read [`artifacts/arithmetic_pilot/summary.md`](artifacts/arithmetic_pilot/summary.md).
Raw evaluations are JSONL, metadata and aggregates are JSON, plots are PNG, and
successful masks are bit-packed in a compressed NPZ file.

The mean-ablation comparison includes both a task-conditioned mean over the 24
arithmetic extraction prompts and a general-corpus mean over 50,000 streamed
WikiText-103 windows (550,000 tokens). For the latter, units are ranked by
`|gradient × (activation − corpus mean)|`, matching the actual replacement
intervention.

The headline pilot result is a median 90%-margin circuit containing 23.7% of
the 1,284,096 layer–position–channel activation instances under gradient ×
activation ranking. Collapsing token positions, those circuits touch a median
83.8% of unique layer–channel neurons: compactness is primarily about *when* a
neuron is retained, not a tiny global neuron subset.

## Language pilot

The language extension tests indirect-object identification (IOI),
subject–verb agreement, and bracket closing on fixed-length, intact-correct
prompts. It compares zero replacement with position-specific means estimated
from 50,000 WikiText-103 windows, and records full-vocabulary KL together with
task-aware error categories.

```bash
for positions in 11 13 18; do
  uv run python scripts/compute_corpus_mean.py \
    --positions "$positions" --windows 50000 \
    --output "artifacts/language_pilot/wikitext103_mean_pos${positions}.pt"
done
uv run python scripts/run_language_pilot.py
MPLBACKEND=Agg uv run python scripts/analyze_language_pilot.py
```

The phone-friendly report is
[`artifacts/language_pilot/report.html`](artifacts/language_pilot/report.html),
with a compact written summary in
[`artifacts/language_pilot/summary.md`](artifacts/language_pilot/summary.md).

## Counterfactual neuron patching

To measure how many changes redirect an answer rather than merely preserve it,
the counterfactual experiment pairs minimally changed prompts and copies source
MLP activations into the base run. It tests both position-specific activation
instances and layer–channel neurons patched at every position. The primary
ranking is the signed base gradient of the answer contrast multiplied by the
actual source-minus-base activation change.

```bash
uv run python scripts/run_counterfactual_patching.py
MPLBACKEND=Agg uv run python scripts/analyze_counterfactual_patching.py
```

See the phone-friendly
[`artifacts/counterfactual_patching/report.html`](artifacts/counterfactual_patching/report.html)
or the concise
[`artifacts/counterfactual_patching/summary.md`](artifacts/counterfactual_patching/summary.md).

### Attention counterfactuals

The attention extension uses the same counterfactual pairs and separately
patches query heads, physical grouped-query key/value heads, normalized
attention rows, individual source-to-destination messages, and pre-output-
projection head vectors. It also evaluates joint query-plus-key patches.

```bash
uv run python scripts/run_attention_counterfactuals.py
MPLBACKEND=Agg uv run python scripts/analyze_attention_counterfactuals.py
```

Results are summarized in
[`artifacts/attention_counterfactuals/summary.md`](artifacts/attention_counterfactuals/summary.md).

### Combined MLP and attention view

The combined analysis puts scalar MLP patches beside attention routing and
content patches for all three task families. It also reports how often each
intervention succeeds, compares targeted patches with random controls, and
lists a representative MLP neuron and attention route for every task.

```bash
MPLBACKEND=Agg uv run python scripts/analyze_combined_counterfactuals.py
```

See
[`artifacts/combined_counterfactuals/summary.md`](artifacts/combined_counterfactuals/summary.md)
or the single-file
[`artifacts/combined_counterfactuals/report.html`](artifacts/combined_counterfactuals/report.html).

### Location-restricted counterfactuals

The nontriviality check repeats the scalar MLP counterfactual sweep after
excluding layer 23, layers 20–23, the final token position, or both the late
layers and final position. Excluded activations remain at their base-run values.

```bash
uv run python scripts/run_restricted_counterfactuals.py
MPLBACKEND=Agg uv run python scripts/analyze_restricted_counterfactuals.py
```

See
[`artifacts/restricted_counterfactuals/summary.md`](artifacts/restricted_counterfactuals/summary.md)
for the paired results and sparse-budget success rates.

The recurrent-unit example analysis ranks positive and negative prompt pairs by
the signed `gradient × donor-minus-base activation` score. It uses the strictest
restriction and a 100-activation sparse budget.

```bash
uv run python scripts/analyze_sparse_neuron_examples.py
```

See
[`artifacts/sparse_neuron_examples/summary.md`](artifacts/sparse_neuron_examples/summary.md),
with all five recurrent units per family in JSON and CSV alongside it.

### Arithmetic counterfactual examples

The arithmetic counterpart holds the operation and left operand fixed, changes
the right operand, and asks how many eligible MLP activation patches are needed
to make the donor answer the model's full-vocabulary top-1 prediction. It uses
the same strict restriction—no layers 20–23 and no final token position—and the
same 100-activation sparse threshold.

```bash
uv run python scripts/run_arithmetic_counterfactuals.py
uv run python scripts/analyze_arithmetic_neuron_examples.py
```

See
[`artifacts/arithmetic_neuron_examples/summary.md`](artifacts/arithmetic_neuron_examples/summary.md),
with the complete recurrent-unit examples in JSON and CSV alongside it.

### Natural neuron examples

The counterfactual example tables above rank causal-direction scores. To inspect
what the same recurrent neurons respond to naturally, the following scan runs
every intact IOI, agreement, bracket, and arithmetic prompt and ranks examples
by the signed raw post-gating MLP activation itself:

```bash
uv run python scripts/analyze_natural_neuron_examples.py
```

See
[`artifacts/natural_neuron_examples/summary.md`](artifacts/natural_neuron_examples/summary.md),
with every neuron's five most positive and five most negative natural examples
in JSON and CSV alongside it.

### Hierarchical algorithm circuits

The hierarchical experiment first patches complete attention and MLP outputs
to localize a conventional module-level circuit. It then restricts the neuron
search to the three strongest MLP layer-position modules and measures exact
single-neuron and cumulative counterfactual effects. The current tasks are
bracket closing, maximum, and minimum over four distinct digits. A separate
capability screen records that second-largest and second-argmax are not reliable
enough in the 0.5B model to support circuit interpretation.

```bash
uv run python scripts/screen_algorithmic_capabilities.py
uv run python scripts/run_hierarchical_circuit_analysis.py
uv run python scripts/analyze_hierarchical_circuits.py
```

See
[`artifacts/hierarchical_circuits/summary.md`](artifacts/hierarchical_circuits/summary.md)
and the complete per-pair interventions in `results.json` and `analysis.json`.

The four neurons highlighted by the concrete `8 -> 2` maximum counterfactual were also
searched over 10 million streamed FineWeb tokens. See
[`artifacts/maximum_natural_corpus/summary.md`](artifacts/maximum_natural_corpus/summary.md)
and `examples.json` for literal extrema and ten high/low natural contexts per neuron.

A separate final-MLP study compares each sampled neuron's natural activation conditions
with its decoder/unembedding vocabulary direction, using held-out next-token enrichment
and exact RMSNorm-aware ablations. See
[`artifacts/last_mlp_monosemanticity/summary.md`](artifacts/last_mlp_monosemanticity/summary.md)
and `results.json`.

### Bilinear memory and clean communication

The bilinear-memory experiment stores up to 16 random key→value mappings per shared
bilinear neuron, then tests lookup, value classification, and max/min on held-out key
pairs. Its analytic weight decomposition measures whether adding reusable computation
turns a merely decodable memory into a cleaner value interface. Run it with:

```bash
uv run python scripts/run_bilinear_memory_experiment.py
```

See [`artifacts/bilinear_memory/summary.md`](artifacts/bilinear_memory/summary.md), the
complete `results.json`, and the saved converged first-seed model in `full_model.pt`.

### Bilin18 layers 0–2 weight-first atlas

The 546M-parameter FineWeb tensor-transformer study audits all 13,824 native
bilinear MLP channels in layers 0–2. It combines direct weight contractions,
held-out activation selectivity over 10.24M positions per layer, exact named
residual-source products, and QK×OV attention paths. The evidence supports a
shift from direct lexical quadratic separators in layer 0 toward neurons that
operate on reusable upstream MLP states and local attention context in layers 1
and 2.

Read the [plan](LAYER_0_2_INTERPRETATION_PLAN.md) and
[results report](artifacts/bilin18_layers_0_2/REPORT.md). The interactive
[neuron viewer](artifacts/bilin18_layers_0_2/neuron_viewer/index.html) covers
all layers and channels; serve its directory over HTTP so it can load
`data.json`.

### Bilin18 causal edges and output-side layers

The follow-up validates QK×OV edge predictions with finite interventions through
the full 18-layer model, matched random and routing-only controls, a disjoint
transfer panel, and target-neuron mediation. See the
[causal report](artifacts/bilin18_qkov_causal_edges/REPORT.md).

Layers 17→16→15 were then audited from the unembedding backward using 2.56M
FineWeb positions per layer. The study also tests whether the 4,608-channel final
MLP can be replaced by a fixed smaller subset. Read the
[late-layer report](artifacts/bilin18_layers_15_17/REPORT.md), the
[compression report](artifacts/bilin18_final_mlp_compression/REPORT.md), and
open the [late-layer neuron viewer](artifacts/bilin18_layers_15_17/neuron_viewer/index.html)
through a local HTTP server.

The follow-up [learned final-MLP replacement](artifacts/bilin18_final_mlp_student/REPORT.md)
distills the 4,608-channel bilinear block into 8–256 newly optimized bilinear
channels while respecting factor symmetries. Across two disjoint panels, a
32-channel replacement uses 0.70% of the MLP parameters with a mean +0.073 CE
penalty, substantially outperforming a fixed 32-native-channel subset.

The expanded replacement comparison adds normalized-state, KL, hybrid, and
CE-only objectives; low-rank linear and full-affine controls; symmetric-square
bilinear students; and fixed WikiText-103 and TinyStories distribution shifts.
The best compact general-bilinear result is a 64-channel KL student
(+0.066/+0.072/+0.024 CE on held-out FineWeb/WikiText/TinyStories), while a
128-channel symmetric-square student is slightly better on average
(+0.068/+0.067/+0.022). Linear and affine controls are much worse, showing that
the compact replacement depends on quadratic structure rather than merely a
low-dimensional residual correction. The same report documents an important
red-team result: the square student's apparently cleaner individual neurons
often work in opposing pairs, so the pair—not either scalar channel alone—is
the faithful causal unit.

The [breadth-first whole-model MLP atlas](artifacts/bilin18_full_mlp_atlas/REPORT.md)
then scans all 82,944 MLP channels over 46.1 million layer-position
observations. It finds a U-shaped depth profile: lexical alignment is strongest
at layers 0–1, weakest through much of the middle, and rebounds at layers
16–17. Finite mean- and zero-replacement tests make layers 0, 1, 2, 3, and 17
the robust first-pass priority set, while also showing that layer 1's
mean-replacement score is partly an adversarial-baseline artifact. The atlas is
an allocation map, not a claim that low-ranked middle layers or individual
connections are unnecessary; joint redundancy and edge-level importance still
require finite validation.

The ensuing [three-iteration layer-1 deep dive](artifacts/bilin18_layer1_deep/REPORT.md)
tests that priority rather than trusting the atlas score. Sparse native-neuron
sufficiency is falsified: roughly half of the 4,608 channels are required for a
low-CE approximation, and two complementary large subsets can each preserve
most behavior. Mean replacement exaggerates the failure, but real product
vectors shuffled between examples or positions are also damaging. In contrast,
the input-conditioned QK×OV calculation successfully predicts causal attention
edges on held-out examples. A one-neuron intervention then fans out across
roughly 2,000 downstream channels, yet its source-specific two-dimensional
propagation span transfers substantially better than spans from other source
neurons. This motivates interpreting conditional edges and population
directions alongside individual neuron meanings.

The next [four-circuit and ordinary-subspace study](artifacts/bilin18_four_circuits/REPORT.md)
freezes boundary/reset, capitalization, number-format, and determiner neuron
groups before testing them. It collects 2,048 moderate-activation interventions
and four 102,400-position semantic panels. All four input interpretations reach
0.95–1.00 AUROC on held-out FineWeb and remain strong on WikiText-103 and
TinyStories. Cross-group ablations causally isolate boundary, capitalization,
and number processing, while the determiner group fails that stricter test.
Ordinary propagation is substantially more high-dimensional than extreme-tail
examples suggested. Only capitalization shows convincing leave-one-neuron-out
subspace transfer; learned PCA-like spans weakly replay output effects but are
not sufficient causal mediators under finite restoration.

The [weight-first circuit study](artifacts/bilin18_weight_only_circuits/REPORT.md)
then freezes six new layer-1 hypotheses using only isolated vocabulary items and
local model weights before exposing them to natural text. The strongest result
is a four-neuron `-ing` code: it transfers across FineWeb, WikiText-103, and
TinyStories, its joint ablation is 20.4× more specific than other group
ablations, and its weight-ranked layer-2 targets change 4.9× more than average.
Only 9.7% of its output effect is repaired by restoring the top 32 targets,
however, so it is a clean local predicate feeding a distributed computation.

```bash
uv run python scripts/discover_bilin18_weight_only_circuits.py
uv run python scripts/validate_bilin18_weight_only_circuits.py
MPLBACKEND=Agg uv run python scripts/report_bilin18_weight_only_circuits.py
```

The subsequent [layer-1→2 canonicalization study](artifacts/bilin18_canonicalization/REPORT.md)
compares five shorter descriptions of that frozen `-ing` circuit on disjoint
FineWeb, WikiText-103, and TinyStories sites. A one-dimensional post-attention
MLP-input change reconstructs 82% of the held-out dense product change, while
the causally complete object is the full post-layer-2 residual state. The study
separates activation compression from finite downstream recovery and documents
strong bf16 error amplification.

```bash
uv run python scripts/collect_bilin18_canonicalization.py
uv run python scripts/fit_bilin18_canonicalizations.py
uv run python scripts/validate_bilin18_canonicalizations.py
MPLBACKEND=Agg uv run python scripts/report_bilin18_canonicalizations.py
```

The [layer-1→2 activation/interference study](artifacts/bilin18_activation_interference/REPORT.md)
then tests the superposition hypothesis directly while intentionally stopping at
MLP2 products. Natural activity is heavy-tailed, with roughly 190–200 effective
write participants per token but thousands of different neurons reused across
tokens. Turning on 32 quiet channels at `2σ` creates only 6.3% algebraic
interaction in a float32 diagnostic, versus 181% apparent nonadditivity in
native bf16; the amplitude sweep identifies most of that gap as finite-precision
noise rather than bilinear feature collision.

```bash
uv run python scripts/run_bilin18_activation_interference.py
uv run python scripts/run_bilin18_activation_interference.py --float32
MPLBACKEND=Agg uv run python scripts/report_bilin18_activation_interference.py
```

The follow-up experiment uses self-bilinear writes in a constant-width residual stream,
`x_(l+1) = x_l + D((Lx_l) ⊙ (Rx_l)) + b`, stacked from one to three layers and followed
by a bias-free readout. It memorizes random binary facts, varies the number of distinct
compositional queries, evaluates unseen 2/4/8-fact parity queries, and measures shared
and input-specific neuron requirements at 80%, 90%, and 99% accuracy, margin fidelity,
and recovery of improvement over chance cross-entropy. CE circuits receive a separate,
numerically stable softmax-aware importance ranking. A matched 2×2 ablation independently
removes the identity residual and decoder biases.

```bash
uv run python scripts/run_self_bilinear_compositional_memory.py
```

See
[`artifacts/self_bilinear_memory/summary.md`](artifacts/self_bilinear_memory/summary.md)
and `results.json` for the computation-load, depth, residual/bias ablation,
monosemanticity, interference, and neuron-sufficiency measurements.

The maximum-load residual/bias comparison also has a 20-seed replication. It reports
sample variance, bootstrap confidence intervals, Wilson convergence intervals,
heavy-tailed held-out cross-entropy, and distributions of accuracy- versus CE-based
circuit sizes:

```bash
uv run python scripts/run_self_bilinear_seed_statistics.py
```

See
[`artifacts/self_bilinear_seed_statistics/summary.md`](artifacts/self_bilinear_seed_statistics/summary.md)
and its complete per-seed `results.json`.

A red-team audit separates the results that survive scrutiny from claims the current
design cannot establish, adds random-rotation and untrained-network cleanliness nulls,
and specifies a confirmatory experiment with explicit falsification criteria. See
[`artifacts/self_bilinear_redteam/summary.md`](artifacts/self_bilinear_redteam/summary.md).
The proposed 10-/100-class one-correct-label benchmark is specified separately in
[`artifacts/multiclass_clean_neurons/design.md`](artifacts/multiclass_clean_neurons/design.md).

That benchmark has now been run through replicated pilot, locked 100-class, and stricter
context-holdout rounds. The final result is weak raw-axis alignment but much stronger
learned distributed subspaces; class counterfactuals fail the preregistered threshold for
a practically substantial privileged-neuron basis when tested across unseen operations.
See the [locked confirmation](artifacts/multiclass_clean_neurons/locked_confirmation100/conclusion.md),
[context-holdout conclusion](artifacts/multiclass_clean_neurons/context_holdout100/conclusion.md),
and their preregistrations in `artifacts/multiclass_clean_neurons/`.

A subsequent 104-model scaling study varies classes, operation types, network depth,
and neuron width, then red-teams the result with a complete class-count × width
factorial. It introduces a rotation-normalized Clean Axis Index while retaining
absolute axis advantage and 80%/90% subspace-efficiency metrics. See the
[scaling summary](artifacts/clean_axis_scaling/summary.md),
[one-factor figure](artifacts/clean_axis_scaling/clean_axis_scaling.png), and
[factorial figure](artifacts/clean_axis_scaling/load_factorial.png).

## Interpretation boundary

The original sufficiency experiments keep attention intact; the newer
counterfactual experiments intervene on MLP activations and attention
components directly. All rankings are answer-informed heuristic searches, not
certified global minima or evidence that every selected unit has a simple
semantic label. Dense masked execution also does not imply a runtime or FLOP
reduction.

## Factual capacity and neuron organization

The amended synthetic factual pilot separates semantic traits from entity identity by
creating four independently identified replicas of every base-8 trait tuple.  It uses
stable, nested, exactly balanced random targets and compares native post-SwiGLU axes
with exact random-rotation nulls.

```bash
uv run pytest tests/test_factual_capacity.py -q
uv run python scripts/run_factual_pilot.py
uv run python scripts/audit_factual_pilot.py
MPLBACKEND=Agg uv run python scripts/analyze_factual_pilot.py
```

The original four-replica stress configuration is recorded in
`configs/factual_pilot.yaml`; the adaptive two-replica Stage 1 configuration that
reached the fidelity gates is in `configs/factual_pilot_r2.yaml`.  The run writes an
immutable compressed manifest, compact per-example predictions, checkpoints, training
curves, held-out explanation tables, exact sampled interventions, component ablations,
figures, and a gate-based report under `artifacts/factual_capacity_pilot/`.

The follow-up signed constraint-network audit uses those frozen checkpoints to test
whether layerwise margin-gradient overlap predicts real finite parameter-edit effects.
Its causal predictor validates, but the proposed load-induced strengthening of the web
is falsified: arbitrary high-fit models have broader yet substantially weaker blockwise
coupling and collateral than structured models. Run it with:

```bash
uv run python scripts/run_constraint_network_experiment.py
```

See the [locked protocol](artifacts/constraint_network_preregistration.md),
[primary conclusion](artifacts/constraint_network_pilot/conclusion.md), and the
larger-panel sensitivity result under `artifacts/constraint_network_sensitivity_n128/`.
The decisive rho 0 versus 1 comparison was subsequently repeated with two fresh
manifests and initializations; see the
[three-seed confirmation](artifacts/constraint_network_replication_summary/summary.md).

The subsequent matched mechanism-transition factorial uses literal nested fact
tables, answer-only loss, AdamW versus SGD, common fact/competitor panels, a
gradient sign null, and current-weight versus initialization-displacement KKT
fits. Doubling learned load does not reproducibly tighten the web; raw support
increases while supranull support falls. See the
[factorial report](artifacts/mechanism_transition/factorial/REPORT.md) and
[red-team log](artifacts/mechanism_transition/factorial/design_changes.md).

## Grammar regimes and reverse-engineering map

The current ranked research program is summarized in
[`HIGH_INFORMATION_EXPERIMENTS.md`](HIGH_INFORMATION_EXPERIMENTS.md). It separates
the already-completed phase diagram from the missing mechanism-transition,
frozen causal-transfer, identifiability, and blind-completeness tests.
The competing theories and their strongest falsifiers are tracked in
[`COMPUTATION_HYPOTHESIS_LEDGER.md`](COMPUTATION_HYPOTHESIS_LEDGER.md), with the
required safeguards for interpreting nulls in
[`NEGATIVE_RESULT_REDTEAM.md`](NEGATIVE_RESULT_REDTEAM.md).
The assay-qualification suite is locked before implementation in
[`artifacts/computation_assay_benchmark/preregistration.md`](artifacts/computation_assay_benchmark/preregistration.md).

The newest experiment trains one- to four-layer language models on a controlled
autoregressive grammar with nested arbitrary exceptions. It compares semantic
counterfactual transfer, native axes against 100 random rotations, residual
fact-specific activation state, and finite parameter-edit constraint networks.
The decisive depth-four conditions use three data/model seeds and both SwiGLU and
bilinear MLPs.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_grammar_regime_pilot.py \
  --output artifacts/grammar_regimes/easy_pilot
PYTHONPATH=src .venv/bin/python scripts/audit_grammar_axes.py \
  artifacts/grammar_regimes/easy_pilot --rotations 100
PYTHONPATH=src .venv/bin/python scripts/audit_grammar_semantic_transfer.py \
  artifacts/grammar_regimes/easy_pilot
PYTHONPATH=src .venv/bin/python scripts/audit_grammar_constraints.py \
  artifacts/grammar_regimes/easy_pilot
PYTHONPATH=src .venv/bin/python scripts/analyze_grammar_regimes.py
```

See the [regime-map synthesis](REVERSE_ENGINEERING_REGIME_MAP.md),
[aggregate results](artifacts/grammar_regimes/summary/summary.md), and
[four-panel figure](artifacts/grammar_regimes/summary/regime_summary.png).
For a plain-language reconstruction of the motivating question, KKT/silence-code
mechanism, experimental results, real-model regimes, and practical workflow, start
with the [easy interpretation guide](EASY_INTERPRETATION_GUIDE.md).

The follow-up interpolation adds 5% and 20% arbitrary-load conditions over three seeds
and both MLP types. See the
[interpolation summary](artifacts/grammar_regimes/interpolation_summary/summary.md),
[behavior figure](artifacts/grammar_regimes/interpolation_summary/behavior_interpolation.png),
[representation figure](artifacts/grammar_regimes/interpolation_summary/representation_interpolation.png),
and [worked neuron-patching example](artifacts/grammar_regimes/worked_examples/worked_example.md).

The fixed-task capacity sweep holds depth and 50% arbitrary load constant while
varying width over 64, 96, 192, and 384, with three seeds and both MLP types. Its
CMP-90 audit measures exact ranked-prefix sufficiency (S90), necessity (N90),
random-rotation sensitivity, same-rule reuse, and cancellation reversals.

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_grammar_causal_circuits.py \
  artifacts/grammar_regimes/easy_pilot --panel-per-group 12
MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python scripts/analyze_grammar_capacity.py
```

See the [metric preregistration and red-team log](artifacts/grammar_regimes/capacity_preregistration.md),
[capacity results](artifacts/grammar_regimes/capacity_summary/summary.md),
[behavior graph](artifacts/grammar_regimes/capacity_summary/behavior_by_width.png),
[causal profile graph](artifacts/grammar_regimes/capacity_summary/cmp90_heldout_unselected.png),
and [two worked circuits](artifacts/grammar_regimes/capacity_summary/worked_examples.md).

## Exact joint-neuron tests across 1–4 layers

The causal-synergy sweep directly tests whether a neuron's effect depends on another
neuron. For every selected pair it evaluates the full 2×2 intervention table and
computes `J = m11 - m10 - m01 + m00`. The fixed self-bilinear memory experiments use
the same lookup and parity tasks at every depth, three seeds per cell, and 24/24
models reached perfect task accuracy. The one-layer additive architecture gives the
expected near-zero interaction while deeper models develop non-additive composition.
Strong partner-dependent sign reversals remain rare, however, so the current evidence
fits stable discriminating “verbs” composed into distributed codes better than
widespread contextual multiplexing.

Run and inspect it with:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_self_bilinear_synergy_models.py
PYTHONPATH=src .venv/bin/python scripts/audit_causal_synergy.py
MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python scripts/analyze_causal_synergy.py
```

See the [plain results and red-team limits](artifacts/causal_synergy/summary.md),
[locked metric](artifacts/causal_synergy_preregistration.md), and
[worked factorial interventions](artifacts/causal_synergy/factorial_examples.png).

## Tensor-decomposition causal ontology

The layer-1 tensor study now separates global operator fidelity, natural-language
value fidelity, tangent/curvature fidelity, downstream causal fidelity, conditional
sparsity, and factor stability. A disjoint validation shows that an 80% language /
20% global-tensor blend improves behavior and two intervention metrics across
FineWeb, WikiText-103, and TinyStories. Attempts to recompress that blend expose an
important failure of ordinary MSE and PCA: more than 99.9% local agreement can still
lose downstream effects, and the leading 32 correction directions explain 60% of
variance while worsening causal fidelity.

Start with the [reverse-engineering ontology](TENSOR_REVERSE_ENGINEERING_ONTOLOGY.md),
then see the [tangent analysis](artifacts/bilin18_tensor_decomposition/TANGENT_ONTOLOGY_REPORT.md),
[blend validation](artifacts/bilin18_tensor_decomposition/VALUE_TANGENT_BLEND_VALIDATION.md),
[recompression audit](artifacts/bilin18_tensor_decomposition/BLEND_RECOMPRESSION_REPORT.md),
[factor rank curve](artifacts/bilin18_tensor_decomposition/DOWNSTREAM_RANK_SELECTION_REPORT.md),
[basis-invariant subspace test](artifacts/bilin18_tensor_decomposition/CORRECTION_SUBSPACE_REPORT.md),
[learned downstream quotient](artifacts/bilin18_tensor_decomposition/DOWNSTREAM_CORRECTION_QUOTIENT_REPORT.md),
and [intervention-specialist red-team](artifacts/bilin18_tensor_decomposition/CORRECTION_QUOTIENT_SPECIALIST_REPORT.md).

The newest weight-first analysis conditions the layer's exact quadratic form on
a proposed feature direction and covariance-cleans it. One or two interaction
modes preserve five token-form discriminators, and the leading direction matches
category across two independent SAE fits (`p = 1/120`). Crucially, the power-law
bulk itself is reproduced by sign-flipped nulls, so a scree plot or outlier alone
is not accepted as circuit evidence.

Read the [spectral-spike pilot](artifacts/bilin18_tensor_decomposition/SPECTRAL_SPIKE_PILOT.md),
[disjoint rank red-team](artifacts/bilin18_tensor_decomposition/SPECTRAL_SPIKE_REDTEAM.md),
and [independent-seed/OOD confirmation](artifacts/bilin18_tensor_decomposition/SPECTRAL_SPIKE_CONFIRMATION.md).

A prospective follow-up then selected three previously untested categories using
only cross-seed leading-mode recurrence. It is an important negative result:
the modes remained semantic predictors (median AUROC 0.830), but none passed its
finite causal gate and two had negative causal fidelity. See the
[prospective spectral atlas](artifacts/bilin18_tensor_decomposition/SPECTRAL_ATLAS.md).
The current rule is therefore strict: spectral magnitude, recurrence, and
semantic decoding generate candidates. The original downstream test is now
classified as scalar-coordinate reconstruction, not eigenmode causality: in
float32 its rank-4 fidelity rises to 0.975–0.986, revealing that much of the
bfloat16 result was edit quantization.

The corrected experiment intervenes on the actual post-RMSNorm residual
eigenspace and recomputes every native product channel. At a local 10%-of-input-
norm edit, known rank-4 modes change their selected feature 16.7–20.0× more than
matched low modes and cause 2.2–5.4× more downstream effect. SAE-feature-path
mediation fidelity is only about 0.41 (a normalized prediction score, not a
literal effect percentage), so these are genuine local semantic paths embedded
in a broader distributed transformation—not standalone circuits. See the
[precision red-team](artifacts/bilin18_tensor_decomposition/SPECTRAL_PRECISION_REDTEAM.md),
[true input intervention](artifacts/bilin18_tensor_decomposition/SPECTRAL_INPUT_INTERVENTIONS.md),
and [local norm-controlled red-team](artifacts/bilin18_tensor_decomposition/SPECTRAL_LOCAL_INPUT_REDTEAM.md).

The eight-site local result does not describe average category behavior. A
64-site-per-category, three-corpus replication finds much weaker mediation on
random positives than high-scoring positives. An exact SAE-gate follow-up shows
that inactive sites are near zero, but gate activity is not sufficient: the same
SAE feature can be reached through multiple input routes. Read the
[stratified replication](artifacts/bilin18_tensor_decomposition/SPECTRAL_STRATIFIED_REPLICATION.md),
[gate validation](artifacts/bilin18_tensor_decomposition/SPECTRAL_GATE_VALIDATION.md),
and [activation-matched semantic test](artifacts/bilin18_tensor_decomposition/SPECTRAL_MATCHED_SEMANTICS.md).

Decomposing the remaining write reveals a reusable middle-scale object. The
named SAE feature alone reaches 0.297 downstream fidelity; adaptive top-16
changed SAE features reach 0.640. A frozen category-specific core reaches 0.606
on held-out high-active sites, and a 32-feature core transfers at 0.514 to a
separate random-active semantic panel versus 0.062 on matched negatives. This
supports category-specific companion ensembles rather than isolated features or
the complete SAE code change. See the
[companion decomposition](artifacts/bilin18_tensor_decomposition/SPECTRAL_COMPANION_PATHS.md),
[frozen-core validation](artifacts/bilin18_tensor_decomposition/SPECTRAL_FIXED_COMPANIONS.md),
and [independent transfer test](artifacts/bilin18_tensor_decomposition/SPECTRAL_COMPANION_TRANSFER.md).

An independent seed-102 discovery confirms that the canonical object is a
decoded-write subspace rather than a list of SAE feature IDs. Same-category
16-feature core subspaces match 5/5 across seeds (`p = 1/120`); removing the
named feature still gives 5/5. A frozen seed-102 program reaches 0.540 downstream
fidelity, while oracle projection into the corresponding seed-101 subspace
reaches 0.636 versus 0.026 for another category. See the
[cross-seed canonicalization](artifacts/bilin18_tensor_decomposition/SPECTRAL_CROSS_SEED_COMPANIONS.md).

A donor-transfer sequence now supplies a stricter causal boundary. Copying only
four category spectral coordinates from high-active donors into inactive
recipients transfers 48% of the donor feature activation and turns its sparse
gate on in 95% of cases; matched low and wrong-category modes remain near zero.
The feature gate is compact, but downstream execution is conditional: native
frozen-core fidelity is 0.281, while retaining the encoder-selected support and
refitting its coefficients reaches 0.784. On gate-selected OOD high-active
examples, frozen FineWeb cores reach 0.457 on WikiText and 0.413 on TinyStories
and beat activity/write-norm-matched changed-feature controls in all five
categories. Random-active OOD sites are much weaker (0.171/0.127), falsifying a
single fixed-program account for every use of the feature. Read the
[donor-transfer test](artifacts/bilin18_tensor_decomposition/SPECTRAL_DONOR_TRANSFER.md),
[dose/decomposition red-team](artifacts/bilin18_tensor_decomposition/SPECTRAL_DONOR_DECOMPOSITION.md),
[decoder-refit dissection](artifacts/bilin18_tensor_decomposition/SPECTRAL_DECODER_REFIT.md),
[OOD frozen-core validation](artifacts/bilin18_tensor_decomposition/SPECTRAL_OOD_CORES.md),
and [hard matched controls](artifacts/bilin18_tensor_decomposition/SPECTRAL_MATCHED_CORE_CONTROLS.md).

The output-facing extension reveals a different regime. Five exact layer-17
unembedding-conditioned quadratic forms have compact semantic projections, but
their leading eigenspaces are not class-specific under hard cross-category
interventions. Instead, the forms are almost linearly independent while sharing
input geometry: a common 4D quadratic representation retains 0.679–0.739 median
AUROC across FineWeb, WikiText, and TinyStories. The forms do not commute, and
deleting cross-coordinate interactions in the shared basis drops 32D AUROC by
0.12–0.20. This is a concrete compact multiplexed code—shared scalar state,
different pairwise interaction matrices—rather than one neuron or eigenspace per
output concept. Follow the full red-team chain in the
[output spectral pilot](artifacts/bilin18_tensor_decomposition/LAYER17_OUTPUT_SPECTRA.md),
[cross-category falsification](artifacts/bilin18_tensor_decomposition/LAYER17_CROSS_CATEGORY.md),
[rank sweep](artifacts/bilin18_tensor_decomposition/LAYER17_RANK_SPECIFICITY.md),
[direct-versus-final localization](artifacts/bilin18_tensor_decomposition/LAYER17_DIRECT_VS_FINAL.md),
[joint-form analysis](artifacts/bilin18_tensor_decomposition/LAYER17_JOINT_FORMS.md),
and [diagonal-only red-team](artifacts/bilin18_tensor_decomposition/LAYER17_DIAGONAL_REDTEAM.md).

The next generalization round freezes that shared basis and separates semantic
prediction from numerical reconstruction. At 32 dimensions, the original five
scalar form scores have language-state correlation 0.96–0.97 and centered
`R² ≈ 0.91`; nine held-out concepts have median correlation 0.66–0.87 and retain
oriented AUROC in 88–100% of eligible cells despite only 5% median global form
energy. A weight-only approximate joint diagonalizer then recovers stable axes
across six starts. It makes layer-0 embedding-facing lexical computations almost
fully neuron-aligned (0.987–0.988 score correlation with unchanged AUROC), but
layer-17 diagonal semantics still fail for OOD and held-out concepts. The full
query-specific interaction matrix is therefore the more portable late-layer
object. See the
[held-out concept test](artifacts/bilin18_tensor_decomposition/LAYER17_HELDOUT_CONCEPTS.md),
[score-fidelity red-team](artifacts/bilin18_tensor_decomposition/LAYER17_SCORE_FIDELITY.md),
[layer-0 readout comparison](artifacts/bilin18_tensor_decomposition/LAYER0_JOINT_FORMS.md),
[joint diagonalization](artifacts/bilin18_tensor_decomposition/JOINT_DIAGONALIZATION.md),
and [multi-start stability test](artifacts/bilin18_tensor_decomposition/JOINT_DIAGONALIZATION_STABILITY.md).

Large-corpus activation and causal tests refine that conclusion. Fourteen token
predicates give concise descriptions for many stable axes, including strong
layer-0 morphology directions, but finite interventions show two regimes. A
dominant axis behaves approximately like a standalone quadratic operation;
ordinary weak coordinates act through the full interaction matrix. Random
sketches of the complete residual write then expose a major depth difference:
layer-0 contextual roles require about 49 directions for 90% energy in a 96D
sketch, while layer 17 requires about four. This gap transfers to WikiText and
TinyStories. Random rotations of the selected subspace can lower within-axis
rank, but only by duplicating roles; the recovered joint basis retains roughly
twice their effective role diversity. The current invariant is therefore the
selected subspace and its query-conditioned interaction forms, with axes as a
useful simplicity/diversity compromise rather than unique model neurons. See the
[axis atlas](artifacts/bilin18_tensor_decomposition/JOINT_AXIS_ATLAS.md),
[finite interventions](artifacts/bilin18_tensor_decomposition/JOINT_AXIS_INTERVENTIONS.md),
[semantic contextual roles](artifacts/bilin18_tensor_decomposition/JOINT_AXIS_CONTEXTUAL_ROLES.md),
[residual-sketch red-team](artifacts/bilin18_tensor_decomposition/JOINT_AXIS_RESIDUAL_ROLES.md),
[basis controls](artifacts/bilin18_tensor_decomposition/JOINT_AXIS_BASIS_CONTROLS.md),
and [role-diversity check](artifacts/bilin18_tensor_decomposition/JOINT_AXIS_BASIS_DIVERSITY.md).

Downstream propagation rejects a second shortcut: clean early input predicates
do not acquire fixed end-to-end meanings. Across twelve selected layer-0 axes,
median directional rank90 rises from 16.5 in the immediate MLP write to 25 in
the final normalized residual state. The same final rank is already present at
10% coordinate removal, 10%-to-100% effect directions align at cosine 0.939,
and median KL scales with exponent 1.90. This supports a context-dependent
downstream Jacobian rather than a large-ablation artifact. See the
[downstream role test](artifacts/bilin18_tensor_decomposition/LAYER0_DOWNSTREAM_ROLES.md)
and [scale red-team](artifacts/bilin18_tensor_decomposition/LAYER0_DOWNSTREAM_SCALE_REDTEAM.md).

The layer-17 query-conditioned extension now gives an executable hierarchy.
The weights contract exactly into a 1,152→32×32 map from arbitrary output
readouts to compact interaction matrices. Its leading four spectral operations
have 0.944–0.957 natural-score correlation across three corpora, carry 22–26×
the middle-spectrum variance, have 92–97% of input-form energy in four modes,
and pass norm-matched causal specificity tests. The power-law bulk itself is not
semantic evidence: sign and pairing nulls retain it, and broad token-predicate
AUROC also survives sign nulls. See the
[query interpreter](artifacts/bilin18_tensor_decomposition/QUERY_CONDITIONED_INTERPRETER.md),
[spectral nulls](artifacts/bilin18_tensor_decomposition/QUERY_MAP_SPECTRAL_REDTEAM.md),
[natural-state component validation](artifacts/bilin18_tensor_decomposition/QUERY_MAP_COMPONENT_VALIDATION.md),
and [finite top-component circuits](artifacts/bilin18_tensor_decomposition/QUERY_MAP_OUTLIER_CIRCUITS.md).

The same audit across depth rejects a universal “few clean spectral circuits per
layer” story. Centered natural MLP writes have effective rank about 19–22 at
layer 0, 140–226 at layer 8, and only 2.7–3.5 at layer 17 across FineWeb,
WikiText, and TinyStories. The layer-17 top-four output subspace transfers almost
perfectly OOD; the layer-8 subspace is much more corpus-dependent. Full-width
weight spectra have power-law-like bulks at all three layers and under sign and
pairing nulls, but strict positive spikes occur cleanly only at layer 17. Those
four spikes beat every tested coordination null, overlap the natural-write
subspace at canonical cosine 0.930, and pass large finite causal-specificity
tests, while failing the stronger claim that all four are simple four-mode
semantic circuits. See the
[standardized depth audit](artifacts/bilin18_tensor_decomposition/DEPTH_QUERY_MAP_REPLICATION.md),
[basis/scale red-team](artifacts/bilin18_tensor_decomposition/DEPTH_SPECTRAL_REDTEAM.md),
[centered natural spectrum](artifacts/bilin18_tensor_decomposition/DEPTH_NATURAL_SPECTRUM.md),
[OOD regime replication](artifacts/bilin18_tensor_decomposition/DEPTH_NATURAL_SPECTRUM_OOD.md),
[full-width power-law nulls](artifacts/bilin18_tensor_decomposition/DEPTH_POWERLAW_NULLS.md),
and [outlier validation](artifacts/bilin18_tensor_decomposition/FULLWIDTH_OUTLIER_VALIDATION.md).

Query conditioning sharpens the regime map. In a common 512D state basis,
leading natural output queries need 64 eigenmodes at layer 0, 256–512 at layer
8, and only 4–16 at layer 17 to predict held-out scores at correlation 0.90 and
centered R² 0.80. Random layer-17 readouts are also compact, so late low rank is
partly a generic natural-write bottleneck rather than semantic evidence.
Layer-8 computation is multiplexed under one definition but dense under
another: an oracle per-token top 16 predicts signed scores at correlation
0.925–0.931, versus 0.604–0.649 for fixed modes, yet about 193 modes are needed
for 90% of absolute contribution mass and the adaptive slots range over roughly
400 mode identities. See the
[query-wise weight audit](artifacts/bilin18_tensor_decomposition/DEPTH_QUERYWISE_COMPACTNESS.md),
[natural-score truncation](artifacts/bilin18_tensor_decomposition/DEPTH_QUERYWISE_TRUNCATION.md),
and [per-token multiplexing test](artifacts/bilin18_tensor_decomposition/QUERY_MODE_SPARSITY.md).

Oracle sparsity is not yet an explanation of the middle layer. Three frozen
routers all transfer OOD but leave a large gap to oracle top-16 correlation
0.925–0.931: a 64-cluster router reaches 0.686–0.727, a 128D weight-derived
quadratic router reaches 0.822–0.847 with exact selected modes, and a
preregistered 128D nearest-context router reaches 0.749–0.761. The quadratic
router is strongest and improves monotonically with state dimension, suggesting
continuous high-dimensional route information; cluster saturation and failed
nearest-neighbor retrieval argue against a small categorical router. See the
[cluster router](artifacts/bilin18_tensor_decomposition/LAYER8_MODE_ROUTER.md),
[quadratic router](artifacts/bilin18_tensor_decomposition/LAYER8_QUADRATIC_ROUTER.md),
and [nearest-context red-team](artifacts/bilin18_tensor_decomposition/LAYER8_KNN_ROUTER.md).

A 2.14M-parameter supervised 128D router also fails, reaching only
0.753–0.775 and underperforming the weight-derived quadratic rule. Extending
that strongest rule localizes the address complexity: 192 of 512 state
dimensions are first to reach correlation 0.85 on every corpus, 256 are first
within 0.05 of oracle with 62.5% support recall, and 384 close the gap to 0.02.
The resulting interpretation is **sparse payload, dense address**: 16 modes can
predict the signed score once chosen, but choosing them reliably requires about
half of the tested natural-state quotient. See the
[learned-router failure](artifacts/bilin18_tensor_decomposition/LAYER8_LEARNED_ROUTER.md)
and [router dimension sweep](artifacts/bilin18_tensor_decomposition/LAYER8_ROUTER_DIMENSION_SWEEP.md).

A natural-state basis changes the compression picture. State-SVD-32 beats the
original semantic basis on whole-write reconstruction and the strongest
spectral operations; their 64D union reaches write R² 0.961–0.964. But it is not
a high-fidelity behavioral replacement: KL remains 0.040–0.046. KL below 0.01
appears around 768 dimensions, and near-exact behavior needs about 1,024 of
1,152 dimensions. Read the
[basis tradeoff](artifacts/bilin18_tensor_decomposition/INPUT_BASIS_TRADEOFFS.md),
[64D behavioral test](artifacts/bilin18_tensor_decomposition/FINAL_MLP_BASIS_REPLACEMENT.md),
[small dimension sweep](artifacts/bilin18_tensor_decomposition/FINAL_MLP_BASIS_DIMENSION_SWEEP.md),
and [large dimension sweep](artifacts/bilin18_tensor_decomposition/FINAL_MLP_LARGE_DIMENSION_SWEEP.md).

Finite causal prediction needs an extra context term. The exact effect of an
in-subspace edit is `2 δᵀMx + δᵀMδ`; a 32D score interpreter omits cross terms
with outside state. A 64D hybrid context quotient repairs the worst top
component from correlation 0.290 to 0.991 and generalizes across sixteen
components, random/high sites, and all three corpora. See the
[tangent derivation](artifacts/bilin18_tensor_decomposition/QUERY_MAP_TANGENT_INTERPRETER.md)
and [top-16 OOD replication](artifacts/bilin18_tensor_decomposition/QUERY_MAP_TANGENT_GENERALIZATION.md).

The quotient is also executable without evaluating native channels. Exact
Hybrid-64 contraction matches the projected MLP to `7.4e-7`. Global tensor SVD
is behaviorally inefficient, while natural-write PCA cuts its additional KL by
70–95% at matched rank and compute. The resulting frontier is real but not
near-exact: a 64×128 program is about 21× cheaper by counted dense multiplies at
KL 0.041–0.047; a 128×64 program is about 12× cheaper at KL 0.034–0.041; and a
256×128 program reaches KL 0.027–0.032 with only about 1.8× reduction. These are
algorithmic counts, not wall-clock benchmarks. See the
[global factorization](artifacts/bilin18_tensor_decomposition/HYBRID64_EFFICIENT_FACTORIZATION.md),
[natural-write factorization](artifacts/bilin18_tensor_decomposition/HYBRID64_NATURAL_FACTORIZATION.md),
and [compute–fidelity frontier](artifacts/bilin18_tensor_decomposition/EFFICIENT_FACTORIZATION_PARETO.md).

The newest backward-path audit clarifies why a later symmetric PCA quotient
failed despite having a compact internal tensor spectrum. The two inputs to a
bilinear derivative are different objects on natural execution: a context
state and a propagated tangent. Separate rank-64 context, tangent, and output
bases yield an explicit tensor that matches projected native execution to
`6.4e-7` and retains 0.89–0.95 of individual score effects on held-out
FineWeb, WikiText, and TinyStories. The same learned bases assigned to the
wrong roles fail. See the
[tangent-bundle quotient](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_TANGENT_BUNDLE_QUOTIENT.md)
and [split red team](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_TANGENT_BUNDLE_SPLIT_REDTEAM.md).

The tensor has 90%-energy ranks 15–17 per role and Tucker-16 retains at least
0.85 of the individual score paths, but its coordinates are not uniquely
identified: output modes transfer more stably than context/tangent modes, and
behavior-optimized cores vary substantially across objectives and seeds while
behaving similarly. The top1-versus-runner-up margin also remains materially
harder than either score. The resulting object is a compact executable
equivalence class, not a unique monosemantic rotation. See the
[core-stability audit](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_TANGENT_BUNDLE_CORE_STABILITY.md)
and [behavior-core red team](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_TANGENT_BUNDLE_BEHAVIOR_CORE.md).

The output and context sides now have separate semantic audits. The output
subspace is strongly lexical and answer-facing, but native axes do not beat
rotations of the same subspace. Raw unembedding roles become much closer to
exact natural effects after applying the local final-RMSNorm metric. On the
input side, contracting sixteen context coordinates produces a local 16×16
operator with median rank-90% of only three; four context modes retain about
0.97 operator fidelity and eight retain about 0.993. Again, individual axes do
not beat matched rotations. See the
[output semantic atlas](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_TANGENT_OUTPUT_SEMANTICS.md),
[output-role metric red team](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_OUTPUT_ROLE_METRIC_REDTEAM.md),
and [context-operator atlas](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_CONTEXT_OPERATOR_ATLAS.md).

Routing into that operator family is partly recoverable from observable lexical
state. Exact-token lookup reaches 0.81–0.89 operator fidelity and beats
class/frequency-matched permutations; a linear native-embedding router reaches
0.85–0.90. On token identities absent from discovery it still reaches
0.81–0.83 and beats matched response permutations. However, the fitted
embedding-input span has mean canonical cosine only 0.26 with the tensor
context span. The best current ontology is therefore a composition of compact
interfaces—lexical geometry, a low-rank router, a joint context register, and a
low-rank local operator—not one global canonical feature basis. See the
[router red team](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_CONTEXT_ROUTER_REDTEAM.md)
and [unseen-identity red team](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_EMBEDDING_ROUTER_REDTEAM.md).

An architectural source decomposition then corrects the causal interpretation
of that predictor. Although Layer 17 explicitly mixes the carried state and
original embedding with coefficients 1.02 and 8.0, direct final embedding
variation has essentially zero operator effect. Carried-state variation alone
retains 0.988–0.995 fidelity; mean-replacing it drops fidelity to 0.54–0.70,
and cyclic substitutions drop it to 0.38–0.58. Replacing or swapping the final
embedding term while exactly recomputing attention leaves more than 0.99999
fidelity. Embedding geometry predicts the compiled lexical state; the carried
residual is the causal router at this boundary. See the
[source decomposition](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_CONTEXT_SOURCE_DECOMPOSITION.md)
and [attention-mediated red team](artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_CONTEXT_SOURCE_MEDIATION_REDTEAM.md).

## Bilin18 native-MLP discriminator atlas

The first real-model pass audits all 4,608 native product channels in layer 0 of
the local 545.9M-parameter FineWeb tensor transformer. It separately records each
channel's Left factor, Right factor, signed product tails, coarse token-class
partition, held-out lift, and availability through the Down projection.

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_bilin18_mlp_discriminators.py
```

See the [10x layer-0 report](artifacts/bilin18_mlp_discriminators/layer_0_10x/README.md)
and [complete 10x per-channel atlas](artifacts/bilin18_mlp_discriminators/layer_0_10x/results.json).
This run uses 204,800 discovery and 102,400 disjoint held-out FineWeb tokens;
it supersedes the original 20,480/10,240-token pilot.

The [layer-0 QK×OV path report](artifacts/bilin18_mlp_discriminators/layer_0_qkov/README.md)
then traces six representative neurons through attention on the 102,400-token
held-out panel. It exactly decomposes each factor into direct-token and signed
head/source paths and includes finite removal of the complete attention-0 write
at each selected site.
The report also records the checkpoint's learned per-block embedding
reinjection and the exact embedding-only / context-only / cross-term
decomposition available for each bilinear channel. Layer 0 is unchanged after
RMS normalization; later-layer audits must explicitly retain these sources.

## Sparse-code test for the layer-1→2 path

The sparse-code pilot compares native product neurons with a Top-K SAE, then tests
PCA, native-coordinate sparsity, additive transcoders, and an interaction-aware
bilinear transcoder on matched finite interventions. The current result is a useful
separation: with 131,072 training positions, an SAE compresses the downstream write
better than matched native neurons, but its individual features remain non-identifiable
across seeds and weaker on strong morphology controls. A causal-delta SAE modestly
beats PCA and native residual coordinates, but no learned K=32 transcoder generalizes.
The exact weight-derived bilinear equation does generalize with >99.9% median
fidelity, locating the missing causal object in a context-conditioned operator rather
than a fixed sparse output dictionary.

```bash
uv run python scripts/collect_bilin18_sparse_code_data.py
uv run python scripts/train_bilin18_sparse_codes.py
MPLBACKEND=Agg uv run python scripts/report_bilin18_sparse_codes.py
```

See the [plain-language report](artifacts/bilin18_sparse_codes/REPORT.md),
[summary figure](artifacts/bilin18_sparse_codes/sparse_code_summary.png), and
[machine-readable results](artifacts/bilin18_sparse_codes/results.json).

## Bilinear SAE and sparse bilinear replacement

The matched follow-up uses bilinear features `(a·x)(b·x)` rather than applying only
a linear SAE to the native products. A random-initialized bilinear SAE recovers known
morphology well but remains non-identifiable across seeds. The strongest practical
result is a teacher-initialized layer-1 replacement with 2,048 candidate bilinear
features and 128 active per token: it uses 44.5% of the original MLP parameters,
explains 77.5% of held-out write energy, and raises FineWeb cross-entropy by 0.045
nats. The same architecture trained from scratch fails under the matched recipe,
so this is currently a successful compression/refinement result rather than an
independent rediscovery result.

```bash
uv run python scripts/collect_bilin18_large_layer1_inputs.py
uv run python scripts/train_bilin18_bilinear_codes.py
uv run python scripts/train_bilin18_bilinear_scratch_control.py
uv run python scripts/evaluate_bilin18_bilinear_codes.py
MPLBACKEND=Agg uv run python scripts/report_bilin18_bilinear_codes.py
```

See the [bilinear-code report](artifacts/bilin18_bilinear_codes/REPORT.md),
[four-panel figure](artifacts/bilin18_bilinear_codes/bilinear_code_summary.png), and
[full-model evaluation](artifacts/bilin18_bilinear_codes/evaluation_results.json).

## Direct CP and LL1 tensor decomposition

The explicit weight-tensor follow-up decomposes layer 1 as
`Σ d_k ⊗ sym(a_k ⊗ b_k)` without materializing its 1,152³ tensor. It compares CP
and LL1 block terms using held-out isotropic bilinear probes, then performs two
iterations: natural-activation refinement and Top-K sparse routing with a mixed
tensor/behavior objective. CP2048 directly preserves 54% of tensor-probe energy.
Activation-refined CP2048 preserves 87% of natural write energy with a 0.014-nat
FineWeb cross-entropy increase. The best sparse compromise uses CP2048/K128 and a
mixed objective: 77% natural-write fidelity, 19% tensor fidelity, and 0.041-nat
FineWeb cross-entropy increase. LL1 is decisively worse at the tested budgets because
the layer's output mode and within-block input matrices are both broad.

```bash
uv run python scripts/run_bilin18_tensor_decomposition.py
uv run python scripts/refine_bilin18_tensor_decomposition.py
uv run python scripts/route_bilin18_tensor_decomposition.py
uv run python scripts/evaluate_bilin18_tensor_decomposition.py
MPLBACKEND=Agg uv run python scripts/report_bilin18_tensor_decomposition.py
```

See the [tensor-decomposition report](artifacts/bilin18_tensor_decomposition/REPORT.md),
[summary figure](artifacts/bilin18_tensor_decomposition/tensor_decomposition_summary.png),
and [full-model results](artifacts/bilin18_tensor_decomposition/full_model_results.json).

The later depth audit also tests conditional spectral routing in the broad
middle layer. A hierarchical coarse-to-fine router saves candidate-projection
work, but finite interventions show that matching a layer-8 output query is not
enough to guarantee the same final-logit effect—especially on WikiText. See the
[hierarchical router report](artifacts/bilin18_tensor_decomposition/LAYER8_HIERARCHICAL_ROUTER.md)
and [finite causal validation](artifacts/bilin18_tensor_decomposition/LAYER8_ROUTER_CAUSAL_VALIDATION.md).

The follow-up path split clamps either the next-layer residual input or
attention write and recomputes the Layer-9 bilinear MLP. The direct residual/MLP
path explains most of the routed edit; attention is a small, nearly additive
correction, and the hierarchy's OOD mismatch is concentrated in the direct
path. This remains true from 1% through 20% input-state edits. See the
[path decomposition](artifacts/bilin18_tensor_decomposition/LAYER8_PATH_DECOMPOSITION.md)
and [scale red-team](artifacts/bilin18_tensor_decomposition/LAYER8_PATH_SCALE_REDTEAM.md).

A downstream-Fisher follow-up separates three notions that a scalar importance
ranking would conflate. Local tensor modes best implement the chosen Layer-8
query; Fisher-diagonal modes have much larger individual final-distribution
effects; but the latter cancel when combined. Exact joint interventions retain
only 20–23% of their summed individual Fisher energy, and the near-linear signed
sum predicts that cancellation. The next executable object is therefore the
full signed mode-interaction Gram matrix, not a top-k list of diagonal scores.
See the [stratified Fisher comparison](artifacts/bilin18_tensor_decomposition/LAYER8_STRATIFIED_FISHER_ORACLE.md)
and [joint cancellation audit](artifacts/bilin18_tensor_decomposition/LAYER8_FISHER_CANCELLATION_REDTEAM.md).

Using the off-diagonal Gram structure rather than its diagonal yields a much
stronger conditional causal program: a greedy 16-mode set produces 3–7× the
local-query route's exact KL across 1%–20% edits. Independent Fisher sketches
often choose different native modes—especially on WikiText—but produce nearly
the same downstream effect. This favors an effect-space equivalence class over
a unique neuron/mode list. It is still a per-example gradient oracle, not a
learned router. See the [joint program](artifacts/bilin18_tensor_decomposition/LAYER8_JOINT_FISHER_PROGRAM.md)
and [finite-scale red-team](artifacts/bilin18_tensor_decomposition/LAYER8_JOINT_FISHER_SCALE_REDTEAM.md).

Frozen global and nearest-neighbor suffix metrics fail to recover that
conditional program, especially OOD. The successful replacement is
architecture-native: analytically propagate Layer-8 mode tangents through the
learned residual, RMSNorm, bilinear MLP, squared-bilinear attention, value-bus,
and unembedding equations. This matches the per-example causal program at
0.983–0.994 effect fidelity and predicts 5%-norm finite edits at 0.984–0.986.
For the chosen eigenmode-removal perturbations, the resulting 512-mode Fisher
Gram is compact: four directions nearly recover the finite program and sixteen
are effectively exact. See the
[full tensor Jacobian](artifacts/bilin18_tensor_decomposition/LAYER8_FULL_TENSOR_JACOBIAN.md)
and [Gram rank sweep](artifacts/bilin18_tensor_decomposition/LAYER8_FULL_TENSOR_GRAM_RANK.md).

A subsequent red-team changes the scope of that claim. Random readouts at
random sites are equally compact, because each readout supplies a complete
rotated mode basis. Under the basis-invariant unit-input definition, rank 4
captures only 46–51% and rank 16 only 69–75% of causal Fisher energy; roughly
rank 64 is needed for 88–92%. The four leading input, logit, and Fisher axes do
not recur globally across contexts. They are nevertheless excellent *local*
causal directions: tangent fidelity remains 0.995–0.997 at 5%-norm edits and
their KL is about 48–50× matched random directions. Read the
[random-readout red-team](artifacts/bilin18_tensor_decomposition/LAYER8_GRAM_RANDOM_REDTEAM.md),
[basis-invariant bottleneck audit](artifacts/bilin18_tensor_decomposition/LAYER8_BEHAVIORAL_BOTTLENECK.md),
and [finite local-axis validation](artifacts/bilin18_tensor_decomposition/LAYER8_LOCAL_AXIS_VALIDATION.md).
Even conditioning on the exact native prediction (` the`, `.`, `,`, or newline)
only stabilizes the Fisher-weighted output axes; unweighted logit and Layer-8
input directions remain essentially unrecurrent. See the
[exact-prediction recurrence audit](artifacts/bilin18_tensor_decomposition/LAYER8_EXACT_PREDICTION_RECURRENCE.md).

Named natural counterfactuals sharpen the middle-layer picture. Nineteen
one-token substitutions across five semantic families do not produce recurring
global state or sensitivity directions, even after fixing position/distance and
selecting high-effect examples. Yet each context's rank-4 causal chart is
0.985–0.997 stable under its nearby edit. The changed behavior crosses Layer 8
mainly through the full residual sequence (0.934–0.956 fidelity), not the final
MLP write (0.107–0.134). The positionwise carrier is nearly additive: oracle
top-four positions reach 0.860–0.907 fidelity, top eight 0.955–0.972, and the
sum of individual effects reaches 0.996. See the
[semantic counterfactual panel](artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_COUNTERFACTUALS.md),
[fixed-position/high-effect red-team](artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_COUNTERFACTUALS_FIXED.md),
[boundary-carrier decomposition](artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_PATH_DECOMPOSITION.md),
and [position localization](artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_POSITION_PATHS.md).

Exact attention-tensor tracing resolves that positional broadcast into
`(QK1)(QK2)V` edges. The eight locally strongest edges form a real but partial
layer-4 route (0.110/0.139 downstream target projection on FineWeb/TinyStories),
and the top 64 reach 0.294/0.241. Local rank is not a calibrated downstream
score, however, and whole-head/direct-source explanations fail stronger causal
gates. A 64-edge address skeleton learned only on disjoint FineWeb contexts
retains 0.183 in-domain and 0.108 OOD—evidence for a reusable broadcast layout
with context-dependent edge weights, not a fixed semantic circuit. See the
[exact attention atlas](artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_ATLAS.md),
[finite edge patches](artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_CAUSAL.md),
[rank-resolved red-team](artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_RANK_REDTEAM.md),
and [frozen-route transfer](artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_FROZEN_ROUTES.md).

Factorial path interventions show what that skeleton carries. Exact QK1/QK2/V
Shapley recomposition attributes 87% of the layer-4 frozen-route effect to value
content, with nearly identical factor mixes in-domain and OOD. Splitting value
again into the current layer and inherited layer-0 bus falsifies a tempting
weight-sign interpretation: although layer 4 uses
`5.1875 V_current - 4.1875 V_bus`, the two paths have the same signed downstream
effect rather than cancelling. Current value dominates, the bus is a smaller
cooperative path, and QK changes are secondary. See the
[three-factor causal split](artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_FACTOR_CAUSALITY.md)
and [current-value/bus red-team](artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_VALUE_BUS_CAUSALITY.md).

The current-value content itself is reproducible: identical lexical edits have
cosine 0.55–0.58 before routing and 0.43–0.44 after it, while different edits in
the same semantic family remain more aligned than controls. The frozen relay
preserves rather than creates this structure. A further 32-world red-team shows
that the large direct layer-4 embedding coefficient is misleading: direct
embedding injection contributes below 1% of downstream route attribution,
whereas the incoming residual from layers 0–3 contributes 65–69%. See the
[value-code recurrence report](artifacts/bilin18_tensor_decomposition/LAYER4_CURRENT_VALUE_CODE_RECURRENCE.md)
and [embedding/residual causal split](artifacts/bilin18_tensor_decomposition/LAYER4_EMBEDDING_RESIDUAL_CAUSALITY.md).

An exact upstream telescope further localizes that residual: MLP3 supplies
about 65% of the routed current-value delta, MLP2 about 20%, and MLP1 about 7%;
four of fourteen terms carry 96% of local attribution. A matched cross-family
target derangement then sharply limits the interpretation. The null has 0.999
component-profile cosine and the same concentration and downstream strength.
Thus the MLP3→MLP2 hierarchy is a generic token-edit transport program, while
semantic evidence lives in the directions transported through it. See the
[exact residual telescope](artifacts/bilin18_tensor_decomposition/LAYER4_RESIDUAL_TELESCOPE.md)
and [cross-family null](artifacts/bilin18_tensor_decomposition/LAYER4_RESIDUAL_TELESCOPE_NULL.md).

The dominant Layer-3 MLP term is broad in its native product basis: oracle
top-256 channels recover only 57–61% of the local write but 65–71% of its
downstream effect. Those addresses are not arbitrary. Frozen pair and semantic-
family channel sets transfer across held-out contexts and TinyStories, and a
leave-one-pair-out family set beats random and all wrong-family controls in the
aggregate. The red team prevents a stronger conclusion: generic global channels
beat family-only channels OOD, exact-pair removal is costly there, and one of
five families fails. This is evidence for overlapping family-conditioned native
address priors, not separated family modules. See the
[native-channel audit](artifacts/bilin18_tensor_decomposition/LAYER3_NATIVE_CHANNEL_CIRCUITS.md),
[frozen transfer](artifacts/bilin18_tensor_decomposition/LAYER3_FROZEN_CHANNEL_TRANSFER.md),
and [leave-one-pair-out red team](artifacts/bilin18_tensor_decomposition/LAYER3_FROZEN_CHANNEL_LOPO.md).

A full-width Layer-3 spectrum then directly tests whether these reused native
addresses are Thomas-Dooms-style spikes. They are not isolated spikes: all top-
four deviations remain below every extrapolated bulk curve in a fifteen-window
sensitivity sweep. Nevertheless, each top singular value exceeds all 32
coordination nulls by 7–12%, so the top shoulder contains learned coordination.
The phrase “power-law bulk” is also only descriptive here—an exponential-rank
curve predicts held-out tail points substantially better. The defensible result
is a continuous spectrum with an exceptional learned shoulder, inside which
reusable family computations can live. See the
[Layer-3 spectrum audit](artifacts/bilin18_tensor_decomposition/LAYER3_POWERLAW_SPIKES.md)
and [fit/null sensitivity red team](artifacts/bilin18_tensor_decomposition/LAYER3_POWERLAW_SENSITIVITY.md).

Conditioning the tensor on a leave-one-pair-out semantic-family output basis
does expose exact compact quadratic forms: 64 input modes per output axis
recover about 89–94% of the family patch. But that clean rank-4 patch carries
only 12–19% of the full MLP3 effect. Expanding the family output basis to rank
32 raises fidelity only to 22–30% and erases its reliable advantage over global
and wrong-family controls. Meanwhile, the frozen native family group remains
stronger. The reusable object is therefore not one fixed semantic write vector;
it is a bank of bilinear operations whose coefficients and resulting output
direction depend on context. See the
[family-conditioned forms](artifacts/bilin18_tensor_decomposition/LAYER3_FAMILY_INTERACTION_MODES.md)
and [output-rank red team](artifacts/bilin18_tensor_decomposition/LAYER3_FAMILY_OUTPUT_RANK_SWEEP.md).

The fixed native bank nevertheless has a strong weight-defined internal
hierarchy. SVD of its 256 `Down` columns recovers roughly 51–63% of the bank’s
downstream effect in four axes and 72–80% in sixteen; bottom and random axes do
almost nothing. A basis learned from discovery activations is worse, making this
a genuine weight-based result. Thirty-two axes still lose a material part of
the bank’s full-model relevance, however, so the clean leading operations sit
above a causal tail rather than replacing it. See the
[native-bank output SVD](artifacts/bilin18_tensor_decomposition/LAYER3_NATIVE_BANK_OUTPUT_SVD.md).

Those dominant axes are family-specific rather than a generic high-gain output
basis. At rank 32, the correct family bank recovers 79–82% of its exact effect,
versus 65–71% for a global-bank basis and 40–41% for the best wrong family;
family/global canonical cosine is only 0.413. The reusable object is thus a
coupled pair: a family-conditioned bank of bilinear channel addresses and a
characteristic medium-rank write subspace, with context choosing coordinates
inside it. Rank 64 still leaves about 0.10 full-model fidelity unexplained, so
the object has a causal tail rather than a tiny sufficient circuit. See the
[basis-specificity test](artifacts/bilin18_tensor_decomposition/LAYER3_NATIVE_BANK_BASIS_SPECIFICITY.md).

The corresponding input side is even more compact for the clean output core.
The rotation-invariant metric formed from the bank’s top four output-conditioned
quadratic forms has a recurring family-specific eigenspace. Projecting the real
held-out state change through only four leading input directions recovers
79–90% of the core’s downstream effect; 32 directions recover 97%, including
on TinyStories. Low, random, global, and wrong-family directions are much
weaker. Sign-flipped weights retain nearly the same eigenvalue concentration,
however, so the causal and family-transfer tests—not spectral concentration—are
the evidence for a circuit. See the
[input-form audit](artifacts/bilin18_tensor_decomposition/LAYER3_NATIVE_BANK_INPUT_FORMS.md).

The input-form red team separates a robust subspace from fragile leading axes.
Rank 32 remains at 95–98% fidelity from 10% through full edits, while rank four
falls to 0.49 on FineWeb near the tangent regime. Sign-null and unweighted
factor-span bases retain much of the four-axis effect, but pairing `Down`
columns with the wrong bilinear forms destroys it. Family-specific causal K90
then reveals several regimes: 4 for polarity and full-scale sentiment, 8 for
tense, 32–64 for gender, and 128–256 for increasing-scale. See the
[causal-null/scale red team](artifacts/bilin18_tensor_decomposition/LAYER3_INPUT_FORM_CAUSAL_NULLS.md)
and [family K90 curves](artifacts/bilin18_tensor_decomposition/LAYER3_INPUT_FORM_K90.md).

An exact polynomial split explains why the relevant complexity changes with
scale. For an edit inside the selected subspace,
`Δq = 2δᵀAx + δᵀAδ`: a context-conditioned linear term plus a within-edit
quadratic interaction. At 10% scale the linear term has 92–95% norm share and
99% causal fidelity. At full scale the quadratic term grows to 37–47%; some
families, especially FineWeb tense, require cancellation between the two.
Replacing the current residual context with a discovery mean fails badly.
Float64 verifies the identity to `5.2e-14`; the stricter float32 gate misses only
because of `2.1e-5` cancellation error. See the
[exact polynomial split](artifacts/bilin18_tensor_decomposition/LAYER3_INPUT_POLYNOMIAL_SPLIT.md).

Within that full-scale quadratic term, the signed interaction web is sparse—but
only relative to the family-specific causal basis. Pair K90 ranges from 16 for
sentiment and tense, through 256 for gender, to 1,024–4,096 for increasing-
scale. Weight magnitude selects almost exactly as well as discovery activations
and decisively beats random. The type of operation is not universal: gender is
strongly off-diagonal, sentiment and tense are largely diagonal/cancellation
regimes, and increasing-scale uses both. See the
[signed-pair sparsity audit](artifacts/bilin18_tensor_decomposition/LAYER3_QUADRATIC_INTERACTION_SPARSITY.md).

Natural-text labeling separates semantic **subspaces** from canonical semantic
axes. Eight leading input-metric modes per family were frozen before examining
524,288 FineWeb and 153,600 TinyStories positions. FineWeb-selected seed-token
labels transfer to TinyStories at AUROC 0.80--0.99 and beat isotropic random
directions, but a random rotation of the exact same family subspace is nearly
as good. Moreover, increasing-scale's selected axis is a stronger polarity
detector than scale detector. The native eigensystem is therefore a useful
coordinate choice inside a family-relevant subspace, not a set of uniquely
privileged monosemantic neurons. See the
[natural mode-label audit](artifacts/bilin18_tensor_decomposition/LAYER3_INPUT_MODE_SEMANTICS.md).

A disjoint-lexicon red team confirms that this is more than lookup of the seed
IDs, while locating the boundary of the result. Frozen sentiment, gender,
tense, and polarity axes classify unseen words on both TinyStories and
WikiText-103 at AUROC 0.86--1.00. At the word itself, however, direct embedding
projection is equally good or better: the model adds the embedding at every
layer, and these coordinates inherit that lexical geometry. One token later,
sentiment and polarity remain strong, tense is corpus-dependent, gender
vanishes, and the current-token embedding baseline is beaten. Increasing-scale
fails on unseen words (0.57--0.60) and is dominated by a wrong-family polarity
axis. See the
[unseen-word and offset test](artifacts/bilin18_tensor_decomposition/LAYER3_UNSEEN_LEXICAL_TRANSFER.md).

Allowing a rotation-invariant linear decoder over progressively larger family
subspaces makes every concept decodable: OOD K80 is 1 for sentiment/polarity,
2 for tense, 4 for gender, and 32 for increasing-scale. This does **not** rescue
module ownership. At rank 256 the scale family reaches 0.95--0.96 AUROC, but
the best wrong family and a random 256D projection reach 0.95 as well; the
one-token-ahead scale score is only 0.46/0.60. Wide-subspace linear
decodability is generic residual-stream information. The useful specificity
window is compact and target-dependent: gender and tense have clear low-rank
correct-family advantages, while sentiment and polarity are shared lexical
backbone directions. See the
[semantic subspace-rank red team](artifacts/bilin18_tensor_decomposition/LAYER3_SEMANTIC_SUBSPACE_RANK.md).

An explicit shared/residual quotient shows that “shared semantics” is itself
clustered. The mean-projector top eight align strongly with sentiment,
increasing-scale, and polarity, but barely with gender or tense. Rotating each
family top-32 space into shared-like eight plus residual 24 is exact, yet the
rank-eight boundary has no eigengap and a 14–15% bilinear cross term remains.
Sentiment/polarity edits primarily use shared-like coordinates, while
gender/tense use family residuals. The global mean basis is not sufficient. See
the [shared-backbone quotient](artifacts/bilin18_tensor_decomposition/LAYER3_SHARED_BACKBONE_QUOTIENT.md)
and its [rank/cluster red-team](artifacts/bilin18_tensor_decomposition/LAYER3_SHARED_QUOTIENT_RANK_REDTEAM.md).

Finite interventions on disjoint unseen words broadly preserve those four
regime labels, but the locked TinyStories panel lacks polarity and gender
coverage. A separately labeled repeated-context sensitivity repairs context
counts without pretending that repeated “yes” examples add lexical diversity.
The follow-up reveals an aggregation trap: TinyStories increasing-scale is
slightly residual-like when token occurrences are weighted, but strongly
shared-like when lexical pairs are balanced. Regime claims must name their
sampling unit. See the [locked unseen causal test](artifacts/bilin18_tensor_decomposition/LAYER3_UNSEEN_CAUSAL_QUOTIENT.md),
[coverage sensitivity](artifacts/bilin18_tensor_decomposition/LAYER3_UNSEEN_CAUSAL_QUOTIENT_COVERAGE.md),
and [hierarchical aggregation audit](artifacts/bilin18_tensor_decomposition/LAYER3_LEXICAL_REGIME_HIERARCHY.md).

Thomas Dooms’s power-law-plus-outliers observation becomes substantially more
precise in this model. Layer 3 has no strict global above-power-law spike, but
contracting the tensor onto a frozen semantic-family output subspace exposes
exceptional low-rank input forms. A tensor-Frobenius canonicalization makes the
four axes invariant to the original basis. All five semantic subspaces beat 64
equally canonicalized random output subspaces in top-four interaction energy.
After raising sign and pairing nulls to 64 draws per axis, four families retain
at least three robust axes and gender retains two. Yet only 2/20 axes lie above
the extrapolated power-law curve. Thus output-conditioned learned coordination
is a productive circuit prior; literal power-law extrapolation has poor recall,
and neither replaces finite causal validation. See the
[conditioned spike test](artifacts/bilin18_tensor_decomposition/LAYER3_CONDITIONED_SPECTRAL_SPIKES.md)
and [fair-control red-team](artifacts/bilin18_tensor_decomposition/LAYER3_CONDITIONED_SPIKE_REDTEAM.md).

Held-out finite interventions then draw the crucial boundary. The correct
family output-4 space beats wrong-family and exceptionally concentrated random
subspaces, and its leading tensor-energy axis is individually necessary across
all five families. But that axis is usually insufficient, two axes are not
uniformly sufficient, rank-64 input forms miss some unseen effects, and spectral
concentration correlates only 0.29 with causal necessity. The parent family
patch explains at most 0.247 of the complete MLP effect. Weight spectra therefore
provide a strong breadth-first search order over causal neighborhoods, not the
final causal ranking within a circuit. See the
[held-out axis-level causal test](artifacts/bilin18_tensor_decomposition/LAYER3_CONDITIONED_AXIS_CAUSALITY.md).

Repeating the same construction across depth reveals a distinct final-layer
regime. Median conditioned top-four energy is only 0.05–0.07 at layers 0, 3,
and 8, then jumps to 0.51 at layer 17; 19/20 late axes become strict fitted
spikes. This is not persistence of semantic family modules. Cross-family
output-space overlap jumps to 0.76, rotation-invariant input-space overlap to
0.93, and family coordinates inside the global top four classify below chance.
The shared output-4 patch still explains only 0.08–0.13 of unseen full-MLP
effects. See the [depth replication](artifacts/bilin18_tensor_decomposition/DEPTH_CONDITIONED_SEMANTIC_SPIKES.md)
and [late-collapse red-team](artifacts/bilin18_tensor_decomposition/DEPTH_FAMILY_COLLAPSE_REDTEAM.md).

Most importantly, a matched label-shuffle null overturns the semantic spectral
interpretation. It preserves the same 19 natural counterfactual write
trajectories and group sizes while randomly reassigning pairs to pseudo-
families. No layer's semantic grouping exceeds this null. At layer 17, shuffled
groups reproduce the 0.51 concentration, 0.76 subspace collapse, and essentially
all strict spikes; their 95th percentile is roughly twelve times the isotropic-
random threshold. The depth effect is a real natural-write/readout bottleneck,
but the five semantic labels do not explain its spectral strength. See the
[matched semantic-label null](artifacts/bilin18_tensor_decomposition/DEPTH_SEMANTIC_LABEL_NULLS.md).

An unrestricted follow-up now resolves what remains after those four raw
Layer-17 outliers. They capture about 85% of centered write variance but are
not behaviorally sufficient (median KL 0.13--0.16 when retained alone). The
orthogonal 14--15% tail has effective rank only 4--6, transfers across FineWeb,
WikiText, and TinyStories, carries independently decodable structural roles,
and is causally important. Thus the observed regime is not “four circuits plus
random noise”; it is a dominant raw-weight bottleneck plus a smaller compact
contextual program. See the
[bottleneck/tail ontology](artifacts/bilin18_tensor_decomposition/LAYER17_BOTTLENECK_TAIL_ONTOLOGY.md)
and [tail causal red-team](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CIRCUIT_REDTEAM.md).

Thomas Dooms's “cleaning” qualifier matters sharply. FineWeb covariance
weighting raises the tail forms' top-four interaction energy from 0.12--0.29 to
0.990--0.994 and gives 0.94--1.00 OOD score correlation. But high-resolution
nulls show that covariance creates most of this apparent rank: only one of four
axes beats channel-sign nulls, although all four beat coefficient-pairing
nulls. The learned coefficient-to-form assignment is exceptional; generic
signed compactness is not. See the
[metric sweep](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CLEANING_SWEEP.md)
and [128-draw null red-team](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CLEANING_NULL_REDTEAM.md).

The correct compact object is the whole four-output tensor map rather than four
named PCA coordinates. A basis-invariant common rank-eight input program captures
99.0% of its cleaned tensor energy and reaches 0.96--0.98 held-out score
correlation. Cross terms are essential, and final-distribution KL remains
0.004--0.011, so the program is a strong explanation but not an exact behavioral
replacement. Its canonical axes also rotate on TinyStories. See the
[tail tensor canonicalization](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_TENSOR_CANONICALIZATION.md).

Weighting that program by the exact local final-logit softmax Fisher clarifies
what “behaviorally important” means, but does not make a discrete axis prefix
exact. The quadratic Fisher prediction matches finite 10% interventions at
correlation 0.998--1.000 and remains strong at full scale, while Fisher-greedy,
score-MSE, and Frobenius selection converge to essentially the same top-eight
set. See the [Fisher basis test](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_FISHER_BASIS.md)
and [finite-dose validation](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_FISHER_FINITE_REDTEAM.md).

Continuous subspace optimization does uncover a distinct behavioral core. Two
split fits agree at mean canonical cosine 0.974 and reduce exact mean KL across
FineWeb, WikiText, and TinyStories. Random and deliberately orthogonal restarts
recover the same six-ish-dimensional core, although the last one or two
directions are unstable. This is evidence for a reusable behavioral subspace,
not eight canonical neurons. See the
[continuous Fisher fit](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CONTINUOUS_FISHER.md)
and [restart red team](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CONTINUOUS_FISHER_BASIN_REDTEAM.md).

The behavioral frontier then exposes the limit of that compression. Increasing
continuous rank from 2 to 32 monotonically lowers held-out Fisher error, but no
tested rank reaches the universal median-KL gate; WikiText plateaus near
0.0052--0.0062 and observed-token CE remains worse. Combining the variance and
Fisher spaces also fails because a bilinear union creates new cross terms.
Those terms are causally material, but suppressing them still does not
consistently beat Fisher-eight. In this setting, dimensions are not independent
feature slots: the interaction graph is part of the circuit budget. See the
[rank frontier](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CONTINUOUS_RANK_FRONTIER.md),
[dual-metric union](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_DUAL_METRIC_UNION.md),
and [cross-term red team](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CROSS_TERM_REDTEAM.md).

The fixed-rank failure is not the end of the compression story. A linear router
over 32 frozen natural-state coordinates selects one of four rank-eight tensor
experts. Hard routing beats a parameter-matched global rank-32 program on
FineWeb, WikiText, and TinyStories and reaches the locked median-KL gate. Route
shuffling, uniform/global/position-only routing, and every fixed expert fail;
the gain is genuinely context-specific. A compute-matched sweep from `1x32` to
`16x2` identifies `4x8` as the common three-corpus optimum. See the
[routed program](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_ROUTED_EXACT_PROGRAM.md),
[specificity controls](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_ROUTED_PROGRAM_CONTROLS.md),
and [granularity sweep](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_ROUTED_GRANULARITY_SWEEP.md).

Four independent fits then establish the appropriate canonical level.
Performance varies by only 3.7%, and expert subspaces recur moderately, but
exact route labels and route partitions are not stable. The same-context local
subspace field is only weakly more stable than a context-shuffled null. On a
larger semantic atlas, however, current-token and structural-shape routing
signals transfer OOD and coarse expert profiles align across seeds at cosine
about 0.98. The correct object is therefore a reusable lexical/structural
routing geometry whose four experts are one useful quantization—not four
unique modules. See the
[seed replication](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_ROUTED_SEED_STABILITY.md),
[local-field red team](artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_ROUTED_LOCAL_FIELD.md),
and [semantic atlas](artifacts/bilin18_tensor_decomposition/LAYER17_ROUTED_SEMANTIC_ATLAS.md).

The corresponding middle-layer test rejects a simple “one routed address equals
one compact verb” interpretation. Direct Layer-8 sensitivity subspaces admit a
useful 64-prototype vocabulary, but the exact restricted
`address × context → write` tensor remains broad: all eight frequent routes need
equal Tucker rank 768 to exceed held-out derivative `R²=0.90`. The native CP
basis is not a hidden shortcut. Although individual-energy and exact Shapley
attributions have effective support in only tens of channels, fixed sufficient
sets require 2,048–4,096 of 4,608 channels; per-state oracle selection and
train-fitted gain/coordinate calibration do not repair the gap. This is a
concrete regime of **sparse attribution but broad cooperative sufficiency**.
See the [restricted tensor test](artifacts/bilin18_tensor_decomposition/LAYER8_RESTRICTED_TENSOR_PROGRAM.md),
[rank/metric red team](artifacts/bilin18_tensor_decomposition/LAYER8_RESTRICTED_TENSOR_REDTEAM.md),
[native-channel test](artifacts/bilin18_tensor_decomposition/LAYER8_NATIVE_CHANNEL_PROGRAM.md),
[cooperative attribution audit](artifacts/bilin18_tensor_decomposition/LAYER8_CHANNEL_SHAPLEY_REDTEAM.md),
and [calibration red team](artifacts/bilin18_tensor_decomposition/LAYER8_CHANNEL_CALIBRATION_REDTEAM.md).

An independently trained 12-layer, width-768 tensor transformer then separates
the reproducible and checkpoint-specific parts of the spectral story. The
power-law-like bulk and its coordination-null controls replicate, but the
18-layer model’s four-spike late phase does not: the strongest single spike is
at layer 5, and layer 11 has only two positive fitted deviations. These raw
directions are naturally used—most strikingly, the middle leading spike recovers
89% of optimal rank-one write variance—but retaining four directions is nowhere
near behaviorally sufficient. Covariance cleaning also requires enough samples
and explicit shrinkage: small-sample whitening generated a false dramatic
failure, while 8,192 states with ridge 0.1 reveal compact late forms and broad
middle forms. See the [independent raw replication](artifacts/bilin18_tensor_decomposition/BILIN12_CHECKPOINT_SPECTRUM_REPLICATION.md),
[natural/causal validation](artifacts/bilin18_tensor_decomposition/BILIN12_NATURAL_OUTLIER_VALIDATION.md),
[fourth-moment diagnostic](artifacts/bilin18_tensor_decomposition/BILIN12_CLEANING_FOURTH_MOMENT_REDTEAM.md),
and [covariance-stability red team](artifacts/bilin18_tensor_decomposition/BILIN12_CLEANING_STABILITY_REDTEAM.md).

At Layer 5, contracting the tensor with the leading output mode produces stable
covariance-cleaned input addresses and exact local operators that transfer
across corpora. Their native implementations are not small: a few dominant
channels sit above broad cooperative tails, and sign-scrambled controls show
that the recurring backbone is structural rather than uniquely owned by the
spectral spike. Channels 196/202 are nevertheless 16--34x more causal than
activation-energy-matched controls. Fresh semantic edits isolate a robust
boundary-suppression role only for channel 202; channel 196 does not replicate.
Finally, the pair's superadditive KL is almost completely reproduced by adding
their separate logit changes, so it reflects aligned effects in the softmax
metric rather than a nonlinear two-channel circuit. See the
[weight-conditioned addresses](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER5_WEIGHT_CONDITIONED_ADDRESSES.md),
[channel decomposition](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER5_ADDRESS_CHANNEL_DECOMPOSITION.md),
[shared backbone](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER5_SHARED_CHANNEL_BACKBONE.md),
[recurrence red team](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER5_BACKBONE_RECURRENCE_REDTEAM.md),
[core-channel audit](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER5_CORE_CHANNEL_AUDIT.md),
and [mechanism red team](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER5_CORE_MECHANISM_REDTEAM.md).

An all-layer sweep then asks whether raw RMT outlier strength predicts any of
these stronger circuit criteria. It does not. Power-law bulk fits hold at all
12 layers, but leading excess has only 0.46 Spearman correlation with natural
use, 0.24 with local operator compactness, and 0.04 with native-channel support.
The strongest outliers occur in broad, naturally used middle programs; the
sharply compact-program phase appears at layers 10--11 despite weaker outliers.
This conclusion survives first/minimum/median/maximum address aggregation and
leave-one-layer-out checks. RMT outliers are therefore excellent discovery
coordinates, not self-certifying clean circuits. See the
[outlier-to-circuit depth sweep](artifacts/bilin18_tensor_decomposition/BILIN12_OUTLIER_CIRCUIT_DEPTH_SWEEP.md)
and [metric-sensitivity red team](artifacts/bilin18_tensor_decomposition/BILIN12_OUTLIER_CIRCUIT_METRIC_REDTEAM.md).

The compact late-program follow-up narrows that phase to Layer 11: Layer 10's
third and fourth addresses become broad on an independent covariance slice,
whereas all four Layer-11 operators remain K90 1--2. Exact full operators predict
finite edits almost perfectly, but Frobenius K90 is not behavioral sufficiency;
one K90-two program still leaves about 30% relative KL at rank two. All four
Layer-11 address predicates nevertheless pass literal newline substitutions
across FineWeb, WikiText, and TinyStories. Their top operators share a nearly
identical boundary-reading input covector (balanced common fraction 0.982) and
fan into partly distinct output directions, while realized activation scores
retain roughly two dimensions. This is a concrete shared-abstraction/multiplexed-
verb circuit rather than either one monosemantic neuron or four unrelated codes.
See the [late address audit](artifacts/bilin18_tensor_decomposition/BILIN12_LATE_ADDRESS_PROGRAM_AUDIT.md),
[rank and semantic red team](artifacts/bilin18_tensor_decomposition/BILIN12_LATE_PROGRAM_RANK_SEMANTIC_REDTEAM.md),
and [boundary-code audit](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_BOUNDARY_CODE_AUDIT.md).

A stronger natural-path test then replaces actual final-position tokens with
newlines. The weight-derived address span captures 43--53% of the resulting
Layer-11 state change and 83--91% of its MLP write change, far above random and
slightly better than ordinary state PCA; it approaches a supervised boundary-
delta PCA upper bound and transfers OOD. Yet it leaves 51--58% of isolated
MLP-path logit KL and explains only about 10--15% of the full token edit, so it
is a real component rather than a complete circuit. Most strikingly, 78% of
address-vector energy lies in a causally weak shared direction. Contrast PCs 3
and 4 contain only 3.0% of address energy but 78.6% of natural operator energy
and 78--92% of captured write R2. This is a concrete discriminator code: a
redundant semantic family plus strongly amplified low-energy distinctions. See
the [natural-path red team](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_BOUNDARY_NATURAL_PATH_REDTEAM.md),
[rank-stability audit](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_ADDRESS_RANK_STABILITY_REDTEAM.md),
and [contrast-gain audit](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_ADDRESS_CONTRAST_GAIN_AUDIT.md).

Upstream tracing identifies the route that constructs those discriminator bits.
At Layer 11, exact factorial attribution assigns about 99% of the PC3--PC4
contrast to the carried residual; direct embedding and attention are negligible.
The same coordinates are visually small before the final block, but contrast-
only residual patches recover 44--52% of the final code from Layer 8, 75--81%
from Layer 9, and 99.2--99.6% from Layer 10. Layer-10's MLP supplies about 71%
of its local contrast write while attention supplies approximately zero. Thus
this interpretable subcircuit is a late MLP-to-MLP residual route: Layer 10 writes
a small discriminative precursor and Layer 11 amplifies it into boundary-
conditioned output verbs. The patch still leaves 61--92% of full-token KL, so
the route is semantically clean but behaviorally partial. See the
[final-block path decomposition](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_BOUNDARY_UPSTREAM_DECOMPOSITION.md)
and [layerwise causal trace](artifacts/bilin18_tensor_decomposition/BILIN12_LAYERWISE_BOUNDARY_CONTRAST_TRACE.md).

The unresolved whole-token effect is now traced through the checkpoint's exact
architectural carriers. At the Layer-0 boundary, residual/embedding/value-bus
Shapley projection is about 48--53% / 39--46% / 6--8%; by Layer 10 the residual
alone has 0.994--0.997 Fisher fidelity. Metric and insertion-order red teams
preserve that progressive-compilation story but show that the early residual and
embedding routes are positively synergistic, so the allocations are not literal
storage percentages. See the
[carrier depth audit](artifacts/bilin18_tensor_decomposition/BILIN12_BOUNDARY_CARRIER_DEPTH_AUDIT.md)
and [metric red team](artifacts/bilin18_tensor_decomposition/BILIN12_BOUNDARY_CARRIER_METRIC_REDTEAM.md).

Opening the residual into carried state, current embedding injection, attention,
and bilinear-MLP writes shows that the MLP is the largest new writer at every
layer for the local newline edit. Layer 0 is about 99% MLP; late state magnitude
and behavioral importance diverge sharply. Exact squared-attention factorization
then separates query, key, current value, persistent Layer-0 value bus, and heads.
Attention is only 0.1--3.0% of full KL when the changed token is the prediction
position, but rises to 13--35% for edits 8--64 tokens in the past, with larger key
roles and alternating current-value versus value-bus transport. See the
[residual-component audit](artifacts/bilin18_tensor_decomposition/BILIN12_RESIDUAL_COMPONENT_DEPTH_AUDIT.md),
[QK/OV depth audit](artifacts/bilin18_tensor_decomposition/BILIN12_ATTENTION_QKOV_DEPTH_AUDIT.md),
and [source-distance red team](artifacts/bilin18_tensor_decomposition/BILIN12_ATTENTION_SOURCE_DISTANCE_REDTEAM.md).

All attention writes form an exact cross-position causal cut, but the route is
compressible rather than localized. Held-out search finds a recurrent Layers 5--6
core with condition-dependent completion layers. One shared `{5,6,8,10}` set
reaches 0.839 mean / 0.704 minimum Fisher fidelity across nine corpus-distance
conditions; six shared layers reach 0.911 / 0.815 and eight reach 0.975 / 0.944.
Partial replacement can be non-monotone, while additive-delta patches fail badly,
so native attention writes should be treated as state-dependent message
interfaces rather than independent additive packets. See the
[multilayer routing audit](artifacts/bilin18_tensor_decomposition/BILIN12_MULTILAYER_ATTENTION_ROUTING_AUDIT.md),
[held-out subset search](artifacts/bilin18_tensor_decomposition/BILIN12_ATTENTION_LAYER_SUBSET_SEARCH.md),
and [universal-backbone audit](artifacts/bilin18_tensor_decomposition/BILIN12_UNIVERSAL_ATTENTION_BACKBONE_AUDIT.md).

Destination-level decomposition further shortens the route. For past-token
edits, 46--71% of a selected attention-delta's state projection can remain near
the edited position while only 0.3--5.2% lands at the final prediction token.
Nevertheless, the final-position write receives 34--101% of downstream
behavioral allocation across five Fisher/KL insertion and removal definitions,
and reaches approximately 98--101% by Layer 10 in every tested corpus and edit
distance. Thus the network uses large local representation updates and small
high-gain destination messages. The universal early-dominance hypothesis fails,
but the state/behavior inversion and late prediction bottleneck survive the red
team. See the
[relay-topology audit](artifacts/bilin18_tensor_decomposition/BILIN12_ATTENTION_RELAY_TOPOLOGY_AUDIT.md)
and [destination metric red team](artifacts/bilin18_tensor_decomposition/BILIN12_ATTENTION_DESTINATION_METRIC_REDTEAM.md).

Opening that small final-position message reveals a cooperative tensor-edge
program rather than a tiny native module. Two heads average only 0.596 fidelity,
three average 0.708, and a frozen rank-16 message PCA reaches 0.386. Exact
head/source factorization instead exposes signed source contributions. A
one-sided “helpful edge” ranking fails because negative contributions cancel
positive overshoot; fresh-split cancellation ratios are 1.40--1.52. Building the
implicit downstream Fisher Gram and selecting edges by residual error reduction
recovers 0.898 mean / 0.753 minimum fidelity with 16 sources and 0.929 / 0.823
with 32. Roughly 20% of selected edges initially oppose the target. Exact
routing/value Shapley then shows why only part of this web is architectural:
normalized attention conserves signed routing mass, but changed-value transport
supplies 44% of behavior on average and often dominates early layers. See the
[head program](artifacts/bilin18_tensor_decomposition/BILIN12_FINAL_DESTINATION_HEAD_PROGRAM.md),
[subspace audit](artifacts/bilin18_tensor_decomposition/BILIN12_FINAL_DESTINATION_SUBSPACE_AUDIT.md),
[cancellation red team](artifacts/bilin18_tensor_decomposition/BILIN12_SOURCE_CANCELLATION_REDTEAM.md),
[cooperative source program](artifacts/bilin18_tensor_decomposition/BILIN12_SOURCE_FISHER_GRAM_PROGRAM.md),
and [routing/value audit](artifacts/bilin18_tensor_decomposition/BILIN12_SOURCE_ROUTING_VALUE_AUDIT.md).

The oracle program now has a predictive local counterpart. Across 19 ordinary
lexical substitutions in five semantic families, simply keeping the source
contractions with largest residual-space change reaches 0.967 fidelity with 16
sources and 0.993 with 32, without using target logits, the unembedding, or a
downstream gradient. On a fresh third context per lexical pair, top-32 mean /
fifth-percentile / minimum fidelity is 0.991 / 0.976 / 0.878 across 152
layer-example effects. Pattern-only ranking is slightly weaker, and recency is
far weaker at small budgets. Same-family source-position overlap is no higher
than different-family overlap, so the reusable abstraction is the tensor-local
selection rule, not a semantic address list. A 0.125×--1.5× scale sweep rejects
small effect size as the sole explanation. See the
[semantic transfer](artifacts/bilin18_tensor_decomposition/BILIN12_SEMANTIC_SOURCE_PROGRAM_TRANSFER.md),
[individual red team](artifacts/bilin18_tensor_decomposition/BILIN12_SEMANTIC_SOURCE_INDIVIDUAL_REDTEAM.md),
and [scale red team](artifacts/bilin18_tensor_decomposition/BILIN12_NEWLINE_SOURCE_SCALE_REDTEAM.md).

The independent Layer-11 follow-up sharpens that result. Its raw four-space is
a genuine cross-corpus natural bottleneck, and the orthogonal residual tail is
variance-compact and stable across FineWeb, WikiText, and TinyStories. It is not
behaviorally compact: even 68 fixed directions leave median final-distribution
KL around 0.054--0.072. Exact-KL fitting gives small transferable improvements
while sacrificing some write variance, so downstream metric mismatch is real
but does not reveal a hidden small fixed circuit. See the
[bottleneck/tail replication](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_BOTTLENECK_TAIL_REPLICATION.md),
[exact-behavior subspace test](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_BEHAVIOR_SUBSPACE.md),
and [optimizer red team](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_BEHAVIOR_OPTIMIZER_REDTEAM.md).

A conditional follow-up then tests whether that broad fixed tail is sparse per
token. A four-expert hard router gives a modest, repeatable gain over fixed rank
20, and routed top-two unions use 36 directions while coming within 10% of the
full learned 68-dimensional union. This is only relative compression: absolute
KL remains high, oracle experts are insufficient, and matched fixed-rank spaces
remain competitive. Across three restarts, aggregate performance is extremely
stable but expert unions, route assignments, selected projectors, reconstructed
writes, and approximate output distributions are not. The result therefore
supports a reproducible family of noncanonical lossy approximations—not a
recovered sparse native circuit. See the
[conditional router](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_CONDITIONAL_OUTPUT_ROUTER.md),
[top-k execution frontier](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_ROUTER_TOPK_FRONTIER.md),
[seed-stability audit](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_ROUTER_SEED_STABILITY.md),
and [functional-equivalence red team](artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_ROUTER_FUNCTIONAL_EQUIVALENCE.md).

### Bilin18 decomposition scorecard

The tensor follow-up now has a preregistered scorecard covering in-distribution
and OOD behavior, softmax-Fisher causal intervention fidelity, compression and
actual routing cost, symmetry-aware factor stability, and held-out token-form
semantics. It compares direct CP-2048, activation-refined CP-2048, mixed Top-K
routed CP-2048, and lower-capacity CP/LL1 controls.

```bash
MPLBACKEND=Agg uv run python scripts/evaluate_bilin18_decomposition_metrics.py
MPLBACKEND=Agg uv run python scripts/redteam_bilin18_decomposition_causal_metrics.py
```

Read the
[evaluation scorecard](artifacts/bilin18_tensor_decomposition/EVALUATION_SCORECARD.md)
and its
[causal scale/OOD red-team](artifacts/bilin18_tensor_decomposition/CAUSAL_METRIC_REDTEAM.md).
Activation-refined CP-2048, not direct tensor CP, is the best current behavioral
and causal replacement, but it still fails the predeclared causal gates.
