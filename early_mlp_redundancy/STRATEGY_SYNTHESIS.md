# What the toy and real-model studies currently say about computational strategies

2026-10-28. A compact reading guide after the fixed capacity/noise/dropout grid. This is a synthesis,
not a declaration that the LLM is reverse engineered.

## Name the object before naming the strategy

- **Function geometry:** which inputs get similar outputs? Token clustering, re-clustering and
  context-dependent conjunctions live here.
- **Implementation geometry:** how do product-channel writes combine? Distributed aligned
  contributions, overlap and signed cancellation live here.
- **Behavioral use:** which changes do readers notice? Preservation of a binary answer, output
  distribution and intermediate vector are different claims.

The same clustered function can have duplicated or cancelling implementations. A class-specific
function can be implemented across many channels. These labels are coordinates of a profile,
not exclusive species of MLP.

## Evidence map

| Finding | Supported conclusion | Not established |
|---|---|---|
| Same-function hand-built toys differ under channel deletion | Intact function/clustering does not identify implementation | Which implementation an optimizer or LLM will choose |
| MLP1 quarter removal preserves binary accuracy but changes top predictions | Task-level removal robustness can hide distributional damage | Unchanged computation or96% removable channels |
| Matched MLP0/1/2 contributions are net constructive on the task panel | Simple “MLP1 redundancy versus MLP2 cancellation” is not a general layer distinction | No cancelling pairs, or corpus-wide absence of cancellation |
| Contribution penalty improves toy deletion robustness when rank permits spreading | Training objective and capacity can change implementation at similar clean fit | Universal superiority of distributed coding |
| Input-noise training increases class specificity and selective channel vulnerability | Different robustness pressures can favor different toy implementations | Noise always causes localization, or an unavoidable tradeoff |
| Width32 plus channel dropout improves both input-noise and deletion classification | The previous toy tradeoff can be reduced by objective/capacity | Exact functional fidelity, severe-noise robustness or an optimal frontier |

## Current explanation, scoped to the toys

The relevant variable is not width alone but how width compares with the dimensionality of the
required output function, plus which perturbations training penalizes. Spare capacity can be used
to spread useful contributions. Training against input ambiguity constrains the function between
clean prototypes; it can produce selective detectors. Training against channel loss can distribute
those detectors' consequences again. The regularizer changes the cost of coordinated large writes.
These observations support an objective-dependent implementation account, rather than one best
representation independent of task and metric.

Even here, we have not isolated every mediating cause. Parameter count and initial-function variance
change with width; the toy inputs privilege a simple supervised prototype geometry. No held-out
feature combinations or label corruption were tested. Nuclear-norm bounds constrain possible
factorizations but do not prove why gradient descent found one solution.

## What would connect this to the language model?

First establish whether empirical function dimensionality and contribution coordination distinguish
the early layers on a shared input distribution. Then measure selective consequences with task
competence and whole-task holdouts intact. Finally, compare representations that encode the
surviving mechanism: shared dictionary directions for genuine reuse, signed grouped contributions
where cancellation matters, and conditional coordinates for context-dependent functions.
A good featurizer must predict/intervene faithfully on additional tasks, not merely reconstruct
activation variance or steer one answer. The toy grid is now closed; this mapping is the next job.

Start with [M1 real-layer comparison](MATCHED_LAYER_MECHANISMS.md),
[T2 capacity/noise](TOY_CAPACITY_RESULTS.md), [T3 width/specialization](TOY_WIDTH_RESULTS.md),
and [T4 combined robustness](TOY_DROPOUT_RESULTS.md). The historical re-clustering correction and
literal-removal distinction are in [the removal audit](REMOVAL_REDUNDANCY.md).
