# Two concrete circuits from the width-96 SwiGLU model

These are the two seed-0 examples plotted in `worked_circuit_curves.png`. They were selected mechanically as the median-S90 valid example in each group, not because they looked especially clean.

## Example 1: a rule the model generalized

Prompt content:

- subject: `syllable_1 syllable_4 syllable_2`
- object: `syllable_7 syllable_4 syllable_2`
- query: `relation_0`
- correct answer: `answer_4`

`relation_0` reads the subject's first syllable. Here that value is 1, and the relation's fixed output permutation maps 1 to `answer_4`. This pair was never trained and was not randomized, so the prediction is genuine rule generalization.

The intact correct-versus-rival margin was 8.36. With every MLP channel silenced it was still 5.43, so the MLPs added 2.93 margin on top of what attention/residual pathways already supplied.

- Native S90: 25 of 768 channels, spread across layers as `[5, 8, 6, 6]`.
- Rotated S90: 37 channels, spread as `[9, 16, 11, 1]`.
- Axis advantage: 0.55 bits, or about 1.46× fewer native than rotated coordinates.
- N90: removing 336 ranked native channels was needed to erase 90% of the MLP-added margin.
- The donor circuit recovered at least 90% on 3 of 6 other naturally correct examples with the same relation and answer.

The first eight ranked coordinates included `(layer, channel)` pairs `(1,71), (2,23), (0,50), (1,140), (2,17), (1,29), (1,132), (2,129)`. Their signed first-order margin contributions were `-2.46, +1.89, -1.69, +1.44, -1.40, -1.38, -1.37, +1.24`. Large effects of both signs explain why the exact blue curve rises and falls: the answer depends on a balance of promoting and suppressing terms, not 25 independent votes for the answer.

Interpretation: this example has a moderately privileged native circuit and partial semantic reuse. It is evidence for an axis-aligned reusable computation, but not a one-neuron/one-rule representation.

## Example 2: a memorized exception

Prompt content:

- subject: `syllable_7 syllable_7 syllable_7`
- object: `syllable_6 syllable_2 syllable_6`
- query: `relation_3`
- rule answer: `answer_3`
- trained arbitrary answer: `answer_4`

`relation_3` normally reads the object's second syllable. The underlying value is 2 and the fixed permutation maps it to `answer_3`, but this training fact was randomized to `answer_4`. The model correctly recalled the exception.

The intact correct-versus-rival margin was 2.02. Silencing every MLP changed it to -3.12, so the MLPs supplied the 5.13 margin that reversed the decision.

- Native S90: 84 of 768 channels, spread across layers as `[29, 23, 10, 22]`.
- Rotated S90: 288 channels, spread as `[83, 87, 58, 60]`.
- Axis advantage: 1.77 bits, or about 3.4× fewer native than rotated coordinates.
- N90: removing 168 ranked native channels erased 90% of the MLP-added margin.
- The circuit transferred above 90% on only 1 of 11 other changed exceptions with the same answer but a different relation.

The first eight ranked coordinates were `(2,150), (0,86), (3,56), (0,84), (1,49), (3,128), (3,67), (2,49)`, again with mixed signed contributions: `-3.13, -2.29, -1.79, +1.74, +1.73, -1.44, +1.38, -1.38`.

Interpretation: this exception is strongly axis-aligned and causally sparse relative to a random basis, yet it barely reuses across other facts with the same output. Calling it “monosemantic” from sparsity alone would be misleading. It looks more like a privileged, fact-sensitive circuit embedded in a web of cancellations.

## What the jagged graph means

The curves are exact forward interventions after ranking channels by `|activation × margin gradient|`. Moving right retains more ranked channels. If channels were independent positive contributors, recovery would rise smoothly. It does not: both curves repeatedly overshoot and reverse. Restoring one member of a cooperating or cancelling set before its partners can temporarily make the margin worse.

Therefore S90 is reported as the smallest crossing on a fixed dense grid, not as a proven globally minimal circuit. The jaggedness is itself mechanistic evidence: even when a native coordinate system is privileged, the coordinates still interact.
