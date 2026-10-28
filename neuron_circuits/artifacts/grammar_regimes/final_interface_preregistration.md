# Preregistration: final-layer semantic rule versus generic answer interface

The validated grammar result found that final-layer mean activation deltas for a
known rule transition transfer causally and are sparse in native channels. This
can have two explanations:

1. the channels implement the reusable grammar rule;
2. the final MLP exposes a generic answer-output interface, so any delta that
   suppresses answer A and promotes answer B transfers regardless of why the
   answer changed.

Use the seven gate-passing depth-4 bilinear checkpoints from three seeds. For
each checkpoint, learn final-layer activation prototypes for the eight answer
classes from correctly predicted training facts on template 0. Evaluate class
transition deltas on disjoint facts using unseen template 6. Compare full,
native-top-32, random-native-32, and top-32-after-random-rotation patches.

Separately repeat the existing direct-rule counterfactual construction. On the
same eligible recipients compare:

- the rule-transition delta learned from paired latent edits;
- the generic class-prototype delta for the same old/new answers;
- their exact finite logit effects, target accuracies, native top-32 supports,
  and support overlap.

All identities used to estimate prototypes are disjoint from evaluation facts.
Discovery and evaluation templates differ. Fully arbitrary models are primary
evidence only for the generic answer interface; structured semantic transitions
are not capability targets there. Underfit seed-2 mixed/arbitrary checkpoints
remain excluded by the original fidelity gate.

If generic class deltas reproduce the pure semantic patch, the earlier causal
result remains valid but its ontology changes from “rule neuron” to “native-axis
output verb.” A semantic advantage beyond the class interface is required for a
rule-specific final-layer claim.
