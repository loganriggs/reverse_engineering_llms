# Multiclass clean-neuron benchmark

## Why multiclass is the right next benchmark

Binary classification makes CE and correct-logit margin nearly redundant for neuron ranking. If `m` is the correct-versus-incorrect margin, then for one example

`|dCE/da_i| = sigmoid(-m) |dm/da_i|`.

The softmax only reweights examples; it does not change the within-example ordering of neurons.

For `C > 2`, a last-layer neuron with class-effect vector `w_i` has

`dCE/da_i = w_i^T (p - one_hot(y))`.

Importance now depends on which wrong classes compete, how probability is distributed among them, and whether the neuron promotes one class or suppresses a family of alternatives. Ten- and hundred-class problems therefore distinguish accuracy, confidence, class similarity, and output-role specificity in a way binary parity cannot.

## Data-generating process

### Memorized facts

- Sample `K` keys.
- Assign every key a balanced latent class `c_k in {0, ..., C-1}`.
- Train an identity task for every key so the model must store the base class.

### Related classification tasks

Each task is a reusable transformation of the base class:

- cyclic shift: `y = (c_k + s_t) mod C`;
- affine permutation: `y = (a_t c_k + b_t) mod C`, with invertible `a_t`;
- coarse-to-fine projection for hierarchical classes;
- paired-input composition in a later phase: `y = (c_i + c_j) mod C`.

The initial experiment should use identity plus cyclic and affine tasks. These have known algorithms, balanced outputs, and exact counterfactuals. Random class permutations are a memorization control: they have the same input/output statistics but less reusable algebraic structure.

### Inputs

Use a key representation and a random distributed task code. Do not give each operation a dedicated scalar coordinate in the confirmatory condition. Compare three instruction regimes:

1. explicit one-hot task marker;
2. fixed random distributed task code;
3. task inferred from a short context or demonstrations.

This reveals whether apparent verb neurons merely copy an exposed task bit.

### Train/test construction

- Every key appears in the identity task.
- Every transformation is trained on a balanced subset of keys/classes.
- Test on held-out key–transformation combinations.
- Hold out some transformations that are compositions of trained generators.
- Keep discovery, intervention-validation, and locked confirmation examples disjoint.

The central generalization test is not unseen keys; it is applying a learned operation to a memorized class on an unseen key–operation pairing.

## Class counts

Run both:

- `C = 10`: easy enough for broad width/depth/load sweeps and many seeds;
- `C = 100`: closer to a vocabulary slice and much more diagnostic of distributed class evidence.

For 100 classes, organize labels as ten superclasses with ten subclasses each. The model still trains with ordinary one-hot CE: exactly one class is correct. The hierarchy is used only for additional evaluation, so standard classification performance remains interpretable.

## Metrics

### Predictive behavior

- exact top-1 accuracy;
- top-5 accuracy;
- cross-entropy;
- Brier score and expected calibration error;
- correct-versus-runner-up margin;
- KL divergence from the intact model after intervention;
- cyclic or tree distance between predicted and correct classes;
- fraction of errors within the correct superclass.

CE treats every wrong label as wrong. Class distance separately distinguishes a nearby error from a semantically unrelated error without changing the training target.

### Sparse circuits

For each metric, report the smallest held-out shared mask found at 80%, 90%, and 99% recovery. Mask discovery and final measurement must use different examples.

Use at least three search methods:

- activation-times-gradient ranking;
- greedy deletion on a validation set;
- learned hard-concrete/L0 masks.

Agreement across methods is evidence that circuit size is not a ranking artifact. Disagreement is itself a result.

### Confidence normalization

Raw CE can be dominated by arbitrary logit scale. Report both raw CE and CE/KL after fitting one temperature on a separate calibration split. Never fit temperature on the circuit-test examples.

## What a clean neuron would mean

### Class-evidence neuron (“noun-like”)

A candidate class neuron should:

- activate for one class or coherent class family across tasks;
- be invariant to which transformation produced that class;
- have a decoder effect that promotes that class relative to appropriate alternatives;
- causally affect that class contrast on held-out examples;
- remain specific when other latent variables are changed.

### Operation neuron (“verb-like”)

A candidate operation neuron should:

- respond consistently to an operation across many base classes;
- generalize to held-out key–operation combinations;
- when counterfactually patched, change predictions according to the same class transformation across many starting classes;
- have little effect on matched examples where that operation is irrelevant;
- outperform rotated directions and equal-dimensional operation subspaces under the same intervention budget.

A fixed final-layer decoder direction is naturally class-specific. Operation neurons are more likely to occur in earlier layers, where their writes can alter downstream routing or transformations. Layer location should therefore be a preregistered prediction rather than interpreted after the fact.

## Counterfactuals

Construct matched pairs such as:

- same key, operation changes from shift `+2` to shift `+3`;
- different keys with the same base class and operation;
- same output class reached through different base-class/operation combinations;
- subclass changes within one superclass;
- superclass changes while lower-level nuisance statistics are matched;
- relevant key changes versus irrelevant-context changes.

Patch one neuron, a neuron set, a rotated direction, and a learned subspace from donor to base. Recompute all downstream layers. Measure the complete output-distribution change, not only whether argmax flips.

## Cleanliness certificate

Freeze each neuron explanation on discovery data. On locked confirmation data require:

1. activation selectivity at least 0.8;
2. counterfactual consistency at least 0.8;
3. nuisance invariance at least 0.8;
4. causal specificity at least 0.8;
5. nontrivial causal effect;
6. input-output prediction agreement at least 0.8;
7. score above the 99th percentile of rotation and matched-direction nulls.

Threshold definitions must be fixed before confirmatory evaluation. Report every component rather than only the pass/fail result.

## Primary falsifiable hypotheses

### Raw-neuron hypothesis

At high task diversity, certified raw neurons account for at least 50% of held-out causal effect, with a seed-level 95% lower confidence bound above a preregistered floor.

### Communication-pressure hypothesis

At fixed example count and model capacity, increasing the number of reusable transformations increases clean causal coverage by at least 0.10.

### Verb-versus-noun prediction

Earlier layers contain more certified operation neurons; the final bilinear layer contains more class-evidence neurons. This must be tested using the same certificate and multiple-comparison correction.

### Falsification

The raw-neuron hypothesis is rejected if raw neurons fail to outperform rotated directions, clean causal coverage remains below the preregistered level, explanations fail on locked counterfactuals, or learned subspaces succeed substantially better than individual neurons.

## Initial experiment matrix

| factor | values |
|---|---|
| classes | 10, 100 |
| keys | 256, 1,024 |
| transformations | 2, 8, 32 |
| instruction | one-hot, random distributed |
| architecture | standard MLP, bilinear residual |
| depth | 1, 2, 3 |
| seeds | at least 20 |

Start with `C=10`, 256 keys, 8 transformations, three layers, and 20 seeds to validate optimization. Then lock analysis choices before running `C=100`.

