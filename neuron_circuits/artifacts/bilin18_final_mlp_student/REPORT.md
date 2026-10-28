# Learned narrow replacement for Bilin18's final MLP

## Result

The 4,608-channel final bilinear MLP can be approximated surprisingly well by
a newly trained bilinear layer with tens, rather than thousands, of channels.
This is substantially better than retaining the same number of native channels.

Across two disjoint FineWeb train/validation/evaluation panels:

| Student channels | Original channel fraction | Parameters | Mean write cosine | Mean CE increase | Perplexity multiplier | Mean KL from full | Top-1 agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.17% | 28,800 | 0.9633 | +0.1213 | 1.129× | 0.1279 | 82.0% |
| 16 | 0.35% | 56,448 | 0.9768 | +0.0855 | 1.089× | 0.0884 | 83.8% |
| 32 | 0.69% | 111,744 | 0.9823 | +0.0727 | 1.075× | 0.0728 | 84.0% |
| 64 | 1.39% | 222,336 | 0.9853 | +0.0620 | 1.064× | 0.0638 | 83.6% |
| 128 | 2.78% | 443,520 | 0.9869 | +0.0492 | 1.050× | 0.0551 | 84.9% |

The original layer has 15,926,400 parameters. A 32-channel student therefore
uses 0.70% as many MLP parameters and increases perplexity by about 7.5% on
average. A 128-channel student uses 2.78% as many and increases perplexity by
about 5.0%.

This validates the narrow-layer hypothesis in the approximate, average-loss
sense. It falsifies the stronger version that a few dozen channels reproduce
the exact full-model distribution: even the 128-channel student leaves about
0.055 nats of KL and changes the top prediction about 15% of the time.

## What was optimized

The teacher computes

`f(x) = Down((Left x) ⊙ (Right x)) + bias`.

The student has the same form but only `k` product channels. Training minimizes

`mean(1 - cosine(student_write, teacher_write))`

plus a 0.25-weighted squared log-norm error. The norm term is necessary because
cosine alone does not constrain how strongly the layer changes the residual
stream. Training used all 128 positions in 400 FineWeb rows (51,200 states), a
separate 12,800-state validation panel, and 400 disjoint document-final
next-token examples for end-to-end evaluation. The entire procedure was
repeated on non-overlapping row ranges with a different optimization seed.

Students were initialized from the strongest native channels and then all
three factors were allowed to rotate. The comparison is therefore a test of a
learned low-channel replacement, not a claim that the same result is always
found from random initialization.

## Symmetries

A bilinear term has no unique factor coordinates. It is invariant to swapping
its left and right readers, reciprocal positive rescaling, paired sign moves,
and channel permutation. The student normalizes every left/right row during its
forward pass and carries the factor magnitude in the down vector. The objective
compares complete 1,152-dimensional writes rather than attempting to pair
student neurons with teacher neurons.

A numerical check on the 32-channel checkpoint changed the output by at most
`4.9e-4` under arbitrary positive factor rescaling, exactly zero under a global
left/right swap and paired sign move, and `9.2e-5` under channel permutation.
The nonzero values are float32 accumulation-order effects, not functional
differences.

The exact symmetrized *global operator* cosine tells a very different story
from FineWeb-state cosine:

| Channels | FineWeb write cosine | Full bilinear-operator cosine |
|---:|---:|---:|
| 8 | 0.963 | 0.054 |
| 16 | 0.977 | 0.088 |
| 32 | 0.982 | 0.113 |
| 64 | 0.985 | 0.137 |
| 128 | 0.987 | 0.165 |
| 256 | 0.987 | 0.202 |

The global cosine is computed exactly from
`T = sum_j d_j ⊗ sym(l_j r_jᵀ)` without materializing the third-order
tensor. It has the same swap, scale, sign, and permutation invariances. Thus the
small student does **not** approximate the teacher on arbitrary directions in
1,152-dimensional space. It approximates the teacher extremely well on the
thin, structured state distribution produced by the first 17 blocks. This is
strong evidence for distributional compression rather than globally low tensor
rank.

## Learned replacement versus native pruning

The fixed-native baseline was rerun on the exact two evaluation panels.

| Channels | Learned mean CE increase | Fixed-native mean CE increase | Learned improvement |
|---:|---:|---:|---:|
| 32 | +0.0727 | +0.1753 | 59% lower |
| 64 | +0.0620 | +0.1188 | 48% lower |
| 128 | +0.0492 | +0.0843 | 42% lower |

This matters mechanistically. The final computation is more compressible than
the native basis suggests: useful quadratic directions can be recombined into
new factors. Consequently, “how many native neurons are needed?” and “what is
the bilinear rank needed on the data distribution?” are different questions.

## Red-team interpretation

- **It is distributional compression.** The student was fit and tested on
  FineWeb states. It need not match the teacher on arbitrary residual vectors,
  rare domains, longer contexts, or adversarial inputs. Exact global operator
  cosines of only 0.05–0.20 confirm that this caveat is substantive.
- **The initialization is favorable.** Starting from ranked teacher channels
  establishes existence and local learnability, not optimizer-independent
  minimal bilinear rank. Random initialization remains a useful control.
- **Cosine is not behavior.** Write cosine rises monotonically with width, but
  CE improves much more slowly after 32 channels. Large write directions can be
  behaviorally redundant after the residual addition and final RMSNorm.
- **CE is the primary capability metric here.** KL and top-1 agreement answer
  the stricter question of reproducing this exact model. A 16–32 channel layer
  may be adequate for understanding most average task contribution while still
  being an incomplete emulator.
- **Only the final MLP was replaced.** This does not show that earlier MLPs or
  the whole 546M-parameter model have comparable low rank.

## Follow-up objectives, architectures, and OOD evaluation

The replacement was subsequently trained with several other objectives and
architectures. Evaluation used 1,200 examples each from a disjoint FineWeb
range, WikiText-103 test, and TinyStories validation. The latter two are real
distribution shifts, although they are not guaranteed decontaminated from all
web content that could appear in FineWeb.

### Objective and initialization comparison at 32 channels

| Variant | FineWeb ΔCE | WikiText ΔCE | TinyStories ΔCE |
|---|---:|---:|---:|
| Ranked initialization, normalized-state cosine | +0.088 | +0.096 | +0.028 |
| Ranked initialization, output KL | +0.079 | +0.075 | +0.040 |
| Ranked initialization, KL + CE | +0.076 | +0.073 | +0.029 |
| Ranked initialization, CE only | +0.107 | +0.132 | +0.068 |
| Random initialization, write cosine | +0.659 | +0.844 | +0.233 |
| Random initialization, normalized-state cosine | +0.660 | +0.985 | +0.233 |

The compact solution was not found from random initialization at this compute
budget. Pure CE training also generalized worse. Matching the normalized state
or the intact distribution provided the best robust objectives.

### Alternative architectures

| Replacement | Parameters | FineWeb ΔCE | WikiText ΔCE | TinyStories ΔCE |
|---|---:|---:|---:|---:|
| Rank-32 linear | 74,880 | +0.483 | +0.613 | +0.190 |
| Rank-64 linear | 148,608 | +0.446 | +0.508 | +0.196 |
| Full affine 1,152×1,152 | 1,328,256 | +0.362 | +0.528 | +0.118 |
| 64 symmetric squares | 148,608 | +0.080 | +0.081 | +0.025 |
| 128 symmetric squares | 296,064 | +0.068 | +0.067 | +0.022 |
| 64 general bilinear, KL | 222,336 | +0.066 | +0.072 | +0.024 |

The final computation is therefore not merely affine on the observed state
manifold. Quadratic structure is essential. The 64-channel general bilinear
student is the most parameter-efficient robust replacement. The 128-square
student has the best mean CE across the three panels by a small margin and is
algebraically simpler per channel.

### Original versus 128-square neurons

On the same 2.048M/0.512M-position FineWeb atlas, the active half of native
layer-17 neurons has median class lift 1.370 and median signed class gap 0.547
SD. The 128-square replacement reaches 1.641 lift and 1.351 SD; 87.5% of its
active half exceeds a one-SD gap, compared with 19.3% of native channels.

That apparent gain in monosemanticity is real descriptively but misleading if
read one channel at a time. The square units inherit a paired decomposition of
general bilinear terms. Across 64 pairs:

- paired output directions have median cosine −0.972;
- activation correlation ranges widely (10th–90th percentile −0.47 to 0.89);
- 43.9% of separate pair contribution magnitude cancels on average;
- several pairs cancel more than 70%.

For example, replacement units 59 and 123 both fire strongly at document
boundaries, but write in nearly opposing directions. Their difference encodes
the signed interaction. Consequently the replacement offers cleaner scalar
detectors while making the *pair*—or the equivalent general bilinear neuron—the
more faithful causal object. This is a concrete case where neuron-level
monosemanticity and module-level simplicity come apart.

## Reproduction

```bash
PYTHONPATH=src:scripts .venv/bin/python \
  scripts/train_bilin18_final_mlp_student.py
```

Primary checkpoints and measurements are in this directory. The independent
replicate is in `../bilin18_final_mlp_student_replicate/`; 8/16-channel runs are
in the corresponding `*_tiny*` directories. Matched fixed-native controls are
in `../bilin18_final_mlp_compression_matched/` and its replicate.
