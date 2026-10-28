# Bilinear memory: capacity, computation, and clean interfaces

The model stores 512 random key→value mappings using 64 shared bilinear neurons (8.0 mappings per neuron). Max/min evaluation uses the 20% of key pairs never shown during training.

Values in this table are means over three seeds.

| regime | final loss | lookup | held-out max | held-out min | within-value cosine | neurons with value η² > .8 | centroid max |
|---|---:|---:|---:|---:|---:|---:|---:|
| lookup | 1.88e-05 | 1.000 | nan | nan | 0.223 | 0.000 | 0.054 |
| lookup_property | 4.26e-05 | 1.000 | nan | nan | 0.235 | 0.010 | 0.047 |
| lookup_compute | 1.50e-04 | 1.000 | 1.000 | 1.000 | 0.405 | 0.417 | 1.000 |
| full | 1.65e-04 | 1.000 | 1.000 | 1.000 | 0.384 | 0.354 | 1.000 |

All 27 trained models reached the convergence criterion: three successive checks with loss below 1e-3 and minibatch accuracy above 99.9%. The capacity sweep achieved 100% lookup at every tested load from 1 to 16 mappings per neuron. Results for every seed are in `results.json`.

## Operational cleanliness

Neuron cleanliness is the product of (1) activation variance explained by the stored value, (2) positive correlation between its value-conditional activation and decoder write, and (3) decoder concentration. This is intentionally strict: a distributed but linearly clean interface can score well at the population level while individual neurons remain mixed.

For the full model's first seed, the five cleanest neurons are:

| neuron | cleanliness | activation η² | activation/output corr. | decoder purity | activation value | promoted value |
|---|---:|---:|---:|---:|---:|---:|
| N24 | 0.131 | 0.939 | 0.880 | 0.158 | 3 | 3 |
| N39 | 0.105 | 0.932 | 0.869 | 0.130 | 2 | 2 |
| N7 | 0.093 | 0.862 | 0.787 | 0.137 | 0 | 7 |
| N16 | 0.090 | 0.746 | 0.863 | 0.140 | 9 | 9 |
| N28 | 0.089 | 0.781 | 0.874 | 0.130 | 3 | 3 |

## Interpretation

The experiment separates capacity from usability. The capacity sweep tests whether lookup can converge when mappings greatly outnumber neurons. Held-out max/min then tests whether the retrieved codes support a shared computation rather than memorized key-pair answers. Replacing every key embedding by its value centroid is an additional causal compression test: preserved pair accuracy means identity-specific details are unnecessary once the value interface is formed.

Lookup alone kept the representation distributed: within-value cosine was 0.223, no neuron had value η² above .8, and 48/64 neurons were needed for 99% lookup. Adding max/min raised within-value cosine to 0.405, made 41.7% of neurons strongly value-selective, and preserved 100% held-out computation after replacing every key by its value centroid. Yet approximately 33 neurons participated in each maximum output tensor, so the clean interface remained a population code rather than one-neuron-per-value storage.

This is not an information-capacity paradox: the 512 learned key embeddings are parameters in the memory table. The result is that the shared 64-neuron bilinear computation can address and use far more mappings than its width, not that 64 scalar parameters contain 512 arbitrary facts.

Every bilinear neuron has an explicit symmetric rank-at-most-two matrix `Q_j`; `results.json` reports the effective rank and neuron participation ratio of each task's analytically reconstructed output tensors. The trained first-seed full model is saved in `full_model.pt`.
