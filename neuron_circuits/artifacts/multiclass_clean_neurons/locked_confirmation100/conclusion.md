# Round 4: locked 100-class confirmation

All 10 preregistered seeds converged and achieved 100% accuracy on 128,000 fresh confirmation examples in total. Mean confirmation cross-entropy was `1.68e-4`.

## Preregistered endpoints

| Counterfactual | Layer | Raw minus rotation | 95% CI | PCA minus raw | 95% CI |
|---|---:|---:|---:|---:|---:|
| Operation | 1 | 0.044 | [0.032, 0.055] | 0.496 | [0.482, 0.510] |
| Operation | 2 | 0.063 | [0.050, 0.075] | 0.399 | [0.372, 0.425] |
| Class | 1 | 0.050 | [0.046, 0.054] | 0.174 | [0.164, 0.185] |
| Class | 2 | 0.050 | [0.038, 0.061] | 0.247 | [0.236, 0.258] |

No family/layer met the preregistered strong clean-axis criterion (`CI lower > 0.10`). Every family/layer met the distributed-subspace criterion (`PCA-minus-raw CI lower > 0.10`). The stricter falsification rule for any axis benefit (`CI upper < 0.05`) was narrowly missed: estimates indicate a small but repeatable 4–6 percentage-point raw-axis advantage.

For operation counterfactuals, PCA reached 80% KL recovery with 16 directions and 90% with 32 (one seed needed 64 in layer 1). Raw and rotated coordinates needed all 256. For class counterfactuals, PCA generally needed 64–256 directions; raw and rotation again needed all 256.

## Interpretation

The result supports **weak axis alignment plus strong distributed structure**, not a picture in which a few individually privileged neurons carry the computation. Operations are more compressible than class identity, but there are only eight operations versus 100 classes, so that comparison does not by itself show that neurons are intrinsically “verbs, not nouns.”

The sealed numerical output is [locked_summary.json](locked_summary.json).

