# Round 3 conclusion: noun/verb counterfactual pilot

This single-seed adaptation added a second counterfactual family before locking the large confirmation:

- **Operation / verb:** identical noisy content, different operation instruction.
- **Class / noun:** same operation, independent examples from adjacent prototype classes.

The held-out accuracy was 100%. For both counterfactual families, raw neurons were close to random rotated coordinates, while fitted PCA subspaces were substantially more efficient. For example, at rank 16:

| Family | Layer | Raw | Rotation | PCA |
|---|---:|---:|---:|---:|
| Operation | 1 | 0.235 | 0.230 | 0.765 |
| Operation | 2 | 0.295 | 0.292 | 0.950 |
| Class | 1 | 0.291 | 0.293 | 0.946 |
| Class | 2 | 0.236 | 0.222 | 0.879 |

This is still exploratory because the class pairing and endpoints were added after seeing Round 2. The next round is locked before observing any 100-class results.

