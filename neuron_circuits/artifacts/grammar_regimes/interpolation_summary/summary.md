# Grammar-to-memorization interpolation

Filled graph markers pass the original recall plus unseen-template gate. Open markers are exploratory audits on naturally correct examples and are not used as matched-fidelity confirmation.

## Seed-level behavioral table

| MLP | arbitrary fraction | gate passes | train recall | exception recall | new structured pairs | unseen template |
|---|---:|---:|---:|---:|---:|---:|
| swiglu | 0 | 3/3 | 0.999 | n/a | 0.999 | 0.965 |
| swiglu | 0.05 | 0/3 | 0.981 | 0.663 | 0.979 | 0.888 |
| swiglu | 0.2 | 1/3 | 0.994 | 0.986 | 0.802 | 0.885 |
| swiglu | 0.5 | 2/3 | 0.988 | 0.984 | 0.431 | 0.929 |
| swiglu | 1 | 3/3 | 0.976 | 0.976 | n/a | 0.953 |
| bilinear | 0 | 3/3 | 0.998 | n/a | 0.997 | 0.968 |
| bilinear | 0.05 | 2/3 | 0.981 | 0.657 | 0.981 | 0.941 |
| bilinear | 0.2 | 0/3 | 0.995 | 0.990 | 0.763 | 0.866 |
| bilinear | 0.5 | 2/3 | 0.983 | 0.977 | 0.429 | 0.926 |
| bilinear | 1 | 2/3 | 0.981 | 0.982 | n/a | 0.945 |

## Interpretation

- Rule generalization declines smoothly once exceptions become numerous: near 1.00 at 0%, about 0.98 at 5%, about 0.78 at 20%, and about 0.43 at 50%.
- Exception acquisition is not smooth across seeds at 5%: some models preserve the rule and largely ignore rare exceptions, while others fit exceptions but become template-specific. This is a competing-solution or rare-exception optimization effect.
- At 20%, exceptions are usually learned, while structured generalization is damaged but far from destroyed. This supplies the requested intermediate regime.
- Representation curves use all seeds for visualization, but open points remain exploratory because prompt-interface brittleness can contaminate mechanistic comparisons.

## Figures

- `behavior_interpolation.png`: task performance and gate status.
- `representation_interpolation.png`: neuron transfer, instance state, constraint breadth, and semantic predictability.
- `patch_interpolation.png`: direct explanation of full/native/rotated/random patch success.
