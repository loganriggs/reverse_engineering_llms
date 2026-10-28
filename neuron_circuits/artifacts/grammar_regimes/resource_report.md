# Measured resource report

Hardware: NVIDIA GeForce RTX 5080, 16,303 MiB.  Software execution used the
repository `.venv`, PyTorch CUDA, and bfloat16 autocast during training.

## Training measured in this run

| run group | cells | total training seconds | peak allocated MiB |
|---|---:|---:|---:|
| initial SwiGLU depth × load pilot | 9 | 300 | 178 |
| initial bilinear depth-2/4 × load pilot | 6 | 197 | 178 |
| seed-1 depth-4 confirmation, both MLPs | 6 | 143 | 179 |
| seed-2 depth-4 confirmation, both MLPs | 6 | 271 | 179 |
| total retained grammar runs | 27 | 910 (15.2 min) | 179 |

The 60 MiB grammar artifact directory includes checkpoints, manifests,
predictions, activation/constraint summaries, tables, and the final figure.
Checkpoints account for approximately 46 MB.

The measured time excludes the two rejected generator/calibration designs,
test execution, analysis development, and report writing.  The completed
causal audits took seconds per condition at their current panels; the
100-rotation axis audit was the largest audit batch but remained small relative
to an hour on this GPU.

## Planning estimates

- Repeating the final 18-cell depth-four confirmation on comparable hardware:
  about 10–20 minutes of training plus audit time.
- A 54-cell small-model grid with these sequence lengths and widths: roughly
  0.5–1.5 GPU-hours including the current audits, with substantial uncertainty
  from gate failures that run to the 20,000-step ceiling.
- Storage scales at roughly 1.7 MiB per retained checkpoint in this setup;
  streamed summaries keep activations from dominating disk or VRAM.
- The 500M FineWeb model should be budgeted separately.  Existing local work
  establishes that one-layer/head-at-a-time probes fit on a 16 GB GPU, but this
  run did not benchmark a new 500M protocol and therefore makes no new
  wall-clock promise for it.

These are measurements and local extrapolations, not cross-GPU guarantees.
