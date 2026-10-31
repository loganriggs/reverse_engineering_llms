# Causal supervision helps sparse coding, but physical heads transfer better

The preceding [multi-composition benchmark](CONTEXTUAL_COMPOSITION_CANONICALIZATION_RESULTS.md)
trained competent eight-head bilinear attention models on four bit-majority,
six XOR, and six AND tasks. A [shared-dictionary follow-up](CONTEXTUAL_COMPOSITION_SHARED_GROUP_RESULTS.md)
showed no group advantage over the physical heads. Here we ask whether a
more relevant prior—preserving the model's **finite response to an input
edit**—can improve the same 12-atom, top-four sparse featurizer.

All featurizer fitting, checkpoint selection, and physical-baseline
calibration use only primitive and AND queries. **All six XOR queries are
withheld as a whole family.** The native model was trained on XOR; this is
an out-of-family *featurizer* test, not zero-shot model learning. No task
answer labels enter the featurizer objective. We freeze the native model
and use its logits as a causal-distillation target.

For each allowed-query sequence, we make a second sequence by toggling
one bit of one key token while leaving the query fixed. Let `h` and `h'`
be the native eight-head outputs and `f(h,q)` the frozen model's final
logit. We fine-tune the same flat 12-atom top-four autoencoder from the
same unsupervised checkpoint with combinations of:

- relative head reconstruction error on `h,h'`;
- relative logit error on `f(h,q),f(h',q)`;
- relative finite edit-path error on `f(h',q)-f(h,q)`.

The last term measures whether the featurizer preserves what the *full
model* changes in response to the token edit, rather than merely
reconstructing either endpoint. Every candidate must keep allowed-query
validation head error within 5% of its unsupervised starting checkpoint.
Checkpoint choice uses only that validation panel and the candidate's
predeclared objective. A candidate that never improves within the gate
falls back to the starting checkpoint.

## Exploratory sweep and independent confirmation

We swept output and finite-edit weights on model seeds 0–4. The larger
edit weights failed the reconstruction gate and fell back to baseline.
We then **fixed** output weight 0.3 and finite-edit weight 0.1 before
training five new models, seeds 5–9, and their separate flat
featurizers. All five new native models reach essentially 100% accuracy
on every task family. The table reports the independent seeds on fresh
XOR-query sequences; smaller error is better.

| Featurizer | XOR accuracy | XOR head error | XOR logit error | XOR finite edit-path error |
|---|---:|---:|---:|---:|
| Flat 12-atom reconstruction baseline | 95.63% | 0.098 | 0.188 | 0.317 |
| Same flat model, output weight 0.3 | 96.06% | 0.096 | 0.166 | 0.296 |
| Same flat model, finite-edit weight 0.1 | 96.00% | 0.102 | 0.164 | **0.288** |
| Physical heads, fixed top 3 | 93.50% | 0.123 | 0.245 | 0.410 |
| Physical heads, fixed top 4 | 97.75% | 0.059 | 0.112 | 0.215 |

The small causal weights improve the flat model on average while staying
within the **seen-query** reconstruction gate in all five confirmation
seeds. Finite-edit weight 0.1 improves XOR edit-path error over its own
flat baseline in four of five new seeds. Under a shifted token-frequency
distribution, its XOR edit-path error is 0.333 versus 0.370 for the flat
baseline; output weight 0.3 gives 0.341. This is useful evidence that a
causal prior can transfer beyond the specific edit examples used for
training. It is not yet a win over the physical-head representation.

## A genuinely matched physical-head control

Fixed top-three and top-four physical-head baselines have different
reconstruction errors. We therefore calibrated a physical gate on the
**allowed-query validation head states only**. For each model and learned
featurizer, it keeps four heads plus the fifth when that head's energy
exceeds a threshold. The threshold is chosen to match the featurizer's
paired-state head error; it uses no task labels, logits, edit-path scores,
or XOR examples. Validation-error mismatch is below 7×10⁻⁶. It keeps a
mean of 4.24–4.27 physical heads in the confirmation models, compared
with four learned atoms.

| Independent seeds 5–9 | Allowed-query validation head error | XOR head error | XOR accuracy | XOR edit-path error |
|---|---:|---:|---:|---:|
| Flat, reconstruction only | 0.0509 | 0.098 | 95.63% | 0.317 |
| Matched physical gate | 0.0509 | 0.038 | 98.75% | 0.167 |
| Flat, output weight 0.3 | 0.0513 | 0.096 | 96.06% | 0.296 |
| Matched physical gate | 0.0513 | 0.038 | 98.73% | 0.167 |
| Flat, finite-edit weight 0.1 | 0.0527 | 0.102 | 96.00% | 0.288 |
| Matched physical gate | 0.0527 | 0.039 | 98.66% | 0.172 |

The physical gate has better XOR prediction and edit-path fidelity in
**all five** independent seeds. The comparison is matched on the seen
family's reconstruction error, not the held-out family's: that is the
point of the transfer test. The learned dictionary's head error roughly
doubles from allowed validation to XOR test, whereas the physical
head gate's XOR error is lower than its allowed-validation error. The
same direction holds under shifted token frequencies. A fixed four-head
control also beats the four-atom learned models on XOR, though its
allowed-query reconstruction error is not matched.

This result does **not** prove individual physical heads are semantic
features or canonical across seeds. Their indices may permute, and
their better generalization could reflect the architecture's separation
of task-conditioned computations. It does show that an intervention-aware
training loss can improve a sparse basis without repairing missing
composition-family coverage. For this model, the physically learned
head decomposition is a strong function-level prior that the tested
data-fitted dictionary and overlapping groups have not surpassed.

Stop penalty tuning on this toy. A next method should identify
independently verified operations, or preserve *multiple* withheld
intervention families across tasks, then test whether its groups transfer
better than physical heads at equal active-coordinate and fidelity
budgets. On the 546M model, use the same separation of reduction fidelity,
finite causal fidelity, and human legibility; none of these toy results
names a real-model attention circuit.

Reproduce the exploratory sweep with `contextual_composition_causal_prior.py`,
the fixed-hyperparameter new-seed panel with
`contextual_composition_causal_confirmation.py`, and the physical
calibration with `contextual_composition_matched_physical.py`. Matching
JSON files contain per-seed checkpoint choices, train/validation/test
separations, uniform and shifted results, and calibrated thresholds.
