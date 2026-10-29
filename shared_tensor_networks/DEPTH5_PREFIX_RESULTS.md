# Depth5 prefix alignment: first-layer information is insufficient

Nine privileged initialization controls close the current depth5 optimization detour. The result supports the relevance of deeper feature alignment in this toy family; it does not supply a method for discovering that alignment from labels.

## Experiment

Reuse the three calibrated depth5 teachers and frozen1024-example importance-weighted training sets. Starting from the original random student, copy the first1,3,or4 bilinear trunk layers from the teacher, including the corresponding fixed scale buffers. Keep the random readout. Independently normalize each modified student's head energies with32768 fresh importance-weighted calibration samples, then train all parameters for600Adam steps at1e-4, clip10. The copied layers are not frozen.

The readout never receives teacher weights. Teacher prefix copying, calibration and proposal fitting are privileged access. Inputs/labels stay fixed across these controls. This is a matched-data diagnostic, not a task-heldout comparison or equal-total-label learning claim. The original training proposal was fitted to the teacher and original random student; it may not cover the response regions of these modified students equally well.

Fresh final proposals and six16384-point batches estimate relative error, where1 is zero prediction. Model/data states, calibration summaries and reproducible evaluation seeds are archived.

## Results

|Teacher index|Copy1layer|Copy3layers|Copy4layers|
|---|---:|---:|---:|
|0|1.93910|0.66883|0.002178*|
|1|9.44603|0.32789|0.003273*|
|2|2.02472|0.12532|0.003886*|

Asterisk denotes failure of the absolute-energy/error batchCV<10% precision gate. All three full-prefix estimates fail: residual-error CVs are12.95–16.62%; one also has teacher-energy CV13.02%. They indicate very small errors relative to1, but do not support highly precise estimates or fine rankings. All prefix1/3 evaluations pass this descriptive gate. All9 fits complete600steps without nonfinite failures.

Prefix1 training losses drop to.316/.245/.0349, yet population error exceeds1 in every case. Its population function cosines are.366/.157/.476, so there is some alignment, but substantial excess/misaligned prediction energy. Prefix3 gives cosines.735/.843/.936 and errors below1. Full-prefix cosines are about.998. These are three fixed teachers, not independent proof of monotonic improvement for every teacher.

Copying a first layer alone is insufficient under this training configuration. Deeper teacher information helps substantially. However copying changes feature semantics, fixed buffers, output normalization and the relation between the model's error regions and its frozen training set. Do not attribute all differences uniquely to abstract manifold alignment. Nor does copying a full trunk demonstrate discovery of reusable computation.

## Audit

Independent NumPy replay verifies all initial/final objectives and regenerates all final evaluation points from saved axes/seeds. Prefix factors and buffers are checked exactly against the teacher; remaining factors against the original random student; readout scales against saved independent calibration estimates. Frozen-data hashes are unchanged. Maximum relative replay discrepancy9.80e-16. All producers/auditors terminal; no GPU work.

## Stop this detour and return to scaling

The bounded depth5 sequence has now tested coverage, output calibration, smaller rates, teacher-near starts and teacher-prefix starts. It shows a serious initialization/coverage/generalization limitation for unprivileged learning of degree32 functions. Do not continue indefinitely adding privileged rescues. Retain depth4/5 as documented difficult cases and return to the original sharing question at feasible depths2/3.

SCALING_GRID_DRAFT.json records the next representative design: d16/h8 and d32/h16, depths2/3, shared and independent teachers, matched-width and near-budget students, three teacher replicates, twelve source tasks and four whole-task holdouts. Expected72fits. It is a draft, not a launched or finalized protocol. Resolve normalization, proposal isolation from heldout tasks, exact budget widths and residual-focused evaluation before freezing it. Source-task prediction features remain the primary transfer representation for comparability; with larger hidden dictionaries, twelve source tasks need not span all target functions, so an oracle source-output-span baseline is essential.

The full goal remains incomplete: representative sharing/transfer scaling and synthesis still remain. No evidence here justifies a universal learnability advantage or circuit-recovery claim.

Artifacts: results/depth5_alignment. Scripts: run_depth5_alignment.py and audit_depth5_alignment.py.
