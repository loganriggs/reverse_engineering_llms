

## H326 — Native versus inherited versus refitted output weights

Compare identical frozen H324 ranked/random factor subsets with native Down output rows, inherited H323 full-fit rows, and H324 subset-refit rows. No new fitting or selection. Evaluate discovery896:1024; save768:1024 rows. Targets remain missing residual cross responses after the compiler and partner baseline. Consequently native Down is a known-weight witness that omits the compiler cross correction, not an exact target oracle. Ranking was based on inherited contributions and is not neutral between readouts.

|Subset|Target|Native fidelity|Inherited fidelity|Subset-refit fidelity|Inherited−native (pp), conditional95%|
|---|---|---:|---:|---:|---:|
|ranked256|actual|8.76%|9.83%|3.01%|1.07 [0.86,1.31]|
|ranked256|angle|8.58%|9.62%|3.02%|1.04 [0.75,1.34]|
|ranked512|actual|16.58%|17.34%|8.39%|0.76 [0.52,1.03]|
|ranked512|angle|16.22%|16.89%|7.88%|0.68 [0.40,0.98]|
|ranked1024|actual|29.03%|30.14%|14.13%|1.12 [0.75,1.50]|
|ranked1024|angle|28.63%|29.41%|13.40%|0.78 [0.37,1.21]|
|ranked2048|actual|51.50%|52.01%|33.50%|0.51 [-0.02,1.09]|
|ranked2048|angle|51.09%|50.65%|31.19%|-0.43 [-0.97,0.10]|
|random256|actual|5.41%|5.12%|1.61%|-0.29 [-0.38,-0.18]|
|random256|angle|5.56%|5.22%|1.56%|-0.34 [-0.47,-0.22]|
|random512|actual|10.98%|10.43%|3.34%|-0.55 [-0.68,-0.38]|
|random512|angle|11.04%|10.40%|2.85%|-0.64 [-0.76,-0.50]|
|random1024|actual|21.63%|21.23%|9.52%|-0.40 [-0.69,-0.06]|
|random1024|angle|21.87%|20.96%|8.99%|-0.91 [-1.17,-0.64]|
|random2048|actual|42.03%|41.65%|23.90%|-0.38 [-0.78,0.08]|
|random2048|angle|42.17%|41.03%|21.46%|-1.15 [-1.71,-0.63]|
|full|actual|84.79%|90.69%|90.69%|5.90 [4.27,7.77]|
|full|angle|86.84%|89.97%|89.97%|3.12 [1.62,4.64]|

Locked gates: inherited1024_gain5: FAIL, native1024_80: FAIL. inherited1024_gain5 requires>=5pp inherited advantage over native with positive paired95% lower bounds in both families; native1024_80 requires>=80% native fidelity in both.

**Limits.** Native and inherited full readouts have different training objectives. A small inherited correction at one subset size does not identify its semantic role, and near-equal energy fidelity does not imply equal predictions. Reported correction energy is saved in JSON. This is one ranked order and one random order on reused documents. Bootstrap conditions on all training, selection and adaptive research choices. No intact-model or causal semantic claim.

**Audit.** Every inherited and refitted error replays H325; maximum relative discrepancy1.62e-16. Full/native witness is also compared against H323 below by the report audit. Frozen reference, full-readout and subset hashes verified; native weights unchanged.

**Next discriminator.** Fresh confirmation should compare full native-factor learned readout, native Down, and inherited1024/2048 subsets under locked evaluation. If these replicate, study response-weighted mixtures and the width needed across layers and tasks. Broad research goal active.
