# Mixed natural-carrier bilinear pilot and OV reductions (2026-11-04)

## Why this experiment

The [surname-only model](FINEWEB_SURNAME_COPY_PILOT.md) nearly fit its training edits but generalized poorly. The [joint-coverage audit](CORPUS_SURNAME_JOINT_COVERAGE.md) found that a subset of natural FineWeb contexts carry repeated surnames alongside bracket, repeated-code, or spelling-style rule sites. This pilot trains on ordinary prose from **every** source document and samples marked sites from the same source-disjoint corpus. It tests whether a wider task mix yields a behaviorally learned rule suitable for weight-based decomposition.

The [cache builder](build_corpus_surname_joint_cache.py) takes FineWeb sample-10BT source offsets 100000–109999 for training and 120000–121999 for validation/test, splitting the latter by full-source SHA parity. Exact full-source hashes and preview SimHash screen duplicates across splits. Each accepted source supplies one random ordinary 256-token background chunk. At most one local site per family/source is separately indexed. For an eligible PERSON surname, the [paired sampler](corpus_surname_joint_dgp.py) changes all copies to one of two same-length surnames and keeps other sampled edits identical. It requires a 20+ word mention gap and a one-BPE source/target within the same context. Bracket/list edits have 0.5 application probability; literal code and spelling-style occurrences are tagged without changing their source text. These are research-chosen rates, **not measured natural task frequencies**. All sampled training rows use full next-token cross entropy, with extra loss at tagged rule sites; the eight-row batch allocation is two background, two surname arms, two bracket, one code, one style. Training therefore intentionally reweights rare tasks, while the background pool remains source-unconditioned.

The full [aggregate manifest](results/corpus_surname_joint_train10k_test2k_manifest.json) has 10,000/972/1,028 accepted train/validation/test sources and as many background rows. Training has 912 paired surname examples, 1,268 bracket rows, 510 code rows, 428 style rows, and 44 list rows. Validation/test each have 81 surname pairs; test has 131 bracket, 46 code, 34 style, and 7 list rows. A surname pair has another aligned site in 128 train and 8 test examples; this is diverse but still sparse at the **same-context** level. Raw prose and token caches remain local. The original corpus distribution is unknown, so there is no exact all-token oracle. Surname, bracket, list, and literal-code sites have local rules; the spelling-style cue is a probabilistic association, not a deterministic target oracle. PERSON tagging is heuristic, not verified identity resolution.

## Matched small bilinear models

Both arms use the same depth-2, width-64, four-head RMS bilinear Transformer (6.58M parameters), 6,000 AdamW updates, seed 46451, training cache, batch allocation, and site weights. The only difference is an auxiliary loss in the second arm: for each paired surname edit it asks the two answer tokens to reverse their rank when the earlier surname changes. The auxiliary weight is 2; it does not replace next-token loss. Training is resumable, and the full [baseline](results/corpus_surname_joint_depth2_seed46451.json) and [paired-loss](results/corpus_surname_joint_depth2_pair2_seed46451.json) histories are saved every 1,000 steps.

| Step-6,000 behavior | Next-token/site loss only | Plus paired surname loss |
| --- | ---: | ---: |
| Test surname: both edited tokens outrank the opposite candidate | 0/81 | 15/81 |
| Test surname: both are full-vocabulary top-1 | 0/81 | 1/81 |
| Test surname: edit changes the two-token margin in expected direction | 47/81 | 59/81 |
| Test bracket site: full-vocabulary top-1 | 42/131 | 49/131 |
| Test ordinary background token NLL, first 128 rows | 7.435 | 7.431 |

The paired loss improves *relative* surname sensitivity without establishing reliable copying. On the first 128 training surname pairs, paired candidate ranking rises from 1/128 to 41/128, while full-vocabulary paired top-1 rises from 1/128 to 15/128. It also raises training surname-site NLL from 2.41 to 3.86 and lowers training bracket top-1 from 122/128 to 104/128; thus it changes the balance among objectives rather than offering a free improvement. These training-slice figures are diagnostic, not random whole-training estimates. Held-out code and style full-vocabulary top-1 remain near zero, so no code/style weight circuit is eligible for interpretation. List co-occurrence and test support are too sparse for a joint rule claim.

An [explicit bracket opener-edit evaluation](results/corpus_surname_joint_bracket_edit_depth2_6000.json) on 200 source-disjoint test examples is more revealing than observational top-1. It compares an original round-bracket passage with a same-source square-bracket counterfactual. Baseline/paired-loss models get the correct closing bracket above the alternative in **171/200 and 183/200 pairs**, and changing the opener moves the two-way closing-bracket margin in the expected direction in **196/200 and 199/200**. Full-vocabulary paired top-1 remains **39/200 and 40/200**. This establishes a substantial causal *two-choice* bracket response, while vocabulary competition limits full next-token accuracy. The test source range was monitored during exploratory training, so a new untouched range is needed for confirmation.

## Weight-based OV reduction on the learned bracket response

Using only the first 128 **training background** rows to estimate activation metrics, [the OV audit](results/corpus_surname_joint_depth2_pair2_ov_rank.json) replaces each head's `O_h V_h` product with rank-4, rank-8, or full-rank factors. It compares ordinary weight-matrix SVD to SVD weighted by the input-activation Gram matrix. QK, MLP, embeddings, and unembedding remain fixed. All methods are evaluated on the same 200 bracket counterfactuals. The full-rank refactor reproduces the teacher (effect mean absolute error 0.000003 and near-zero KL), checking the implementation.

| OV method | Paired two-choice bracket | Paired full-vocab top-1 | Bracket-edit effect MAE versus teacher | Site KL versus teacher |
| --- | ---: | ---: | ---: | ---: |
| Original teacher | .915 | .200 | 0 | 0 |
| Weight-only rank 4/head | .915 | .200 | 1.130 | .0710 |
| Activation-weighted rank 4/head | .910 | .235 | 1.128 | .0595 |
| Weight-only rank 8/head | .920 | .200 | .708 | .0240 |
| Activation-weighted rank 8/head | .915 | .200 | .579 | .0227 |

The activation metric gives modestly lower KL and rank-8 effect error, but neither basis has a clear causal-behavior advantage at these ranks. The preserved two-choice behavior is encouraging for compression; it does not show that singular modes correspond to a legible bracket circuit. Nor can a weight-only result on a model that has not learned surname copying reveal that task's algorithm.

## Fresh-source replication

After freezing both checkpoints and the rank settings, a new FineWeb panel at offsets 130000–131999 was fetched with the same 2,500-character previews. All 2,000 source documents passed exact full-source hash and preview SimHash Hamming≤3 screens against the earlier 12,000 sources. The [fresh behavioral result](results/corpus_surname_joint_fresh_130000_2000.json) contains 195 aligned surname pairs and 200 bracket counterfactuals. There was no training or checkpoint selection on this panel.

| Fresh behavior | Next-token/site loss only | Plus paired surname loss |
| --- | ---: | ---: |
| Surname paired two-candidate rank | 0/195 | 32/195 |
| Surname paired full-vocabulary top-1 | 0/195 | 4/195 |
| Surname edit moves two-candidate margin correctly | 108/195 | 139/195 |
| Bracket paired two-choice accuracy | 183/200 | 192/200 |
| Bracket paired full-vocabulary top-1 | 40/200 | 46/200 |

Thus the narrow result replicates: paired supervision improves relative surname sensitivity but does not yield reliable full-vocabulary copying, while bracket choice responds strongly to an opener edit. On the fresh panel's independent 200 bracket pairs, the [rank audit](results/corpus_surname_joint_depth2_pair2_ov_rank_fresh130k.json) uses the same training activation Gram and frozen rank settings. Weight-only/data-aware rank-4 effect MAE is 1.056/.938 with site KL .0728/.0569; rank-8 effect MAE is .607/.513 with site KL .0223/.0213. Both rank-4 methods retain .950 paired two-choice accuracy versus .950 for this evaluator's teacher subset. The activation metric again improves functional approximation modestly, with no demonstrated legibility advantage. This OV evaluator selects source-hash-odd examples from the fresh panel, while the behavioral table uses all eligible fresh sources; the two 200-pair bracket sets are therefore not identical.

## Decision

Keep the bracket response as a candidate for a fresh-source causal and weight-circuit replication. Keep surname as a partial transfer result and code/style as negative capability controls. Do not extend these same small-source runs merely to lower training loss: the previous surname-only run and this joint run both show substantial train/heldout gaps. The next data/model experiment should add independent sources, preserve ordinary-source sampling, and separately test whether paired causal supervision helps full-vocabulary generalization. The broader **fully sampled** multi-rule language DGP still needs a more varied semantic carrier; this natural-text conditional arm is complementary, not a substitute.
