## H737 — Random coordinates explain much of the high-capacity cross-task success

We evaluated signed adaptive Top-K in three fixed random orthogonal bases of the unwhitened write-PCA coordinates, at K=8,16,32,64. Gaussian QR with diagonal-sign correction generates each basis; the same basis is used at every K. Adaptive PCA-coordinate Top-K is a second control. Neither fits sparse decoder directions or uses task labels. All controls use the same training mean/write coordinate system, 128 directions and active budgets as the learned models. They are approximation controls, not parameter-matched learned models. This differs from H724's random rotation after whitening.

Natural validation relative write MSE (4,096 vectors/head):

| Head / K | Learned scalar | Adaptive PCA axes | Random orthogonal axes |
|---|---:|---:|---:|
| Agreement / 8 | 0.3465 | 0.3691 | 0.6817 |
| Agreement / 16 | 0.2319 | 0.2316 | 0.5033 |
| Agreement / 32 | 0.1223 | 0.1106 | 0.2807 |
| Agreement / 64 | 0.0373 | 0.0253 | 0.0733 |
| Recall head / 8 | 0.4420 | 0.5545 | 0.6792 |
| Recall head / 16 | 0.3365 | 0.3748 | 0.5007 |
| Recall head / 32 | 0.1927 | 0.1868 | 0.2788 |
| Recall head / 64 | 0.0565 | 0.0434 | 0.0726 |

The learned dictionaries beat random directions in natural reconstruction, particularly at low budgets. They do not uniformly beat adaptive PCA, especially at higher budgets. Natural reconstruction and functional effects must still be assessed separately.

Fresh-panel retained exact-head task effect at K=64, competent strata only:

| Head and task / format | Learned scalar | Adaptive PCA axes | Random orthogonal axes |
|---|---:|---:|---:|
| Agreement, agreement A | 98.2% | 100.0% | 92.3% |
| Agreement, agreement B | 99.5% | 99.7% | 90.3% |
| Recall, capital recall B | 96.5% | 95.7% | 91.0% |
| Recall, binding A | 90.4% | 92.7% | 90.1% |
| Recall, binding B | 92.2% | 96.7% | 93.4% |
| Recall, copying A | 92.4% | 93.7% | 92.1% |
| Recall, copying B | 93.4% | 93.6% | 93.8% |

These are mean margin-effect ratios, not accuracy; learned and random results average their respective three seeds. Near-zero-effect ratios remain suppressed, and all failed-competence cells remain diagnostic. High-K random coordinates preserve approximately as much recall-head binding/copying effect as learned directions. Individual-effect errors support the same limited conclusion: at K=64, relative RMS errors for learned versus random directions are 12.5% versus 13.0% (binding A), 12.4% versus 11.6% (binding B), 8.9% versus 8.9% (copy A), and 8.1% versus 8.0% (copy B). No precise significance claim follows from three random bases and this finite templated panel.

There are stronger low-budget distinctions. For agreement at K=8, adaptive PCA retains approximately 95% in both formats, learned features 88.5%/94.3%, and random axes only 22.8%/26.2%. The agreement computation is genuinely aligned with the natural-data coordinate structure measured here. For competent capital recall at K=8, learned features retain 65.1%, versus adaptive PCA 47.1% and random axes 29.4%. Learned low-budget directions help factual recall but do not by themselves establish semantic units or cross-task reuse.

Output KL adds nuance. At K=64, learned versus random mean KL is 0.0026 versus 0.0043 on capital recall, approximately 0.0007 versus 0.0007 on binding A, 0.0008 versus 0.0009 on binding B, and about 0.0004 versus 0.0004–0.0005 on copying. Adaptive PCA often improves it further. Thus learned directions retain some advantages, but high-capacity functional preservation alone is weak evidence of discovered structure.

Correctness: two tests pass, including exhaustive enumeration of the best orthogonal support on a small example and seeded/full-coordinate round-trip checks. Runtime orthogonality error was at most 2.38e-7, full coordinate round-trip relative error at most 2.49e-7, and all active budgets were exact. All input/source hashes remained unchanged and task/KL values were finite. Identity task replay maximum error was 6.68e-6; identity native KL was zero. The run completed 15,708 full-model example forwards, using no new natural documents or original reserved-test evaluations.

Decision: do not promote K=64 learned atoms to fundamental units on the strength of their reconstruction or whole-head transfer. Much of their binding/copying preservation is generic high-dimensional approximation. Agreement's low-dimensional alignment and the recall head's low-budget factual advantage are better targets for interpretation.

Next move from compression to the head's actual operation. Inspect and reproduce L13H0's signed bilinear attention pattern and its mixed local/persistent-bus values, then cross donor/receiver routing and values on structurally position-matched H732 pairs, with full reconstruction and identity controls. Report unequal-length exclusions independently of task performance. This directly distinguishes changes in which values are read from changes in the values themselves, and tests the shared-retrieval-operation/distributed-value-code hypothesis across recall, binding and copying. Reuse the established reader-decomposition machinery, but do not treat prior L14H4/L8H3 findings as evidence for L13H0. L11H3 provides the selective agreement comparison. Meaningful manifold and selective-component claims still require evidence beyond compression; the sentiment diagnostic remains outstanding.
