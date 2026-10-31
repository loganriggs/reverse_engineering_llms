# A separately fitted pre-RoPE rank-96 cut preserves the 546M model better

The [relocation control](PREROPE_CUT_TRANSFER_RESULTS.md) showed that taking an existing post-RoPE cut and moving it before RoPE makes some new-position errors much worse. That is an initializer test, not a comparison of the two reduction geometries at matched training objectives. Here both geometries are fitted **from their own pair features** with the same rank, sampled token/position pairs, weighting, and weighted-trace-gauge (WTG) algorithm. The separately fitted pre-RoPE cut wins the broad pair objective and downstream full-model KL. Specific held-out position gaps remain exceptions.

## Matched object and training

For each causal query/key pair, the post-RoPE geometry uses the two rotary QK1 vectors \((u_p,v_t)\). The pre-RoPE geometry uses \((u_pR_p^{-1},v_tR_p^\top)\), which preserves the *uncut* scalar QK1 score exactly even with the checkpoint's rounded rotary tables. Its maximum relative identity residual is below \(3.1\times10^{-16}\) in each fit. A rank-96 bond matrix is inserted in either coordinate system. The pre-RoPE matrix is one fixed matrix before query RoPE; its equivalent post-RoPE matrix varies with query position.

For each of three independent fit seeds (41, 43, 47), draw 32,768 uniform-vocabulary and 32,768 FineWeb-training-document cross-token pairs across causal positions in length-128 contexts. Exclude relative gaps **32 and 48** before fitting. The pair metric weights squared QK1-score error by the unchanged QK2 score squared and the exact OV-output norm squared. Each law is normalized by its own uncut target energy and receives half the objective weight. An independent selection sample has 8,192 pairs per law, with FineWeb documents 15,000–17,500; test documents are 17,500–20,000. WTG truncates the physical width-128 bond to rank 96 separately in the two geometries. Boundary eigenoperator residuals are below \(1.3\times10^{-8}\), and the saved cuts are numerically rank 96. WTG is a reproducible truncation method, not a global-optimum certificate.

The selection relative pair errors are:

| Fit seed | Fixed post-RoPE | Fixed pre-RoPE |
|---:|---:|---:|
| 41 | .05862 | **.03212** |
| 43 | .05724 | **.03201** |
| 47 | .05798 | **.02873** |

Two old rank-96 cuts, originally fitted in the post geometry, were also relocated and optimized in both geometries for 300 Adam steps. They remained substantially worse as pre-RoPE initializations (.086–.089 best selection error). Starting from the geometry-specific WTG cut gives .029–.032 before Adam; subsequent Adam steps degraded selection error in this setting. This is an **initialization and optimization warning**: the relocation failure did not establish that the pre-RoPE function class was worse.

## Independent pair and branch-edit tests

On a new 65,536-pair sample per law, three-seed mean relative pair MSE:

| Token law / branch condition | All gaps, post / pre | Excluded gaps 32 + 48, post / pre |
|---|---:|---:|
| Uniform / natural | .07651 / **.04520** | .07989 / **.05448** |
| Corpus / natural | .02918 / **.01337** | .03848 / **.01836** |
| Uniform / QK2 set to one | .09323 / **.05865** | .08952 / **.05812** |
| Corpus / QK2 set to one | .07793 / **.04725** | .08903 / **.05644** |

The QK2-constant test reweights the same QK1 pair error by OV norm alone; it does not retrain the cuts or establish downstream effects of a branch edit. The excluded-gap subset has 1,413 pairs per law, while the all-gap test has 65,536. All three seeds have the same ordering in each row.

The bulk average hides a concrete stress failure. On a **held-out query/key anchor panel** at key/query positions (16, 64), relative gap 48, the pre cut is worse: uniform-key complete-head response error averages **.02175 versus .01806** post; corpus-key error **.05602 versus .04994**. At held-out gap 32 and positions (32, 64), pre is much better under corpus keys (.03312 versus .04171), but slightly worse under uniform keys (.01211 versus .01173). These are different conditionings of the same function; the position-aware representation is not uniformly dominant.

## Downstream 18-layer model

The cuts are installed in layer 0, head 0 only, and all 18 layers are run. An identity pre-RoPE edit reproduces the native model (zero maximum logit difference in the main run). Two independent document samples of 24 length-128 windows per corpus give **6,144 next-token positions per corpus**. The three fit seeds use the same document windows for paired comparisons. Mean original-to-edited KL per token:

| Held-out corpus | Post-RoPE | Pre-RoPE | Post / pre |
|---|---:|---:|---:|
| FineWeb | 1.685×10⁻⁵ | **4.401×10⁻⁶** | 3.83 |
| WikiText-103 | 2.127×10⁻⁵ | **6.403×10⁻⁶** | 3.32 |
| TinyStories | 1.218×10⁻⁵ | **2.785×10⁻⁶** | 4.37 |

Pre-RoPE has lower KL in **all 18 paired fit-seed × document-sample × corpus comparisons**. Both edits are relatively faithful; this is a compression-fidelity advantage, not evidence of better language modeling. Mean next-token CE shifts are tiny and vary by corpus and sample. No semantic component, sparse gauge, or selective full-model group deletion has yet been established for the new cut.

## Why this geometry might help—and what remains open

One possible explanation is that raw query directions are concentrated but their post-RoPE orientations vary across positions, forcing a fixed post-RoPE rank-96 subspace to cover a broader union. A supporting **marginal** check on seed 41 finds that the top 96 weighted query-covariance eigenvectors contain **.926** of uniform-law energy before RoPE versus **.882** after; under corpus weighting, **.948** versus **.914**. Key-side differences are smaller. This is consistent with the explanation, but marginal spectra do not determine the paired bilinear optimum or establish the mechanism of the WTG gain.

The next factorial test should fit uniform-only, corpus-only, and mixed metrics in both geometries, then add weight, activation, ordinary-response, and intervention-aware legibility priors *within cuts that clear the same KL and branch-fidelity gates*. Position-covariant components are now technically viable on this head; whether they expose reusable, human-legible computation remains untested.

Reproduce with `flagship_matched_rope_fit.py --n-train 32768 --n-val 8192 --steps 0 --wtg-only --seed 41` (and seeds 43, 47), followed by `flagship_matched_rope_bulk_test.py --n-pairs 65536`, `flagship_matched_rope_full_model.py --n-docs 24 --seeds 23 29 --model-seeds 41 43 47 --corpora fineweb_test wikitext103 tinystories`, and `flagship_matched_rope_spectra.py`. The companion JSON/NPZ files in `full_width_results/` store per-seed cuts, fit diagnostics, pair tests, full-model records, and spectra.
