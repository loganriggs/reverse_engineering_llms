# Full-model propagation of first-layer attention-bond cuts

The [sequence-output study](SEQUENCE_OBJECTIVE_RESULTS.md) showed that a rank-64 QK1 bond cut fitted to the summed layer-0/head-0 response better preserves that head than pair-response cuts. This experiment asks whether that ordering survives the **entire 18-layer language model**. It compares the same saved width-128-to-64 bond cuts, inserted after RoPE in the first QK score branch of layer 0, head 0. The second QK branch, all other heads, all OV weights, and layers 1–17 remain at their checkpoint values. A zero-bond edit supplies a scale reference.

The evaluator loads the saved 546M checkpoint into the repository's bilinear GPT implementation and patches only this head's first attention-score branch. The unedited custom logits reproduce the model's native cross-entropy to below `1e-5`; inserting \(X=I\) gives exactly the same logits (`0.0` maximum absolute difference in the recorded run). This verifies the full-model edit path. It uses CPU only because another project occupied the GPU.

FineWeb documents 0–14,999 supplied the fitting data; documents 15,000–17,499 supplied checkpoint selection for the direct fits. Final FineWeb documents 17,500–19,999 are disjoint from both. WikiText-103 and TinyStories are separate tokenized corpora. For each corpus and each of seeds 23 and 29, the evaluator samples 24 length-129 windows, predicts each next token from the preceding 128 tokens, and compares every cut on the **same windows**: 48 windows and 6,144 scored positions per corpus, 18,432 positions total.

The main fidelity measure is mean \(D_{\mathrm{KL}}(p_{\mathrm{original}}\Vert p_{\mathrm{edited}})\) across token positions. This compares the full next-token distributions, rather than a single target token. Values below are nats per token.

| Bond condition | FineWeb KL | WikiText-103 KL | TinyStories KL |
|---|---:|---:|---:|
| **Zero head-0 QK1 bond** | .001638 | .001935 | .001000 |
| Half uniform–corpus WTG, rank 64 | .0000768 | .0000586 | .0000399 |
| Thirds metric WTG, rank 64 | .0000872 | .0000567 | .0000416 |
| Direct pair-metric refit, rank 64 | .0000641 | .0000502 | .0000319 |
| **Natural sequence-output fit, rank 64** | **.0000306** | **.0000410** | **.0000209** |
| Sequence + 25% QK2 intervention fit, rank 64 | .0000343 | .0000414 | .0000233 |

The natural sequence fit has the lowest full-model KL in **both independent sampling seeds on all three corpora**, and the intervention-aware sequence fit is close behind. The pair refit improves on the initial WTG cut, so additional optimization matters; the further sequence-fit improvement is consistent with optimizing the function that the downstream model actually receives. On these samples, completely removing this head causes roughly 47–54 times the KL of the natural sequence fit, depending on corpus. The small absolute KL for rank 64 therefore does not merely mean that the head is dispensable under this measure.

The same run reports next-token cross-entropy (CE) and original-versus-edited argmax flips:

| Corpus | Original CE | Zero-bond CE change | Natural sequence-fit CE change | Zero / sequence-fit flip fraction |
|---|---:|---:|---:|---:|
| FineWeb | 3.2767 | +.00327 | −.000092 | .0186 / .0029 |
| WikiText-103 | 4.3248 | +.00387 | −.000215 | .0241 / .0047 |
| TinyStories | 2.7205 | +.00161 | −.000010 | .0163 / .0023 |

These finite-sample CE differences are much smaller than the baseline CE and can change sign with corpus or cut. They do **not** show that truncation improves language modeling. The zero baseline raises CE and flips substantially more top tokens than the rank-64 cuts. The KL ordering is the clearer functional result here; the individual seed records are retained so the two-seed stability can be inspected.

There are important boundaries. This tests one head, one retained rank, length 128, and 18,432 next-token positions. The intervention-prior variant was trained against an internal QK2-constant edit, but this full-model run evaluates **natural** text only; it does not propagate that branch intervention through the model. None of the bond directions is given a semantic label, and lower KL is not evidence of a unique or human-legible decomposition. An [exact-gauge sparse-prior follow-up](EXACT_GAUGE_SPARSITY_RESULTS.md) now compares factor-weight and activation sparsity while holding this cut and its downstream behavior fixed; its weak cross-seed alignment prevents treating sparsity alone as a canonical circuit.

Reproduce with `flagship_full_model_eval.py` using `--n-docs 24 --length 128 --batch-size 8 --threads 4`, and the separate `--cuts head0_zero` pass. The evaluator uses fixed seeds 23 and 29. The JSON records are `full_width_results/head0_full_model_l128_n24.json` and `head0_full_model_l128_n24_zero.json`; the earlier one-document-per-corpus smoke test and eight-document pilot are not used for the table. External model and corpus files are referenced rather than copied into this repository.
