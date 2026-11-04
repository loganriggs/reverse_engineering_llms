# Training distribution form and causal-only counterfactuals

The [joint natural-carrier pilot](CORPUS_SURNAME_JOINT_MODEL_AND_OV.md) puts controlled rule sites inside varied FineWeb prose, but its training minibatch duplicates a surname source passage across two counterfactual arms. That can affect both form diagnostics and effective sample diversity, even though each individual edited passage looks natural. This audit separates the **marginal text form** from **cross-example dependence** and tests a changed loss allocation at matched model width, seed, data, and training updates.

The [form audit](results/corpus_surname_joint_training_form_n512.json) samples 512 first-100-word windows per panel using the GPT-2 tokenizer and the training cache. The original eight-row allocation has two source-unconditioned background rows, two arms of one surname pair, two bracket rows, one code row, and one spelling-style row. The changed allocation has three background rows and only one surname arm in next-token CE; its counterfactual arm is forwarded separately for the paired causal-ranking loss **without** next-token CE. Both still use full-token CE on their eight LM rows and the same site weights. The source documents, rule-site cache, and paired loss weight 2 are unchanged.

| 512 matched 100-word windows | Word types | Top-ten word share | Cross-document repeated-trigram occurrence share |
| --- | ---: | ---: | ---: |
| Fresh FineWeb source previews, offset 130000 | 9,982 | .211 | .045 |
| Unselected training background chunks | 9,978 | .212 | .043 |
| Original allocation, both counterfactual arms as LM rows | 9,221 | .213 | .282 |
| Same family mix but one arm from each distinct name source | 9,971 | .213 | .065 |
| Causal-only counterfactual allocation | 9,962 | .213 | .074 |

The top-ten lexical mass is almost unchanged. The very high repeated-trigram share in the original allocation primarily records two near-identical **examples from one source** as separate documents; it is a training correlation, not proof that any single passage is ungrammatical. The comparison of sampled chunks to first-100-word fresh document previews also has a position confound. Nonetheless, it identifies a concrete way in which the task-balanced training process differs from unselected natural text.

Two depth-2, width-64, four-head RMS bilinear models were trained for 6,000 updates with the same paired surname loss weight 2. Only the treatment of the counterfactual arm changed. On the monitored test split, the original/causal-only models have surname paired two-candidate ranking 15/81 versus 8/81, bracket site full-vocabulary top-1 49/131 versus 54/131, and background token NLL 7.431 versus 7.290. On the separate [fresh panel comparison](results/corpus_surname_joint_fresh_130000_2000_onearm_comparison.json), surname paired ranking is **32/195 versus 12/195**, surname paired full-vocabulary top-1 **4/195 versus 1/195**, and bracket opener-edit paired two-choice **192/200 versus 187/200**. Fresh surname-context full-token NLL improves **7.322 to 7.211** for the causal-only allocation. The [training history](results/corpus_surname_joint_depth2_onearm_pair2_seed46451.json) is saved every 1,000 steps; the checkpoint and reconstructable cache remain local.

This is a tradeoff, not an overall win: removing the duplicate counterfactual from ordinary LM loss makes the training text distribution less correlated and improves ordinary token prediction, but weakens the behaviorally relevant edit response at this fixed task-sampling budget. The fresh range was used after the earlier model comparison to evaluate this variant, so it should be treated as exploratory for choosing any further hyperparameters. More independent source documents and a new confirmation panel are preferable to tuning loss coefficients on these monitored sources. Neither arm yet has reliable full-vocabulary surname copying, so no surname weight circuit should be claimed. This corpus-conditioned DGP also does not solve the separate fully sampled semantic-language realism problem.

## Depth-4 comparison on a second fresh panel

Before touching the next panel, a depth-4, width-64, four-head RMS bilinear model was trained with the original both-arm allocation and paired loss weight 2 for the same 6,000 updates. Its [training history](results/corpus_surname_joint_depth4_pair2_seed46451.json) uses the same cache and seed. The depth increase adds nonlinear computation steps; it does not change the source set or training objective. A new FineWeb panel at offsets 140000–141999 then passed full-source hash and preview SimHash Hamming≤3 screens against all 14,000 earlier sources. The [aggregate frozen-model comparison](results/corpus_surname_joint_fresh_140000_2000_depth_comparison.json) has 178 aligned surname pairs and 200 bracket edits:

| Fresh 140k–142k behavior | Depth 2, both arms | Depth 2, causal-only arm | Depth 4, both arms |
| --- | ---: | ---: | ---: |
| Surname paired two-candidate ranking | 32/178 | 11/178 | 48/178 |
| Surname paired full-vocabulary top-1 | 3/178 | 1/178 | 2/178 |
| Surname edit moves two-candidate margin correctly | 130/178 | 124/178 | 134/178 |
| Bracket opener-edit paired two-choice | 191/200 | 189/200 | 187/200 |
| Bracket opener-edit paired full-vocabulary top-1 | 52/200 | 37/200 | 40/200 |

Depth 4 improves **relative** surname ranking on this source range and lowers surname-site NLL from 7.82 to 7.10, but full-vocabulary paired prediction remains near zero. It also loses some bracket full-vocabulary accuracy. One seed and two fresh contiguous source ranges are not enough for a universal architecture conclusion. The evidence argues against more same-data training or sparsity analysis of a putative surname circuit; first improve independent source/task coverage and the actual full-vocabulary capability gate. The bracket two-choice response remains the valid candidate for mechanistic tests, with prior [weight/activation OV reductions](CORPUS_SURNAME_JOINT_MODEL_AND_OV.md) as baselines.
