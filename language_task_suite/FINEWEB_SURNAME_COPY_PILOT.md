# FineWeb-carried surname-copy pilot (2026-11-04)

## Question and scope

Can we put a controlled long-context rule inside varied real prose, train a small bilinear language model on it, and only then inspect weights for that rule? This pilot uses unchanged FineWeb passages except for replacing every occurrence of one repeated surname. A source name such as `Alice Smith ... Smith` becomes `Alice Jones ... Jones`, and a paired counterfactual uses another surname. The intended test is whether changing the earlier name changes the later surname prediction. The rule is **surface-string reuse**, not a verified person-coreference rule. It supplies a known target at selected sites, not a complete next-token oracle for the surrounding prose.

This complements the fully sampled DGP work and the earlier [first-name corpus carrier](CORPUS_TRANSFORMED_NAME_REUSE_PILOT.md) and [sports alias pilot](SPORTS_ENTITY_ALIAS_PILOT.md). It does not replace the goal of a diverse sampled generator with known interacting rules.

## Construction and audits

`fineweb_name_copy_dgp.py` selects source documents containing a spaCy `PERSON` span matching a given name and surname, a later exact surname string, and at least 60 characters between first and last occurrence. It rejects several common non-name uses, conflicting detected given names, lowercase uses, and URL-like occurrences. It uniformly samples from 182 common replacement surnames absent from the passage, then replaces every exact occurrence. The stricter model cache requires at least 20 words from the preceding surname mention to the scored target, both alternatives to be distinct single GPT-2 BPE tokens, and the target to fit in a 256-token context. Full source hashes plus a 64-bit SimHash near-duplicate screen separate train/validation/test sources. Raw FineWeb prose, review excerpts, token caches, and model checkpoint remain local under `neuron_circuits/artifacts`.

A broader capitalized-word regex was rejected during manual review: it substituted words such as *Christmas* and *American* as if they were people. The PERSON filter improved examples but is still heuristic. Manual review caught `Jesus Christ` becoming `Jesus Daniels`; the final filter excludes that surname and some other obvious false positives. Historical names, names embedded in brands, and other semantic errors may remain. A possessive-collision bug was also fixed: replacement `Martin` is excluded if the source contains `Martin's`.

Before the last exclusion-list refinement, the independent 10,000/2,000-source audits found 1,747/312 eligible documents and 4,428/811 later surname occurrences; every tested edit and counterfactual reversed byte-for-byte to its source. Median first-target distance was 43/42 words. These are **pre-refinement** audit counts. On 256-document first-100-word panels, selected original versus edited text had top-ten word mass .2121/.2121 and cross-document repeated-trigram occurrence share .02884/.02876 (10,000-source panel). The near identity is expected because only surnames change. Form is close to its natural carrier by construction; this does not establish that surname substitutions or the selected subset preserve the natural distribution, nor that the uniform replacement-name prior is Zipfian.

The final filtered, long-gap cache contains 1,100 distinct training sources / 2,171 paired examples, 93 validation sources/pairs, and 100 test sources/pairs. The source ranges are FineWeb sample-10BT offsets 100000–109999 and 120000–121999. Held-out test gaps run 20–163 words (median 43). The aggregate [cache manifest](results/fineweb_surname_copy_cache_manifest_10k_2k_gap20.json) records counts, gap bands, source hashes, and configuration without source text.

## Bilinear model gate

A depth-2, width-64, four-head bilinear Transformer with RMS normalization and 6.58M parameters was trained from scratch for 6,000 AdamW steps. Loss is next-token cross entropy over all available context tokens, with a 500-fold weight at the marked later surname site. The [full result history](results/fineweb_surname_copy_depth2_weight500_seed45251.json) is saved every 500 steps; the local optimizer/model/RNG checkpoint reached step 6,000 before the computer stopped.

| Held-out test metric | Step 0 | Step 6,000 |
| --- | ---: | ---: |
| Both edited surnames rank above the other candidate | 0/100 | 27/100 |
| Both edited surnames are full-vocabulary top-1 | 0/100 | 2/100 |
| Earlier-name edit moves the two-candidate logit difference in the right direction | 60/100 | 90/100 |
| Mean later-surname negative log probability | 10.85 | 9.11 |

The model has learned a directional response to the controlled edit, but reliable copying is absent. At step 6,000 the first 100 **training** pairs score 40/100 for paired candidate ranking and 24/100 for paired full-vocabulary top-1, versus 27/100 and 2/100 on test. Thus part of the remaining gap is generalization, and the model is not even fully fitting this initial training slice. The test surname NLL improved early and then worsened as training proceeded; ordinary token loss and controlled-rule fidelity do not move together. These figures are one seed, and the first-100 training slice is a diagnostic rather than a random training-panel estimate.

No weight circuit or canonicalization success is claimed for this checkpoint. The behavioral gate is insufficient. A useful next comparison is a larger or pretrained bilinear model and a stronger corpus-diverse training set, with the same whole-source heldout paired edits and explicit train/test task-site curves. The uniform surname prior should be compared with a corpus-frequency prior; manual semantic review and target-conditioned form audits should continue. Weight-only and data-aware decomposition should be attempted only on rules the model demonstrably performs, while this run can serve as a partial/negative control.

## Reproduction

Use `fetch_fineweb_long_panel.py` to obtain the two source ranges into **local-only** preview JSON; the Hugging Face source is `HuggingFaceFW/fineweb`, config `sample-10BT`. Install dependencies including PyTorch, `datasets`, `tokenizers`, `spacy==3.8.18`, `en_core_web_sm==3.8.0`, and `faker==40.44.0`. Run `audit_fineweb_name_copy_dgp.py` on each preview, then `build_fineweb_surname_copy_cache.py` with `--min-gap-words 20 --n-ctx 256`. Run `train_fineweb_surname_copy_bilinear.py` with `--depth 2 --d-model 64 --heads 4 --steps 6000 --batch-size 8 --site-weight 500 --lr 0.0005 --seed 45251`; point `--cache` and `--state` at local artifact paths. The trainer resumes from its saved step if interrupted.
