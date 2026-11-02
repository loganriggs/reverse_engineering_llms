# Longer document-world v2: generator and first models (2026-11-02)

## What changed

`document_world_v2.py` preserves the v1 latent state and oracle but realizes each world as longer story, log, or summary sentences. It fixes the indefinite article before `orange`, uses `in`/`on` with locations, gives location answers a grammatical determiner, and varies phrasing without changing event semantics. This is a **versioned new distribution**; v1 checkpoints continue to use v1 wording.

The v2 verifier checked 32,000 examples for oracle agreement, non-conflicting targets, causal edits that change the answer, nuisance edits that preserve it, no exact evaluation-prefix overlap with training, no `a orange` forms, and no unseen tokens in the held-out summary style. All passed. Contexts have 126–280 generator tokens (including punctuation) in the train and fresh-world splits, compared with 61–95 in v1.

The surface audit against the frozen FineWeb panel shows partial improvement, not realism. Story sentences average 14.7 words, summary 20.4, and logs 22.3, versus 17.8 in the panel. But 1,000 story documents use only 74 distinct alphabetic words, versus 17,932 in 480 FineWeb windows; the generated sentences also reuse a few visible templates. The panels have different unit lengths and are not a controlled human naturalness test. **We cannot claim that v2 matches real web language.** See `world_v2_surface_comparison.json`.

## Answer-only depth sweep

Reproduce the generator and surface checks with:

```bash
PY=/home/loganriggs/Coding/neuron_circuits/.venv/bin/python
$PY language_task_suite/validate_world.py --world-version v2
$PY language_task_suite/compare_realism.py --world-version v2 \
  --output language_task_suite/world_v2_surface_comparison.json
```

The trainer and paired-edit runner accept the same `--world-version v2` flag; see the saved JSON files for exact hyperparameters and checkpoint paths.

Each model trained for 5,000 updates on 20,000 v2 worlds, with 48 hidden dimensions and four heads. The 2,000-example evaluation splits use full-vocabulary answer accuracy. All evaluation prefixes are new. One seed only.

| Blocks | Fresh worlds | Held-out object–answer pairs | Held-out summary style |
|---:|---:|---:|---:|
| 1 | 43.9% | 6.3% | 26.8% |
| 2 | 83.6% | **58.4%** | 53.4% |
| 3 | 84.0% | 51.7% | 30.1% |
| 4 | **85.3%** | 51.9% | **59.8%** |

The longer wording does not make the task impossible, but no depth here solves transfer. Relative to v1's same-size depth sweep, the 2-block v2 model scores higher on all three splits. The data distributions differ, so that comparison does not isolate a mechanism or prove that longer sentences caused improvement.

A matched-latent check scored the v1 and v2 4-block checkpoints on the *same 2,000 underlying worlds per split*, realized in each generator's own wording. V1/v2 scores were 73.1%/84.3% on fresh worlds, 18.1%/52.5% on held-out combinations, and 38.2%/59.6% on held-out style (`world_v1_v2_matched_latent.json`). This removes test-world sampling variation, but still does not control training distributions or optimization trajectories.

For 300 causal and 300 nuisance edit pairs per family, the 4-block model gets both answers correct on 66–79% of fresh-world causal pairs and 76–85% of fresh-world nuisance pairs. Held-out-combination causal pair accuracy is 47% for color, 54% for place, and 53% for owner (`world_v2_block4_pairs.json`). These are stricter than ordinary single-example accuracy: the edit requires the model to update its answer. The 2-block model has similar fresh-world pair scores but is somewhat better on held-out place pairs (`world_v2_block2_pairs.json`).

The 96-dimensional, eight-head, four-block capacity follow-up trained for 8,000 updates on 100,000 worlds. It reaches **89.9% fresh-world accuracy but only 24.7% held-out-combination accuracy**, and 54.2% on the held-out style. More data and width improved fresh-world fit while making composition transfer much worse in this one run (`world_v2_capacity_block4.json`). Paired causal edits remain weak on held-out combinations: 18% color, 30% place, 25% owner both-correct (`world_v2_capacity_pairs.json`). This argues against a simple capacity shortage as the only explanation; it does not establish a unique internal shortcut.

On that larger model, exact orthogonal changes of basis at the final attention pre-output bond reproduce the native logits at full width. With only 16 of 96 rotated coordinates retained per example, activation varimax lowers relative bond MSE from 0.054 (physical axes) to 0.031 while both retain 89.0% fresh-world accuracy on the 200-example gauge panel. An edit-supervised varimax fitted on owner/color edits reaches 0.989 post-fit equal-family causal-energy purity, versus 0.931 for physical axes and 0.951 for label-free activation varimax. **Place was withheld from basis fitting**, but used in this post-fit signature evaluation. All 16-coordinate bases retain only about 24–30% held-out-combination accuracy; none repairs the native model's transfer failure. These are local bond gauges and sparse gates, **not a canonical form of the full attention function or a recovered human-readable circuit** (`world_v2_capacity_bond_gauges.json`).

## Next tests

- Return to the existing `toy_models_of_tensor_networks/attn_circuits` mixed-rule generator as the primary full next-token LM sandbox. Its bigram/skip/induction conflicts instantiate the original mechanistic objective; this document-world benchmark is a complementary semantic stress test. See `EXISTING_GENERATOR_BRIDGE.md`.
- Increase lexical diversity and discourse structure using a manually reviewed natural-text sample; surface length alone is weak evidence of realism.
