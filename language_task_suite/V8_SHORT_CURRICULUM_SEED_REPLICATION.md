# Replication of the short section-lookup curriculum

The [first short-stage result](V8_SCOPE_CURRICULUM_TRANSFER.md) reached
near-perfect keyed retrieval after 1,500 updates. We repeated the exact
architecture, two-section data generator, batch, learning rate, update
budget and development-selection rule at initialization/data seeds 2–5.
Each selected checkpoint was evaluated on an independent 512-world test
set. The query-blind reference has 50% answer accuracy and key-edit TV 0;
the exact oracle changes the answer under the alternate section query.

| Seed | Selected step | Test accuracy | Query-key edit TV | Test answer CE |
|---:|---:|---:|---:|---:|
| 1 | 1,500 | 1.000 | .9997 | .0004 |
| 2 | 1,250 | .495 | .0039 | .697 |
| 3 | 1,500 | .650 | .2342 | .579 |
| 4 | 1,500 | 1.000 | .9987 | .0014 |
| 5 | 1,500 | .921 | .8012 | .178 |

This is **not a reliably converged five-seed recipe**. Two seeds are nearly
perfect, one is strong but incomplete, one partly learns, and one remains
key-blind. Several runs stay near the 50%/log-2 plateau through 1,000
updates and then change abruptly; seed 4 reaches 89.6% at step 1,250 and
99.9% on development at step 1,500. The paired-query construction rules out
a fixed color-frequency answer on the short test but does not by itself
guarantee that every initialization finds the key-sensitive solution.

The next optimization study should separate initialization from data-order
seeds and compare longer training or a staged shorter-context curriculum.
Any mixed-language follow-up using a successful short checkpoint is
**conditional on short-stage success**; it does not establish an
unconditional multi-seed success rate for the full procedure. This toy
curriculum is grammar-valid under the exact v8 prefix oracle but is not a
prevalence-matched natural web-text sample.

The five [training records and selected weights](results/) are in
`v8_scope_short_pretrain_seed1` through `v8_scope_short_pretrain_seed5`.
