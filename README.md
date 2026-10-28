# reverse_engineering_llms
An additional attempt at reverse engineering LLMs, with an initial start from neuron interp &amp; trying to include manifolds from Block sparse featurizers.


## Research snapshot through H1033

The [neuron_circuits](neuron_circuits/) directory contains experiment code, configurations, reports, and compact result artifacts from the research workspace.

- [Latest research checkpoint](neuron_circuits/RESEARCH_CONTINUATION_2026-10-21.md)
- [Results timeline](neuron_circuits/CURRENT_COMPUTATION_REGIME_UPDATE.md)
- [Hypothesis ledger](neuron_circuits/COMPUTATION_HYPOTHESIS_LEDGER.md)
- [Interpretation guide](neuron_circuits/EASY_INTERPRETATION_GUIDE.md)
- [Reports and compact artifacts](neuron_circuits/artifacts/bilin18_tensor_decomposition/)
- [Experiment scripts](neuron_circuits/scripts/) and [configurations](neuron_circuits/configs/)
- [Latest reviewed report: finite causal supervision](neuron_circuits/artifacts/bilin18_tensor_decomposition/BILIN18_FINITE_SUPERVISED_SELECTORS.md)
- [H1034 raw task-holdout results](neuron_circuits/artifacts/bilin18_tensor_decomposition/bilin18_selector_task_holdouts.json) — awaiting independent audit and interpretation.

This is a publication snapshot, not a full data backup or a verified portable reproduction environment. Large tensors, activation captures, arrays, and JSON artifacts above 1 MiB are excluded. [The manifest](neuron_circuits/PUBLICATION_MANIFEST.json) lists included file hashes and excluded paths/sizes. The original full workspace remains at `/home/loganriggs/Coding/neuron_circuits`.

See [publication notes](neuron_circuits/PUBLICATION_NOTES.txt) for external checkpoint/code dependencies, absolute-path assumptions, and archival scope. Historical pause/publication-status notes record the state at snapshot creation.
