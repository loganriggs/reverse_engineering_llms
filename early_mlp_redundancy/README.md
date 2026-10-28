# Why does the second MLP repeat so much early token information?

A bounded research handoff for the 18-layer bilinear language model, assembled 2026-10-28.
Start with **MLP1**, the **second block** (all indices here are zero based).
MLP0 is the preceding control; MLP2 is the following comparison case.

**Question:** What does MLP1 contribute beyond the token information already present after MLP0,
and can we isolate that contribution with a faithful, reusable intervention?

The word *redundant* has covered several different observations. This packet separates them;
it does not assume that MLP1 is dispensable, that its neurons are duplicates, or that a low-rank
replacement already exists. It includes historical evidence, runnable inspection code, and seven
small exploratory native-model studies.

## Current answer, within the tested panel

MLP1 is causally active even when its information overlaps earlier layers. It amplifies upstream
changes, including random ones ([P3](MLP1_RESPONSE.md)). Amplification alone is not semantic evidence.

Natural prefix MLP0-write transfer predicts the noun-position MLP1 context change much better than
local-only transfer:83% versus20% reduction in squared target error ([P4](MLP1_MEDIATION.md)).
This supports a cross-position contribution through attention1, without identifying a particular head.

The induced MLP1 writes across positions also carry substantial donor-output recovery: about34%
on their own versus38–40% for the full prefix intervention; freezing them leaves only1–3% recovery
([P5](PREFIX_RESPONSE_FACTORIAL.md)). Matched random-prefix edits do not recover the donor distribution.
This is evidence of a direction-dependent early computation on a small reused authored panel, **not
a verified compact feature basis or broad task understanding**. She/he transfer is notably weak.

Position localization ([P6](RESPONSE_POSITIONS.md)) shows that the changed word's MLP1 response
alone recovers about29% of donor discrepancy, versus34% for all positions and4% for the complement.
This is a source-token response, not a discovered neuron or compact basis. The four later nouns
provide only eight distinct source contexts, so further feature fitting on this panel would be
uninformative. Next work must vary prefixes before the edit and broaden competent-task coverage.

A [training-task screen (P7)](TRAINING_TASK_SCREEN.md) now tests this source response on76
single-token pairs across nine competent task families. Agreement, sentiment, pluralization and
past tense show partial donor recovery; binding and factual recall do not. Only sentiment and
binding have repeated token edits under multiple pre-edit contexts. This is a scoped causal target,
not a universal featurizer objective; whole-task evaluation exclusions remain intact.

The earlier [P1](NATIVE_MLP2_RESPONSE_PILOT.md) and [P2](MLP2_TOKEN_TRANSFER.md) studies distinguish
native MLP2 amplification from approximation-specific error attenuation. Read the parent contracts
below before comparing those experiments with historical compressors.

## Read this first

**Follow-up correction:** [the history audit](HISTORY_AUDIT.md) adds completed table-composition,
MLP2 compensation, and compiler results. Read it before planning a new experiment: the older
aligned-repair hypothesis was not established, and the compiler parent differs from this native inspector. The C512 compensation parent
is native apart from its MLP0.Down replacement; see [parent contracts](PARENT_CONTRACTS.md).

| Observation | Evidence | What it establishes / leaves open |
|---|---|---|
| An MLP0 article-related cluster can be changed without substantially changing MLP1's article-related signal. | [Original result](evidence/prior/mlp1_reads_mlp0_results.json); [historical synthesis](evidence/prior/EXCERPTS_RESULTS.md), §847 | Supports parallel encoding for that signal under that intervention. It does not show all of MLP1 ignores MLP0. |
| Small article-carrying clusters can be sufficient under patching without being necessary under removal. | [MLP1 localization](evidence/prior/article_mlp1_localization_results.json), [MLP0 localization](evidence/prior/article_mlp0_localization_results.json) | Readable clusters are handles on a distributed computation; this is not evidence that individual channels are identical. |
| A current-token table recovers 96.01% of MLP1's mean-ablation CE stake on covered positions, versus 90.27% for MLP0 and 76.98% for MLP2. | [Attribution result with protocol](evidence/prior/mlp1_residue_attribution_results.json); [corrected historical account](evidence/prior/EXCERPTS_FINDINGS.md) | Strong token-conditioned replaceability on the measured distribution. Not 96% explained variance, output KL fidelity, or semantic understanding. |
| MLP1 has a large mean-ablation stake: about 7.005 nats; the remaining 3.99% table gap is about 0.280 nats. | Same attribution result | High table recovery does not make the contextual remainder negligible or the whole module unnecessary. |
| Freezing attention0 alone barely changes that table gap; freezing attention1 removes about 19%; freezing both removes it. | Same attribution result | Consistent with distributed/context-dependent routes. Each arm refits a different model and denominator; these percentages are not additive causal shares. |
| Stable native gate selections at budgets 32/128/512 did not pass the registered causal comparison. | [Gate findings](evidence/prior/MLP1_GLOBAL_GATE_FINDINGS_2026-08-28.md) | Stable rankings are insufficient for a faithful sparse circuit. Does not exclude learned alternative coordinates. |
| MLP1's coefficient tensor is full multilinear rank, with 90% coefficient energy requiring 835 output / 937 input directions. | [Weight-only finding](evidence/prior/MLP1_IMPLICIT_FOLDED_TENSOR_V2_FINDINGS.md) | Simple Euclidean Tucker compression is unattractive. Weight complexity is distinct from computation used on data. |
| This session's small input-based predictors explain only 26.5% of MLP1 write variance at linear rank32; output-PCA32 oracle explains 36.8%. | [Early MLP prediction](evidence/current/BILIN18_EARLY_MLP_PREDICTION.md) | Token-table CE success and local vector prediction measure different things, on different panels. These results are not contradictory. |

The prior repo also contains later, more specific source/reader studies. In the included
rung489–492 receipts, the common native-state reader classification fails; source decomposition
at rung491 passes its gates; selective attention1→MLP1 reader intervention at rung492 fails its
selectivity/transfer gates. **Do not turn a source-attribution success into an established removable
semantic circuit.** See [the evidence guide](EVIDENCE_GUIDE.md).

### Secondary puzzle: MLP2 cancellation

The [MLP2 dossier](evidence/prior/MLP2_CURRENT_UNDERSTANDING.md) records that selecting 512 native
channels was worse than deleting the module: local selection added about .265 CE, versus .162
for deletion. A learned 512-product program improved this to about .051 CE in a separate refit
comparison, but still failed its replacement threshold. This is a useful nearby test of
**coordinate choice and cancellation**, not the same phenomenon as MLP1's token-class overlap.

## Run the model inspection

The directory is movable: the runnable entry point does not import either original workspace.
It contains the exact local model implementation and pinned loader in `vendor/`.
Weights are **not** bundled; the pinned checkpoint is about 2.1 GB on disk. Float32 parameters
alone need roughly 2.1 GB; loading and activations require additional memory. Use short prompts
first. GPU is recommended; CPU is supported by the interface but was not separately benchmarked.

Use Python 3.11+ and a PyTorch build appropriate for your hardware:

```bash
cd early_mlp_redundancy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python inspect_model.py --download --device cuda --output my_smoke.json
```

`--download` permits Hugging Face to download only the pinned model config and weights.
Without that flag, the model must already be cached. The GPT-2 tokenizer uses `tiktoken`;
its first use can fetch tokenizer assets. For offline use, prepopulate its cache as well.
To use an explicit existing checkpoint directory:

```bash
python inspect_model.py --snapshot /path/to/snapshot --device cuda \
  --layers 0 1 2 --text 'The dogs near the house are playing.' \
  --save-activations activations.pt --output inspection.json
```

The CLI:

1. Verifies the checkpoint SHA256 and exact topology.
2. Captures normalized MLP inputs, Left/Right factors, products, writes, and surrounding residual states.
3. Checks its forward pass against the original `GPT.forward`/`Block.forward` path.
4. Verifies exact product reconstruction and the finite bilinear difference identity.
5. Measures individual and joint **zero-write** ablations, recomputing all later layers.
6. Reports output KL, next-token CE, and top1 agreement separately.

Zero-write removal includes the bias. It is **not** the historical mean-ablation protocol.
The default prompt is only an instrumentation check; its removal results are not a corpus-level
redundancy claim. The bundled [validation receipt](validation/smoke_results.json) passed with
exact native logit replay on 13 tokens. See [validation notes](VALIDATION.md).

### Use in a notebook or another script

```python
from inspect_model import load_model, encode, trace
import torch

model, receipt = load_model(download=True)
ids = encode('The dogs are playing.', next(model.parameters()).device)
logits, sites = trace(model, ids)
mlp1 = model.transformer.h[1].mlp
L, R, D = mlp1.Left.weight, mlp1.Right.weight, mlp1.Down.weight
z = sites[1]['normalized_input']  # [batch, token, 1152]
h = sites[1]['products']         # [batch, token, 4608]

# Custom interventions return a WRITE, not a replacement residual state.
edited, _ = trace(model, ids, replace=lambda layer, z, write:
    torch.zeros_like(write) if layer == 1 else write)
```

`trace` is inference-only (`no_grad`). For gradient experiments, adapt it deliberately and verify
native replay again. To test a learned token table, return the table write at covered positions
and the original write elsewhere. Fit the table on separate documents before evaluating it.

## Architecture facts that matter

- 18 blocks; residual width 1152; 9 heads of width128; 4608 bilinear products per MLP.
- For normalized input `z`, `MLP(z) = Down(Left(z) * Right(z)) + Down_bias`.
- Every block mixes its incoming residual with the original normalized token embedding using
  learned `lambdas`, then adds attention and MLP writes with RMSNorm at their inputs.
- Attention reuses the first block's value bus. A residual-only intervention does not erase all
  token information available to later layers.
- Attention uses a product of two QK scores with a causal mask, not softmax attention.
- Final logits use RMSNorm, unembedding, and `30*tanh(logits/30)`.
- Tokenizer vocabulary is 50,257; output vocabulary is padded to 50,304. KL here uses the full
  output distribution, including padded logits, to match native behavior.

Ignoring the embedding reinjection, value bus, normalization, bias, or softcap can manufacture
apparent redundancy or disagreement between implementations.

## A bounded next experiment

First read [HISTORY_AUDIT.md](HISTORY_AUDIT.md). The initial proposal in
[RESEARCH_QUESTION.md](RESEARCH_QUESTION.md) describes one fixed set of token-conditioned replacements
and a 2×2 MLP0/MLP1 intervention factorial, scored on untouched documents and competent behavioral
contrasts. No new featurizer architecture is needed to ask this question. The first goal is to
separate shared predictable content from a context-dependent correction and downstream compensation.

## Contents and provenance

- `evidence/current/`: selected reports, JSON results, figures, and historical scripts from this session.
- `evidence/prior/`: focused earlier results, findings, dossiers, and line-numbered corrections.
- `vendor/`: source copies used by the runnable loader; these are code, not model weights.
- `PROVENANCE.json`: original locations, source hashes, and source repository commit.
- `MANIFEST.json` and `verify_packet.py`: integrity check for all distributed files.
- `validation/`: instrumentation results only; no new research benchmark is claimed.

The old research scripts are archival evidence. They can require external datasets, fitted arrays,
and other source files; they are not standalone reproductions. Original links inside copied reports
may refer to that larger repository. The guide supplies the local entry points and provenance
needed to recover those dependencies. This packet excludes training corpora, activation banks,
fitted compressors, and checkpoint weights. It is a curated handoff, not a complete research backup.
