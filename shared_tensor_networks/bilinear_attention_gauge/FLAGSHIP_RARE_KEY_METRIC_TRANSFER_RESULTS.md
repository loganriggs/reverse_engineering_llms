# Mixed attention reduction protects rare natural keys better than corpus-only fitting

The [pre-RoPE metric factorial](ROPE_METRIC_FACTORIAL_RESULTS.md) fitted
three rank-96 QK1 cuts on layer 0/head 0 of the 546M bilinear language
model: uniform vocabulary pairs, FineWeb corpus pairs, and a 50/50 mixed
measure. On ordinary documents, the mixed cut had the lowest whole-model
KL. Earlier [pair-level frequency tests](FLAGSHIP_CONTEXT_TRANSFER_RESULTS.md)
showed that data-heavy fitting can lose rare-token fidelity. This test asks
whether the three fixed cuts preserve a **naturally occurring rare key's**
contribution when propagated through the whole model.

We define rarity using only FineWeb training documents 0–14,999: a rare
token ID occurs 1–9 times there; a common ID occurs at least 1,000 times.
We select one occurrence at position 16 of a length-64 natural window in
each held-out document. FineWeb test documents are 17,500–19,999;
WikiText-103 and TinyStories are external corpora. We replace only this
key's QK1 score contribution in head 0 with one fixed rank-96 cut. All
other keys and all other heads remain native. The comparison is paired:
uniform, mixed, and corpus cuts see **identical tokens and positions**.
Identity and null-mask edits reproduce native logits exactly in the saved
run; the causal prefix has negligible KL. There are no labels or
model-performance-based choices in key selection.

## Broad exact first-layer output assay

For each corpus and frequency band, we sampled **128 distinct documents**.
The exact selected-key contribution after the learned `c_proj` output
projection was computed for the three cuts and three independently
sampled **cut fits** (seeds 41, 43, 47). These are three fits of the
same model, not three independently trained models. The local formula
agrees with the full-model hook's removed output energy on 12 checked
cases to 2.2×10⁻⁶ relative error.

Aggregate relative squared error of the selected key's native output,
averaged over the three cut fits:

| Corpus / key band | Uniform | Mixed | Corpus-only |
|---|---:|---:|---:|
| FineWeb rare | .0189 | **.0158** | .0207 |
| FineWeb common | .0254 | .0128 | **.0102** |
| WikiText rare | .0189 | **.0168** | .0218 |
| WikiText common | .0308 | .0210 | **.0194** |
| TinyStories rare | .0118 | **.0089** | .0109 |
| TinyStories common | .0269 | .0080 | **.0056** |

The same rare/mixed and common/corpus ordering holds in **all nine**
fit-seed × corpus cells. On the seed-41 panel, mixed beats uniform on
101/128 FineWeb, 93/128 WikiText, and 103/128 TinyStories rare keys;
it beats corpus-only on 85/128, 87/128, and 75/128 respectively. A
paired document bootstrap gives negative mixed-minus-uniform and
mixed-minus-corpus aggregate-error intervals in every rare cell for
that fit. These intervals condition on the fixed cut and sampled
documents; they do not make the three fit seeds independent evidence.
Common-key output energy is especially concentrated: the effective
number of documents is only about 5–11 despite 128 sampled documents.
The paired wins and per-case records should therefore accompany
energy-weighted aggregates.

This is a useful correction to a simplistic “data-aware harms rare
tokens” story. Uniform fitting protects the *uniform token-pair law*,
but rare tokens in natural documents are not uniformly paired with
queries and positions. The mixed measure appears to retain structure
from both domains. Corpus-only can still overconcentrate on common keys.

## Full 18-layer propagation

The first full-model panel used 24 rare and 24 common natural windows
per corpus. Its rare FineWeb mean favored uniform, in tension with the
broader local result, and its paired interval for mixed-minus-uniform
crossed zero. We therefore ran a **fresh 64-document rare-only panel**
per corpus with the unchanged seed-41 cuts. Mean native-to-cut KL on
the common suffix beginning at the selected key is shown in units of
10⁻⁶:

| Rare-key corpus | Uniform | Mixed | Corpus-only | Mixed − uniform, paired 95% interval | Mixed − corpus, paired 95% interval |
|---|---:|---:|---:|---:|---:|
| FineWeb | .186 | .181 | .263 | −.004 [−.023, .017] | −.082 [−.146, −.030] |
| WikiText-103 | .565 | .457 | .713 | −.108 [−.334, .028] | −.256 [−.413, −.115] |
| TinyStories | .135 | .116 | .174 | −.019 [−.034, −.006] | −.058 [−.120, −.013] |

Mixed has lower mean rare-key KL than corpus-only in all three corpora,
with paired document-bootstrap intervals below zero. Mixed versus
uniform is unresolved on FineWeb and WikiText-103 and favors mixed on
TinyStories. Uniform beats corpus-only on FineWeb rare keys in the
expanded panel. The new panel's effective sample counts for KL are
roughly 10–26 of 64 because a few keys carry much larger effects.

For common keys, the original 24-window full-model panel gives lower
mean KL to corpus-only than to uniform or mixed on all three corpora.
Those effects are also tiny and variable; the 128-document exact-output
assay is the stronger evidence for the common-key tradeoff. No
answer-level task accuracy or linguistic role was measured here.

## Interpretation and next gate

The rank-96 mixed pre-RoPE cut is a defensible **fidelity prior** for
this physical bond: it protects naturally occurring rare keys better
than corpus-only fitting while preserving much of corpus-only's
common-key advantage. It is not a canonical function decomposition or
evidence that any axis is human-legible. The local metric and downstream
KL agree directionally against corpus-only on rare keys, but downstream
effects of one key in one first-layer head are only about 10⁻⁷–10⁻⁶
KL per token. A useful legibility experiment must make a prospective,
selective prediction about a larger behavior or multiple heads, and
compare its intervention at matched **post-projection dose** on fresh
contexts. More sparse-axis tuning on these tiny single-key KL effects
would not answer that question.

Reproduce with `flagship_rare_key_metric_transfer.py` for the 24-window
all-band panel and the fresh 64-window rare-only panel, then
`flagship_rare_key_local_transfer.py --fit-seed 41`, 43, and 47, and
`analyze_flagship_rare_key_transfer.py`. The `full_width_results/`
JSON files retain every selected token ID and training frequency,
document/window identity, cut response energy, KL, CE shift, and
paired-bootstrap interval.
