# Bilin18 final-MLP compression

## Verdict

A small fixed subset captures most of the final MLP's **next-token loss benefit**, but it does not faithfully reproduce the full output distribution. The final layer is therefore small in an operational performance sense and still broad in a mechanistic emulation sense.

Channels were ranked on a disjoint discovery split by activation variance times the squared norm of their direct `Down→unembedding` direction. Omitted channels were replaced by their discovery mean, which can be folded into a replacement layer's bias. Results below are shown as initial / independent-replication values.

|retained channels|fraction of layer|Δ cross-entropy|KL from full|top-1 agreement|MLP loss benefit recovered|
|---:|---:|---:|---:|---:|---:|
|0|0.0%|+0.388 / +0.415|0.391 / 0.388|86.0% / 81.4%|0.0% / 0.0%|
|8|0.2%|+0.315 / +0.326|0.310 / 0.309|85.8% / 82.7%|18.8% / 21.4%|
|16|0.3%|+0.271 / +0.271|0.263 / 0.258|84.5% / 82.2%|30.2% / 34.6%|
|32|0.7%|+0.175 / +0.181|0.187 / 0.180|85.2% / 82.9%|55.0% / 56.3%|
|64|1.4%|+0.119 / +0.121|0.129 / 0.127|86.0% / 83.7%|69.2% / 70.9%|
|128|2.8%|+0.080 / +0.086|0.095 / 0.091|86.8% / 84.9%|79.2% / 79.3%|
|256|5.6%|+0.061 / +0.070|0.076 / 0.074|88.8% / 86.6%|84.3% / 83.0%|
|512|11.1%|+0.047 / +0.054|0.058 / 0.055|89.5% / 87.6%|87.9% / 87.1%|
|1024|22.2%|+0.020 / +0.035|0.038 / 0.037|90.8% / 88.8%|94.7% / 91.5%|
|2048|44.4%|+0.012 / +0.020|0.020 / 0.019|92.8% / 92.5%|96.8% / 95.2%|
|4096|88.9%|-0.005 / +0.001|0.002 / 0.002|97.0% / 97.6%|101.3% / 99.7%|
|4608|100.0%|+0.000 / +0.000|0.000 / 0.000|100.0% / 100.0%|100.0% / 100.0%|

## What is concentrated

The activation-conditioned ranking has effective support 45.6 channels. Its top 32 channels hold 58.4% of ranking mass and the top 64 hold 69.1%. By contrast, raw `Down→unembedding` column energy has effective support about 4,485 channels. Concentration is therefore created by which channels vary on data, not by a visibly sparse output matrix.

The ranking is stable across discovery samples: top-64 overlap is 100.0%, top-128 overlap is 98.4%, and top-256 overlap is 97.7%.

## Red-team interpretation

- **Strongly supported:** 64 fixed channels (1.4% of the native layer) recover about 69–71% of the mean-replacement loss benefit; 128 recover about 79%. Both greatly beat random subsets of the same size.
- **Not supported:** a few dozen channels faithfully replace the final MLP. At 64 channels KL remains about 0.127 nats and top-1 agreement about 84–86%.
- Roughly 1,024 channels are needed for KL below 0.04 nats on both runs; 4,096 are needed for near-exact KL around 0.002 and roughly 98% top-1 agreement.
- Cross-entropy measures task performance and can hide compensating logit changes. KL to the intact model is the stricter replacement criterion.
- The evaluation uses 400 and 1,600 held-out document-final positions at context length 128. It should be replicated at other positions and longer contexts before pruning.
- This is a fixed native-channel subset, not a learned low-rank refactorization. A newly trained smaller bilinear layer might compress the computation further.
