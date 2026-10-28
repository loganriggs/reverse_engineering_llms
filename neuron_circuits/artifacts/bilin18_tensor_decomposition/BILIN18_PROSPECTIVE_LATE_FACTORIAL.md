## H640 — Prospective replication survives new items and one-token instruction changes

H639 was progress: later bilinear MLP context explained much of cross-wording transfer conditional on matched incoming response/readout, while item transfer often required attention and context together. H640 prospectively tests whether that pattern depends on the old items or broad formatting changes.

Freeze64 prompts before model inference: two families, eight new items each, four templates. Copy items are fish, horse, apple, chair, water, stone, flower, table. Fact subjects are Poland, Sweden, Greece, Turkey, Norway, Finland, Austria, Portugal; targets Warsaw, Stockholm, Athens, Ankara, Oslo, Helsinki, Vienna, Lisbon. Items and prompt strings are absent from the H633 task panel; some country names occurred in earlier language experiments. No claim of training-data novelty is made. All targets are single tokens with their declared leading space.

Matched copying templates are “Repeat this word: {}.\nAnswer:” and “Copy this word: {}.\nAnswer:”. Matched facts use “Name the capital of {}.\nAnswer:” versus “Give the capital of {}.\nAnswer:”. Each matched pair has identical token length, identical answer-introducing suffix, and exactly one differing token. This controls sequence length and answer position, though the changed instruction token remains a legitimate contextual difference.

Broad templates reproduce H633's contrasting formats on the new items: “Repeat the word: {}. The word is” versus “Copy exactly: {}\nCopy:”; “The capital of {} is” versus “What is the capital of {}?\nAnswer:”. Cross donors swap within each template pair at fixed item. Within donors shift item by four at fixed template. Use the frozen original/span2/orthogonal2 probes and the same full C/N/A factorial over blocks10–17, crossed with receiver/donor block9 incoming tangent and readout. All scripts and the prompt manifest were prepared before inference. No output-driven prompt revision or filtering occurred.

**The main conditional mechanism replicates, including for one-token changes.** With donor incoming tangent and donor final readout, exchanging bilinear MLP context C is more effective than normalization N or normalized attention A alone in every family/contrast cell. C lowers mean remaining-gap ratios from0.621–0.721 to0.205–0.336. Joint factors further reduce the gap, and the complete donor operator recovers the endpoint.

| Cross-wording condition; donor input/readout | Native later operators | C | N | A | C+N | C+A | N+A | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|copy / matched|0.621|0.331|0.618|0.568|0.251|0.261|0.547|0.000|
|copy / broad|0.706|0.336|0.732|0.604|0.272|0.220|0.624|0.000|
|fact / matched|0.721|0.306|0.672|0.639|0.233|0.155|0.601|0.000|
|fact / broad|0.642|0.205|0.608|0.638|0.087|0.167|0.610|0.000|

Values are means of per-example gap-norm ratios under receiver-native probabilities; they are not accuracy scores or causal shares. The primary table intentionally conditions on donor input/readout, matching H639. All other input/readout combinations are retained in the full result. This is not a claim that context replacement alone starting from receiver components reproduces the donor. The original, learned-span and orthogonal probes each have mean context-only cross-wording remaining ratios below1 in all four cells; the result is not solely driven by the historically selected vector.

**Attention–context coordination also replicates for item exchange, with an important conditional qualification.**

| Within-wording condition; donor input/readout | Native later operators | C | N | A | C+N | C+A | N+A | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|copy / matched|0.863|0.697|0.857|0.621|0.715|0.152|0.623|0.000|
|copy / broad|0.763|0.556|0.768|0.557|0.552|0.120|0.551|0.001|
|fact / matched|0.883|0.727|0.860|0.625|0.821|0.162|0.611|0.000|
|fact / broad|1.303|1.530|1.245|0.861|1.755|0.396|0.793|0.000|

C+A produces large improvements for both families and both contrast sets. The earlier factual-item worsening from C alone replicates in broad formats (1.303→1.530), but not in matched-ending formats (0.883→0.727). Thus the claim that factual item transfer is always harmed by context-only exchange is false on this new panel. The more robust observation is that coordinated attention and MLP context outperform either alone for these item exchanges. This is still not a universal task/component division of labor.

**The matched contrast is smaller but well above the numerical recovery error.** Mean absolute weighted logit-derivative gaps for cross-template pairs are0.162 for matched copying versus0.339 broad, and0.252 for matched facts versus0.428 broad. Corresponding mean gap/native-response norm ratios are0.557/1.408 and0.714/1.262. Thus equal length and a shared ending reduce the contrast but do not eliminate it. The largest all-donor endpoint error relative to the gap is0.00132, ruling out tiny-denominator numerical noise as the explanation for the reported mean ratios.

**Candidate ranking must not be confused with immediate answer execution.** Every intended target wins among the declared eight candidates, but exact first-token accuracy varies:

| Family / template | Exact top1 | Candidate accuracy |
|---|---:|---:|
|copy / 0|25.0%|100.0%|
|copy / 1|25.0%|100.0%|
|copy / 2|0.0%|100.0%|
|copy / 3|100.0%|100.0%|
|fact / 0|25.0%|100.0%|
|fact / 1|0.0%|100.0%|
|fact / 2|100.0%|100.0%|
|fact / 3|12.5%|100.0%|

For matched copying, some top tokens capitalize the requested word or start “The”; the exact-copy labels deliberately preserve case. Matched fact prompts often start “The” or name another entity, and “Give…” has no correct immediate top1 targets. Broad explicit copying and declarative capital completion are each100% exact top1. First-token failure does not prove the full generated answer would be wrong, but candidate success does not prove it would be right. The operator result therefore concerns local response transfer across all these contexts; it is not a demonstration of universally successful semantic tasks.

The prospective result strengthens the hypothesis that bilinear operating-context changes capture a recurring component of wording-dependent late transformations. It weakens explanations based only on the old item set, changed sequence length or different answer-position suffixes. It does not rule out the lexical instruction token or continuation-format preference as the source of that context change. Attention–context coordination for item exchange also recurs, while the detailed effect of context alone depends on format.

Next move from native donor operators to prediction. On this frozen panel, fit an item-independent estimate of the normalized MLP-context change for each instruction/format pair using only the first four items. Predict donor contexts for the other four, then run the same downstream transformation without reading those held-out donor MLP contexts. Compare with exact donor-context transfer and no replacement, keeping input/readout conditioning explicit. If a shared context change transfers, that is a concrete step toward a reusable operator representation; if it fails, quantify identity-dependent residuals before increasing complexity. This is a new planned fit, not a result of H640.

Audits:448 whole-model forwards for native/±0.02 captures;432 boundary-control JVP calls plus4608 factorial JVP calls;512 cached attention reconstructions; all-donor finite-response recovery below0.000538 relative error; factorial whole-block replay below9.50e-7; exactly one changed token and equal length for every matched pair; all input/runner hashes unchanged; finite metrics. Panel, boundary control and absolute-gap audit have separate artifacts and provenance hashes. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.
