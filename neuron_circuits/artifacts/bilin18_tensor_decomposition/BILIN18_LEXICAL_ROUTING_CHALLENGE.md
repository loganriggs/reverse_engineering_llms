## H1024 — Position selection transfers; competing mentions expose routing-gain failure

Previous turn was progress: H1023exported a weight-defined lexical write without native prefix/QKinputs at write inference. This turn freezes its hashes and coefficients and challenges its assumptions with new contexts, retaining the native effect denominator and capability checks.

Panel:8historical country pairs, twoendpoints, fiveformats.16baselinecontrol prompts repeat old material;64constructed challenge contexts introduce source distance4,source distance7,competitor-before(source distance1),competitor-after(distance6). Vocabulary is familiar and prompts are new to this frozen-program challenge, not asserted novel across all research history. Panel/config written before model evaluation; no outcome filtering or coefficient refit. Native both-endpoint answer accuracy100%allformats; fullvocabularytop1baseline/distance/longdistance93.75%,competitor-before68.75%,competitor-after56.25%. Allmeet the predeclared75%binary/50%top1gate, but competitor free-generation competence is visibly weaker.

Fivearms: native isolatedcached13.0input; exact native-routing write; literal exported positions(source,source+1); a predeclared no-fit adapter placing the same fixedcoefficients at(source,final); and oracle native final-token write only. The adapter changes placement, not weights or routing values; it still needs the correct source/finalpositions supplied. Literal export was documented only for adjacent source/finaltokens, so its distance failure is an out-of-scope limit, not evidence of faulty original implementation.

| Format | Literal export error | Fixed source/final adapter error | Live-routing final-only error |
|---|---:|---:|---:|
| Baseline,distance1 |2.6%|2.6%|0.2%|
| Distance4 |98.4%|2.6%|2.8%|
| Distance7 |99.1%|4.2%|3.2%|
| Competitor before,distance1 |22.2%|22.2%|0.2%|
| Competitor after,distance6 |98.9%|15.4%|2.9%|

Relative errors target the isolated13.0cachepath, whose aligned recovery of full natural country-swap effect remains60.2%baseline,63.9%distance4,63.6%distance7,69.7%competitor-before,77.3%competitor-after. Increased relative share with competitors does not imply improved total model competence; their fullvocabularyaccuracy falls. Exact live-routing program replays every path within4.29e-6absolute. Lexical cache-value delta replay remains1.88e-7relative, so payload-map failure is not the cause.

Final-token signed attention coefficient mean/std:baseline-0.6744/0.0155,distance4-0.6573/0.0163,distance7-0.6690/0.0202,competitor-before-0.5557/0.0173,competitor-after-0.5737/0.0190. Frozenprogrammean is-0.6738. Competing mentions reduce coefficient magnitude; simple distance by itself does not produce comparable attenuation in these templates. Pattern is a product of two query-key scores without softmax, so this is not evidence for a softmax probability-competition mechanism. Query changes, key changes, normalization and rotary/positioneffects remain to separate.

Interpretation for computational units: fixed lexical value map, source/destination selection and context-dependent scalar strength are distinct pieces. We now have evidence that the value map transfers, destination must be explicit, and a constant scalar is inadequate with competing entities. Treating this all as one static feature would hide the failures. The adapter is not an autonomous source selector, and native-routing oracle still runs the model; neither is a completed replacement circuit.

Audits: export hashes unchanged;zero routingfitrows;80unique panelprompts;independentTorch effecterrors exact;finite;80prefix and480suffix sequences;zero newgradientfits. No reservedBSFtest data or natural-document expansion; pool26112unchanged. Saved all arms,coefficients,promptidentities,capability and full-effect coverage. No live process,H961pending,broad goal active.

Next use discovery-side matched single/competing contexts to factor the routing coefficient into query, key and rotary contributions, with finite swaps or exact score recomputation. Do not simply fit one correction factor to these now-open challenge results. Preserve H1023as frozen baseline; any source/destination-aware executor is a separately versioned candidate. Seek a compact selector/gain rule then test it on fresh competition structures and a distinct task sharing head13.0. Representation architecture should reflect these separately tested components rather than infer a universal manifold from one prompt family.
