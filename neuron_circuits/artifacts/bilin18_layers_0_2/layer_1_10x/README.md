# Bilin18 layer-1 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 1 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 204,800; held-out tokens: 102,400.
- Left/right factors choose the same coarse class for 45.9% of all channels and 48.4% of the more usable half.
- Among the more usable half, median held-out class lift is 1.63; 65.8% exceed 1.5× and 25.0% exceed 2×.

- Their median held-out signed class-to-class gap is 0.69 product standard deviations; 24.7% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 640 | subword | capitalized | punctuation → + / capitalized → − | 0.55 | 1.35 | + `[ig]eli V of Rwanda,`; `[ad]ol addictive. Compr`<br>− `SS is documented\nin[ HP] Customer Notice: HPS`; `), Ms Olena Nab[oka] (National Technical University �` |
| 570 | pronoun | determiner | punctuation → + / content_word → − | 0.60 | 1.17 | + `There are not many[ Thanksgiving] horror movies. It�`; ` divulge the secrets of[ Carnival]. I sent Joe off`<br>− ` accommodated. That is[ something] Felipão must consider`; ` (AAA), Edgar S[oto] (Hi A)\n` |
| 506 | determiner | subword | determiner → + / punctuation → − | 0.67 | 2.24 | + `TURT[LE] BAY, Hawaii —`; `SSSSSSSSSS[SS]`<br>− `deal[�]�s on. deal`; `There[�]�s no doubt of` |
| 4394 | subword | subword | punctuation → + / subword → − | 0.54 | 2.18 | + ` you[�]� (2008, p`; `[�]�A Fine Addition`<br>− ` typically 4-6 p[ups]. The young are prec`; ` protect and provision the p[ups]. Degus nest commun` |
| 624 | subword | capitalized | content_word → + / subword → − | 0.22 | 1.58 | + `�re working at some[ sort] of hippy, ultra`; ` it was delivered in the[ mail] this morning and she has`<br>− `[uing] the system and to suggest`; `[HOW]LAND, Maine — Fox` |
| 1782 | number | subword | determiner → + / subword → − | 0.30 | 1.17 | + `, Lord?!”[ �]�Why me!?!`; ` county time,”[ �]�illegal voting,�`<br>− `There isn[�]�t much that gets`; `deal[�]�s on. deal` |
| 759 | punctuation | capitalized | punctuation → + / determiner → − | 0.56 | 2.42 | + `There isn[�]�t much that gets`; `: “Why aren[�]�t you responding to`<br>− `[|]Commissioners put off decision`; `[|]Tommy Pi - Trance` |
| 2976 | content_word | content_word | capitalized → + / content_word → − | 1.94 | 2.20 | + ` before, though it doesn[�]�t stop me wondering`; ` he was feeling better,[�]� says Boudreau`<br>− `[ used] as primary diagnoses. He`; `[ scientific] "discovery."\n` |
| 896 | subword | subword | subword → + / conjunction → − | 0.47 | 2.13 | + `[uing] the system and to suggest`; `Serv[ing] diverse populations has never been`<br>− ` optimistic, were actually more[ likely] to still be alive.`; `. Delaine is very[ light] on his feet to be` |
| 3942 | punctuation | punctuation | conjunction → + / punctuation → − | 0.50 | 2.75 | + `, let’s[ talk] about the practical impact of`; `Now let’s[ look] at DOMA\nThe`<br>− `deal[�]�s on. deal`; `[�]�A Fine Addition` |
| 1690 | capitalized | capitalized | capitalized → + / subword → − | 2.16 | 4.76 | + `[Cl]utch Hitting and Strong`; `[T]SA employees have posted that`<br>− `That[�]�s Random! A`; ` a large bowl, s[ift] together the flour, cocoa` |
| 3289 | capitalized | subword | number → + / capitalized → − | 0.15 | 1.36 | + `Dem[et]rious Johnson vs. John`; ` anatomicae (R[ome], 1741).\n`<br>− `[ even] if carried to term,`; `[ just] after 10 pm, others` |
| 1641 | subword | subword | conjunction → + / subword → − | 1.61 | 4.05 | + ` been ‘ineffect[ual] as a podium for oppos`; `, “Theat[res] are so worried about upsetting`<br>− `adol coupon. Tr[am]adol online cod.`; `ol online cod. Tr[am]adol photos.\n` |
| 2510 | subword | punctuation | content_word → + / punctuation → − | 0.47 | 1.82 | + ` DO NOT have my email[ address]. Hahahaha.`; ` Windows Live Hotmail email[ address] (and Windows Live ID`<br>− `USE[�]�s Sumo Carbon`; ` “experimented[�]� with drugs together in` |
| 2570 | subword | subword | subword → + / punctuation → − | 1.23 | 2.29 | + ` online cod. Buy X[al]atan online cod. Ser`; ` if Urusei Y[ats]ura or DBZ.`<br>− `Results tagged[ �]�Tops”`; `ans must not let their[ �]�Washington Intervention”` |
| 2298 | capitalized | number | number → + / determiner → − | 0.38 | 1.86 | + `[These] wise words from Tolst`; `[Many] technologies and tools in use`<br>− `deal[�]�s on. deal`; ` ‘ipod touch[�]�\nApple has recently` |
| 1535 | number | number | punctuation → + / number → − | 3.38 | 7.85 | + `It[�]�s probably the height`; ` you[�]� (2008, p`<br>− `. In the ‘[70]s France chose to develop`; `[ 40] dbs, 50 d` |
| 1775 | determiner | determiner | determiner → + / capitalized → − | 2.18 | 5.44 | + ` everyone better. He gave[ the] QB extra time. He`; ` 13, 2013\nHow[ the] Buckeyes Won This Game`<br>− `. “All my[ life] I danced.”`; ` quilter my entire[ life]. But certainly not enough` |
| 4361 | determiner | determiner | determiner → + / conjunction → − | 2.47 | 5.51 | + ` old. Now I'm[ a] Lord of the Rings fan`; ` the answers. That includes[ a] vast wealth of town historians`<br>− ` We got up, got[ ready], got everything packed up`; ` own, but a good[ idea] that I'll use in` |
| 4562 | number | number | auxiliary → + / number → − | 3.41 | 7.09 | + `There isn[�]�t much that gets`; `[�]�A Fine Addition`<br>− `-back runs in the[ 2008] and 2009 campaigns.\n`; ` runs in the 2008 and[ 2009] campaigns.\nGame notes` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 1 already includes attention-1 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
