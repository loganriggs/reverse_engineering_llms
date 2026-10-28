# Bilin18 layer-1 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 1 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 8,192,000; held-out tokens: 2,048,000.
- Left/right factors choose the same coarse class for 47.3% of all channels and 49.2% of the more usable half.
- Among the more usable half, median held-out class lift is 1.63; 64.4% exceed 1.5× and 24.9% exceed 2×.

- Their median held-out signed class-to-class gap is 0.69 product standard deviations; 24.3% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 640 | subword | capitalized | punctuation → + / capitalized → − | 0.50 | 1.34 | + `[il]ienne A, Ponc`; `[ur]iously soft cotton fabric.`<br>− ` Lakeview Trl from[ Sara] Dailey to William`; `ars JW, Pr[ins] MH, van Oers` |
| 570 | pronoun | determiner | punctuation → + / subword → − | 0.55 | 1.15 | + ` in social media[ networking] but by social media networking`; `[ Thanksgiving] weekend.\nI met`<br>− ` the latest Microsoft® Windows[ Mobile] Version 5.0 Software`; ` two goals and the last[ two] goals of a 6-` |
| 1825 | conjunction | conjunction | punctuation → + / conjunction → − | 0.78 | 3.76 | + ` A. ZORINA[<|endoftext|>]It’s hard`; ` 02114-2696[<|endoftext|>]A:A single's`<br>− `ink boots in the summer[ time] like Will.I.`; ` Prospectus has not been[ and] will not be registered under` |
| 4574 | subword | determiner | determiner → + / punctuation → − | 0.32 | 1.25 | + `‘Rolling Out[�]� Intern Erica Smith Gets`; ` your facebook friends.<|endoftext|>[�]�Rolling Out�`<br>− `[ades] and barbeques,`; ` ENDORSE THE ACC[UR]ACY, CURRENTNESS` |
| 4394 | subword | subword | punctuation → + / subword → − | 0.55 | 2.56 | + ` didn[�]�t have to go`; `ura[�]� councils, at which`<br>− ` Park Service. Yellowstone National[ Park] is the nation’`; ` Vet Team Answers Your Pupp[y] and Kitten Questions\n` |
| 506 | determiner | subword | determiner → + / punctuation → − | 0.67 | 2.57 | + ` high chair\n, travel[ high] chair target\n, fisher`; `, long, long,[ long]) ways to go until`<br>− `u[�]�s Hair Endorse`; `di[�]�s signature recipe fills` |
| 291 | subword | subword | punctuation → + / content_word → − | 0.38 | 1.83 | + ` next year.”[<|endoftext|>]I still don’`; ` my lifetime.”[<|endoftext|>]THE DESIGN PROCESS`<br>− ` the support you need to[ stream]line the homebuying`; ` told him he has to[ hand] over his soul.\n` |
| 3289 | capitalized | subword | number → + / punctuation → − | 0.14 | 1.16 | + ` of generic Muscad[et]`; ` claiming that Muscad[et] was divided into only three`<br>− `['d]. Where glossy magazines once`; `[ few] repetitions, the dogs` |
| 1782 | number | subword | determiner → + / subword → − | 0.23 | 1.16 | + `, 2006)\nPR[OM]ETHEUS 103 (`; `, 2006)\nPR[OM]ETHEUS 104 (`<br>− `[ reduce] labor, increase information security`; `[ reduce] the brightness of the radiation` |
| 2976 | content_word | content_word | punctuation → + / content_word → − | 1.93 | 2.18 | + ` any. So it doesn[�]�t matter if you`; `. The universe certainly doesn[�]�t stop for any`<br>− `[ happen]. That is how I`; `[ break]age of the container 10` |
| 2590 | number | subword | punctuation → + / number → − | 0.37 | 2.07 | + `?\n| Report this[<|endoftext|>]On July 26, HUD`; `.\nShow All Answers[<|endoftext|>]Headstones are not only`<br>− ` we are being ceas[eless]ly attacked and vilified`; ` decades have been ceas[eless]ly gnawing away` |
| 896 | subword | subword | subword → + / conjunction → − | 0.46 | 2.10 | + `[aying] in the Spirit means we`; `[aining] conscious of the importance of`<br>− ` it’s more[ likely] to be a better fit`; `’s much more[ likely] to be called “` |
| 624 | subword | subword | conjunction → + / subword → − | 0.33 | 1.80 | + ` disposal means, renting some[ sort] of dumpster through a`; ` to be avoided in any[ respect] cost. People imp source`<br>− `[ish], Jason Southard said`; `[ish] pooch. After waiting` |
| 759 | punctuation | capitalized | punctuation → + / determiner → − | 0.61 | 2.65 | + ` aren[�]�t being as productive`; ` aren[�]�t seated together.`<br>− `[|]Clayton Grue`; `[|]\n|Includes a full` |
| 3942 | punctuation | punctuation | conjunction → + / punctuation → − | 0.61 | 2.83 | + ` sincere and creative potter[\n]Take care my dear friend`; `But let’s[ be] realistic: NBC (and`<br>− `[ �]�so small and loosely`; `[ �]�short extraction”` |
| 210 | capitalized | capitalized | subword → + / capitalized → − | 1.97 | 4.01 | + ` with aff[lict]ions like chronic pain,`; `ătrat�[�] – Totem @ Lente`<br>− `[ Again] refrain from the waters of`; `[ Recent] Woes Sushi is` |
| 1535 | number | number | punctuation → + / number → − | 3.44 | 7.96 | + ` High[�]�s boys win their`; ` Franklin[�]�s father, Rev`<br>− `ed in the late '[60]`; ` the tumultuous late ’[60]s, when peace consciousness` |
| 2119 | conjunction | conjunction | conjunction → + / number → − | 3.10 | 6.63 | + `[ particularly] unusual about it. But`; `[/]Services or interfere with any`<br>− `tanenbaum.org[/]wp-content/uploads`; ` http://doi.org[/]10.3886/` |
| 4361 | determiner | determiner | determiner → + / conjunction → − | 2.48 | 5.46 | + ` within you.\nLike[ a] roadmap, life has scenic`; ` column address. Depending on[ the] operating mode of the memory`<br>− ` UI. At Happy Cam[per] we almost always use the`; ` Center, because the whole[ thing] got started with the erg` |
| 1775 | determiner | determiner | determiner → + / capitalized → − | 2.18 | 5.45 | + `Fit finger stitching to give[ the] best grip and feel\n`; ` the first half to give[ the] Gators a sizeable lead.`<br>− ` pass[ away]. The scripture records that`; ` be heard, the following[ exchange] occurred:\n"MR` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 1 already includes attention-1 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
