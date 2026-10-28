# Bilin18 layer-16 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 16 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 51,200; held-out tokens: 25,600.
- Left/right factors choose the same coarse class for 31.5% of all channels and 34.4% of the more usable half.
- Among the more usable half, median held-out class lift is 1.31; 27.8% exceed 1.5× and 7.8% exceed 2×.

- Their median held-out signed class-to-class gap is 0.41 product standard deviations; 10.2% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 2098 | number | pronoun | pronoun → + / preposition → − | 0.85 | 1.25 | + `.\nPlease confirm or[ modify] the email address to which`; ` sent.\nFor a[ more] personalized experience, please supply`<br>− ` more and more inclusive order[.] Invent some manner of`; ` vanquished party's interests lay[.] The course of history is` |
| 3031 | punctuation | pronoun | preposition → + / punctuation → − | 1.31 | 1.72 | + ` fact[ that] our students have purchase plane`; `Swami leads the opening[ of] the 2012 Festoon Wind`<br>− `amobile.com/[<|endoftext|>]Try right clicking the entry`; `li Orafi Hotel[<|endoftext|>]Thursday, July 8,` |
| 4364 | pronoun | punctuation | pronoun → + / punctuation → − | 1.85 | 1.84 | + ` the old ladies was at[ once] turned to the\nmy`; ` soon enough.”[ I] knew then that I was`<br>− `instead of pastoral Buckinghamshire[.]\nThen his thoughts travelled`; `ha (13th century[).]\nIn this short article` |
| 2926 | auxiliary | auxiliary | capitalized → + / auxiliary → − | 1.71 | 1.54 | + ` 3-1 (25[-]13, 25-20`; ` head to head |…[<|endoftext|>]`<br>− `, swearing, laughing mass[ of]\nmore or less dil`; `… we’re[ not] just three All-Pro` |
| 799 | preposition | preposition | number → + / preposition → − | 0.91 | 1.43 | + `ley) before Drebin[ (]Leslie Nielsen) hits`; ` to hold up Jane Spencer[ (]Priscilla Presley`<br>− `A pained expression came[ into] her eyes for a moment`; `\n"I do,"[ was] the reply, delivered with` |
| 2536 | punctuation | punctuation | punctuation → + / auxiliary → − | 1.74 | 3.26 | + `es 4.[<|endoftext|>]Hip-Hop Wired`; `amobile.com/[<|endoftext|>]Try right clicking the entry`<br>− ` income). Don’[t] be shy about asking for`; ` Vellacott.[ To]-night his lips gave` |
| 3043 | punctuation | punctuation | punctuation → + / conjunction → − | 1.65 | 3.31 | + ` requested other members of team[\n]Promote positive attitude and`; `es 4.[<|endoftext|>]Hip-Hop Wired`<br>− ` answering this question[,] there is an inevitable tendency`; ` individual[ who] wants you to offer in` |
| 769 | pronoun | auxiliary | punctuation → + / auxiliary → − | 1.86 | 1.59 | + `es 4[.]<|endoftext|>Hip-Hop`; ` Version 3[.]5 is now released!`<br>− ` bowl game. I kind[ of] feel like you can be`; `\nAt this moment Sidney[ came] up and interrupted them.` |
| 1388 | conjunction | auxiliary | auxiliary → + / punctuation → − | 1.77 | 1.33 | + ` that is why HE did[ not] pressure me to be HIS`; ` the birds' wants had[ been] attended to, it was`<br>− `HeartRadio Music Awards.[<|endoftext|>]– Japan dismissed US warnings`; ` the amount is stolen.[\n]When you are generating a` |
| 3467 | preposition | pronoun | pronoun → + / preposition → − | 0.85 | 1.24 | + `Purchase a trial Online Edition[ subscription] and receive unlimited access for`; ` 1963.\nPurchase a[ trial] Online Edition subscription and receive`<br>− ` situation suddenly demanded their presence[.]\nThe following extract from`; ` order to secure that seat[.] From his\npocket he` |
| 2687 | auxiliary | auxiliary | auxiliary → + / number → − | 1.95 | 1.66 | + ` be punished.\nItalian[ researchers] say pasta eating is linked`; `incoln - The Nebraska soccer[ team] returns home with momentum after`<br>− `-17-2013 12[:]40 PM|\n|`; `-13-2013 02[:]52 PM|\nThis` |
| 4569 | pronoun | pronoun | punctuation → + / pronoun → − | 2.10 | 1.78 | + ` or taking 16 147g[.] Flexibility but did x`; ` my credentials i and lots[.] Shoe polish to databases`<br>− ` been\nironical.[ He] was just the sort of`; ` of the more imperative ideals[ now] begin to speak with an` |
| 1355 | punctuation | punctuation | punctuation → + / auxiliary → − | 1.81 | 3.39 | + ` Jefferson 12, Cambridge 4[—]After throwing a no-`; ` most on time airlines.[<|endoftext|>]Not sure what neighbourhood you`<br>− ` on itself from each edge[,] again creating little ball-`; ` its six-[game] losing streak on Friday in` |
| 762 | conjunction | preposition | punctuation → + / number → − | 0.99 | 1.66 | + `?"\n"What?"[ exclaimed] Christian with perfunctory`; `"But,"[ exclaimed] the butcher, emphatically banging`<br>− ` of those rare amazing.[ The] excellent was great all over`; ` are not put forward with[ any] idea of teaching those who` |
| 4034 | preposition | preposition | punctuation → + / pronoun → − | 1.06 | 1.89 | + `"But,"[ exclaimed] the butcher, emphatically banging`; ` that fact some years ago[,"] replied the sub-editor`<br>− ` little dough over on itself[ from] each edge, again creating`; `.\nUSAToday[ may] have thought they were making` |
| 3711 | auxiliary | auxiliary | capitalized → + / auxiliary → − | 1.46 | 1.28 | + ` you should know.\n[D]iversity is ultimate game plan`; ` on Wednesday, Southern Virginia[ (]3`<br>− ` received the compliment\nof[ being] banned by the Russian censor`; ` "The Sowers"[ is] still the first book the` |
| 3270 | punctuation | punctuation | preposition → + / punctuation → − | 1.04 | 4.42 | + `. Rebel losses were put[ at] 70, plus 5 who`; ` bottom of World Cup qualifying[ groups] F and A respectively.`<br>− ` to send a DM."[<|endoftext|>]Google does innovate in some`; ` head to head |…[<|endoftext|>]` |
| 4250 | punctuation | punctuation | conjunction → + / punctuation → − | 1.07 | 4.03 | + ` plugging the Strait of[ H]ormuz, a vital`; ` Rep. Sheila Jackson Lee[ expressed] concern last week about`<br>− ` ethic for other team members[\n]Sr. Representative EX`; ` and proper lane usage.[\n]Keep up with the recommended` |
| 4236 | punctuation | punctuation | auxiliary → + / punctuation → − | 1.65 | 4.10 | + ` indeed presents itself, and[ has] as a matter of history`; `inently intelligent. He\n[had] recognised the footstep while`<br>− `HeartRadio Music Awards.[<|endoftext|>]– Japan dismissed US warnings`; ` crap or not yet.[<|endoftext|>]Endowment Assets and Similar` |
| 74 | determiner | determiner | determiner → + / auxiliary → − | 1.24 | 4.74 | + ` should leave the apartment in[ a] clean state.\nA`; ` the left, and after[ a] few paces you\n`<br>− ` said Morot, in[ an] indifferent way which was`; ` note and slipped it into[ an] envelope, which he addressed` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 16 already includes attention-16 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
