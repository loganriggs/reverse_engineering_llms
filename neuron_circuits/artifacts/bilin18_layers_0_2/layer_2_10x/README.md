# Bilin18 layer-2 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 2 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 204,800; held-out tokens: 102,400.
- Left/right factors choose the same coarse class for 39.8% of all channels and 43.9% of the more usable half.
- Among the more usable half, median held-out class lift is 1.52; 52.8% exceed 1.5× and 16.6% exceed 2×.

- Their median held-out signed class-to-class gap is 0.56 product standard deviations; 13.5% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 526 | pronoun | subword | capitalized → + / determiner → − | 0.18 | 1.82 | + `[-] Action research (6 posts`; `[-] November 15th, 2008`<br>− ` Wetzel, former VP[-]consumer marketing at Pfizer`; ` your community. Click here[ to] get the Patch app for` |
| 3013 | preposition | content_word | pronoun → + / auxiliary → − | 0.35 | 1.43 | + ` doesn't mean he wouldn['t] be a useful target for`; ` weekend, there is little[ doubt] in the mind of Europeans`<br>− ` It’s far[ from] the troubling the lightest`; ` from 1939. They decided[ against] registering “nylon` |
| 2273 | punctuation | punctuation | conjunction → + / punctuation → − | 0.53 | 2.49 | + ` put five stars?�[�] Clearly it’s`; `m the player.�[�]’\nWhen reporters`<br>− ` a fan of today�[�]s over-sexualized`; ` of some of today�[�]s postings on Seeking Alpha` |
| 4412 | number | capitalized | number → + / punctuation → − | 1.29 | 2.39 | + `SSSSSSSSSS[SS]`; `SSSSSSSSSS[SS]SS`<br>− ` announced today that more than[ 85] hotels will earn the EN`; ` efforts from Brooks Law ([113]), Isaac Whittem` |
| 598 | number | subword | subword → + / conjunction → − | 0.45 | 1.37 | + ` don’t have[ parchment] paper. It might have`; ` Clannad, Cl[ann]ad: After Story and`<br>− ` — The Boyne City[ Glen]’s Market location`; `, CVE-2012-[11]65, CVE-2012` |
| 2907 | content_word | subword | subword → + / pronoun → − | 0.48 | 1.95 | + ` be able to make some[ plays] for the Tigers going forward`; ` acknowledged, so other CM[Ms] have been\nadopted`<br>− ` arrival of a UFO r[ouses] Godzilla from his underwater lair`; ` a broad subject that really[ interests] you. A massive effort` |
| 1142 | punctuation | punctuation | capitalized → + / punctuation → − | 0.58 | 2.17 | + ` staying on the field since[ then] and is currently on the`; ` ran the company until 1895[ when] he was forced to retire`<br>− ` called the Stage. It[�]�s an activity like`; ` 12th), but it[�]�s a chance to` |
| 901 | number | subword | conjunction → + / number → − | 0.21 | 1.38 | + ` is that Upton and U[gg]la are low-contact`; ` entire draft. His twin[ brother] Maurkice made the`<br>− ` Syracuse's top free-[throw] shooter, and in his`; `iche's steal and lay[up] with 18:57` |
| 1737 | capitalized | capitalized | capitalized → + / punctuation → − | 2.40 | 4.54 | + `[Multi] Multi award winning film\n`; `[Multi]-Channel Retailing Today`<br>− ` there's one in color[ado]!!!!!!!!!Â Â`; `derón de la cult[ura] mexicoameric` |
| 605 | subword | subword | conjunction → + / subword → − | 0.73 | 2.64 | + ` of our copies of Er[go] 3 in this shipment have`; `s smart historical satire Ar[go], based loosely on the`<br>− ` winning film maker Alast[air] Lee. Alastair`; `air Lee. Alast[air] is renowned for his innovative` |
| 899 | number | capitalized | number → + / pronoun → − | 0.60 | 1.68 | + ` also returned.\n-[ Assassin] recruiting will be more diverse`; ` against the cage.\n[Dem]etrious Johnson Has Too`<br>− ` TV on the cellphone.[ Videos] will catch on too.`; ` 4 times before the White[ Spot]\nrelents completely.` |
| 2852 | punctuation | pronoun | conjunction → + / capitalized → − | 0.44 | 1.61 | + ` in the Mail on Sunday[),] for which he makes an`; `NYSE: RTN)[ could] all hang in the balance`<br>− `5 batches for better coverage[)]\n4 large egg whites`; `-44 on the season[ —] the Mountaineers missed` |
| 2572 | punctuation | punctuation | number → + / punctuation → − | 1.22 | 3.13 | + `. His Linked-[in] profile lists historical organization after`; ` miserable.\nBut then[ there]’s that one`<br>− ` currently, Hanley said[.]\nThe firm`s`; `, Hefler said[.]\nEach suite will average` |
| 66 | content_word | subword | capitalized → + / content_word → − | 0.26 | 1.67 | + ` I accidentally stumbled into an[ anchors]' meeting where they discuss`; ` went to see a MD[s], and as it turned`<br>− ` those pathetic little medioc[rities] who in another time and`; ` of Crumb's song[books], a collection of Civil` |
| 3080 | number | pronoun | conjunction → + / capitalized → − | 0.30 | 1.62 | + `ized Victims," (Part[ I]), who are either "`; `September 25th, 2007[ §] § permalink\nIn`<br>− `SSSSSSSSSS[SS]`; `SSSSSSSSSS[SS]SS` |
| 3337 | punctuation | punctuation | pronoun → + / punctuation → − | 0.38 | 2.49 | + `[\n]It researches programs on`; `[\n]On the practical side,`<br>− ` retailers offer electronic coupons ([�]�favorites�`; ` that people ask…\n[�]�Loco, just` |
| 223 | number | number | number → + / determiner → − | 2.46 | 8.27 | + ` northeast Arkansas will cost $[215],000 to create conservation`; `.8mm): £[89].99\nMore information`<br>− ` for two years in a[ row] a resolution calling for integrity`; ` the second year in a[ row]. Business adoption is strong` |
| 4027 | auxiliary | auxiliary | conjunction → + / auxiliary → − | 2.88 | 6.71 | + `-release, there can[ and] will be bugs, so`; ` to me but it could[ and] seems likely) so I`<br>− ` war for us? How[ is] it that these paradig`; ` the year.\nWhy[ am] I telling you this?` |
| 3610 | auxiliary | auxiliary | auxiliary → + / number → − | 3.00 | 6.19 | + `, it seemed like we[ could] all finally breathe a bit`; ` show. Sure, she['s] a cute and funny kid`<br>− ` and has been having he[aps] of fun! The word`; ` the animal.\nWhere[ does] it come from:\n` |
| 3455 | preposition | preposition | preposition → + / capitalized → − | 2.52 | 5.32 | + ` funding, and engages experts[ in] science and economics who can`; `formists. He believed[ in] demons, angels and spirits`<br>− ` fountain pen.\nFrom[ time] immemorial, the`; ` the sun came out.[ Sarah] and I came out of` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 2 already includes attention-2 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
