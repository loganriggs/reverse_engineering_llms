# Bilin18 layer-15 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 15 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 51,200; held-out tokens: 25,600.
- Left/right factors choose the same coarse class for 36.4% of all channels and 38.8% of the more usable half.
- Among the more usable half, median held-out class lift is 1.30; 22.4% exceed 1.5× and 5.2% exceed 2×.

- Their median held-out signed class-to-class gap is 0.33 product standard deviations; 4.8% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 2070 | punctuation | auxiliary | pronoun → + / punctuation → − | 1.26 | 1.57 | + `, Arctic Sea Ice Sets[ a] New Low That Leads`; `, and probable consequences.[\n]Key actions: Identifies`<br>− ` — but not without adaptations[.]\nActivestills,`; ` usual this first half year[.]\nAnd I'm not` |
| 2416 | punctuation | punctuation | pronoun → + / punctuation → − | 1.31 | 2.44 | + `\n|Authors:[||]Durack, Cindy Ma`; `YRIC VIDEO FOR �[�]MIRACLES (`<br>− ` LYRIC VIDEO FOR[ �]�MIRACLES`; `t. (AP)[ -] State officials say Vermont will` |
| 978 | preposition | punctuation | determiner → + / punctuation → − | 1.42 | 1.90 | + ` scary!<|endoftext|>The focus[ around] the house lately has been`; ` about how long it took[ for] him as a young aspiring`<br>− ` for not only core services[ (]available to everyone), but`; ` further growth.\n�[�]My goal is to make` |
| 1577 | punctuation | punctuation | punctuation → + / pronoun → − | 1.52 | 2.04 | + ` eminently intelligent. He[\n]had recognised the footstep`; `stalls. You will[\n]probably go slowly because of`<br>− ` winning streak.\n"[I] feel like we have a`; `, as I’[d] rather listen to what people` |
| 2350 | punctuation | punctuation | conjunction → + / punctuation → − | 1.08 | 3.37 | + ` long way to go…[�]�\nThe dog owner`; ` much violence. Many Jama[icans] had left the island because`<br>− ` forward Shajuan Lindsey[ (]Midlothian,`; ` head to head |…[<|endoftext|>]` |
| 438 | punctuation | auxiliary | auxiliary → + / punctuation → − | 1.52 | 1.72 | + `Deposited By:[||]Concordia University`; ` to him by seafaring[\n]ancestors. The`<br>− ` requested other members of team[\n]Promote positive attitude and`; ` to send a DM."[<|endoftext|>]Google does innovate in some` |
| 3599 | punctuation | punctuation | auxiliary → + / punctuation → − | 1.27 | 3.66 | + `ant all hot, and[ was] slumberously enjoying a`; ` color to your look without[ breaking] the bank.\nMarc`<br>− ` most on time airlines.[<|endoftext|>]Not sure what neighbourhood you`; ` crap or not yet.[<|endoftext|>]Endowment Assets and Similar` |
| 1018 | number | number | determiner → + / content_word → − | 0.29 | 1.31 | + `, but also how a[ lot] will be billed for shipping`; ` one week to all the[ content] on nybooks.`<br>− `i on a vacation and[ fell] in love with the spirit`; ` he won't let anything[ stand] in his way!\n` |
| 311 | pronoun | auxiliary | pronoun → + / capitalized → − | 1.94 | 2.13 | + ` heart’s ability[ to] pump blood, Kameron`; `an Dhan Payo[ right] now for easy access this`<br>− `. Priced at only[ $]130 USD these are a`; ` has won a $10[,]000 lottery prize.\n` |
| 3506 | punctuation | auxiliary | auxiliary → + / capitalized → − | 1.71 | 1.11 | + ` the form of the lips[ that] lent`; `"What?" exclaimed Christian[ with] perfunctory interest,`<br>− ` 7, 2016 - 14[:]19 by Netceter`; ` South African GDP at 2[.]1% or R 50` |
| 1575 | auxiliary | punctuation | auxiliary → + / capitalized → − | 1.72 | 1.11 | + `Life Expectancy Shr[inks] for Less-Educated`; ` on defining the underlying mechanisms[ that] determine health and disease.`<br>− `� said Telltale CEO[ Dan] Conn`; ` reserved by the original copyright[ holders].<|endoftext|>Bobby Lab` |
| 3355 | preposition | auxiliary | auxiliary → + / capitalized → − | 1.68 | 1.11 | + ` smaller heifers because[ they] have less growth remaining to`; ` Virginia Tech found heif[ers] experience a body weight loss`<br>− `\nAdvertise on[ R]MD\nMixart`; `� said Telltale CEO[ Dan] Conn` |
| 901 | capitalized | auxiliary | auxiliary → + / capitalized → − | 1.80 | 1.16 | + `omo Kosho had[ already] told Koriron that the`; ` which ordinary men follow when[ they] are disputing with one`<br>− ` information to more than 3[.]84 million members, including`; `\nHOW ARE THESE GU[YS]?!? u forgot the` |
| 2036 | pronoun | auxiliary | auxiliary → + / capitalized → − | 1.73 | 1.11 | + ` the lumber. Kosh[o] had been sent to start`; ` the form of the lips[ that] lent`<br>− ` reserved by the original copyright[ holders].<|endoftext|>Bobby Lab`; `. May contain small amounts[ of] writing and/or highlighting` |
| 2271 | pronoun | auxiliary | pronoun → + / capitalized → − | 1.50 | 2.00 | + `nothing about Hilda?"[ she] said interrogatively.\n`; `.\n"Yes,"[ he] replied; "he has`<br>− `stam; Editing by[ Matt] Driskill)<|endoftext|>`; ` personal use only. All[ rights] reserved. All rights in` |
| 1693 | number | number | conjunction → + / number → − | 0.78 | 1.86 | + `nd Floor, Providence RI[ 0]2903. Directions and`; `nd Floor, Providence RI[ 0]2903\n- Saturday`<br>− ` over 3m (9[.]9ft) across on`; ` South African GDP at 2[.]1% or R 50` |
| 3417 | pronoun | pronoun | pronoun → + / determiner → − | 1.81 | 4.59 | + `. You know, when[ you]'re ready.\nAfter`; ` great starting off point.[ They] are tart, sometimes slightly`<br>− ` South African GDP at 2[.]1% or R 50`; ` facets.<|endoftext|>Stayed[ here] in November 2012 to celebrate` |
| 3173 | pronoun | pronoun | preposition → + / pronoun → − | 1.51 | 3.66 | + `\nladies rubbed their[ hands], nodded their palsied`; `? bactrim cost[ without] insurance 3 1 or f`<br>− ` Ohio State teaching awards.[ He] has been responsible for`; ` Todd Hido\nThis[ item] is currently unavailable from photo` |
| 3178 | determiner | determiner | capitalized → + / determiner → − | 0.55 | 3.72 | + ` that EADS had submitted[ an] unsolicited proposal to the`; ` in the Learning Network,[ an] umbrella organization financed by $`<br>− `."\nCharles Goldberg is[ a] Senior Writer at AuburnT`; ` band have also just premiered[ a] new lyric video for Mir` |
| 1106 | number | number | conjunction → + / number → − | 1.80 | 4.76 | + ` his crew chief, Chad[ K]naus, have proved`; ` your immune system supplements easy[ to] swallow. Protect your body`<br>− ` Bender. The 713[,]000-square-foot`; ` supported as many as 3[,]000 highly skilled, well` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 15 already includes attention-15 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
