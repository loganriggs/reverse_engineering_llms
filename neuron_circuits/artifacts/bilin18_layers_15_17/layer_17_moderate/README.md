# Bilin18 layer-17 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 17 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 51,200; held-out tokens: 25,600.
- Left/right factors choose the same coarse class for 31.3% of all channels and 36.2% of the more usable half.
- Among the more usable half, median held-out class lift is 1.38; 37.8% exceed 1.5× and 14.8% exceed 2×.

- Their median held-out signed class-to-class gap is 0.56 product standard deviations; 20.2% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 1239 | punctuation | preposition | pronoun → + / preposition → − | 1.45 | 1.44 | + ` the two old\nlad[ies] rubbed their hands, nodded`; ` in themselves of offence,[ I] see no reason why consistent`<br>− `language propaganda station.\n[Meanwhile] Wikileaks’ grassing`; `hetia.\nMembers[ of] the extended family of former` |
| 2563 | auxiliary | pronoun | capitalized → + / auxiliary → − | 1.54 | 1.35 | + `yet it was always there[--]an unceasing throb`; ` toiling race of men[--]not`<br>− ` 338px Unless noted Copyright[ 2010] Soul||Maka.`; `, gazing\nvague[ly] at the flying clouds.` |
| 137 | number | number | preposition → + / number → − | 1.67 | 1.92 | + ` special accommodation needs, contact[ Debbie] Nelson, commission operations manager`; ` come up."\nL[ori] Smetanka, director`<br>− ` age had less (7[.]1 percent) first-`; `02. She averaged 8[.]3 points, 3.` |
| 3533 | pronoun | auxiliary | auxiliary → + / capitalized → − | 1.81 | 1.55 | + ` size, portable air condition[ers]\natmosphere condition`; `, gazing\nvague[ly] at the flying clouds.`<br>− `ber, H.A[.] Tucker, and R.`; `, "Mechul Bir[ y]azarin…..". Ar` |
| 644 | punctuation | pronoun | pronoun → + / preposition → − | 0.75 | 1.99 | + ` Additional announcements are forthcoming.[<|endoftext|>]Workout Day………….`; ` crap or not yet.[<|endoftext|>]Endowment Assets and Similar`<br>− ` my credentials i and lots[.] Shoe polish to databases`; ` nobody said they've tested[.] EEG downtown so cynical that` |
| 3237 | punctuation | auxiliary | pronoun → + / punctuation → − | 1.37 | 1.58 | + `yet it was always there[--]an unceasing throb`; ` the second compositor as[ he] thinks of the extra five`<br>− ` my credentials i and lots[.] Shoe polish to databases`; ` nobody said they've tested[.] EEG downtown so cynical that` |
| 1811 | auxiliary | pronoun | auxiliary → + / punctuation → − | 1.57 | 1.41 | + ` 338px Unless noted Copyright[ 2010] Soul||Maka.`; `oritomo Kosh[o] had already told Koriron`<br>− ` crap or not yet.[<|endoftext|>]Endowment Assets and Similar`; ` head to head |…[<|endoftext|>]` |
| 3343 | preposition | pronoun | capitalized → + / auxiliary → − | 1.42 | 1.40 | + ` crap or not yet.[<|endoftext|>]Endowment Assets and Similar`; ` head to head |…[<|endoftext|>]`<br>− ` 338px Unless noted Copyright[ 2010] Soul||Maka.`; ` assured air of superiority\n[which] the people once tried in` |
| 1043 | punctuation | punctuation | punctuation → + / determiner → − | 1.52 | 3.76 | + ` crap or not yet.[<|endoftext|>]Endowment Assets and Similar`; `HeartRadio Music Awards.[<|endoftext|>]– Japan dismissed US warnings`<br>− `. And this one has[ great] depth and luscious`; `|\nThat system utilizes[ a] combination valve. No other` |
| 919 | determiner | auxiliary | punctuation → + / auxiliary → − | 1.71 | 1.46 | + ` crap or not yet.[<|endoftext|>]Endowment Assets and Similar`; `HeartRadio Music Awards.[<|endoftext|>]– Japan dismissed US warnings`<br>− ` sites. It is ideal[ to] stick to online shops with`; `.\nAll these feature[ updates] at present are exclusive to` |
| 2479 | auxiliary | punctuation | auxiliary → + / capitalized → − | 1.58 | 1.47 | + `creeping rot, finding[ its] source at the point of`; ` size, portable air condition[ers]\natmosphere condition`<br>− ` my credentials i and lots[.] Shoe polish to databases`; ` nobody said they've tested[.] EEG downtown so cynical that` |
| 3504 | punctuation | punctuation | punctuation → + / auxiliary → − | 1.24 | 4.47 | + ` crap or not yet.[<|endoftext|>]Endowment Assets and Similar`; `HeartRadio Music Awards.[<|endoftext|>]– Japan dismissed US warnings`<br>− ` email address to which you[ will] have subscription offers sent.`; ` online access to all content[ on] nybooks.com` |
| 2902 | pronoun | preposition | punctuation → + / pronoun → − | 1.41 | 1.50 | + ` upon that water,"\n[said] he in a conversational`; `"But,"[ exclaimed] the butcher, emphatically banging`<br>− ` 7, 2016 - 14[:]19 by Netceter`; ` 16, 2017 - 15[:]43 by Anna Kary` |
| 59 | pronoun | number | capitalized → + / number → − | 1.06 | 1.26 | + ` Jessica Backhaus, John[ G]ossage, Martin Par`; ` routes,” said[ M]zimela.\n`<br>− ` currently being accessed by more[ than] 400 agency representatives and available`; `day trial, on Oct[.] 2, the jury found` |
| 1111 | auxiliary | number | pronoun → + / capitalized → − | 1.27 | 1.64 | + ` Sci. 82[:]1753.\nL`; ` Anim. Sci. 50[:]173.\nSej`<br>− ` post), something very much[ caught] my eye and it just`; `\nI was supposed to[ meet] with Daniel's grandmother on` |
| 4094 | auxiliary | number | auxiliary → + / number → − | 2.34 | 2.11 | + ` the two old\nlad[ies] rubbed their hands, nodded`; ` said Morot, in[ an] indifferent way which was`<br>− ` 7, 2016 - 14[:]19 by Netceter`; ` 7, 2016 - 14[:]20 by Java Studio\n` |
| 2701 | determiner | determiner | capitalized → + / determiner → − | 1.07 | 6.65 | + ` said Morot, in[ an] indifferent way which was`; ` his face\nassumed[ an] air of childish bewilder`<br>− ` of an excellent imagination and[ a] steady and meticulous drawing hand`; ` flight was not sudden but[ a] steady trickle. The school` |
| 1588 | number | number | number → + / pronoun → − | 1.52 | 7.11 | + ` South African GDP at 2[.]1% or R 50`; ` Anim. Sci. 50[:]173.\nSej`<br>− ` persuaded teachers in the city[�]�s middle schools —`; ` said.\nThe county[�]�s` |
| 938 | number | number | number → + / conjunction → − | 2.19 | 6.34 | + `. Priced at only[ $]130 USD these are a`; ` 2.1% or[ R] 50.9-billion`<br>− ` retail sale of\nnew[sp]apers. At six o`; `economic policy and traditional �[�]prudential”` |
| 4344 | punctuation | punctuation | punctuation → + / preposition → − | 1.17 | 3.70 | + ` crap or not yet.[<|endoftext|>]Endowment Assets and Similar`; `HeartRadio Music Awards.[<|endoftext|>]– Japan dismissed US warnings`<br>− `ch und perfekt[ ge]legen.Es g`; `15th century) and[ N]âsir al-` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 17 already includes attention-17 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
