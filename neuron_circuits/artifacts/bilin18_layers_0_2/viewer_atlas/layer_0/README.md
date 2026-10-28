# Bilin18 layer-0 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 0 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 8,192,000; held-out tokens: 2,048,000.
- Left/right factors choose the same coarse class for 42.8% of all channels and 52.0% of the more usable half.
- Among the more usable half, median held-out class lift is 1.82; 73.7% exceed 1.5× and 40.9% exceed 2×.

- Their median held-out signed class-to-class gap is 0.88 product standard deviations; 41.4% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 1737 | determiner | determiner | determiner → + / capitalized → − | 2.42 | 3.43 | + ` three match affinity dating website[ the] more popular free dating websites`; `… We visited, Sharon[ the] health visitor and myself,`<br>− ` indoor grill + free shipping[<|endoftext|>]January 17, 2008\n`; ` could absolutely ruin matches.[<|endoftext|>]Prior to the establishment,` |
| 2264 | number | capitalized | capitalized → + / preposition → − | 1.08 | 1.32 | + `, Keighley and[ Ot]ley.<|endoftext|>A Thailand`; ` London boundary, then the[ Oy]ster Card (explained`<br>− `le advertisement (zazz[vertisement]?) on every stamp was`; `[ When] these four elements are in` |
| 3326 | capitalized | capitalized | punctuation → + / capitalized → − | 2.40 | 6.16 | + `nie & Marie Show.[�]� During pre-rec`; ` Continental Congress of Lighting.[�]� The focus was on`<br>− ` sex.\nAshley[ H]anes and Ethan Pierce both`; `atch.\n* Steven[ G]rieveson, 43` |
| 1492 | determiner | conjunction | capitalized → + / determiner → − | 2.12 | 3.15 | + ` the president of the Mic[hen]er Awards Foundation said the`; ` work straight away. Join[ us] for a few special days`<br>− `, to education, to[ the] environment.\n- 6`; ` approach to movement and to[ the] infinite corrections that need to` |
| 3926 | capitalized | capitalized | capitalized → + / content_word → − | 2.36 | 5.08 | + ` more than a million alleged[ Communists], ethnic Chinese and intellectuals`; ` was composed of 933[ Communists], eight Revolutionary-Commun`<br>− `12. Points To Keep[ In] Mind Before Hiring A`; ` I Try To Turn Everything[ Into] A Positive. I Give` |
| 3574 | punctuation | punctuation | punctuation → + / preposition → − | 1.24 | 4.42 | + ` those who simply love wood[.]<|endoftext|>St Joseph's Primary`; ` and excellent resistance to light[.]\nColor Swatch created`<br>− ` From Cd\n5[.] The Online Project Management Tools`; ` your personal data\n3[.]1 We collect your personal` |
| 1376 | pronoun | determiner | punctuation → + / capitalized → − | 0.54 | 1.27 | + `�s for everyone.[<|endoftext|>]Stay at home with our`; ` man at 13th.*[<|endoftext|>]The UT Dallas Callier`<br>− ` of study which deals with[ properties] and transformation of matter.`; ` to yesterday’s[ addition] of a handful of new` |
| 3069 | content_word | content_word | content_word → + / subword → − | 1.80 | 1.69 | + `.\n[Art]ificial lures have begun`; ` HOTTIE!!\n[Art]emis Once in a lifetime`<br>− `”\nThe women[�]�s division is also`; ` Along the way, they[�]�ll no doubt preserve` |
| 3120 | punctuation | punctuation | punctuation → + / preposition → − | 1.88 | 4.30 | + `IgM[)-]induced human B-cell`; ` exceeding 1.8 pounds[/]day for prepubert`<br>− ` Be Making You Sick ([The] Atlantic). The architecture critic`; `\nThe Episodes/[The] Designers\nBecause each` |
| 3389 | content_word | content_word | content_word → + / preposition → − | 1.60 | 2.50 | + `s products.\nAfter[ t]iptoeing around the`; `celeration Services\nby[ d]-wise | November 03`<br>− ` to market research firm I[RI], the industry witnessed a`; ` data item. The I[RI] data item now has an` |
| 906 | determiner | determiner | content_word → + / determiner → − | 2.64 | 5.49 | + `\nA study of the[ strain] by its developers at the`; ` Access Program to grant a[ select] group of academics and researchers`<br>− `t sell out or reach[ our] minimum we reserve the right`; `How we come to parent[ our] children is typically made up` |
| 1277 | determiner | determiner | determiner → + / capitalized → − | 1.42 | 2.11 | + `horror-war/[ that] had a great premise and`; `golf-course project[ that] also includes a mall and`<br>− ` Surendranath Mit[ra], the young man having`; ` Studieskolen in Bor[ger]gade, Copenhagen.` |
| 212 | conjunction | conjunction | capitalized → + / conjunction → − | 1.99 | 3.19 | + ` arks, where the[ straw] could be put on top`; ` sections, including over the[ tail]bone and at the upper`<br>− `\n"Maenads[ &] Muses: A Celebration`; ` absolutely right\nTerms[ &] Conditions\nAdvertise` |
| 3128 | auxiliary | auxiliary | determiner → + / auxiliary → − | 0.55 | 3.11 | + ` managed to remove the Sons['] VP, despite him being`; ` of power for both Mercedes['] which suddenly brought half-`<br>− ` tell\nher that he['s] not a traitor to his`; ` it's great that he['s] recognized and I think it` |
| 3722 | capitalized | pronoun | capitalized → + / punctuation → − | 0.95 | 3.40 | + `[And] now you’re`; `[And] I said to the savior`<br>− `USTEE, CALIF[O]\nFree format text:`; `IT POWERED BY AL[O]FT HOTELS:\n` |
| 591 | punctuation | punctuation | subword → + / punctuation → − | 0.83 | 3.01 | + ` started utilizing LikeToKnow[.]it to make my inst`; ` register at LikeToKnow[.]it and then when you`<br>− `.04.005.[\n]- Cotton, Christopher (`; `. grades saved me.[\n]I wasn’t` |
| 1642 | number | number | number → + / preposition → − | 4.24 | 13.35 | + `ic opened for business at[ 319] Broadway. IL FARO`; `[ 309]: 1559–15`<br>− ` with a new ten-[year] deal already negotiated to start`; ` after about the ten-[year] point, things just kind` |
| 3088 | number | number | determiner → + / number → − | 2.53 | 10.21 | + `o-standard Parallel Virtual[ Machine] (PVM) communication`; ` to occur during a real[ mission] to Mars was not tested`<br>− ` the season (694-[729]). OAK’`; `:52 | 6,[985] Views\nThese are Game` |
| 2869 | number | number | number → + / punctuation → − | 4.38 | 11.69 | + `. 1991, 32:[ 364]-372. 10.`; ` Act and punishable under Section[ 364] and 368 of the same`<br>− ` Stadium track, including the[ ever] popular Australian Jason Crump`; ` gig is more difficult than[ ever], and once you get` |
| 3305 | number | number | preposition → + / number → − | 3.75 | 10.91 | + `-month period ending Sept[.] 30,`; `earance that will expire Sept[.] 15. Around the same`<br>− ` following hockey for at least[ 78] years.\n“`; `% less hospital admissions\n[59]% less days in the` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 0 already includes attention-0 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
