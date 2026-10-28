# Bilin18 layer-0 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 0 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

This supersedes the smaller 20,480/10,240-token pilot. It uses exactly ten
times as many tokens for both discovery and held-out evaluation. The metric,
coarse class vocabulary, selection rule, and model are unchanged.

## Architecture note

Every block reinjects the RMS-normalized token embedding through
`alpha_l * x_l + beta_l * x0`. At layer 0, `x_l` and `x0` are the same vector,
so their positive common scaling is removed by the following RMSNorm. The
layer-0 MLP nevertheless receives the attention-0 write as well as that lexical
path, so this atlas describes contextual layer-0 states rather than a pure
embedding lookup.

## Aggregate

- Discovery tokens: 204,800; held-out tokens: 102,400.
- Left/right factors choose the same coarse class for 41.8% of all channels and 51.1% of the more usable half.
- Among the more usable half, median held-out class lift is 1.84; 74.2% exceed 1.5× and 41.1% exceed 2×.

- Their median held-out signed class-to-class gap is 0.90 product standard deviations; 42.3% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Stability against the original 1x pilot

The aggregate conclusions did not shrink with ten times more data. Median
held-out lift in the more usable half changed from 1.83 to 1.84, the fraction at
or above 2x changed from 40.1% to 41.1%, and the median signed gap changed from
0.88 to 0.90 standard deviations.

Several preselected examples were highly stable: unit 1737 retained its
determiner/determiner factors and a 2.40-SD signed gap; unit 1642 retained its
number/number factors and strengthened from a 3.27- to 4.31-SD gap; units 3574,
3326, and 3069 retained both factor labels and signed separators. Unit 2264 did
not: its factor labels changed from conjunction/conjunction to
number/capitalized and its held-out lift fell from 1.88x to 1.28x. It is an
explicit failed coarse interpretation, not evidence for a clean conjunction
neuron.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 1737 | determiner | determiner | determiner → + / number → − | 2.40 | 3.39 | + ` Black Grand Champion Male form[ the] Hemingway & Bu`; ` approximate sound made on board[ a] US submarine as it rises`<br>− ` John Roberts differs from his[ predecessor], William Rehnquist`; ` BuzzFeed, told Betab[eat] via Gchat.\n` |
| 2264 | number | capitalized | capitalized → + / preposition → − | 1.08 | 1.28 | + ` ratio of drinks prices to[ calib]re of view we move`; `)\n- MKX[ AW]D: 17/23`<br>− `[ when], in his words,`; ` history[ and] facts. Today I�` |
| 3326 | capitalized | capitalized | punctuation → + / capitalized → − | 2.36 | 6.00 | + `Saving Hip Hop.[�]� The FREE`; `Tarrant County).[�]�\nMore than 8`<br>− `\n- Add support to[ Warden] for using ZFS snapshots`; `mins)\nSKYE[ W]ALL (trad)*` |
| 1492 | determiner | conjunction | capitalized → + / determiner → − | 2.12 | 3.13 | + ` jumpers and Peyton S[iva] made a dunk during the`; ` outperformed “Go[ Global]” this year.`<br>− `\nClinging to[ the] Favored Christian life is`; ` expression we can comprehend to[ a] remarkable degree and that the` |
| 3926 | capitalized | capitalized | capitalized → + / content_word → − | 2.30 | 4.94 | + ` all of them.\n[T]revor Weymouth`; ` create a notebook that met[ M]oleskine’`<br>− ` I WILL LOOK YOU UP[ WHEN] I GET THERE SAT..`; `-technologies for tele[communication] and nuclear medicine for radi` |
| 3574 | punctuation | punctuation | punctuation → + / preposition → − | 1.25 | 4.38 | + ` boost voter participation among troops[.]\nThe blame for the`; ` scientists might meet that demand[.] I also became acutely aware`<br>− ` take digestive enzyme\n6[.]30-7pm take`; ` phone power.\n4[.] Get the iPhone app.` |
| 3069 | content_word | content_word | content_word → + / subword → − | 1.81 | 1.72 | + `Below, I essentially[ replicate] his results with the purpose`; ` himself away from the six[ household] dogs), you can pick`<br>− ` in the set above didn[�]�t appear in print`; `. Of course, don[�]�t forget to protect` |
| 3120 | punctuation | punctuation | punctuation → + / preposition → − | 1.89 | 4.36 | + `s postings on Seeking Alpha[ (]a popular financial markets website`; ` that low natural gas prices[ (]not much higher than current`<br>− ` to the original. ([The] latest one: from the`; ` newest book "Scra[ps] & Shirtails 2"` |
| 3389 | content_word | content_word | content_word → + / preposition → − | 1.61 | 2.52 | + `[ d]reary Seattle winter months`; `.\nThe flow and[ d]rape of the Liberty of`<br>− ` Traffic in Women. I[th]aca: Cornell University Press`; `part singing in the I[st]rian scale, in I` |
| 3128 | auxiliary | auxiliary | determiner → + / auxiliary → − | 0.53 | 3.03 | + ` but it's New Orleans['] worst nightmare. Scientists engineer`; ` is a Ulysses['] cave on the island of`<br>− ` "\nCertainly, he['s] remarkably easygoing about how`; ` dissension is that he['s] been lured over to the` |
| 3722 | capitalized | pronoun | capitalized → + / punctuation → − | 0.90 | 3.33 | + `[How] AP reported in all formats`; `[How] To Win At Science F`<br>− ` political leadership other than how[ neat] it is to stone people`; `\nAlthough more requests are[ coming] in every day, the` |
| 906 | determiner | determiner | content_word → + / determiner → − | 2.60 | 5.30 | + `"It's not a[ matter] of their voices being denied`; ` especially poor ones, a[ matter] of public policy.\n`<br>− ` is to lovingly nurture[ our] partnerships – whether they�`; ` how we sleep, and[ our] feelings about ourselves as a` |
| 1277 | determiner | determiner | determiner → + / capitalized → − | 1.49 | 2.15 | + `istry and hormones) and[ what] is going on outside you`; `-1) and account[ for] ~ 90% of the`<br>− `affaaele Sol[le]cito reopens with`; ` Raffaele Sol[le]cito promise huge surprises` |
| 591 | punctuation | punctuation | subword → + / punctuation → − | 0.86 | 3.05 | + ` Clomid For Sale[.] Avodart For Sale`; ` performing Rolling In The Deep[.] (C) 2010 XL`<br>− ` world endures.�[�]\nSeries: Divine Action`; ` the national media conglomerates[."]\nRelenting to pressure` |
| 212 | determiner | conjunction | capitalized → + / conjunction → − | 1.94 | 3.12 | + ` involves wagging of the[ tail] and trembling of the body`; ``` and have a separate[ living] room with microwave oven,`<br>− `N. Hingley[ &] Sons Ltd, Washington Street`; `, West Norwood Library[ &] Assistant Branch Secretary\nAccording` |
| 874 | punctuation | number | punctuation → + / pronoun → − | 0.19 | 2.60 | + `ican America - Introduction\n["]Mexican America" is`; ` Rocky road to Philly\n["]After a short train journey`<br>− `\ninformation is "Hidden["] within the text for those`; `.\nThe "function["] of "gatekeeping"` |
| 1642 | number | number | number → + / preposition → − | 4.31 | 13.68 | + `hew, I only have[ 309] followers.)\nPosted a`; ` of unique users has grown[ 324]% in the same period`<br>− ` on goodness was rather ten[uous], that it was contingent`; ` love life, his ten[uous] friendships, his alcoholism,` |
| 2869 | number | number | number → + / punctuation → − | 4.32 | 11.74 | + ` 13 on flat bench,[ 455] x 5 on decline bench`; `professional content. There were[ 144] (3%) unprofessional`<br>− ` Diego seems more vibrant than[ ever] and the perfect place to`; ` outreach efforts are better than[ ever], and warned against comparing` |
| 3088 | number | number | determiner → + / number → − | 2.69 | 10.83 | + ` unique quality of the student[ body]. What are some of`; `.UU.), convirt[i]éndose en el`<br>− ` the number was 41,[762].\nAlthough more requests`; ` Butterfly Boucher and 5[678]!\nUh, huh` |
| 1107 | number | number | number → + / punctuation → − | 4.23 | 10.72 | + ` state champion Ponies totaled[ 249].5 points to best`; ` Included with general admission ($[11] for adults, $9`<br>− ` it.\nSuch now[-]deceased contemporaries as`; ` state health commissioner and then[-]Gov. David Paterson` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 0 already includes attention-0 context, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
