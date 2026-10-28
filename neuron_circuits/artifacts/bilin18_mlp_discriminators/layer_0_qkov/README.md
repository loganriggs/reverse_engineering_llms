# Layer-0 QK×OV paths into native bilinear MLP neurons

This report uses 102,400 held-out FineWeb token positions from rows 400--599. For each preselected unit, activation sites are selected only within this held-out panel.

For a fixed destination and source, `path L = QK1 × QK2 × OV→L`; the R path is analogous. The sum of these paths plus the direct current-token residual reconstructs the two pre-RMS factor inputs. RMSNorm supplies a shared positive scale, after which the native product is `Left × Right`.

The source paths are exact fixed-input forward decompositions, not causal deletion effects. A large signed term may cancel another term.

## Mechanistic reading

- **Unit 1737 is primarily lexical.** On the clean token ` the`, removing the
  entire attention-0 write changes the product from about 3,202 to 4,367 but
  does not change its sign. Its determiner discrimination is already present in
  the direct embedding path; attention generally calibrates it. Some negative
  subword cases receive much larger contextual amplification.
- **Unit 1642 is a digit discriminator plus a contextual rejection circuit.**
  Actual digits remain strongly positive without attention (4,522 versus 4,489
  for ` 373`). But the negative `ten|ures` and `ten|or` cases become positive or
  nearly zero without attention. Head 3 reads the preceding ` ten` and writes
  opposite signs into Left and Right, changing the product to roughly -467.
- **Unit 3574 distinguishes sentence periods from numbered-list periods.** A
  period alone gives a positive product of about 2,029. Ordinary sentence-final
  contexts raise this to 2,624--2,845. For `8.` and `5.` list markers, heads 2
  and 6 read the preceding newline and numeral and drive Right negative; the
  products become -1,720 and -1,703. Zeroing attention restores both to +2,029.
- **Unit 874 is an attention-constructed quote-role discriminator.** A quote by
  itself has product about -151. After a newline, primarily head 6 writes a
  large positive Right factor and the product becomes +3,641--3,691, matching an
  opening-quotation role. When an earlier opening quote/space is present,
  heads 7, 6, and 8 write Right negative and the closing quote becomes
  -4,720--5,163. The old coarse `punctuation` label hid this clean contextual
  computation.
- **Unit 3326 is mostly lexical on capitalized ` H`, but contextual on malformed
  quote bytes.** Attention barely changes the negative capitalized examples
  (-3,247 to about -3,300), while a preceding period routed mainly through head
  3 amplifies the positive malformed-byte cases from 85 to 399--492.
- **Unit 3069 mixes direct word-piece evidence with boundary context.** ` travel`
  is already positive without attention (1,259) and is reinforced to 1,691 by
  ` to` and ` allowed`. At sentence-initial `Art`, newline and period paths
  through heads 6 and 3 jointly amplify a positive same-sign product from 823
  to 1,930.

These descriptions are stronger than activation labels because they identify
executable QK routing and OV writes and survive a finite all-attention ablation.
They still do not prove that a named path is necessary for language-model loss;
the next causal granularity is to ablate the implicated head/source edges and
measure both the neuron and next-token logits on a broader, predeclared panel.

## Unit 1737

### positive tail: ` are identical twins who combine to form[ the] musical comedy duo, “The`

- Native factors: Left +54.5, Right +59.8, product +3.26e+03.
- Zeroing the entire attention-0 write at this position gives Left +63.8, Right +68.5, product +4.37e+03.
- Raw Left: direct +776 + attention +42.5 = +818.
- Raw Right: direct +834 + attention +61 = +895.
- Attention absolute-path share: Left 28.2%, Right 33.0%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` form`|-1|+0.371|+0.414|+0.153|+208|+32|+627|+96.1|
|3|` the`|0|+0.43|+0.547|+0.235|-227|-53.4|-262|-61.5|
|6|` combine`|-3|-0.467|+0.241|-0.113|-86.3|+9.73|-171|+19.3|
|5|` who`|-4|+0.424|+0.285|+0.121|+67.1|+8.13|-37.2|-4.5|
|6|` form`|-1|-0.223|+0.17|-0.0378|-214|+8.1|-91.5|+3.46|
|6|` identical`|-6|-0.208|+0.169|-0.0352|-66.9|+2.35|-308|+10.8|

### positive tail: `[ the] salted caramel that came on the`

- Native factors: Left +54.5, Right +58.8, product +3.2e+03.
- Zeroing the entire attention-0 write at this position gives Left +63.8, Right +68.5, product +4.37e+03.
- Raw Left: direct +776 + attention -58.6 = +717.
- Raw Right: direct +834 + attention -60.7 = +774.
- Attention absolute-path share: Left 7.5%, Right 7.6%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` the`|0|+0.432|+0.547|+0.236|-227|-53.6|-262|-61.8|
|6|` the`|0|-0.711|+0.125|-0.0889|+48.8|-4.34|+7.96|-0.707|
|8|` the`|0|+0.082|+0.527|+0.0432|-47|-2.03|-31.7|-1.37|
|5|` the`|0|+0.398|-0.243|-0.0967|+4.33|-0.418|-31.4|+3.04|
|7|` the`|0|-0.424|-0.0486|+0.0206|+88.7|+1.83|+39.1|+0.807|
|2|` the`|0|+0.0771|+0.0986|+0.0076|-61.7|-0.469|-56.4|-0.428|

### negative tail: ` Buddha's wife, Yasodh[ara]. Devadatta is said to`

- Native factors: Left -18.8, Right +12.2, product -229.
- Zeroing the entire attention-0 write at this position gives Left -18.4, Right +11.4, product -209.
- Raw Left: direct -224 + attention -89.3 = -313.
- Raw Right: direct +139 + attention +64.3 = +203.
- Attention absolute-path share: Left 40.9%, Right 49.0%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`h`|-1|-0.516|-0.451|+0.232|-309|-71.7|+127|+29.5|
|3|`od`|-2|-0.469|-0.336|+0.157|-87|-13.7|+201|+31.6|
|3|`ara`|0|-0.311|-0.117|+0.0361|-214|-7.72|+232|+8.4|
|3|`'s`|-6|-0.206|-0.0427|+0.00879|-489|-4.3|-329|-2.89|
|3|` Buddha`|-7|-0.139|-0.0635|+0.00879|-118|-1.03|+518|+4.55|
|3|`,`|-4|-0.326|+0.0354|-0.0115|-278|+3.2|-188|+2.17|

### negative tail: `\nalso hired trumpeter Roy Eld[ridge]\n, and the trio combined to`

- Native factors: Left +13.7, Right -14.9, product -204.
- Zeroing the entire attention-0 write at this position gives Left +9.5, Right -8.62, product -81.9.
- Raw Left: direct +115 + attention +151 = +266.
- Raw Right: direct -105 + attention -185 = -290.
- Attention absolute-path share: Left 83.0%, Right 84.7%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` Eld`|-1|+0.551|+0.52|+0.287|+487|+140|-470|-135|
|3|` Roy`|-2|-0.436|-0.144|+0.0625|+511|+32|-449|-28.1|
|6|` contests`|-44|+0.155|+0.151|+0.0236|-99.8|-2.35|-213|-5.02|
|6|`ridge`|0|-0.242|-0.206|+0.0498|+54.2|+2.7|+80.6|+4.02|
|8|` hired`|-5|+0.0977|+0.221|+0.0216|+0.00384|+8e-05|+289|+6.24|
|3|` ex`|-270|-0.0396|-0.312|+0.0123|-347|-4.28|+117|+1.45|

## Unit 1642

### positive tail: `220 U.S.[ 373]\nDR. MILES MEDICAL`

- Native factors: Left -68, Right -66.5, product +4.52e+03.
- Zeroing the entire attention-0 write at this position gives Left -67, Right -67, product +4.49e+03.
- Raw Left: direct -817 + attention -49.9 = -867.
- Raw Right: direct -819 + attention -24.8 = -844.
- Attention absolute-path share: Left 11.3%, Right 14.2%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`.`|-1|-0.316|-0.258|+0.0815|-403|-32.9|+251|+20.5|
|0|` 373`|0|+0.275|-0.314|-0.0864|+93.5|-8.08|+229|-19.8|
|6|`220`|-5|-0.236|+0.162|-0.0383|+307|-11.8|+362|-13.9|
|6|` 373`|0|-0.254|-0.139|+0.0352|+282|+9.91|+247|+8.69|
|7|`220`|-5|-0.25|+0.299|-0.0747|-27.2|+2.03|+190|-14.2|
|6|`.`|-1|-0.318|-0.177|+0.0562|+49.6|+2.78|+136|+7.62|

### positive tail: ` their [220 U.S.[ 373], 375] good will and commend`

- Native factors: Left -62.8, Right -59.8, product +3.75e+03.
- Zeroing the entire attention-0 write at this position gives Left -67, Right -67, product +4.49e+03.
- Raw Left: direct -817 + attention -159 = -977.
- Raw Right: direct -819 + attention -110 = -930.
- Attention absolute-path share: Left 48.3%, Right 50.5%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`.`|-1|-0.316|-0.258|+0.0815|-403|-32.9|+251|+20.5|
|0|` 373`|-227|+0.381|-0.25|-0.0952|+93.5|-8.9|+229|-21.8|
|0|` 373`|0|+0.275|-0.314|-0.0864|+93.5|-8.08|+229|-19.8|
|6|`220`|-5|-0.236|+0.162|-0.0383|+307|-11.8|+362|-13.9|
|6|` 164`|-166|-0.103|-0.188|+0.0193|+405|+7.82|+652|+12.6|
|6|` 373`|0|-0.254|-0.138|+0.0349|+282|+9.84|+247|+8.63|

### negative tail: `. After bad experiences amid brief ten[ures] with Benny Goodman\nand even Raymond`

- Native factors: Left +22.4, Right -20.9, product -467.
- Zeroing the entire attention-0 write at this position gives Left -1.66, Right -5.94, product +9.83.
- Raw Left: direct -20.2 + attention +433 = +413.
- Raw Right: direct -72.2 + attention -313 = -385.
- Attention absolute-path share: Left 97.4%, Right 90.0%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` ten`|-1|-0.5|-0.535|+0.268|+1.51e+03|+405|-1.05e+03|-281|
|3|` brief`|-2|-0.455|-0.226|+0.103|+222|+22.7|-331|-34|
|3|`ures`|0|-0.283|-0.113|+0.032|-502|-16.1|-463|-14.8|
|3|` ex`|-234|-0.0469|-0.159|+0.00745|-589|-4.38|+615|+4.58|
|3|` amid`|-3|-0.346|+0.0544|-0.0188|-314|+5.9|-46.3|+0.871|
|3|` incorporated`|-120|-0.0466|+0.107|-0.00497|+437|-2.18|+785|-3.91|

### negative tail: `amic landscapes. The songs for ten[or] travel in a narrower path—t`

- Native factors: Left +35.8, Right -13.1, product -467.
- Zeroing the entire attention-0 write at this position gives Left +16.9, Right +6.47, product +109.
- Raw Left: direct +206 + attention +448 = +654.
- Raw Right: direct +78.9 + attention -318 = -239.
- Attention absolute-path share: Left 80.4%, Right 89.8%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` ten`|-1|-0.508|-0.57|+0.289|+1.51e+03|+437|-1.05e+03|-303|
|3|`or`|0|-0.492|-0.154|+0.0762|+177|+13.5|+264|+20.1|
|3|`or`|-8|-0.188|+0.271|-0.0513|+177|-9.07|+264|-13.5|
|5|` The`|-4|+0.281|+0.205|+0.0576|-146|-8.4|-96.2|-5.54|
|3|`Peter`|-172|+0.0698|+0.104|+0.00729|+776|+5.66|+528|+3.85|
|3|`min`|-102|-0.0679|-0.2|+0.0136|-438|-5.96|-236|-3.21|

## Unit 3574

### positive tail: ` they happen such as saving his friends[.]\nThe element of friendship is a`

- Native factors: Left +55.2, Right +51.5, product +2.85e+03.
- Zeroing the entire attention-0 write at this position gives Left +56.8, Right +35.8, product +2.03e+03.
- Raw Left: direct +693 + attention +96.8 = +790.
- Raw Right: direct +436 + attention +301 = +738.
- Attention absolute-path share: Left 60.0%, Right 76.4%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|2|`.`|0|-0.209|-0.574|+0.12|-247|-29.7|-411|-49.3|
|3|` friends`|-1|-0.293|-0.232|+0.0679|-369|-25.1|+738|+50.1|
|5|` saving`|-3|+0.406|+0.367|+0.149|+126|+18.8|+147|+22|
|7|`.`|0|+0.494|-0.146|-0.0718|-172|+12.4|-421|+30.2|
|3|`.`|0|-0.432|-0.492|+0.213|+60.2|+12.8|-96.8|-20.6|
|2|` events`|-9|+0.352|+0.334|+0.117|+133|+15.6|+143|+16.8|

### positive tail: ` enemy has focused its attack on you[.] Once you select and use an Art`

- Native factors: Left +55.2, Right +47.5, product +2.62e+03.
- Zeroing the entire attention-0 write at this position gives Left +56.8, Right +35.8, product +2.03e+03.
- Raw Left: direct +693 + attention +115 = +808.
- Raw Right: direct +436 + attention +259 = +695.
- Attention absolute-path share: Left 53.8%, Right 70.2%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|2|`.`|0|-0.21|-0.57|+0.12|-247|-29.5|-411|-49.1|
|3|` you`|-1|-0.342|-0.385|+0.132|+186|+24.5|+287|+37.8|
|7|`.`|0|+0.494|-0.145|-0.0713|-172|+12.3|-421|+30|
|3|`.`|0|-0.432|-0.492|+0.213|+60.2|+12.8|-96.8|-20.6|
|6|`.`|0|+0.754|+0.191|+0.145|-162|-23.3|-35|-5.06|
|8|` you`|-1|+0.326|+0.279|+0.0913|+254|+23.2|+55.9|+5.1|

### negative tail: `od Dream by Ted Wood\n8[.] Overhead projector, overhead and pen`

- Native factors: Left +34.8, Right -49.5, product -1.72e+03.
- Zeroing the entire attention-0 write at this position gives Left +56.8, Right +35.8, product +2.03e+03.
- Raw Left: direct +693 + attention -74.7 = +618.
- Raw Right: direct +436 + attention -1.32e+03 = -879.
- Attention absolute-path share: Left 61.0%, Right 83.6%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|2|`\n`|-2|-0.652|-0.582|+0.379|-195|-73.9|-1.3e+03|-494|
|6|`8`|-1|+0.648|+0.602|+0.391|+116|+45.3|-569|-222|
|2|`.`|-12|-0.365|-0.484|+0.177|-247|-43.6|-411|-72.6|
|6|`\n`|-2|+0.754|+0.169|+0.127|+363|+46.1|-299|-37.9|
|8|`\n`|-2|+0.566|-0.52|-0.295|-40.4|+11.9|+359|-106|
|2|`.`|0|-0.21|-0.574|+0.121|-247|-29.8|-411|-49.5|

### negative tail: `\n2. DVD player\n5[.] Learning center and trade books\n6`

- Native factors: Left +28.6, Right -59.5, product -1.7e+03.
- Zeroing the entire attention-0 write at this position gives Left +56.8, Right +35.8, product +2.03e+03.
- Raw Left: direct +693 + attention -148 = +544.
- Raw Right: direct +436 + attention -1.57e+03 = -1.13e+03.
- Attention absolute-path share: Left 62.5%, Right 84.9%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|2|`\n`|-2|-0.652|-0.582|+0.379|-195|-73.9|-1.3e+03|-494|
|2|`.`|-5|-0.664|-0.703|+0.467|-247|-115|-411|-192|
|6|`5`|-1|+0.648|+0.594|+0.385|+237|+91.1|-335|-129|
|2|`\n`|-7|-0.504|-0.231|+0.117|-195|-22.8|-1.3e+03|-152|
|5|`.`|-5|+0.498|+0.404|+0.201|-276|-55.5|-281|-56.5|
|6|`\n`|-2|+0.754|+0.17|+0.128|+363|+46.4|-299|-38.2|

## Unit 3326

### positive tail: `�Waves Breaking on Rocks.[�]� Though the instrumentation is larger`

- Native factors: Left +24.8, Right +19.9, product +492.
- Zeroing the entire attention-0 write at this position gives Left +17.5, Right +4.84, product +84.8.
- Raw Left: direct +213 + attention +194 = +406.
- Raw Right: direct +58.9 + attention +267 = +326.
- Attention absolute-path share: Left 79.2%, Right 92.3%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`.`|-1|+0.463|+0.406|+0.188|+741|+140|+270|+50.9|
|3|`�`|0|-0.334|-0.451|+0.15|+253|+38.1|+136|+20.4|
|7|`.`|-1|-0.457|+0.516|-0.235|-97|+22.8|-138|+32.6|
|2|`�`|-7|+0.637|+0.621|+0.395|+51.1|+20.2|+37.4|+14.8|
|7|` Rocks`|-2|+0.332|-0.32|-0.106|+13.6|-1.44|-123|+13|
|2|`.`|-1|+0.43|+0.484|+0.208|+34.5|+7.17|-38.1|-7.93|

### positive tail: `�The Roque Dalton Songs.[�]� Garland is an American composer whose`

- Native factors: Left +24, Right +16.6, product +399.
- Zeroing the entire attention-0 write at this position gives Left +17.5, Right +4.84, product +84.8.
- Raw Left: direct +213 + attention +170 = +383.
- Raw Right: direct +58.9 + attention +207 = +267.
- Attention absolute-path share: Left 73.4%, Right 89.1%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`.`|-1|+0.463|+0.408|+0.188|+741|+140|+270|+50.9|
|3|`�`|0|-0.334|-0.449|+0.15|+253|+38.1|+136|+20.4|
|7|`.`|-1|-0.459|+0.512|-0.235|-97|+22.8|-138|+32.6|
|2|`�`|-7|+0.637|+0.625|+0.398|+51.1|+20.4|+37.4|+14.9|
|7|` Songs`|-2|+0.383|-0.359|-0.138|+89.7|-12.3|-52.4|+7.22|
|7|`The`|-6|+0.273|-0.205|-0.0562|-181|+10.2|+134|-7.55|

### negative tail: ` that for sure."\nAdrian[ H]ouser, who helped guide his`

- Native factors: Left -56, Right +59.8, product -3.35e+03.
- Zeroing the entire attention-0 write at this position gives Left -55.5, Right +58.5, product -3.25e+03.
- Raw Left: direct -678 + attention -173 = -851.
- Raw Right: direct +712 + attention +196 = +908.
- Attention absolute-path share: Left 50.4%, Right 48.8%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`Ad`|-2|+0.398|+0.207|+0.0825|+73.2|+6.04|+581|+48|
|3|`rian`|-1|-0.359|-0.147|+0.053|-345|-18.3|+171|+9.08|
|8|`\n`|-3|+0.295|-0.338|-0.0996|+142|-14.2|-91|+9.06|
|3|` H`|0|-0.434|-0.135|+0.0583|-280|-16.3|+71|+4.14|
|4|`ous`|-17|+0.205|-0.159|-0.0327|+324|-10.6|-275|+9.01|
|7|`Ad`|-2|+0.245|-0.283|-0.0693|+172|-11.9|-86.3|+5.98|

### negative tail: `aylor.com.\nKATE[ H]UDSON AND GWYNET`

- Native factors: Left -56.5, Right +58.2, product -3.29e+03.
- Zeroing the entire attention-0 write at this position gives Left -55.5, Right +58.5, product -3.25e+03.
- Raw Left: direct -678 + attention -123 = -801.
- Raw Right: direct +712 + attention +113 = +825.
- Attention absolute-path share: Left 42.1%, Right 38.7%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|`\n`|-3|+0.295|-0.338|-0.0996|+142|-14.2|-91|+9.06|
|3|`ATE`|-1|-0.346|-0.164|+0.0566|+209|+11.9|-151|-8.53|
|3|` H`|0|-0.434|-0.135|+0.0583|-280|-16.3|+71|+4.14|
|3|`.`|-4|-0.338|+0.053|-0.0179|+741|-13.3|+270|-4.85|
|7|`K`|-2|-0.307|+0.242|-0.0742|+66|-4.9|-104|+7.68|
|3|`K`|-2|-0.297|-0.1|+0.0297|-72.2|-2.14|+314|+9.32|

## Unit 3069

### positive tail: ` same time the following week.\n[Art] school may have shown you how to`

- Native factors: Left -46.5, Right -41.5, product +1.93e+03.
- Zeroing the entire attention-0 write at this position gives Left -33.2, Right -24.8, product +823.
- Raw Left: direct -405 + attention -319 = -725.
- Raw Right: direct -301 + attention -343 = -645.
- Attention absolute-path share: Left 58.7%, Right 66.4%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|`\n`|-1|+0.645|+0.439|+0.283|-676|-192|-581|-164|
|3|`\n`|-1|+0.428|+0.342|+0.146|-234|-34.3|-400|-58.6|
|3|`.`|-2|-0.395|-0.104|+0.0408|-842|-34.3|-1.01e+03|-41.4|
|3|`Art`|0|-0.512|-0.11|+0.0562|-366|-20.6|-545|-30.6|
|6|`Art`|0|+0.516|+0.249|+0.128|-174|-22.3|-222|-28.4|
|6|`.`|-2|+0.418|+0.283|+0.118|+122|+14.4|+107|+12.6|

### positive tail: ` whose vision and graft allowed me to[ travel] from London to Elgin (near`

- Native factors: Left +40.8, Right +41.5, product +1.69e+03.
- Zeroing the entire attention-0 write at this position gives Left +36.5, Right +34.5, product +1.26e+03.
- Raw Left: direct +444 + attention +170 = +614.
- Raw Right: direct +421 + attention +201 = +621.
- Attention absolute-path share: Left 45.8%, Right 47.6%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` to`|-1|-0.381|-0.363|+0.139|+485|+67.3|+493|+68.4|
|3|` travel`|0|-0.385|-0.138|+0.053|-512|-27.2|-308|-16.3|
|8|` allowed`|-3|+0.264|+0.385|+0.102|+212|+21.5|+85.6|+8.69|
|3|` me`|-2|-0.35|-0.14|+0.0488|+256|+12.5|+261|+12.7|
|3|` allowed`|-3|-0.275|-0.0576|+0.0159|+476|+7.55|+937|+14.9|
|4|` railway`|-9|+0.295|-0.203|-0.0598|-80.6|+4.82|-248|+14.8|

### negative tail: ` the First Amendment. But universities don[�]�t exist to encourage just any`

- Native factors: Left -19.9, Right +16.8, product -333.
- Zeroing the entire attention-0 write at this position gives Left -21.4, Right +11.3, product -242.
- Raw Left: direct -261 + attention -46.9 = -308.
- Raw Right: direct +138 + attention +123 = +261.
- Attention absolute-path share: Left 78.3%, Right 87.4%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` don`|-1|+0.404|+0.336|+0.136|+120|+16.2|+716|+97.2|
|3|`�`|0|-0.332|-0.451|+0.149|-291|-43.4|-413|-61.8|
|3|`.`|-4|+0.355|-0.0757|-0.0269|-842|+22.6|-1.01e+03|+27.3|
|2|`�`|-18|+0.479|+0.475|+0.228|+118|+26.9|+73.3|+16.7|
|8|` But`|-3|-0.212|-0.312|+0.0664|+272|+18|+287|+19|
|2|`�`|-11|-0.4|-0.412|+0.165|-154|-25.4|+41.2|+6.8|

### negative tail: ` were expecting. Retailers including Macy[�]�s Inc., Abercrom`

- Native factors: Left -16.6, Right +18.4, product -305.
- Zeroing the entire attention-0 write at this position gives Left -21.4, Right +11.3, product -242.
- Raw Left: direct -261 + attention +23.6 = -237.
- Raw Right: direct +138 + attention +123 = +261.
- Attention absolute-path share: Left 62.2%, Right 78.9%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`�`|0|-0.332|-0.451|+0.149|-291|-43.4|-413|-61.8|
|3|` Macy`|-1|+0.432|+0.25|+0.108|+100|+10.8|+590|+63.7|
|3|` including`|-2|+0.295|+0.204|+0.0603|+365|+22|+549|+33.1|
|3|`.`|-5|+0.326|-0.0581|-0.0189|-842|+15.9|-1.01e+03|+19.2|
|6|`\n`|-28|-0.146|-0.121|+0.0176|-676|-11.9|-581|-10.2|
|8|` including`|-2|-0.238|-0.381|+0.0908|+86.7|+7.88|+155|+14.1|

## Unit 874

### positive tail: `1-3-2).\n["]You would like to think it's`

- Native factors: Left +66.5, Right +55.5, product +3.69e+03.
- Zeroing the entire attention-0 write at this position gives Left +74, Right -2.05, product -151.
- Raw Left: direct +901 + attention +267 = +1.17e+03.
- Raw Right: direct -24.9 + attention +999 = +974.
- Attention absolute-path share: Left 53.8%, Right 99.0%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|`\n`|-1|+0.734|+0.648|+0.477|+21.5|+10.2|+1.46e+03|+695|
|6|`"`|0|+0.543|+0.193|+0.105|+460|+48.3|-749|-78.6|
|7|`"`|0|+0.336|-0.33|-0.111|-357|+39.5|+472|-52.3|
|3|`"`|0|-0.404|-0.293|+0.119|-171|-20.3|+767|+91|
|2|`\n`|-1|-0.512|-0.398|+0.204|+116|+23.6|+407|+83|
|1|`\n`|-1|-0.57|+0.688|-0.393|-15.3|+6.01|-261|+102|

### positive tail: `Walking out to the pond\n["]Our" house through the willows`

- Native factors: Left +66.5, Right +54.8, product +3.64e+03.
- Zeroing the entire attention-0 write at this position gives Left +74, Right -2.05, product -151.
- Raw Left: direct +901 + attention +166 = +1.07e+03.
- Raw Right: direct -24.9 + attention +900 = +875.
- Attention absolute-path share: Left 40.2%, Right 98.7%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|`\n`|-1|+0.734|+0.648|+0.477|+21.5|+10.2|+1.46e+03|+695|
|6|`"`|0|+0.543|+0.193|+0.105|+460|+48.3|-749|-78.6|
|7|`"`|0|+0.336|-0.33|-0.111|-357|+39.5|+472|-52.3|
|2|`\n`|-1|-0.512|-0.398|+0.204|+116|+23.6|+407|+83|
|3|`"`|0|-0.406|-0.293|+0.119|-171|-20.3|+767|+91.4|
|1|`\n`|-1|-0.57|+0.688|-0.393|-15.3|+6.01|-261|+102|

### negative tail: ` the latter increasingly makes more "business["] sense.\nThe much heralded`

- Native factors: Left +96.5, Right -53.5, product -5.16e+03.
- Zeroing the entire attention-0 write at this position gives Left +74, Right -2.05, product -151.
- Raw Left: direct +901 + attention +700 = +1.6e+03.
- Raw Right: direct -24.9 + attention -863 = -888.
- Attention absolute-path share: Left 57.0%, Right 98.6%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|7|` "`|-2|+0.527|-0.543|-0.287|-707|+203|+1.17e+03|-337|
|6|` "`|-2|+0.68|+0.418|+0.283|+800|+227|-488|-138|
|8|` "`|-2|+0.516|-0.389|-0.2|-294|+58.9|+811|-162|
|6|`"`|0|+0.543|+0.192|+0.104|+460|+48.1|-749|-78.3|
|3|`"`|0|-0.404|-0.293|+0.119|-171|-20.3|+767|+91|
|2|` "`|-2|-0.598|-0.781|+0.467|+64.2|+30|-148|-69.3|

### negative tail: ` A\nreddish "halo["] is often seen around the base of`

- Native factors: Left +87, Right -54.2, product -4.72e+03.
- Zeroing the entire attention-0 write at this position gives Left +74, Right -2.05, product -151.
- Raw Left: direct +901 + attention +515 = +1.42e+03.
- Raw Right: direct -24.9 + attention -857 = -882.
- Attention absolute-path share: Left 56.6%, Right 98.7%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|7|` "`|-3|+0.508|-0.598|-0.303|-707|+214|+1.17e+03|-355|
|8|` "`|-3|+0.516|-0.373|-0.192|-294|+56.6|+811|-156|
|6|` "`|-3|+0.625|+0.178|+0.111|+800|+89.1|-488|-54.3|
|6|`"`|0|+0.543|+0.193|+0.105|+460|+48.3|-749|-78.6|
|2|` "`|-3|-0.695|-0.797|+0.555|+64.2|+35.6|-148|-82.4|
|3|`"`|0|-0.404|-0.293|+0.119|-171|-20.3|+767|+91|
