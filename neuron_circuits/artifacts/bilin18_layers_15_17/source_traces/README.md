# Bilin18 layers 15–17 named-source and QK×OV traces

The selected sites come from 25,600 held-out FineWeb tokens. Each factor is expanded over the embedding, prior attention/MLP writes, and the current attention write. The current attention term is then expanded over native head/source QK×OV paths.

`Without local attention` removes only the target block's attention write; prior context remains for layers 1–2. Source-pair terms are symmetric off-diagonal sums.

## Layer 17

### Unit 59

**positive tail:** ` at the No. 1 position when[ G]ille defeated Tafelski in`

Product +1.05e+03; without local attention +1.04e+03. Dominant named source-pair share 7.1%; effective support 45.70.

Top named source-pair terms: `mlp16×mlp16` -519; `mlp15×mlp16` -429; `attention5×mlp16` +420; `attention6×mlp16` +286; `mlp13×mlp16` +211.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|` G`|0|-0.438|-0.919|+0.0044|+0.4|+1.32|-0.0158|-0.57|
|1|` G`|0|-0.363|-0.515|-0.00019|+0.187|+1.3|+0.001|-0.473|
|6|` when`|-1|-0.539|-0.174|-0.00359|+0.0959|+1.21|-0.00455|-0.651|
|8|` when`|-1|-0.365|+0.684|-0.00449|-0.248|-0.399|-0.00251|+0.147|
|8|` G`|0|-0.318|+0.743|-0.00212|-0.236|-0.00691|-0.00316|+0.0032|
|6|` 1`|-3|-0.234|-0.499|-0.00095|+0.117|+1.03|-0.0071|-0.24|

**positive tail:** ` Tulsa was well represented, as quarterback[ G].J. Kinne was named`

Product +915; without local attention +877. Dominant named source-pair share 7.8%; effective support 43.51.

Top named source-pair terms: `mlp16×mlp16` -407; `attention5×mlp16` +358; `attention6×mlp16` +277; `mlp6×mlp16` +199; `mlp14×mlp16` -175.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|` G`|0|-0.424|-1.08|+0.00453|+0.457|+1.38|-0.0163|-0.576|
|6|` quarterback`|-1|-0.488|-0.583|+0.00148|+0.284|+1.56|+0.0195|-0.773|
|6|` as`|-2|-0.375|-0.294|+0.0018|+0.11|+1.24|-0.00435|-0.464|
|6|` Tulsa`|-7|-0.203|-0.523|-0.00162|+0.107|+1.41|+0.00301|-0.287|
|1|` G`|0|-0.252|-0.416|-0.0002|+0.105|+1.11|+0.00103|-0.279|
|6|`,`|-3|-0.248|-0.304|+0.00719|+0.0736|+1.17|+0.00104|-0.291|

**negative tail:** ` at gmail.com\nTue[ Dec] 13 10:18:00 PST`

Product -4.83e+03; without local attention -4.82e+03. Dominant named source-pair share 5.4%; effective support 55.94.

Top named source-pair terms: `mlp13×mlp15` -275; `mlp13×mlp14` -252; `mlp14×mlp15` -193; `mlp12×mlp13` -187; `mlp13×mlp13` -179.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|`\n`|-2|+0.32|-0.266|+0.00234|-0.0845|+0.621|-0.00599|+0.197|
|1|`.`|-4|+0.285|-0.3|+0.0017|-0.0849|+0.607|-0.00303|+0.172|
|1|`com`|-3|+0.445|-0.12|-0.00289|-0.0547|+0.463|-0.00034|+0.206|
|1|`mail`|-5|+0.348|-0.121|-0.00384|-0.0434|+0.465|-0.00153|+0.161|
|2|` Dec`|0|-0.367|-0.209|+0.00613|+0.0745|+0.234|+0.00664|-0.0884|
|1|`\n`|-13|+0.154|-0.319|+0.00234|-0.0488|+0.885|-0.00599|+0.136|

**negative tail:** `?\n01-28-2006[,] 06:40 AM\n...to`

Product -4.55e+03; without local attention -4.4e+03. Dominant named source-pair share 4.9%; effective support 63.44.

Top named source-pair terms: `mlp13×mlp14` -236; `mlp12×mlp13` -177; `mlp13×mlp15` -171; `mlp11×mlp13` -149; `mlp12×mlp14` -145.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|`\n`|-6|+0.252|-0.305|+0.00238|-0.0763|+0.867|-0.00611|+0.217|
|1|`-`|-4|+0.326|-0.2|+0.0003|-0.0652|+0.647|-0.00152|+0.211|
|1|`-`|-2|+0.273|-0.292|+0.0003|-0.0799|+0.653|-0.00152|+0.178|
|1|`28`|-3|+0.383|-0.121|-0.00278|-0.0475|+0.518|+0.00143|+0.199|
|1|`01`|-5|+0.359|-0.116|-0.00017|-0.0416|+0.498|-0.00394|+0.178|
|1|`?`|-7|+0.271|-0.12|+0.00377|-0.0316|+0.649|-0.00547|+0.175|

### Unit 4094

**positive tail:** ` nothing. Porfiry Petrov[itch] laughed and was ready to go on`

Product +1.72e+03; without local attention +1.94e+03. Dominant named source-pair share 3.4%; effective support 134.29.

Top named source-pair terms: `mlp15×mlp16` -101; `mlp14×mlp16` -78.7; `mlp16×mlp16` -53.9; `mlp12×mlp14` +42.7; `mlp12×mlp13` +39.9.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|`enance`|-10|+0.285|-0.193|-0.00763|-0.0572|+0.219|+0.00049|+0.0626|
|1|`itch`|0|-0.279|-0.0941|+0.00296|+0.0255|-0.321|-0.00346|+0.0905|
|2|`itch`|0|-0.531|+0.0645|-0.00352|-0.0324|-0.133|-0.00929|+0.0757|
|8|`itch`|0|-0.281|+0.266|+0.00336|-0.0759|-0.023|+0.00382|+0.00539|
|6|`iry`|-3|+0.258|+0.193|+0.00791|+0.0519|+0.096|+0.00323|+0.0256|
|8|` nothing`|-7|-0.185|+0.36|-0.00425|-0.0656|-0.018|+0.00941|+0.00159|

**positive tail:** ` demons. Cúchulain[n] proves to be a formidable opponent with`

Product +1.67e+03; without local attention +1.8e+03. Dominant named source-pair share 2.9%; effective support 134.10.

Top named source-pair terms: `mlp14×mlp16` -87.6; `mlp15×mlp16` -78.2; `mlp10×mlp16` -46.4; `mlp13×mlp16` -46.3; `mlp13×mlp14` +42.8.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|` demons`|-7|+0.252|-0.262|+8e-05|-0.0661|+0.212|-0.0031|+0.0527|
|8|` demons`|-7|-0.229|+0.418|+0.00393|-0.0965|-0.0253|-0.00203|+0.00624|
|8|` stone`|-23|-0.194|+0.408|+0.00331|-0.0799|-0.0141|-0.0041|+0.00353|
|8|`n`|0|-0.209|+0.352|+0.00302|-0.0742|+0.0497|-0.00204|-0.00997|
|3|`.`|-6|+0.328|-0.211|+0.00599|-0.0673|-0.0318|+0.0143|-0.00575|
|0|`n`|0|-0.418|-0.0281|-0.00044|+0.0119|-0.157|-0.00209|+0.0664|

**negative tail:** `01-27-2006, 10[:]44 AM\nWould your buddy be`

Product -4.19e+03; without local attention -4.19e+03. Dominant named source-pair share 7.3%; effective support 33.32.

Top named source-pair terms: `mlp13×mlp16` -374; `mlp16×mlp16` +368; `mlp13×mlp15` -313; `mlp14×mlp15` -271; `mlp15×mlp16` -231.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` 10`|-1|+0.32|+0.111|+0.00177|+0.0361|-0.429|-0.00098|-0.138|
|1|`\n`|-8|+0.153|+0.194|-0.00157|+0.0295|-0.975|+0.00574|-0.149|
|1|`2006`|-3|+0.299|-0.124|-0.00051|-0.0372|-0.415|+0.00304|-0.123|
|6|`27`|-5|-0.291|+0.233|+0.0036|-0.0688|+0.177|-0.00288|-0.0507|
|1|`27`|-5|+0.266|+0.0148|-0.0003|+0.00385|-0.662|+0.00124|-0.176|
|1|`-`|-6|+0.179|+0.139|-0.00076|+0.0248|-0.756|+0.00267|-0.135|

**negative tail:** ` 01-12-2013, 03[:]17 PM #15\nJust to`

Product -4.15e+03; without local attention -4.18e+03. Dominant named source-pair share 7.9%; effective support 33.43.

Top named source-pair terms: `mlp16×mlp16` +407; `mlp13×mlp16` -373; `mlp13×mlp15` -311; `mlp14×mlp15` -263; `mlp13×mlp14` -213.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` 03`|-1|+0.305|+0.123|+0.00015|+0.0375|-0.523|+0.0106|-0.156|
|6|`12`|-5|-0.291|+0.289|+0.00318|-0.0849|+0.206|-0.00073|-0.0598|
|1|`2013`|-3|+0.295|-0.133|+0.001|-0.0389|-0.474|+0.00321|-0.139|
|6|` 01`|-7|-0.268|+0.293|+0.00362|-0.0793|+0.153|-0.0064|-0.0392|
|1|`12`|-5|+0.218|-0.0346|+0.00172|-0.00715|-0.688|+0.00478|-0.149|
|1|`,`|-2|+0.236|-0.155|-0.00167|-0.0371|-0.389|+0.0001|-0.092|

### Unit 1043

**positive tail:** ` October 1, 2009\nReview of[ "]Princess of Cups" on Decker`

Product +5.18e+03; without local attention +5.29e+03. Dominant named source-pair share 5.0%; effective support 54.67.

Top named source-pair terms: `mlp15×mlp16` +267; `mlp13×mlp16` +260; `mlp12×mlp16` +256; `mlp16×mlp16` +239; `mlp14×mlp16` +186.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|` of`|-1|-0.57|-0.747|-0.00273|+0.428|-0.654|-0.00338|+0.375|
|6|`Review`|-2|-0.385|-0.841|+0.00055|+0.323|-0.759|-0.00758|+0.295|
|6|`,`|-5|-0.225|-0.789|-0.00081|+0.177|-0.677|-0.0022|+0.153|
|6|` 1`|-6|-0.195|-0.734|-0.00359|+0.144|-0.665|-0.0011|+0.13|
|6|`,`|-8|-0.188|-0.754|-0.00081|+0.142|-0.688|-0.0022|+0.129|
|6|` 2009`|-4|-0.24|-0.597|-0.00091|+0.144|-0.528|+0.00537|+0.125|

**positive tail:** `Sunday, April 03, 2011[\n]MACON, Ga. (April`

Product +5e+03; without local attention +5.08e+03. Dominant named source-pair share 4.2%; effective support 75.45.

Top named source-pair terms: `mlp13×mlp16` +213; `mlp12×mlp16` +202; `mlp11×mlp16` +154; `mlp16×mlp16` +152; `mlp12×mlp13` +140.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|`,`|-2|-0.531|-0.772|-0.00077|+0.411|-0.683|-0.00211|+0.364|
|6|` 2011`|-1|-0.49|-0.649|-0.00058|+0.319|-0.583|+0.00326|+0.284|
|6|` 03`|-3|-0.34|-0.794|+0.00142|+0.269|-0.717|-0.00703|+0.246|
|6|` April`|-4|-0.301|-0.722|-0.00628|+0.219|-0.621|-0.00318|+0.188|
|6|`,`|-5|-0.248|-0.799|-0.00077|+0.198|-0.736|-0.00211|+0.183|
|6|`Sunday`|-6|-0.209|-0.858|+0.00367|+0.179|-0.746|+0.00282|+0.155|

**negative tail:** `udar Goendagiri ([�]�েল�`

Product -410; without local attention -417. Dominant named source-pair share 7.4%; effective support 44.96.

Top named source-pair terms: `mlp15×mlp16` +532; `attention5×mlp16` -403; `attention5×mlp15` -302; `mlp16×mlp16` +295; `attention6×mlp16` -283.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|`ud`|-7|+0.117|-3.22|-0.00703|-0.378|-2.92|-0.0122|-0.343|
|6|`ar`|-6|+0.105|-3.14|+0.0149|-0.329|-2.68|+0.0152|-0.281|
|6|`iri`|-2|-0.0693|-3.4|+0.0206|+0.234|-2.97|-0.00015|+0.206|
|6|` detective`|-10|+0.0728|-3.16|-0.0097|-0.23|-2.74|-0.0123|-0.2|
|6|` Go`|-5|+0.0742|-2.68|-0.0181|-0.201|-2.26|-0.0257|-0.169|
|6|` in`|-9|+0.063|-2.55|+0.00214|-0.161|-2.1|+0.00459|-0.132|

**negative tail:** `� গো[�]�়ে�`

Product -274; without local attention -298. Dominant named source-pair share 6.0%; effective support 60.84.

Top named source-pair terms: `mlp15×mlp16` +268; `attention5×mlp16` -204; `attention5×mlp15` -195; `mlp15×mlp15` +136; `mlp14×mlp15` +126.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|`�`|-1|-0.2|-2.46|-0.00477|+0.493|-1.87|-0.0239|+0.38|
|1|`�`|-4|+0.234|+0.189|+0.0043|+0.0453|+1.16|+0.0167|+0.276|
|1|`�`|-1|+0.25|+0.131|+0.00136|+0.0331|+1.07|-0.0128|+0.265|
|6|`�`|-13|+0.042|-2.66|-0.00139|-0.112|-1.89|-0.0431|-0.0814|
|2|`�`|-8|-0.107|-1.03|+0.0101|+0.109|-0.767|-0.00311|+0.0828|
|6|`�`|-2|-0.191|-0.836|-0.0258|+0.165|-0.0357|-0.0228|+0.0112|

### Unit 1239

**positive tail:** ` Scaramanga, but it's[ over]-impressed with it's own`

Product +2.19e+03; without local attention +2.24e+03. Dominant named source-pair share 4.4%; effective support 78.96.

Top named source-pair terms: `mlp15×mlp16` +656; `mlp16×mlp16` +600; `mlp11×mlp16` -599; `attention5×mlp16` -472; `mlp11×mlp15` -312.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|` over`|0|-0.379|-1.03|-0.00798|+0.392|+1.05|-0.0126|-0.392|
|6|`'s`|-1|-0.209|-1.39|+0.0225|+0.285|+1.08|+0.00834|-0.228|
|6|` it`|-2|-0.164|-1.01|-0.0066|+0.167|+1.26|+0.00633|-0.207|
|1|` Scar`|-7|+0.0747|-1.68|+0.00041|-0.126|+1.95|-0.0204|+0.144|
|8|`'s`|-1|-0.389|-0.291|-0.0399|+0.129|-0.288|-0.0138|+0.117|
|2|`'s`|-1|-0.5|+0.389|+7e-05|-0.194|-0.102|+0.00272|+0.0498|

**positive tail:** ` demons. Cúchulain[n] proves to be a formidable opponent with`

Product +1.91e+03; without local attention +2.44e+03. Dominant named source-pair share 5.1%; effective support 77.52.

Top named source-pair terms: `mlp16×mlp16` +905; `mlp15×mlp16` +649; `attention9×mlp16` -509; `mlp13×mlp16` -493; `mlp9×mlp16` -451.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|`avi`|-8|+0.25|-0.955|+0.0126|-0.236|+0.729|+0.00538|+0.184|
|6|` demons`|-7|+0.252|-1.44|+0.0106|-0.36|+0.256|-0.00452|+0.0634|
|6|`ain`|-1|+0.143|-0.783|-0.0127|-0.113|+0.971|-0.003|+0.138|
|6|`n`|-14|+0.161|-1.25|+0.00707|-0.201|+0.421|-0.00115|+0.0677|
|8|` demons`|-7|-0.229|+0.552|-0.00557|-0.125|+0.44|-0.00903|-0.0986|
|6|`.`|-6|+0.157|-0.806|+0.00023|-0.127|+0.608|-0.00172|+0.0953|

**negative tail:** ` allowed to pick TWO. My wife[ picked] brains and sanity. Its like I`

Product -4.69e+03; without local attention -4.51e+03. Dominant named source-pair share 9.0%; effective support 38.37.

Top named source-pair terms: `mlp16×mlp16` +1.85e+03; `mlp15×mlp16` +1.13e+03; `mlp12×mlp16` -1.01e+03; `attention5×mlp16` -951; `mlp11×mlp16` -844.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|`.`|-3|+0.291|-1.18|+0.00622|-0.341|+1.26|-0.00855|+0.365|
|1|` wife`|-1|+0.303|-1.21|-0.00615|-0.369|+0.896|+0.00724|+0.274|
|1|`.`|-29|+0.117|-1.26|+0.00622|-0.146|+1.47|-0.00855|+0.171|
|6|` My`|-2|+0.17|-0.847|+0.0114|-0.142|+0.906|+0.00736|+0.155|
|6|` wife`|-1|+0.145|-1.34|+0.00963|-0.193|+0.635|+0.0119|+0.0935|
|1|` mate`|-25|+0.145|-0.98|-0.0114|-0.143|+0.9|-0.0101|+0.129|

**negative tail:** ` Ekéus; from 1997 to[ 1999] its director was Richard Butler.\n`

Product -4.55e+03; without local attention -4.4e+03. Dominant named source-pair share 8.4%; effective support 42.41.

Top named source-pair terms: `mlp16×mlp16` +2.1e+03; `mlp15×mlp16` +1.61e+03; `attention5×mlp16` -1.09e+03; `mlp12×mlp16` -919; `attention9×mlp16` -873.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` director`|-11|+0.138|-1.34|-0.00455|-0.185|+1.04|+0.00479|+0.145|
|1|`;`|-4|+0.146|-0.89|-0.0075|-0.131|+0.759|-0.00993|+0.109|
|1|`.`|-17|+0.0977|-1.18|+0.00691|-0.115|+1.12|-0.00946|+0.108|
|1|` 1997`|-2|+0.124|-1.08|+0.0141|-0.131|+0.761|-0.0164|+0.092|
|1|` production`|-29|+0.11|-0.971|+0.0124|-0.105|+0.718|-0.00753|+0.078|
|6|` 1999`|0|-0.0981|-1.11|+0.00466|+0.108|+0.724|+0.0126|-0.0723|

## Layer 16

### Unit 4364

**positive tail:** `era liberals. Leading American clerics have[ regularly] attacked and bullied liberal Catholics like John`

Product +701; without local attention +661. Dominant named source-pair share 7.7%; effective support 39.99.

Top named source-pair terms: `mlp13×mlp15` +160; `mlp15×mlp15` +144; `mlp14×mlp15` +124; `mlp12×mlp15` +118; `mlp10×mlp15` +74.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` have`|-1|-0.531|-0.00496|+0.148|-0.0759|-0.0631|+0.591|-0.281|
|8|` regularly`|0|-0.691|-0.014|-0.0026|+0.0115|-0.0145|+0.478|-0.32|
|3|` moved`|-38|+0.213|-0.0248|-0.331|-0.0756|+0.0462|-0.704|-0.14|
|3|` allied`|-72|+0.146|-0.0171|-0.336|-0.0517|+0.0234|-0.634|-0.0894|
|4|` bishops`|-17|+0.115|-0.00623|-0.526|-0.0611|+0.0102|-0.519|-0.0583|
|0|` fighting`|-77|-0.185|-0.0236|-0.172|+0.0361|-0.0323|-0.339|+0.0685|

**positive tail:** ` Georgetown moratorium cap. The idea was[ to] address a perception that good restaurants were`

Product +639; without local attention +620. Dominant named source-pair share 8.0%; effective support 42.60.

Top named source-pair terms: `mlp13×mlp15` +156; `mlp15×mlp15` +115; `mlp14×mlp15` +103; `mlp12×mlp15` +94.4; `mlp10×mlp15` +88.5.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` was`|-1|-0.479|-0.00506|-0.0752|+0.0384|-0.0738|+0.7|-0.3|
|8|` to`|0|-0.52|+0.0163|+0.274|-0.151|+0.00237|+0.24|-0.126|
|0|` liquor`|-11|-0.127|+0.00599|+0.372|-0.048|-0.00837|+0.443|-0.0551|
|7|` licenses`|-10|+0.201|-0.018|+0.181|+0.0329|+0.00112|-0.0828|-0.0164|
|8|` idea`|-2|-0.469|+0.0206|-0.0953|+0.035|-0.0533|+0.0392|+0.00659|
|7|` idea`|-2|+0.252|-0.0229|-0.114|-0.0344|-0.0165|+0.016|-0.00012|

**negative tail:** ` and no 007.\n18[)] You Only Live Twice. After some`

Product -2.26e+03; without local attention -2.28e+03. Dominant named source-pair share 3.1%; effective support 110.33.

Top named source-pair terms: `mlp11×mlp13` -70.2; `mlp12×mlp13` -59.5; `mlp13×mlp14` -57.4; `mlp13×mlp13` -52.9; `mlp13×mlp15` -52.7.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|4|` 00`|-5|+0.463|+0.0112|-0.266|-0.118|+0.00154|-0.136|-0.0623|
|3|` Scar`|-41|+0.273|-0.0107|-0.0713|-0.0224|+0.018|-0.486|-0.128|
|5|` 00`|-5|-0.293|+0.0155|+0.218|-0.0685|-0.0249|+0.221|-0.0575|
|5|` Scar`|-41|-0.586|+0.00196|+0.0854|-0.0512|-0.0445|-0.084|+0.0753|
|0|` Scar`|-41|-0.354|+0.0339|-0.216|+0.0643|-0.00212|-0.114|+0.041|
|4|` house`|-24|+0.234|-0.0129|+0.0966|+0.0196|+0.00026|-0.369|-0.0865|

**negative tail:** `ov away from the FA Cup Final[.] The Community Shield (the most important`

Product -2.14e+03; without local attention -2.15e+03. Dominant named source-pair share 2.3%; effective support 135.95.

Top named source-pair terms: `mlp11×mlp13` -50.9; `mlp11×mlp15` -36.7; `mlp12×mlp13` -36.2; `attention6×mlp15` -33.4; `mlp11×mlp12` -33.1.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`Seven`|-32|+0.198|-0.00262|-0.472|-0.094|-0.0105|-0.45|-0.0913|
|5|` Ber`|-9|-0.453|+0.0621|+0.146|-0.0943|-0.089|-0.0716|+0.0727|
|3|` Cox`|-53|+0.133|+0.00782|-0.425|-0.0554|+0.0082|-0.479|-0.0625|
|3|` Michael`|-54|+0.118|-0.0062|+0.505|+0.0589|+0.0444|+0.41|+0.0537|
|0|` Anti`|-44|-0.354|+0.0123|+0.0676|-0.0283|-0.00864|-0.204|+0.0753|
|4|`Player`|-42|+0.237|+0.0204|+0.166|+0.0441|+0.00333|+0.214|+0.0515|

### Unit 1388

**positive tail:** `Like all Saints fans, I[ am] immensely proud of our Academy. For`

Product +1.09e+03; without local attention +1.09e+03. Dominant named source-pair share 11.4%; effective support 35.35.

Top named source-pair terms: `mlp15×mlp15` -342; `attention5×mlp15` +170; `mlp6×mlp15` +115; `mlp11×mlp15` +114; `mlp9×mlp15` +109.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` am`|0|-0.73|+0.0117|+0.279|-0.212|-0.0245|-0.26|+0.208|
|8|`,`|-2|-0.523|-0.0302|+0.254|-0.117|-0.00166|-0.0396|+0.0216|
|4|` Saints`|-4|+0.215|+0.0362|-0.254|-0.0468|-0.00893|+0.231|+0.0476|
|8|` all`|-5|-0.291|+0.0199|+0.24|-0.0757|-0.0227|-0.0265|+0.0143|
|8|` I`|-1|-0.383|+0.00136|+0.0752|-0.0293|+0.0518|+0.0456|-0.0372|
|2|` am`|0|-0.245|+0.0735|+0.0792|-0.0374|-0.0439|-0.0738|+0.0289|

**positive tail:** ` Michael Bullerdick (pictured below[),] forgot that even a “private`

Product +1.11e+03; without local attention +1.08e+03. Dominant named source-pair share 7.8%; effective support 47.34.

Top named source-pair terms: `mlp15×mlp15` -244; `attention5×mlp15` +154; `mlp11×mlp15` +133; `mlp9×mlp15` +112; `mlp12×mlp15` +109.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|`),`|0|-0.535|+0.0192|+0.293|-0.167|-0.0266|+0.0912|-0.0346|
|8|` (`|-3|-0.312|+0.0555|-0.502|+0.139|-0.0262|-0.072|+0.0307|
|4|` pedal`|-14|+0.179|+0.0223|-0.32|-0.0532|-0.0216|+0.291|+0.0481|
|0|` clown`|-40|-0.301|+0.0267|-0.225|+0.0598|+0.0148|-0.0595|+0.0134|
|8|` when`|-13|-0.142|+0.025|+0.213|-0.0337|-0.026|+0.186|-0.0226|
|3|` editors`|-9|-0.0698|-0.0533|-0.299|+0.0246|+0.0302|+0.264|-0.0206|

**negative tail:** `57′18″E / 41[.]01361°N 28.9`

Product -849; without local attention -855. Dominant named source-pair share 16.3%; effective support 18.61.

Top named source-pair terms: `mlp14×mlp15` -181; `mlp14×mlp14` -117; `mlp13×mlp14` -69.8; `mlp15×mlp15` -51; `attention5×mlp14` -46.6.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|`.`|0|-0.338|+0.0454|+0.13|-0.0593|+0.00942|+0.0276|-0.0125|
|3|`Co`|-20|-0.0393|+0.0129|-0.161|+0.00582|-0.0318|+0.527|-0.0195|
|3|` 28`|-9|+0.0554|+0.00404|+0.139|+0.00792|-0.0268|-0.264|-0.0161|
|3|`57`|-7|+0.0405|+0.0167|+0.634|+0.0264|-0.0204|+0.0123|-0.00033|
|1|`stanbul`|-27|-0.0933|+0.0306|+0.0899|-0.0112|-0.0196|-0.0854|+0.0098|
|8|`00`|-14|-0.0603|+0.00413|+0.246|-0.0151|+0.00058|-0.0902|+0.00541|

**negative tail:** ` of reterritorializion[�]� by “`

Product -615; without local attention -611. Dominant named source-pair share 10.4%; effective support 32.21.

Top named source-pair terms: `mlp14×mlp15` -717; `mlp15×mlp15` -531; `attention5×mlp15` +381; `mlp13×mlp15` -262; `attention5×mlp14` +258.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`ir`|-95|+0.0386|-0.0413|+0.726|+0.0264|-0.0261|-1.04|-0.0411|
|4|`iz`|-2|-0.0684|+0.0185|+0.0803|-0.00675|-0.0146|-0.632|+0.0442|
|1|`ion`|-1|-0.141|+0.0224|+0.0995|-0.0171|-0.0484|-0.14|+0.0266|
|0|`�`|0|+0.138|-0.175|-0.0144|-0.0261|+0.136|-0.031|+0.0144|
|0|` Dele`|-28|-0.0344|+0.0129|+0.191|-0.00703|+0.0409|-0.826|+0.027|
|3|`iz`|-2|+0.051|+0.0143|-0.299|-0.0145|-0.0468|-0.312|-0.0183|

### Unit 2687

**positive tail:** `Originally Posted[ by] IminUroutKFP\n`

Product +1.1e+03; without local attention +1.09e+03. Dominant named source-pair share 9.8%; effective support 37.68.

Top named source-pair terms: `mlp13×mlp14` +115; `mlp13×mlp13` +67.1; `mlp14×mlp14` +48.9; `mlp9×mlp13` +39.5; `mlp11×mlp13` +39.3.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` Posted`|-1|-0.309|-0.00598|+0.129|-0.038|+0.0275|+0.11|-0.0425|
|8|`Originally`|-2|-0.439|-0.00165|+0.0265|-0.0109|+0.0315|-0.0855|+0.0237|
|1|` by`|0|+0.11|-0.0186|-0.0418|-0.00664|-0.0514|-0.0509|-0.0112|
|8|` by`|0|-0.248|-0.0246|+0.0803|-0.0138|-0.00254|+0.00317|-0.00016|
|3|`Originally`|-2|+0.0815|-0.0362|+0.0448|+0.0007|-0.00485|-0.159|-0.0133|
|2|`Originally`|-2|-0.118|-0.00872|-0.0646|+0.00867|-0.00595|-0.0208|+0.00316|

**positive tail:** `Originally Posted[ by] Champ2647\nNo matter how`

Product +1.1e+03; without local attention +1.09e+03. Dominant named source-pair share 9.8%; effective support 37.68.

Top named source-pair terms: `mlp13×mlp14` +115; `mlp13×mlp13` +67.1; `mlp14×mlp14` +48.9; `mlp9×mlp13` +39.5; `mlp11×mlp13` +39.3.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` Posted`|-1|-0.309|-0.00598|+0.129|-0.038|+0.0275|+0.11|-0.0425|
|8|`Originally`|-2|-0.439|-0.00165|+0.0265|-0.0109|+0.0315|-0.0855|+0.0237|
|1|` by`|0|+0.11|-0.0186|-0.0418|-0.00664|-0.0514|-0.0509|-0.0112|
|8|` by`|0|-0.248|-0.0246|+0.0803|-0.0138|-0.00254|+0.00317|-0.00016|
|3|`Originally`|-2|+0.0815|-0.0362|+0.0448|+0.0007|-0.00485|-0.159|-0.0133|
|2|`Originally`|-2|-0.118|-0.00872|-0.0646|+0.00867|-0.00595|-0.0208|+0.00316|

**negative tail:** ` on 06/26/2012 1[:]42:40 PM PDT by j`

Product -475; without local attention -473. Dominant named source-pair share 8.9%; effective support 22.20.

Top named source-pair terms: `mlp14×mlp15` -52.2; `mlp12×mlp15` -51.4; `mlp13×mlp15` -48.7; `mlp15×mlp15` -40.7; `mlp12×mlp13` -36.8.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|`/`|-5|-0.122|+0.0111|-0.123|+0.0136|+0.0373|-0.369|+0.0405|
|0|` Voter`|-11|+0.114|-0.00412|+0.197|+0.0221|-0.0119|-0.054|-0.00754|
|8|` 1`|-1|-0.26|+0.0128|-0.0233|+0.00273|+0.0156|-0.123|+0.0278|
|8|` 06`|-6|-0.131|+0.0314|-0.152|+0.0158|+0.00562|-0.0957|+0.0118|
|8|`/`|-3|-0.0728|+0.0435|-0.123|+0.00575|+0.0692|-0.369|+0.0218|
|8|`\n`|-9|-0.0762|+0.0109|-0.107|+0.00732|+0.0467|-0.303|+0.0196|

**negative tail:** ` 01-12-2013, 03[:]17 PM #15\nJust to`

Product -462; without local attention -460. Dominant named source-pair share 7.8%; effective support 29.11.

Top named source-pair terms: `mlp14×mlp15` -49.6; `mlp13×mlp15` -45.5; `mlp12×mlp15` -43.2; `mlp15×mlp15` -33.6; `mlp12×mlp14` -33.2.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` 01`|-7|-0.176|-0.00427|-0.162|+0.0292|-0.0151|-0.211|+0.0397|
|8|`-`|-6|-0.12|-0.00703|-0.143|+0.018|+0.0242|-0.349|+0.039|
|8|`-`|-8|-0.0952|-0.0178|-0.143|+0.0153|+0.00969|-0.349|+0.0323|
|8|`\n`|-9|-0.104|+0.00611|-0.113|+0.0111|+0.0243|-0.319|+0.0307|
|8|`-`|-4|-0.0693|+0.0189|-0.143|+0.00861|+0.0239|-0.349|+0.0226|
|8|`:`|0|-0.348|-0.019|-0.00256|+0.00749|+0.0468|+0.0147|-0.0214|

### Unit 4569

**positive tail:** ` venues, with Dance Village area organ[iser] Malcolm Haynes.\n1.`

Product +945; without local attention +945. Dominant named source-pair share 10.7%; effective support 31.46.

Top named source-pair terms: `mlp13×mlp14` +114; `mlp13×mlp13` +78.7; `mlp12×mlp13` +69.6; `mlp12×mlp14` +52.8; `mlp14×mlp14` +41.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` with`|-5|-0.228|+0.043|+0.0367|-0.0181|+0.0109|+0.299|-0.0705|
|0|` Dance`|-4|-0.256|+0.0095|-0.0435|+0.00869|+0.0109|-0.307|+0.0757|
|5|` Gl`|-12|-0.355|-0.00751|+0.112|-0.0371|-0.0285|-0.0108|+0.014|
|1|` Gl`|-12|-0.355|-0.0169|-0.00114|+0.0064|-0.0308|-0.12|+0.0536|
|3|` Gl`|-12|+0.119|+0.0136|+0.196|+0.0248|+0.0353|+0.0929|+0.0152|
|8|` organ`|-1|-0.137|+0.0109|-0.0304|+0.00267|+0.0701|+0.271|-0.0466|

**positive tail:** ` on 06/26/2012 1[:]42:40 PM PDT by j`

Product +825; without local attention +825. Dominant named source-pair share 12.2%; effective support 17.52.

Top named source-pair terms: `mlp14×mlp15` +110; `mlp13×mlp14` +105; `mlp13×mlp15` +102; `mlp14×mlp14` +57; `mlp15×mlp15` +52.5.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` 1`|-1|-0.26|+0.00711|+0.103|-0.0286|+0.0247|+0.0502|-0.0195|
|8|` 06`|-6|-0.131|+0.00751|+0.108|-0.0152|-0.013|+0.193|-0.0236|
|8|`/`|-5|-0.122|+0.00723|-0.0867|+0.0097|-0.0271|+0.159|-0.0161|
|8|`:`|0|-0.297|+0.0091|-0.00763|-0.00044|-0.0253|-0.0632|+0.0263|
|8|`/`|-3|-0.0728|-0.00722|-0.0867|+0.00683|-0.0083|+0.159|-0.0109|
|8|`Posted`|-8|-0.0957|+0.0175|-0.0132|-0.00042|-0.0304|-0.155|+0.0177|

**negative tail:** ` grit and dust from the ceiling.[ I] decide to press`

Product -722; without local attention -722. Dominant named source-pair share 6.9%; effective support 43.48.

Top named source-pair terms: `mlp13×mlp15` -86.7; `mlp12×mlp15` -85.6; `mlp11×mlp15` -64.2; `mlp14×mlp15` -55; `mlp10×mlp15` -41.5.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|7|` dust`|-5|+0.201|+0.00768|+0.193|+0.0403|+0.0315|+0.166|+0.0397|
|5|` H`|-66|+0.0771|-0.0373|+0.389|+0.0271|+0.00218|+0.362|+0.0281|
|3|` shakes`|-14|-0.0879|-0.0285|-0.214|+0.0214|+0.0638|-0.364|+0.0264|
|4|` news`|-59|+0.0615|+0.0341|+0.296|+0.0203|+0.021|+0.314|+0.0206|
|4|` commuters`|-69|+0.116|-0.0041|-0.128|-0.0152|+0.00886|-0.194|-0.0214|
|3|` streams`|-9|-0.0277|+0.0147|-0.588|+0.0159|+0.0217|-0.755|+0.0203|

**negative tail:** ` want to add a little color but[ we] are afraid we won't like it`

Product -705; without local attention -690. Dominant named source-pair share 7.5%; effective support 47.47.

Top named source-pair terms: `mlp15×mlp15` +113; `mlp11×mlp15` -70.3; `mlp12×mlp15` -62.7; `embedding×mlp15` -57.6; `attention5×mlp15` -54.4.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` but`|-1|-0.586|-0.0198|-0.203|+0.131|+0.0365|+0.0823|-0.0696|
|8|` we`|0|-0.455|-0.00022|-0.163|+0.0741|+0.0425|+0.0989|-0.0643|
|3|` use`|-31|+0.149|+0.0284|-0.473|-0.0664|-1e-05|-0.4|-0.0598|
|3|` channel`|-15|+0.0544|+0.015|-0.194|-0.00975|+0.0235|-1.16|-0.0621|
|0|` hair`|-53|-0.254|+0.0297|+0.0678|-0.0248|+0.0472|-0.226|+0.0455|
|4|` temporary`|-30|+0.125|+0.00119|+0.465|+0.0583|-0.0238|+0.0821|+0.00728|

## Layer 15

### Unit 311

**positive tail:** ` both short-term rental homes,[ long]-term homes for rent Floral`

Product +352; without local attention +386. Dominant named source-pair share 2.4%; effective support 110.50.

Top named source-pair terms: `attention5×mlp14` +11.9; `attention5×mlp11` +11; `attention5×attention6` +10.9; `embedding×attention5` +8.8; `attention5×mlp9` +8.77.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` Owner`|-9|-0.227|+0.336|+0.184|-0.118|+0.335|+0.0469|-0.0866|
|1|`term`|-4|-0.348|+0.0863|+0.12|-0.0718|+0.169|+0.11|-0.097|
|1|` homes`|-2|-0.531|+0.123|-0.0135|-0.0584|+0.25|-0.0953|-0.082|
|1|` Rent`|-11|-0.11|+0.578|+0.0323|-0.0674|+0.243|-0.0577|-0.0205|
|8|` homes`|-2|-0.242|+0.0861|+0.118|-0.0494|+0.121|-0.0105|-0.0268|
|1|` By`|-10|-0.0562|+0.675|+0.135|-0.0455|+0.326|+0.0891|-0.0233|

**positive tail:** ` the U.S. Navy but[ was] refused entry because, at 6 feet`

Product +322; without local attention +347. Dominant named source-pair share 2.2%; effective support 138.48.

Top named source-pair terms: `attention5×mlp13` +9.41; `attention5×mlp10` +7.88; `attention5×mlp12` +7.67; `mlp10×mlp13` +7.49; `mlp12×mlp13` +6.42.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` Navy`|-2|-0.586|+0.18|+0.0389|-0.128|+0.0803|+0.0328|-0.0663|
|1|`S`|-4|-0.305|+0.193|+0.0713|-0.0804|+0.137|+0.0268|-0.0499|
|1|` but`|-1|-0.609|+0.0224|-0.1|+0.0474|-0.0339|-0.065|+0.0602|
|1|`.`|-3|-0.412|+0.102|-0.00044|-0.0418|+0.0861|-0.0121|-0.0305|
|1|`\n`|-18|-0.154|+0.303|+0.102|-0.0625|+0.0905|+0.0187|-0.0169|
|1|`,`|-13|-0.258|+0.211|-0.0108|-0.0516|+0.0601|-0.00892|-0.0132|

**negative tail:** `2\n- Word Count: 7[,]803\nDostoyevsky`

Product -247; without local attention -243. Dominant named source-pair share 8.2%; effective support 36.87.

Top named source-pair terms: `mlp12×mlp14` -40.4; `mlp12×mlp13` -35.5; `mlp13×mlp14` -25.1; `mlp11×mlp14` -24.7; `mlp11×mlp13` -19.7.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|`:`|-2|-0.402|+0.364|+0.0319|-0.159|+0.201|+0.0287|-0.0926|
|1|` 7`|-1|-0.473|+0.246|+0.0371|-0.134|+0.137|+0.00471|-0.0669|
|1|` Count`|-3|-0.252|+0.53|+0.103|-0.159|+0.257|-0.0654|-0.0484|
|1|`,`|0|-0.516|+0.262|-0.0127|-0.128|+0.137|-0.0106|-0.0649|
|1|`-`|-5|-0.139|+0.502|+0.0334|-0.0742|+0.232|+0.022|-0.0352|
|1|` Word`|-4|-0.147|+0.506|-0.0065|-0.0736|+0.172|+0.0533|-0.0333|

**negative tail:** `57′18″E / 41[.]01361°N 28.9`

Product -221; without local attention -215. Dominant named source-pair share 10.6%; effective support 28.62.

Top named source-pair terms: `mlp12×mlp14` -51.7; `mlp13×mlp14` -36.2; `mlp12×mlp13` -28.2; `mlp11×mlp14` -24; `mlp10×mlp14` -23.6.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` /`|-2|-0.197|+0.18|+0.0168|-0.0388|+0.162|+0.0152|-0.0349|
|1|`.`|0|-0.379|+0.116|-0.00043|-0.044|+0.0849|-0.0118|-0.0277|
|1|` 41`|-1|-0.209|+0.137|+0.07|-0.0433|+0.078|+0.0488|-0.0265|
|0|`.`|0|-0.334|-0.00382|-0.0207|+0.00818|-0.0759|-0.0445|+0.0402|
|3|`49`|-12|+0.0835|-0.0346|-0.0218|-0.00471|-0.0677|-0.341|-0.0341|
|2|`57`|-7|-0.13|+0.0529|+0.0248|-0.0101|+0.0485|+0.109|-0.0205|

### Unit 438

**positive tail:** ` Networks|\n|Authors:[ ||]Kalapriya, K|`

Product +494; without local attention +472. Dominant named source-pair share 12.3%; effective support 21.89.

Top named source-pair terms: `mlp13×mlp14` +146; `mlp12×mlp14` +144; `mlp14×mlp14` +79.9; `mlp12×mlp13` +76.1; `mlp13×mlp13` +54.3.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|2|` ||`|0|-0.203|-0.393|-0.115|+0.103|+0.508|+0.074|-0.118|
|1|`|`|-6|-0.199|+0.121|+0.0581|-0.0356|+0.16|+0.0437|-0.0406|
|1|`Auth`|-3|-0.233|-0.0902|+0.00232|+0.0205|+0.268|-0.0249|-0.0568|
|1|`:`|-1|-0.314|-0.0329|+0.038|-0.0016|+0.223|+0.0247|-0.078|
|1|`|`|-4|-0.191|+0.0538|+0.0581|-0.0214|+0.183|+0.0437|-0.0433|
|4|` Indian`|-78|-0.207|+0.00891|-0.0727|+0.0132|-0.0209|+0.277|-0.053|

**positive tail:** ` 1866\n- Language: English[\n]- Country of Origin: Russia\n`

Product +406; without local attention +404. Dominant named source-pair share 8.7%; effective support 33.38.

Top named source-pair terms: `embedding×mlp14` +98.8; `mlp12×mlp14` +72.9; `mlp13×mlp14` +70; `embedding×embedding` -69; `mlp14×mlp14` +54.2.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|2|` Published`|-9|-0.166|-0.0314|-0.074|+0.0175|+0.239|+0.182|-0.0698|
|1|`\n`|0|-0.16|-0.316|-0.0461|+0.058|+0.0929|+0.0329|-0.0201|
|4|` Year`|-10|+0.198|-0.0158|-0.114|-0.0256|+0.00997|+0.0565|+0.0132|
|2|`\n`|0|+0.0742|-0.258|+0.661|+0.0299|+0.48|-0.521|-0.0031|
|4|` Published`|-9|+0.0972|-0.0326|-0.119|-0.0148|-0.00282|-0.17|-0.0168|
|4|`66`|-6|-0.102|-0.0492|-0.144|+0.0197|+0.00929|+0.0781|-0.00892|

**negative tail:** `ibson at gmail.com[\n]Tue Dec 13 10:18:`

Product -699; without local attention -710. Dominant named source-pair share 8.1%; effective support 38.83.

Top named source-pair terms: `embedding×embedding` -61.2; `embedding×mlp12` -55.6; `embedding×mlp13` -36.5; `embedding×mlp9` -30.8; `embedding×mlp10` -28.8.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|`\n`|0|-0.154|-0.195|-0.0433|+0.0367|+0.28|+0.0311|-0.048|
|2|`\n`|0|-0.063|-0.072|+0.622|-0.0346|+0.313|-0.492|+0.0113|
|2|`com`|-19|-0.159|+0.0526|+0.07|-0.0195|+0.149|+0.00149|-0.0239|
|2|` MS`|-24|-0.148|+0.00943|-0.0861|+0.0114|+0.115|+0.116|-0.0344|
|2|`[`|-29|-0.14|-0.0239|+0.0186|+0.00073|+0.199|+0.135|-0.0466|
|5|`\n`|-11|-0.213|+0.0327|-0.0682|+0.00756|+0.0926|+0.0643|-0.0334|

**negative tail:** `1. That all intellectual property rights[ (]including, but not limited to,`

Product -651; without local attention -698. Dominant named source-pair share 3.6%; effective support 86.35.

Top named source-pair terms: `embedding×mlp11` -33.5; `embedding×embedding` -28.1; `embedding×mlp12` -27.1; `embedding×mlp14` -24.1; `mlp11×mlp12` -23.3.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` (`|0|-0.449|-0.0412|-0.0256|+0.03|+0.19|+0.0819|-0.122|
|1|` rights`|-1|-0.387|-0.0727|-0.0228|+0.0369|+0.25|-0.0266|-0.0863|
|1|` property`|-2|-0.404|-0.0744|+0.0358|+0.0156|+0.272|-0.00983|-0.106|
|1|` That`|-5|-0.252|-0.235|-0.0339|+0.0679|+0.077|-0.0486|-0.00716|
|3|` intellectual`|-3|+0.283|-0.0856|-0.00449|-0.0255|+0.045|-0.235|-0.0538|
|1|` intellectual`|-3|-0.311|-0.0607|+0.0532|+0.00233|+0.289|+0.0104|-0.0929|

### Unit 2416

**positive tail:** ` Networks|\n|Authors:[ ||]Kalapriya, K|`

Product +371; without local attention +361. Dominant named source-pair share 7.2%; effective support 42.23.

Top named source-pair terms: `mlp12×mlp14` +40.3; `mlp13×mlp14` +31.5; `mlp12×mlp13` +29.9; `mlp11×mlp14` +24.8; `mlp11×mlp13` +18.2.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` ||`|0|-0.451|-0.132|+0.154|-0.0101|-0.094|-0.102|+0.0884|
|1|`Auth`|-3|-0.233|+0.0661|+0.159|-0.0527|-0.0879|-0.0663|+0.036|
|1|`:`|-1|-0.314|-0.0231|-0.0458|+0.0216|-0.142|-0.0207|+0.0511|
|3|` Indian`|-78|-0.12|+0.0869|+0.174|-0.0313|-0.0306|+0.333|-0.0363|
|1|`|`|-6|-0.199|+0.0339|-0.0126|-0.00425|-0.261|-0.0449|+0.0609|
|1|`\n`|-5|-0.181|-0.0771|+0.0419|+0.00635|-0.0993|-0.14|+0.0432|

**positive tail:** `Originally Posted[ by] Champ2647\nNo matter how`

Product +220; without local attention +228. Dominant named source-pair share 7.8%; effective support 33.61.

Top named source-pair terms: `mlp12×mlp13` +28; `mlp13×mlp14` +27.9; `mlp12×mlp14` +23.6; `mlp13×mlp13` +16.3; `mlp14×mlp14` +11.8.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` Posted`|-1|-0.504|+0.16|+0.206|-0.184|-0.252|-0.197|+0.226|
|1|`Originally`|-2|-0.449|+0.207|+0.0498|-0.116|-0.234|-0.145|+0.17|
|1|` by`|0|-0.59|+0.106|+0.0174|-0.0726|-0.25|-0.0471|+0.175|
|3|`Originally`|-2|+0.136|-0.00465|-0.229|-0.0317|-0.0138|+0.152|+0.0188|
|5|` Posted`|-1|+0.147|+0.0107|+0.0629|+0.0109|-0.0432|+0.0203|-0.00339|
|8|`Originally`|-2|-0.0723|+0.00402|+0.0527|-0.0041|+0.0328|-0.178|+0.0105|

**negative tail:** `icial Watch and True the Vote Seek[ to] Defend Florida’s Eff`

Product -618; without local attention -525. Dominant named source-pair share 6.1%; effective support 64.17.

Top named source-pair terms: `mlp12×mlp13` -39.2; `mlp12×mlp12` -23.9; `mlp10×mlp12` -22.4; `mlp12×attention15` -19.4; `mlp12×mlp14` -18.6.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` to`|0|-0.539|+0.98|+0.0289|-0.544|-0.776|-0.0211|+0.429|
|1|` Vote`|-2|-0.539|+0.495|+0.183|-0.365|-0.432|-0.0915|+0.282|
|1|` Seek`|-1|-0.605|+0.55|-0.141|-0.248|-0.413|-0.00551|+0.253|
|1|` Watch`|-6|-0.426|+0.545|-0.0133|-0.226|-0.528|+0.0284|+0.213|
|1|` the`|-3|-0.432|+0.44|-0.011|-0.185|-0.357|+0.0311|+0.141|
|1|` True`|-4|-0.359|+0.419|+0.0552|-0.171|-0.354|-0.0574|+0.148|

**negative tail:** ` 2009\nMisenheimer, NC[ –] Head women’s soccer coach`

Product -589; without local attention -589. Dominant named source-pair share 4.0%; effective support 76.16.

Top named source-pair terms: `mlp11×mlp12` -24.1; `mlp9×mlp12` -19.8; `mlp10×mlp12` -19.3; `mlp12×mlp14` -19.1; `mlp12×mlp12` -16.7.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` –`|0|-0.33|+0.336|-0.0489|-0.0946|-0.24|-0.03|+0.0892|
|3|`M`|-5|-0.539|+0.101|-0.248|+0.0794|-0.0649|+0.215|-0.081|
|3|` NC`|-1|-0.5|+0.00081|-0.186|+0.0928|-0.0608|+0.161|-0.0501|
|4|` NC`|-1|-0.715|-0.0294|+0.144|-0.0818|-0.00146|+0.0457|-0.0316|
|1|`heimer`|-3|-0.199|+0.169|+0.0352|-0.0408|-0.299|-0.0375|+0.067|
|3|`isen`|-4|-0.287|+0.0308|+0.0323|-0.0181|-0.0498|-0.143|+0.0555|

### Unit 3599

**positive tail:** ` the morning, when he rose to[ break] his fast. The old man had`

Product +166; without local attention +166. Dominant named source-pair share 3.0%; effective support 104.84.

Top named source-pair terms: `mlp12×mlp14` +5.98; `mlp10×mlp14` +5.92; `mlp13×mlp14` +4.84; `mlp14×mlp14` +4.6; `mlp10×mlp12` +3.73.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` when`|-4|-0.496|-0.0787|-0.108|+0.0929|+0.135|+0.0809|-0.107|
|1|` to`|-1|-0.605|-0.0566|-0.00734|+0.0387|+0.132|+0.0159|-0.0896|
|4|`izza`|-38|-0.212|-0.0178|-0.304|+0.0683|-0.0209|-0.168|+0.04|
|1|` rose`|-2|-0.598|-0.0795|+0.0458|+0.0201|+0.114|-0.00537|-0.0648|
|1|`,`|-5|-0.469|-0.0562|-0.00222|+0.0274|+0.118|-0.00093|-0.0549|
|4|` father`|-11|+0.128|-0.0237|-0.396|-0.0537|-0.0342|-0.155|-0.0242|

**positive tail:** ` a new study from Pew's Project[ for] Excellence in Journalism.\nThe PE`

Product +161; without local attention +159. Dominant named source-pair share 9.8%; effective support 36.47.

Top named source-pair terms: `mlp12×mlp14` +25.3; `mlp14×mlp14` +13.3; `mlp13×mlp14` +12.5; `mlp10×mlp14` +11.5; `mlp12×mlp12` +11.2.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` for`|0|+0.271|-0.0348|-0.21|-0.0664|+0.0627|-0.315|-0.0685|
|6|` 2012`|-53|-0.0957|+0.211|+0.162|-0.0357|-0.212|-0.295|+0.0486|
|2|` Project`|-1|+0.328|-0.0103|+0.165|+0.0508|+0.0078|-0.0882|-0.0264|
|1|`'s`|-2|-0.199|-0.0916|+0.0177|+0.0147|+0.197|+0.0764|-0.0544|
|1|` Project`|-1|-0.293|-0.0477|-0.132|+0.0525|+0.0918|-0.0311|-0.0178|
|3|` Project`|-1|+0.0991|+0.0819|-0.473|-0.0388|+0.111|+0.146|+0.0255|

**negative tail:** `\nFebruary 3, 1894|[\n]New York City\n|Died`

Product -732; without local attention -729. Dominant named source-pair share 8.3%; effective support 32.62.

Top named source-pair terms: `mlp12×mlp13` -61.5; `embedding×mlp12` -56; `embedding×mlp13` -51.7; `mlp12×mlp12` -33.3; `mlp13×mlp13` -28.4.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|4|`orman`|-14|-0.279|+0.00538|+0.176|-0.0506|+0.0489|+0.0319|-0.0226|
|4|` Rock`|-10|-0.225|+0.0401|-0.15|+0.0248|+0.0216|+0.149|-0.0382|
|2|`|`|-1|-0.149|-0.1|-0.0418|+0.0212|+0.124|+0.0648|-0.0283|
|3|`February`|-6|+0.0679|-0.0123|-0.0889|-0.00687|-0.00204|+0.428|+0.0289|
|2|` name`|-17|-0.124|-0.0846|-0.0899|+0.0215|+0.0389|+0.0653|-0.0129|
|2|`\n`|-7|-0.106|-0.0985|+0.207|-0.0115|+0.12|-0.289|+0.018|

**negative tail:** `Died||November 8, 1978[\n]|Training||National Academy of Design`

Product -637; without local attention -636. Dominant named source-pair share 8.8%; effective support 40.71.

Top named source-pair terms: `embedding×mlp12` -57; `embedding×mlp13` -34.8; `embedding×embedding` -31.2; `mlp12×mlp13` -29.6; `mlp12×mlp12` -24.9.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|4|`orman`|-27|-0.225|+0.0062|+0.203|-0.0469|+0.0562|+0.0367|-0.0209|
|4|` Rock`|-23|-0.201|+0.0463|-0.174|+0.0256|+0.0248|+0.171|-0.0394|
|4|`New`|-12|-0.361|-0.0819|-0.0954|+0.0641|-0.0129|+0.0127|+4e-05|
|2|`ied`|-6|-0.281|+0.0189|+0.0371|-0.0157|-0.0362|-0.0328|+0.0194|
|2|`||`|-5|-0.0903|-0.188|+0.00979|+0.0161|+0.167|+0.0267|-0.0175|
|3|` Rock`|-23|-0.0825|-0.0348|-0.0631|+0.00808|+0.0752|+0.215|-0.024|
