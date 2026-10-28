# Bilin18 layers 0–2 named-source and QK×OV traces

The selected sites come from 128 held-out FineWeb tokens. Each factor is expanded over the embedding, prior attention/MLP writes, and the current attention write. The current attention term is then expanded over native head/source QK×OV paths.

`Without local attention` removes only the target block's attention write; prior context remains for layers 1–2. Source-pair terms are symmetric off-diagonal sums.

## Layer 0

### Unit 1737

**positive tail:** ` decorations. Such places as laundry rooms[,] office spaces, craft spaces, game`

Product +2.48e+03; without local attention +3.22e+03. Dominant named source-pair share 92.3%; effective support 1.17.

Top named source-pair terms: `embedding×embedding` +2.33e+03; `embedding×attention0` +171; `attention0×attention0` -23.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`,`|0|+0.228|-9.7|-9.7|-4.41|-6.55|-6.55|-2.98|
|5|` as`|-3|+0.4|+1.73|+1.73|+1.39|-2.27|-2.27|-1.81|
|3|` rooms`|-1|+0.0757|+14.2|+14.2|+2.14|-4.55|-4.55|-0.689|

**negative tail:** ` down the attic for decorating at[ Christmas] and all the other decorating holidays`

Product -72.8; without local attention -53.2. Dominant named source-pair share 49.3%; effective support 2.58.

Top named source-pair terms: `embedding×attention0` -35.9; `embedding×embedding` -24.8; `attention0×attention0` -12.1.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` at`|-1|+0.316|+11|+11|+6.94|+2.17|+2.17|+1.37|
|3|` at`|-1|+0.139|+3.02|+3.02|+0.839|-0.698|-0.698|-0.194|
|3|`ating`|-2|+0.0381|-6.96|-6.96|-0.53|-4.35|-4.35|-0.332|

## Layer 1

### Unit 1775

**positive tail:** `, to bring things up and down[ the] attic for decorating at Christmas and`

Product +63.2; without local attention +222. Dominant named source-pair share 35.8%; effective support 3.83.

Top named source-pair terms: `embedding×attention1` +22.7; `embedding×embedding` +19.8; `embedding×mlp0` +8.5; `attention1×attention1` +6.44.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|` the`|0|-0.258|+0.935|+0.0447|-0.252|-5.17|-0.0108|+1.34|
|5|` bring`|-5|+0.123|+13.1|+0.0683|+1.62|-0.482|+0.0331|-0.0552|
|5|` and`|-2|+0.229|-0.665|-0.0567|-0.165|-2.14|+0.0363|-0.482|

**negative tail:** ` in your home where you can add[ high] shelves and use decorative boxes to contain`

Product -15.2; without local attention +13.9. Dominant named source-pair share 47.9%; effective support 3.37.

Top named source-pair terms: `mlp0×attention1` -15.6; `embedding×attention1` -5.47; `mlp0×mlp0` +4.84; `attention1×attention1` +3.71.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|5|` add`|-1|+0.271|+8.57|+0.0591|+2.34|+0.0444|+0.13|+0.0475|
|1|` high`|0|-0.24|-3.73|+0.073|+0.879|-2.01|-0.0302|+0.491|
|1|` add`|-1|-0.176|-3.19|+0.0279|+0.557|-2.28|+0.0763|+0.388|

## Layer 2

### Unit 4027

**positive tail:** ` basement. I'm finding it more[ difficult], pain in the be hind,`

Product +77.4; without local attention +45.2. Dominant named source-pair share 41.4%; effective support 3.72.

Top named source-pair terms: `mlp1×attention2` +33.3; `mlp1×mlp1` +20.7; `attention2×attention2` +13.1; `attention1×mlp1` +3.9.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` finding`|-3|+0.247|-0.743|+3.54|+0.691|+4.81|-0.00316|+1.19|
|2|` finding`|-3|+0.218|+5.14|+0.746|+1.28|+2.35|-0.269|+0.454|
|2|` more`|-1|+0.336|+1.91|+0.768|+0.899|+2.4|-0.203|+0.737|

**negative tail:** ` FL we have no basement. I['m] finding it more difficult, pain in`

Product -583; without local attention -744. Dominant named source-pair share 47.7%; effective support 3.58.

Top named source-pair terms: `mlp1×mlp1` -278; `mlp1×attention2` -86.6; `embedding×mlp1` -67.7; `mlp0×mlp1` -64.3.

|head|source|offset|route|current-V→L|bus-V→L|path L|current-V→R|bus-V→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|5|`'m`|0|-0.285|+1.01|+0.492|-0.428|-3.15|-1.6|+1.35|
|6|` I`|-1|+0.332|-2.99|-0.957|-1.31|+1.94|-1.58|+0.119|
|8|` I`|-1|+0.242|-0.502|+1.74|+0.299|+1.57|+1.54|+0.753|
