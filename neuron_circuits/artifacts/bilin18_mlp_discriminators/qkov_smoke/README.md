# Layer-0 QK×OV paths into native bilinear MLP neurons

This report uses 128 held-out FineWeb token positions from rows 400--401. For each preselected unit, activation sites are selected only within this held-out panel.

For a fixed destination and source, `path L = QK1 × QK2 × OV→L`; the R path is analogous. The sum of these paths plus the direct current-token residual reconstructs the two pre-RMS factor inputs. RMSNorm supplies a shared positive scale, after which the native product is `Left × Right`.

The source paths are exact fixed-input forward decompositions, not causal deletion effects. A large signed term may cancel another term.

## Unit 1737

### positive tail: ` decorations. Such places as laundry rooms[,] office spaces, craft spaces, game`

- Native factors: Left +54.2, Right +45.8, product +2.48e+03.
- Raw Left: direct +679 + attention +96.8 = +776.
- Raw Right: direct +704 + attention -49.3 = +656.
- Attention absolute-path share: Left 30.3%, Right 28.5%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|`,`|0|-0.396|-0.574|+0.228|-278|-63.2|-188|-42.7|
|5|` as`|-3|+0.73|+0.547|+0.4|+49.6|+19.9|-64.9|-26|
|3|` rooms`|-1|-0.268|-0.283|+0.0757|+405|+30.7|-130|-9.87|

### negative tail: ` down the attic for decorating at[ Christmas] and all the other decorating holidays`

- Native factors: Left +17.4, Right -4.19, product -72.8.
- Raw Left: direct +162 + attention +148 = +311.
- Raw Right: direct -48.7 + attention -26.2 = -75.2.
- Attention absolute-path share: Left 63.4%, Right 78.0%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|` at`|-1|+0.516|+0.613|+0.316|+392|+124|+77.4|+24.5|
|3|` at`|-1|-0.379|-0.365|+0.139|+108|+15|-24.9|-3.45|
|3|`ating`|-2|-0.27|-0.142|+0.0381|-249|-9.47|-155|-5.92|

## Unit 1642

### positive tail: `/[ shelves]. These can be handsome pieces of`

- Native factors: Left +22.8, Right +10.3, product +235.
- Raw Left: direct +251 + attention +43 = +294.
- Raw Right: direct +130 + attention +3.29 = +134.
- Attention absolute-path share: Left 15.4%, Right 19.7%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` shelves`|0|-0.277|-0.199|+0.0552|+269|+14.8|-115|-6.36|
|3|`/`|-1|+0.289|+0.293|+0.0845|+206|+17.4|+53.9|+4.55|
|0|` shelves`|0|+0.361|-0.234|-0.0845|-35|+2.95|-69.1|+5.83|

### negative tail: `. Also, look for areas in[ your] home where you can add high shelves`

- Native factors: Left -11.7, Right +9.25, product -108.
- Raw Left: direct -75.5 + attention -95.5 = -171.
- Raw Right: direct +150 + attention -14.7 = +135.
- Attention absolute-path share: Left 78.1%, Right 61.1%.

|head|source|offset|QK1|QK2|route|OV→L|path L|OV→R|path R|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|` in`|-1|+0.373|+0.42|+0.156|-447|-69.8|-103|-16.1|
|3|` your`|0|+0.328|+0.486|+0.159|-110|-17.5|-138|-22|
|6|` in`|-1|-0.594|+0.547|-0.324|+62.8|-20.4|+45.2|-14.6|
