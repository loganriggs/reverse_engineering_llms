# A small inspected dependency-cue panel

`fineweb_dependency_panel.py` selects five pattern-positive windows for each of five cues from the existing frozen 480 × 513-token FineWeb panel. It uses seed 20261102, excludes previously selected rows, pins the source hash, and stores row IDs plus qualitative labels in `fineweb_dependency_panel.json`. It does **not** commit source prose. I inspected the matched text around each cue. “Yes” means the local semantic relationship is reasonably clear, not that it is an unambiguous next-token task or that the model must learn it.

| Cue and intended relationship | Yes | No | Unclear | What the inspection showed |
|---|---:|---:|---:|---|
| Attribution of content to a named source | 4 | 0 | 1 | Direct speech and reported claims are useful forms; one “said as much” lacks a clear local source. |
| Correction replacing an earlier fact | 0 | 5 | 0 | “Instead,” “actually,” and “updated” often mark contrast, evaluation, or changelog structure rather than a corrected entity state. |
| Alias linking two referential forms | 3 | 1 | 1 | Country and procedure aliases occur, but “aka” can be figurative and referents can be unclear. |
| Numeric update with a locally inferable arithmetic result | 1 | 3 | 1 | One payment example supports division of a total over remaining installments; many numbers are dates, ranks, or statistics. |
| Numbered heading/item scope | 5 | 0 | 0 | The sampled items have distinct headings or explanations; the scope of the item is the plausible dependency. |

These are **25 cue-selected development windows**, not a random prevalence sample, and the categories do not have equal numbers of eligible rows. For example, after exclusions there are 207 attribution matches but only 10 alias matches. The windows can start or end mid-document. The labels are a manual pilot and should be rechecked by an independent annotator before using them as a benchmark. Their useful conclusion is qualitative: build a source/claim and numbered-scope task next; do not justify a high-frequency correction or arithmetic task merely from regex counts. A new untouched natural panel is needed for prevalence and generalization estimates.

The synthetic task should sample a latent source–claim graph before wording it, mix its statements with the existing competing-rule narrative, and include at least two claims by different speakers before asking who made one. The query object must be counterbalanced, with a causal edit that changes the key and an irrelevant narrative edit as a no-effect control. This tests actual source binding rather than the presence of a quotation mark. Later forms should vary between direct quotation and reported speech, then test whether a component transfers across those forms and lexical combinations.
