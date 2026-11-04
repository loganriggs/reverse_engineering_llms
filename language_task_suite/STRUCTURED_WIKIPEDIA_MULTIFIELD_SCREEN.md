# Clean article prose supplies many joint sites, but literal overlap is not semantic grounding

The [Wikimedia Structured Contents snapshot](https://huggingface.co/datasets/wikimedia/structured-wikipedia)
contains article leads, later paragraphs, and parsed infoboxes in one record.
Unlike WikiBio, its text retains ordinary capitalization and punctuation.
This screen asks whether two different structured facts are expressed in one
short piece of human-written prose, and whether the corresponding values can
be changed independently without losing token alignment. It is a
development screen on one English Parquet shard, pinned to dataset revision
`417c267bb457fa645c22eb3b5c77764963194c70`; the 371 MB source
shard stays local. No source prose is committed.

The [lead screen](audit_structured_wikipedia_multifield.py) samples 20,000
articles across row groups. It accepts a value only when it occurs exactly
once in the lead, is associated with a named infobox field, is short enough
for a local edit, and does not overlap another accepted value. It finds
3,567 articles with one or more such values, 1,883 with at least two, and
750 with at least three. With the first two matching fields serialized above
the lead, 1,881/1,883 put both literal target spans inside 256 GPT-2 tokens.
The common field pairs include family+genus, category+edition,
champion+runner-up, and category+location. Some are semantically coupled;
their frequency is not evidence that independent edits are valid.

The [four-arm prototype](structured_wikipedia_multifield_dgp.py) renders
the original article and three same-label substitutions: first field,
second field, or both. A visible-header parser checks that only the intended
target values changed, and a span check verifies that all other prose is
byte-identical. In a separately seeded 20,000-article development sample,
1,524 articles passed these **mechanical** checks; 1,515 had all eight
target spans token-aligned and inside 256 tokens. Fourteen target spans
failed the boundary check. An initial boundary check incorrectly treated
GPT-2's leading-space tokens as misaligned; the final check permits only
whitespace outside the exact value in an overlapping token.

Manual review rejected treating those 1,524 as a semantic training bank.
For example, replacing the `Sport` field of a hockey-league article with
ski jumping leaves “Hockey League” in its title. Replacing a person's full
name in the infobox and lead may leave the old article title unchanged.
Changing a country while retaining a province can create an impossible
geographic pair. Conversely, a field value can be mentioned in prose as a
relative's name, a firm, or a generic noun rather than as the article's
own attribute. Exact text overlap and localized edits cannot detect these
contradictions.

| First 40 prose words, 512 samples | Top-ten word share | Cross-document repeated-trigram share | Word types |
| --- | ---: | ---: | ---: |
| Unselected article leads | .288 | .192 | 4,830 |
| Selected two-field leads | .317 | .353 | 3,927 |
| Both-edited selected leads | .317 | .344 | 4,026 |
| Broad FineWeb preview | .208 | .025 | 5,859 |

Requiring two matching fields strongly selects formulaic leads. The edited
and original leads have similar form, but neither matches broad FineWeb.
This is a selection-bias problem at *task-bearing sites*, not merely a
problem with the background mixture.

The [later-section screen](audit_structured_wikipedia_section_sites.py)
offers a more promising surface. Among 10,000 sampled articles, 2,212
contained a 40–220-word later paragraph; 277 had two distinct literal
infobox values in one such paragraph, and 262 of those fit the field prefix
and paragraph inside 256 tokens. On 277 matched 40-word paragraphs,
selected later sections have repeated-trigram share **.105**, versus .064
for other later paragraphs from the same source shard. Manual examples
include a naturally written football-season paragraph with manager and
stadium mentions, but also false alignments such as “infantry” describing
a different unit. Thus later sections improve *form* without solving
semantic correspondence.

**Decision:** Keep the lead and section samplers as candidate sources and
the four-arm code as an unapproved mechanical fixture. Do not train on the
unfiltered edits or claim that the full article has a known symbolic oracle.
The next implementation should validate a small set of typed, corpus-
attested relations in their actual clauses—for example, a manager
associated with a club, a founding year associated with an organization,
or a venue associated with an event—rather than use value occurrence alone.
Pair labels that imply each other must be edited jointly or modeled as a
dependency, not forced into an independent 2×2. Once those relation checks
pass, audit rule-centered form on a new shard, build article-disjoint and
whole-relation-combination holdouts, then train 1–4-layer bilinear pilots.

Aggregate outputs: [lead yield](results/structured_wikipedia_multifield_shard0_n20000.json),
[four-arm mechanics and form](results/structured_wikipedia_four_arm_shard0_n20000.json),
and [later-section yield and form](results/structured_wikipedia_section_sites_shard0_n10000.json).
