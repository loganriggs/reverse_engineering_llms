# Clause patterns and a local semantic judge do not yet certify multi-rule Wikipedia examples

The [structured-Wikipedia multi-field screen](STRUCTURED_WIKIPEDIA_MULTIFIELD_SCREEN.md)
found many articles with two infobox values literally present in prose. Its
failure mode was semantic: a word could refer to another unit, relative,
company, or location, and independent swaps could contradict titles or
linked facts. This follow-up tests two ways to distinguish a *relation
assertion* from mere co-occurrence. Both are exploratory development gates;
no resulting text is a training set.

The [typed clause matcher](typed_wikipedia_relations.py) checks a small set
of explicit lexical relations, such as “founded in YEAR,” “studied at
SCHOOL,” “owned by COMPANY,” “family TAXON,” and “genus TAXON.” It requires
the article title in the same approximate sentence. The
[10,000-article audit](audit_typed_wikipedia_relations.py) found 145
lead-clause witnesses and four later-paragraph witnesses, but only **one
lead with two** and **no later paragraph with two**. The surviving example
was a species lead with explicit family and genus. This high-precision
syntax path is far too small for the desired multi-rule pretraining
distribution. Loosening the title anchor or matching more words would
largely restore the ambiguity that this gate was meant to remove.

The [local model screen](judge_structured_wikipedia_relations.py) sampled
30 two-field leads and 30 two-field later paragraphs from the larger
literal-match pool. Qwen3-4B, prompted to decide whether the prose asserts
each infobox field *about the article subject*, returned parseable JSON for
30/30 leads and 29/30 later paragraphs. It marked both fields asserted in
29 leads and 22 later paragraphs. These are **model judgments**, not measured
truth. Manual inspection found clear false positives in the later set:

* A population paragraph says “According to 2011 Census of India”; this
  does not by itself assert that the village's infobox `Country` is India.
* A military-history paragraph discusses the Australian Army considering
  commando units; that does not establish every `Country` or `Branch`
  attribute of the article's named squadron.
* A paragraph mentioning a bishop's appointment in Carpentras does not
  automatically establish the infobox's `See` relation; the model marked
  both fields true.

Some other cases are genuinely ambiguous rather than plainly wrong. This
small inspection is not a quantified precision estimate, but the visible
errors suffice to reject the model's positive label as a semantic
certificate. Its first prompt also asked whether an unspecified value swap
was safe; that question was ill-defined and led the model to count the
*intended* changed fact as a contradiction. The run was interrupted, the
prompt was narrowed to assertion only, and only the latter completed run
is reported. The local model's reasons are not committed; the JSON stores
hashes, field labels, and boolean decisions, with no source passages.

**Decision:** Stop mining unannotated Wikipedia with ever-broader literal
or model-based filters as the main DGP. The clean prose is valuable as a
form reference and background LM component. For rule-grounded training,
move to human-corrected structured-data/text pairs, where the intended
facts have been annotated or checked, and independently audit any remaining
unmentioned or unsupported facts. [ToTTo](https://github.com/google-research-datasets/totto)
provides a large single-sentence component; the human-revised
[SynthBio](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/9fc3d7152ba9336a670e36d0ed79bc43-Paper-round2.pdf)
set provides a smaller multi-sentence semantic control. Neither alone solves document planning, so keep the
jointly sampled world-state grammar as a correctness fixture and evaluate
the combined corpus/source mixture on rule-centered form and genuine
multi-rule behavior before weight interpretation.

Aggregates: [typed clause yield](results/typed_wikipedia_relations_shard0_n10000.json)
and [local model decisions](results/structured_wikipedia_semantic_judge_shard0_n60.json).
