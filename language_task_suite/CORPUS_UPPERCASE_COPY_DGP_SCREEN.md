# A natural-prose copy rule: high yield, poor semantic substitution

To add a computationally different rule from list continuation and bracket
closure, this screen samples a *document-local literal copying* task. It
finds a three-letter uppercase string repeated 20–160 words later in a
FineWeb document, then replaces every whole-token occurrence consistently
with one of two randomly drawn three-letter codes. The later occurrence is
the prediction target. The two arms differ at exactly the preceding code
tokens through the target; a visible earlier occurrence determines the
sampled answer. The code is a surface string, not an entity identity label.

The [generator](corpus_uppercase_copy_dgp.py) and
[BPE validator](corpus_uppercase_copy_validation.py) enforce same-length
substitution, reversibility, two distinct target token IDs, shared target
position, a previous occurrence inside the 256-token context, and
token-local paired prefix changes. A synthetic fixture checks these
invariants. The replacement prior uses 34 familiar acronym-like codes with
both ordinary and leading-space GPT-2 BPE tokens; obvious all-caps English
headings such as `THE` are excluded as sources. This is a deliberately
simple prior, not a semantic type model.

On FineWeb `sample-10BT` offsets 84,500–85,499, the
[audit](results/corpus_uppercase_copy_dgp_curated_skip84500_n1000.json)
found 158 aligned paired documents from 1,000 sources. The mean gap to
the previous occurrence was 56.5 words (range 20–155). Among 154 selected
first-100-word passages, source and transformed top-ten word mass are both
.220; repeated-across-document trigram share is .0217 for source and .0215
for either sampled arm. Around the target, corresponding 41-word trigram
shares are .0131 for source and .0141/.0131 for the arms. These local
word-level form metrics barely move because the transformation changes only
codes; they cannot certify semantic naturalness.

Manual inspection of twelve deterministic examples found the decisive
problem. Most *source* strings were genuine acronyms, but a global
acronym-like replacement often changed their type or meaning: a search
marketing abbreviation became a cryptocurrency or FAQ label; a historical
computer company became a hardware device; a networking term became a
public-health agency. Some substitutions, such as one sports-league code
for another, are less jarring, but even those can change factual claims.
The audit output shows source and target snippets locally for review; no
source prose is committed.

**Decision:** this high-yield surface rule is a valid synthetic *copying
fixture* but not yet a realistic training distribution. Do not train a
bilinear model on this version or infer an entity/copy circuit from it. The
next generator should either (a) use source-type-conditioned aliases with
an independently reviewed compatibility policy, or (b) use genuinely nonce
codes and evaluate multi-token copying, while explicitly treating the
surrounding text as an alternate-world document. Either route needs a new
manual naturalness sample and paired BPE/causal checks before training.
The independent FineWeb carrier still supplies only a controlled rule-site
oracle, not a whole-document next-token oracle.
