"""V16 pilot: vary articles across every ordinary narrative frame.

The existing six frames and task rules stay in one sampled stream. At their
fixed determiner slots, this/the/that vary; before place and event nouns,
a/an are additionally available. For indefinite articles, the next noun law
is conditioned on its initial letter, so the marginal noun law is unchanged.
This is a measured form intervention, not a natural-language model.
"""

from surface_mixture_v2 import _frame, _split
from surface_mixture_v2_varied import DETERMINER_SLOTS
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v12_price_faq import SPECIAL
from surface_mixture_v14_conditional import (
    SurfaceMixtureV14Conditional, VOCAB_V14,
)


VOCAB_V16 = tuple(dict.fromkeys(VOCAB_V14 + ('an', 'this', 'that')))
TOKEN2ID_V16 = {word: i for i, word in enumerate(VOCAB_V16)}
ID2TOKEN_V16 = {i: word for word, i in TOKEN2ID_V16.items()}
VOWELS = frozenset('aeiou')


def _indefinite_article(word):
    return 'an' if word[0].lower() in VOWELS else 'a'


class SurfaceMixtureV16Determiners(SurfaceMixtureV14Conditional):
    def __init__(self, seed=0, article_variation=.8,
                 indefinite_share=.9, fact_article_variation=.2, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if not 0 <= article_variation <= 1:
            raise ValueError('article_variation must be in [0, 1]')
        if not 0 <= indefinite_share <= 1:
            raise ValueError('indefinite_share must be in [0, 1]')
        if not 0 <= fact_article_variation <= 1:
            raise ValueError('fact_article_variation must be in [0, 1]')
        self.article_variation = article_variation
        self.indefinite_share = indefinite_share
        self.fact_article_variation = fact_article_variation
        self.vocab, self.token2id, self.id2token = (
            VOCAB_V16, TOKEN2ID_V16, ID2TOKEN_V16)

    def _tokens(self, prefix):
        return [ID2TOKEN_V16[word] if isinstance(word, int) else word
                for word in prefix]

    def next_distribution(self, prefix):
        words = self._tokens(prefix)
        history, raw = _split(words)
        content = _content(raw)
        if not content or content[0] in SPECIAL:
            return super().next_distribution(words)
        try:
            frame = _frame(content)
        except ValueError:
            return super().next_distribution(words)
        pos = len(content)
        slots = DETERMINER_SLOTS[frame]
        if pos in slots:
            base = super().next_distribution(words)
            assert base == {'the': 1.0}, (frame, pos, base)
            adjective_slot = pos == OBJECT_INDEX[frame]-2
            fact_site = adjective_slot and bool(self._pending_attributes(history))
            p = (self.fact_article_variation if fact_site
                 else self.article_variation)
            indefinite = 0. if fact_site else self.indefinite_share
            out = {'the': 1-p,
                   'this': p*(1-indefinite)/2,
                   'that': p*(1-indefinite)/2}
            if indefinite:
                next_base = super().next_distribution(words+['the'])
                vowels = sum(mass for word, mass in next_base.items()
                             if _indefinite_article(word) == 'an')
                out['a'] = p*indefinite*(1-vowels)
                out['an'] = p*indefinite*vowels
            return {word: mass for word, mass in out.items() if mass > 0}
        if pos-1 in slots and content[-1] in ('a', 'an'):
            base = super().next_distribution(words)
            out = {word: mass for word, mass in base.items()
                   if _indefinite_article(word) == content[-1]}
            total = sum(out.values())
            assert total > 0, (frame, pos, content[-1])
            return {word: mass/total for word, mass in out.items()}
        return super().next_distribution(words)
