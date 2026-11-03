"""Versioned v3 grammar with shared parenthesis/square-bracket content.

The original joint v3 generator keeps all of its lexical, skip, induction,
category, and priority-dispatch rules. Only the bracket-content conditional
distribution changes: both opener types now use the same ADJ/NOUN grammar.
The closer still depends on the opener. This makes opener-only edits of a
visible content prefix in support of both bracket types.

Import after adding toy_models_of_tensor_networks/attn_circuits to sys.path.
"""

from generator_v3 import LanguageV3


SHARED_BRACKET_CONTENT = {
    'start_cat': 'ADJ',
    'transitions': {
        'ADJ': (('ADJ', .4), ('NOUN', .6)),
        'NOUN': (('ADJ', .75), ('NOUN', .25)),
    },
}


class LanguageV3SharedBracket(LanguageV3):
    """Same joint-rule stream as LanguageV3, with counterfactual-safe brackets."""

    def _sample_bracket_content(self, bracket_type, last_cat, prev_token):
        assert bracket_type in ('paren', 'quote')
        if last_cat is None:
            cat = SHARED_BRACKET_CONTENT['start_cat']
        else:
            cats, probabilities = zip(*SHARED_BRACKET_CONTENT['transitions'][last_cat])
            cat = self.rng.choices(cats, weights=probabilities)[0]
        token = self._sample_from_category(cat)
        if prev_token is not None and token == prev_token:
            for _ in range(5):
                token = self._sample_from_category(cat)
                if token != prev_token:
                    break
        return token, cat


def bracket_close_distribution(opener, content_count):
    """Exact closer mass given visible in-span content, for sampled length 2–4.

    Since both opener types share every content likelihood, the posterior
    over the original uniformly sampled span length depends only on the
    number of content tokens already observed. This function returns only
    the closer mass; the remaining mass is shared content continuation.
    """
    if opener not in ('(', '[') or content_count not in (2, 3, 4):
        raise ValueError((opener, content_count))
    p_close = 1 / (5 - content_count)
    return {')' if opener == '(' else ']': p_close}


def bracket_next_distribution(gen, opener, content):
    """Exact next-token oracle within an open shared-content span.

    Span length is uniform on {2,3,4}. The visible content prefix reveals
    only that the sampled length is at least its current length. Token
    probabilities include the generator's five extra draws when a sampled
    word repeats the preceding word.
    """
    if opener not in ('(', '[') or not 1<=len(content)<=4:
        raise ValueError((opener,content))
    if any(token not in gen.token2cat for token in content):
        raise ValueError(content)
    k=len(content)
    close=0. if k<2 else next(iter(bracket_close_distribution(opener,k).values()))
    distribution={')' if opener=='(' else ']':close} if close else {}
    if close<1:
        last=content[-1]
        last_cat=gen.token2cat[last]
        for cat,p_cat in SHARED_BRACKET_CONTENT['transitions'][last_cat]:
            vocabulary=gen.categories[cat]
            n=len(vocabulary)
            if last in vocabulary:
                same=(1/n)**6
                for token in vocabulary:
                    p_token=same if token==last else (1-same)/(n-1)
                    distribution[token]=distribution.get(token,0.)+(
                        (1-close)*p_cat*p_token)
            else:
                for token in vocabulary:
                    distribution[token]=distribution.get(token,0.)+(
                        (1-close)*p_cat/n)
    assert abs(sum(distribution.values())-1)<1e-10
    return distribution
