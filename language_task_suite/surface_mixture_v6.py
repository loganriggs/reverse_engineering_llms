"""Joint mixed-rule prose with source-attributed claims and source queries.

Statements and questions share the same stream as v5 narratives. Source
queries are available only after two different speakers have made different
claims, preventing a source-blind constant answer from being sufficient.
"""

from surface_mixture_v2 import OBJECTS, PEOPLE, _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v5 import SurfaceMixtureV5, VOCAB_V5


SPEAKERS = PEOPLE[:8]
CLAIM_OBJECTS = OBJECTS[:16]
EXTRA = ('As', 'Who', '"')
VOCAB_V6 = VOCAB_V5 + tuple(word for word in EXTRA if word not in VOCAB_V5)
TOKEN2ID_V6 = {word: i for i, word in enumerate(VOCAB_V6)}
ID2TOKEN_V6 = {i: word for word, i in TOKEN2ID_V6.items()}


def source_pair_heldout(speaker, obj):
    """Checkerboard pair split for compositional source-binding tests."""
    return (SPEAKERS.index(speaker)+CLAIM_OBJECTS.index(obj)) % 4 == 0


def source_facts(history):
    """Object -> speaker for completed source statements in visible history."""
    facts = {}
    for raw in history:
        content = _content(raw)
        if content and content[0] == 'As':
            if content[4] == '"':
                assert len(content) == 11 and content[2:6] == ['said', ',', '"', 'the']
                assert content[7:] == ['was', 'missing', '"', '.']
                obj = content[6]
            else:
                assert len(content) == 9 and content[2:5] == ['said', ',', 'the']
                assert content[6:] == ['was', 'missing', '.']
                obj = content[5]
            assert obj not in facts
            facts[obj] = content[1]
    return facts


class SurfaceMixtureV6(SurfaceMixtureV5):
    def __init__(self, seed=0, quote_rate=.25, source_rate=.20,
                 count_rate=0., exclude_source_pairs=False,
                 unquoted_source_rate=0., **kwargs):
        super().__init__(seed=seed, count_rate=count_rate, **kwargs)
        assert 0 <= quote_rate < 1 and 0 <= source_rate < 1
        assert quote_rate+source_rate < 1
        assert 0 <= unquoted_source_rate <= 1
        self.quote_rate = quote_rate
        self.source_rate = source_rate
        self.exclude_source_pairs = exclude_source_pairs
        self.unquoted_source_rate = unquoted_source_rate
        self.vocab = VOCAB_V6
        self.token2id = TOKEN2ID_V6
        self.id2token = ID2TOKEN_V6

    def _tokens(self, prefix):
        return [ID2TOKEN_V6[i] if isinstance(i, int) else i for i in prefix]

    def _available_claim_objects(self, history, speaker, used=None):
        used = source_facts(history) if used is None else used
        return [obj for obj in CLAIM_OBJECTS if obj not in used
                and (not self.exclude_source_pairs or
                     not source_pair_heldout(speaker, obj))]

    def _start_distribution(self, history, after_newline):
        out = super()._start_distribution(history, after_newline)
        facts = source_facts(history)
        quote_mass = (self.quote_rate if any(
            self._available_claim_objects(history, speaker, facts)
            for speaker in SPEAKERS) else 0.)
        source_mass = (self.source_rate if len(set(facts.values())) >= 2
                       else 0.)
        content_mass = 1. if after_newline else 1.-self.newline_rate
        out = {word: (mass if word == '\n' else
                      mass*(1.-quote_mass-source_mass))
               for word, mass in out.items()}
        if quote_mass:
            out['As'] = content_mass*quote_mass
        if source_mass:
            out['Who'] = content_mass*source_mass
        return out

    def _statement_distribution(self, history, content):
        pos = len(content)
        if pos == 1:
            used = source_facts(history)
            available = [speaker for speaker in SPEAKERS
                         if self._available_claim_objects(history, speaker,
                                                          used)]
            return {speaker: 1/len(available) for speaker in available}
        if pos == 2:
            return {'said': 1.}
        if pos == 3:
            return {',': 1.}
        if pos == 4:
            return ({'the': 1.} if self.unquoted_source_rate == 1 else
                    {'"': 1.} if self.unquoted_source_rate == 0 else
                    {'"': 1.-self.unquoted_source_rate,
                     'the': self.unquoted_source_rate})
        quoted = content[4] == '"'
        obj_pos = 6 if quoted else 5
        if pos == obj_pos:
            available = self._available_claim_objects(history, content[1])
            return {obj: 1/len(available) for obj in available}
        fixed = ({5: 'the', 7: 'was', 8: 'missing', 9: '"', 10: '.'}
                 if quoted else {6: 'was', 7: 'missing', 8: '.'})
        if pos in fixed:
            return {fixed[pos]: 1.}
        raise ValueError(f'Bad source-statement position {pos}')

    def _source_distribution(self, history, content):
        pos = len(content)
        fixed = {1: 'said', 2: 'the', 4: 'was', 5: 'missing',
                 6: '?', 7: 'Answer', 8: ':', 10: '.'}
        facts = source_facts(history)
        if pos == 3:
            return {obj: 1/len(facts) for obj in facts}
        if pos == 9:
            return {facts[content[3]]: 1.}
        if pos in fixed:
            return {fixed[pos]: 1.}
        raise ValueError(f'Bad source-question position {pos}')

    def next_distribution(self, prefix):
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if content and content[0] == 'As':
            return self._statement_distribution(history, content)
        if content and content[0] == 'Who':
            return self._source_distribution(history, content)
        return super().next_distribution(prefix)

    def source_answer(self, prefix):
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if not content or content[0] != 'Who' or len(content) != 9:
            raise ValueError('Not at a source answer slot')
        return source_facts(history)[content[3]]

    def sample_sequence(self, length=128):
        tokens, labels, qs = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            _, raw = _split(tokens)
            content = _content(raw)
            label = 'frame'
            if content:
                if content[0] == 'Who' and len(content) == 9:
                    label = 'source_answer'
                elif content[0] == 'Where' and len(content) == 8:
                    label = 'question_answer'
                elif content[0] == 'How' and len(content) == 11:
                    label = 'count_answer'
                elif content[0] not in ('Who', 'Where', 'How', 'As'):
                    frame = _frame(content)
                    if len(content) == _indices(frame)[1]:
                        label = 'mixed_verb'
                    elif len(content) == OBJECT_INDEX[frame]:
                        label = 'mixed_object'
            word = self.rng.choices(list(q), weights=list(q.values()))[0]
            labels.append(label)
            qs.append(q)
            tokens.append(word)
        return [TOKEN2ID_V6[word] for word in tokens], labels, qs


def render(ids):
    words = [ID2TOKEN_V6[i] if isinstance(i, int) else i for i in ids]
    text, inside_quote = '', False
    for word in words:
        if word == '\n':
            text = text.rstrip()+'\n'
        elif word in (',', '.', '?', ':'):
            text = text.rstrip()+word
        elif word == '"':
            if inside_quote:
                text = text.rstrip()+'"'
            else:
                text += ('' if not text or text[-1].isspace() else ' ')+'"'
            inside_quote = not inside_quote
        else:
            if text and not text[-1].isspace() and not (
                    inside_quote and text[-1] == '"'):
                text += ' '
            text += word
    return text
