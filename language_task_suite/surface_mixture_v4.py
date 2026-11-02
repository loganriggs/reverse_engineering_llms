"""Mixed narrative and object-to-place questions with an exact prefix oracle.

Narrative sentences retain the v3 verb/object mixtures. A question asks for
the place in the most recent narrative sentence mentioning its object. Short
numbered lines vary document layout. Questions are excluded from narrative
history, so their text cannot silently change the original v3 rules.
"""

from surface_mixture_v2 import (INTERIOR_PLACES, TOKEN2ID, VOCAB, _frame,
                                _indices, _split, uniform)
from surface_mixture_v3 import OBJECT_INDEX, SurfaceMixtureV3


EXTRA = ('\n', ':', '?', 'Where', 'Answer', 'did', 'appear') + tuple(str(i) for i in range(1, 10))
VOCAB_V4 = VOCAB + tuple(token for token in EXTRA if token not in TOKEN2ID)
TOKEN2ID_V4 = {token: i for i, token in enumerate(VOCAB_V4)}
ID2TOKEN_V4 = {i: token for token, i in TOKEN2ID_V4.items()}
def _content(clause):
    if clause and clause[0] == '\n':
        if len(clause) == 1:
            return []
        if clause[1] in {str(i) for i in range(1, 10)}:
            return clause[3:] if len(clause) >= 3 and clause[2] == ':' else []
        return clause[1:]
    return clause


def _narrative_history(history):
    return [_content(clause) for clause in history
            if _content(clause) and _content(clause)[0] != 'Where']


def _object_place(history):
    """Last observed place for each observed object, in narrative history."""
    out = {}
    for clause in _narrative_history(history):
        frame = _frame(clause)
        obj = clause[OBJECT_INDEX[frame]]
        place = clause[_indices(frame)[2]]
        out[obj] = place
    return out


class SurfaceMixtureV4(SurfaceMixtureV3):
    def __init__(self, seed=0, weights=None, object_weights=None,
                 object_vocab_size=None, question_rate=.05,
                 newline_rate=.30, numbered_line_rate=.25,
                 hard_query_rate=None, counterbalanced_queries=False):
        super().__init__(seed, weights, object_weights, object_vocab_size)
        assert (0 <= question_rate < 1 and 0 < newline_rate < 1
                and 0 < numbered_line_rate < 1)
        assert hard_query_rate is None or 0 <= hard_query_rate <= 1
        assert not (counterbalanced_queries and hard_query_rate is not None)
        self.question_rate = question_rate
        self.newline_rate = newline_rate
        self.numbered_line_rate = numbered_line_rate
        self.hard_query_rate = hard_query_rate
        self.counterbalanced_queries = counterbalanced_queries
        self.vocab = VOCAB_V4
        self.token2id = TOKEN2ID_V4
        self.id2token = ID2TOKEN_V4

    def _tokens(self, prefix):
        return [ID2TOKEN_V4[i] if isinstance(i, int) else i for i in prefix]

    def _v3_prefix(self, history, clause):
        return [token for old in _narrative_history(history)
                for token in old] + clause

    def _start_distribution(self, history, after_newline):
        narrative = super().next_distribution(self._v3_prefix(history, []))
        places = _object_place(history)
        answerable = (len(set(places.values())) >= 2 if self.counterbalanced_queries
                      else bool(places))
        question_mass = self.question_rate if answerable else 0.0
        content_mass = 1.0 if after_newline else 1.0-self.newline_rate
        out = {token: mass*content_mass*(1-question_mass)
               for token, mass in narrative.items()}
        if question_mass:
            out['Where'] = content_mass*question_mass
        if not after_newline:
            out['\n'] = self.newline_rate
        return out

    def _question_distribution(self, history, content):
        pos = len(content)
        fixed = {1: 'did', 2: 'the', 4: 'appear', 5: '?',
                 6: 'Answer', 7: ':', 9: '.'}
        if pos in fixed:
            return {fixed[pos]: 1.0}
        places = _object_place(history)
        if pos == 3:
            objects = tuple(places)
            if self.counterbalanced_queries:
                return {obj: 1/len(objects) for obj in objects}
            latest = _narrative_history(history)[-1]
            latest_object = latest[OBJECT_INDEX[_frame(latest)]]
            if self.hard_query_rate is not None and len(objects) > 1:
                earlier_objects = [obj for obj in objects if obj != latest_object]
                out = {obj: self.hard_query_rate/len(earlier_objects)
                       for obj in earlier_objects}
                out[latest_object] = 1-self.hard_query_rate
                return {obj: mass for obj, mass in out.items() if mass > 0}
            # The original v4 sampling prior favors the most recent object.
            out = {obj: .5/len(objects) for obj in objects}
            out[latest_object] += .5
            return out
        if pos == 8:
            return {places[content[3]]: 1.0}
        raise ValueError(f'Bad question position {pos}')

    def next_distribution(self, prefix):
        history, raw = _split(self._tokens(prefix))
        if not raw:
            if not history:
                return super().next_distribution([])
            return self._start_distribution(history, after_newline=False)
        if raw[0] == '\n':
            if len(raw) == 1:
                out = self._start_distribution(history, after_newline=True)
                out = {token: mass*(1-self.numbered_line_rate)
                       for token, mass in out.items()}
                out[str((len(history) % 9)+1)] = self.numbered_line_rate
                return out
            if len(raw) == 2 and raw[1].isdigit():
                return {':': 1.0}
        content = _content(raw)
        if not content:
            return self._start_distribution(history, after_newline=True)
        if content[0] == 'Where':
            return self._question_distribution(history, content)
        if content[0] == 'In' and len(content) == 2:
            return uniform(INTERIOR_PLACES)
        return super().next_distribution(self._v3_prefix(history, content))

    def candidates(self, prefix):
        history, raw = _split(self._tokens(prefix))
        return super().candidates(self._v3_prefix(history, _content(raw)))

    def object_candidates(self, prefix):
        history, raw = _split(self._tokens(prefix))
        return super().object_candidates(self._v3_prefix(history, _content(raw)))

    def question_answer(self, prefix):
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if not content or content[0] != 'Where' or len(content) != 8:
            raise ValueError('Not at a question answer slot')
        return _object_place(history)[content[3]]

    def sample_sequence(self, length=72):
        tokens, labels, qs = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            _, raw = _split(tokens)
            content = _content(raw)
            label = 'frame'
            if content:
                if content[0] == 'Where' and len(content) == 8:
                    label = 'question_answer'
                elif content[0] != 'Where':
                    frame = _frame(content)
                    if len(content) == _indices(frame)[1]:
                        label = 'mixed_verb'
                    elif len(content) == OBJECT_INDEX[frame]:
                        label = 'mixed_object'
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            labels.append(label)
            qs.append(q)
            tokens.append(token)
        return [TOKEN2ID_V4[token] for token in tokens], labels, qs


def render(ids):
    text = ' '.join(ID2TOKEN_V4[i] if isinstance(i, int) else i for i in ids)
    for token in (',', '.', '?', ':'):
        text = text.replace(' '+token, token)
    return text.replace(' \n ', '\n')
