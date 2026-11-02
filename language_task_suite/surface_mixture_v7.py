"""Joint mixed-rule stream with section-scoped facts and exact prefix oracle.

A heading opens a section; its detail may follow after unrelated narrative
clauses. Reusing the same badge key with distinct colors across sections makes
the section number necessary for answering the later question.
"""

from surface_mixture_v2 import PLACES, _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v6 import SurfaceMixtureV6, VOCAB_V6


SECTION_IDS = ('1', '2', '3', '4')
COLORS = ('red', 'blue', 'green', 'yellow', 'purple', 'orange')
TOPICS = PLACES[:8]
EXTRA = ('Section', 'Detail', 'What', 'color', 'badge', 'section', 'red',
         'blue', 'green', 'yellow', 'purple', 'orange')
VOCAB_V7 = VOCAB_V6 + tuple(word for word in EXTRA if word not in VOCAB_V6)
TOKEN2ID_V7 = {word: i for i, word in enumerate(VOCAB_V7)}
ID2TOKEN_V7 = {i: word for word, i in TOKEN2ID_V7.items()}


def section_state(history):
    """Return completed section facts and any heading still awaiting detail."""
    facts = {}
    pending = None
    used = set()
    for raw in history:
        clause = _content(raw)
        if not clause:
            continue
        if clause[0] == 'Section':
            assert len(clause) == 5 and clause[2] == ':' and clause[4] == '.'
            assert pending is None and clause[1] not in used
            used.add(clause[1])
            pending = clause[1]
        elif clause[0] == 'Detail':
            assert len(clause) == 7 and clause[1:5] == [':', 'the', 'badge', 'was']
            assert clause[6] == '.' and pending is not None
            assert clause[5] in COLORS and clause[5] not in facts.values()
            facts[pending] = clause[5]
            pending = None
    return facts, pending, used


class SurfaceMixtureV7(SurfaceMixtureV6):
    def __init__(self, seed=0, section_rate=.10, detail_rate=.65,
                 scope_question_rate=.20, **kwargs):
        super().__init__(seed=seed, **kwargs)
        assert 0 <= section_rate < 1
        assert 0 < detail_rate <= 1
        assert 0 <= scope_question_rate < 1
        assert section_rate + scope_question_rate < 1
        self.section_rate = section_rate
        self.detail_rate = detail_rate
        self.scope_question_rate = scope_question_rate
        self.vocab = VOCAB_V7
        self.token2id = TOKEN2ID_V7
        self.id2token = ID2TOKEN_V7

    def _tokens(self, prefix):
        return [ID2TOKEN_V7[i] if isinstance(i, int) else i for i in prefix]

    def _start_distribution(self, history, after_newline):
        out = super()._start_distribution(history, after_newline)
        facts, pending, used = section_state(history)
        section_mass = self.section_rate if pending is None and len(used) < len(SECTION_IDS) else 0.
        detail_mass = self.detail_rate if pending is not None else 0.
        query_mass = (self.scope_question_rate if pending is None and len(facts) >= 2
                      else 0.)
        assert section_mass + detail_mass + query_mass <= 1
        content_mass = 1. if after_newline else 1.-self.newline_rate
        out = {token: (mass if token == '\n' else
                       mass*(1.-section_mass-detail_mass-query_mass))
               for token, mass in out.items()}
        if section_mass:
            out['Section'] = content_mass*section_mass
        if detail_mass:
            out['Detail'] = content_mass*detail_mass
        if query_mass:
            out['What'] = content_mass*query_mass
        return out

    def _heading_distribution(self, history, content):
        pos = len(content)
        if pos == 1:
            _, pending, used = section_state(history)
            assert pending is None
            choices = [key for key in SECTION_IDS if key not in used]
            return {key: 1/len(choices) for key in choices}
        if pos == 2:
            return {':': 1.}
        if pos == 3:
            return {topic: 1/len(TOPICS) for topic in TOPICS}
        if pos == 4:
            return {'.': 1.}
        raise ValueError(f'Bad heading position {pos}')

    def _detail_distribution(self, history, content):
        pos = len(content)
        fixed = {1: ':', 2: 'the', 3: 'badge', 4: 'was', 6: '.'}
        if pos == 5:
            facts, pending, _ = section_state(history)
            assert pending is not None
            available = [color for color in COLORS if color not in facts.values()]
            return {color: 1/len(available) for color in available}
        if pos in fixed:
            return {fixed[pos]: 1.}
        raise ValueError(f'Bad detail position {pos}')

    def _scope_distribution(self, history, content):
        pos = len(content)
        fixed = {1: 'color', 2: 'was', 3: 'the', 4: 'badge', 5: 'in',
                 6: 'section', 8: '?', 9: 'Answer', 10: ':', 12: '.'}
        facts, pending, _ = section_state(history)
        assert pending is None and len(facts) >= 2
        if pos == 7:
            return {key: 1/len(facts) for key in facts}
        if pos == 11:
            return {facts[content[7]]: 1.}
        if pos in fixed:
            return {fixed[pos]: 1.}
        raise ValueError(f'Bad scope-question position {pos}')

    def next_distribution(self, prefix):
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if content:
            if content[0] == 'Section':
                return self._heading_distribution(history, content)
            if content[0] == 'Detail':
                return self._detail_distribution(history, content)
            if content[0] == 'What':
                return self._scope_distribution(history, content)
        return super().next_distribution(prefix)

    def scope_answer(self, prefix):
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if not content or content[0] != 'What' or len(content) != 11:
            raise ValueError('Not at a scope answer slot')
        return section_state(history)[0][content[7]]

    def sample_sequence(self, length=128):
        tokens, labels, qs = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            _, raw = _split(tokens)
            content = _content(raw)
            label = 'frame'
            if content:
                if content[0] == 'What' and len(content) == 11:
                    label = 'scope_answer'
                elif content[0] == 'Who' and len(content) == 9:
                    label = 'source_answer'
                elif content[0] == 'Where' and len(content) == 8:
                    label = 'question_answer'
                elif content[0] == 'How' and len(content) == 11:
                    label = 'count_answer'
                elif content[0] not in ('Section', 'Detail', 'What',
                                        'Who', 'Where', 'How', 'As'):
                    frame = _frame(content)
                    if len(content) == _indices(frame)[1]:
                        label = 'mixed_verb'
                    elif len(content) == OBJECT_INDEX[frame]:
                        label = 'mixed_object'
            word = self.rng.choices(list(q), weights=list(q.values()))[0]
            labels.append(label)
            qs.append(q)
            tokens.append(word)
        return [TOKEN2ID_V7[word] for word in tokens], labels, qs


def render(ids):
    from surface_mixture_v6 import render as v6_render
    return v6_render([ID2TOKEN_V7[i] if isinstance(i, int) else i for i in ids])
