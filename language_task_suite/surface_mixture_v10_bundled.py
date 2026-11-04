"""Bracket-mediated continuation of the existing V10 linked-event stream.

V10 already links claims to prose and later source questions. This variant
makes a bracketed *related* prose mention more likely and, after an unrelated
completed clause, gives that object additional question probability. Every
boost is recoverable from the emitted prefix; no latent episode label is used.
"""

from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents


SPECIAL = frozenset(('As', 'Who', 'Where', 'How', 'Section', 'Detail',
                     'What', 'The'))


def bridge_state(history):
    """Recover bracketed linked objects and later query eligibility."""
    facts, bracketed, asked = {}, {}, set()
    for index, raw in enumerate(history):
        content = _content(raw)
        if not content:
            continue
        if content[0] == 'As':
            obj = content[6] if content[4] == '"' else content[5]
            facts[obj] = content[1]
        elif content[0] == 'Who':
            asked.add(content[3])
        elif content[0] not in SPECIAL:
            frame = _frame(content)
            obj = content[OBJECT_INDEX[frame]]
            speaker = content[_indices(frame)[0]]
            suffix = content[_indices(frame)[3]:]
            if (facts.get(obj) == speaker and suffix and
                    suffix[0] in ('(', '[') and
                    (')' if suffix[0] == '(' else ']') in suffix):
                bracketed[obj] = index
    # Require at least one later completed clause before extra query mass.
    ready = {obj for obj, index in bracketed.items()
             if len(history) - 1 > index and obj not in asked}
    return facts, bracketed, ready


class SurfaceMixtureV10Bundled(SurfaceMixtureV10LinkedEvents):
    def __init__(self, seed=0, linked_bracket_rate=.55,
                 bracket_query_mass=.25, bracket_query_key_mass=.6,
                 **kwargs):
        super().__init__(seed=seed, **kwargs)
        for value in (linked_bracket_rate, bracket_query_mass,
                      bracket_query_key_mass):
            assert 0 <= value < 1
        self.linked_bracket_rate = linked_bracket_rate
        self.bracket_query_mass = bracket_query_mass
        self.bracket_query_key_mass = bracket_query_key_mass

    def _start_distribution(self, history, after_newline):
        out = super()._start_distribution(history, after_newline)
        _, _, ready = bridge_state(history)
        if ready and out.get('Who', 0.) > 0:
            support = tuple(word for word in out if word != '\n')
            return self._boost(out, 'Who', self.bracket_query_mass, support)
        return out

    def _source_distribution(self, history, content):
        out = super()._source_distribution(history, content)
        if len(content) != 3:
            return out
        facts, _, ready = bridge_state(history)
        candidates = [obj for obj in ready if obj in out]
        if candidates:
            # Latest claimed object is a deterministic tie breaker.
            chosen = max(candidates, key=lambda obj: list(facts).index(obj))
            return self._boost(out, chosen, self.bracket_query_key_mass)
        return out

    def next_distribution(self, prefix):
        out = super().next_distribution(prefix)
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if not content or content[0] in SPECIAL:
            return out
        frame = _frame(content)
        if len(content) != _indices(frame)[3]:
            return out
        facts, _, _ = bridge_state(history)
        obj = content[OBJECT_INDEX[frame]]
        speaker = content[_indices(frame)[0]]
        if facts.get(obj) != speaker:
            return out
        old = out.get('(', 0.) + out.get('[', 0.)
        target = max(old, self.linked_bracket_rate)
        if target == old:
            return out
        updated = {token: p*(1.-target)/(1.-old)
                   for token, p in out.items() if token not in ('(', '[')}
        updated['('], updated['['] = target/2., target/2.
        assert abs(sum(updated.values()) - 1.) < 1e-10
        return updated
