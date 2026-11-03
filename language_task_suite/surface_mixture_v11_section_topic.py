"""V11: section topics participate in ordinary narrative place prediction.

A completed visible `Section N: TOPIC.` heading sets the current topic until
the next heading. At a later ordinary narrative place slot, that topic adds
probability mass when it is grammatical in the base place distribution. The
same place remains available through the existing lexical/frame generator,
and it can later be queried by the ordinary `Where` task. There is no hidden
episode schedule or special answer-only task. The exact next-token oracle is
a pure function of the visible prefix.
"""

from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v4 import _content
from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents


SPECIAL = ('Where','How','As','Who','Section','Detail','What')


def active_section_topic(history):
    """Latest completed heading's topic, from visible completed clauses."""
    topic = None
    for raw in history:
        content = _content(raw)
        if content and content[0] == 'Section':
            assert len(content) == 5 and content[2] == ':' and content[4] == '.'
            topic = content[3]
    return topic


class SurfaceMixtureV11SectionTopic(SurfaceMixtureV10LinkedEvents):
    def __init__(self, seed=0, section_topic_mass=.35, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if not 0 <= section_topic_mass < 1:
            raise ValueError('section_topic_mass must be in [0, 1)')
        self.section_topic_mass = section_topic_mass

    def place_rule_components(self, prefix):
        """Named baseline and heading-topic mixture at a narrative place site."""
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if not content or content[0] in SPECIAL:
            raise ValueError('Not a narrative clause')
        frame = _frame(content)
        if len(content) != _indices(frame)[2]:
            raise ValueError('Not at a narrative place slot')
        baseline = super().next_distribution(prefix)
        topic = active_section_topic(history)
        applicable = topic is not None and topic in baseline
        mass = self.section_topic_mass if applicable else 0.
        components = {'base_place':dict(mass=1-mass,distribution=baseline)}
        if applicable:
            components['section_topic']=dict(mass=mass,distribution={topic:1.})
        return components

    def next_distribution(self, prefix):
        history,raw = _split(self._tokens(prefix))
        content = _content(raw)
        if not content or content[0] in SPECIAL:
            return super().next_distribution(prefix)
        frame = _frame(content)
        if len(content) != _indices(frame)[2]:
            return super().next_distribution(prefix)
        parts = self.place_rule_components(prefix)
        base = parts['base_place']['distribution']
        topic = parts.get('section_topic')
        if topic is None:
            return base
        mass = topic['mass']
        out = {token:(1-mass)*prob for token,prob in base.items()}
        target = next(iter(topic['distribution']))
        out[target] += mass
        assert abs(sum(out.values())-1)<1e-10
        return out

    def sample_sequence(self,length=128):
        ids,labels,distributions=super().sample_sequence(length)
        words=[self.id2token[i] for i in ids]
        for t in range(length):
            history,raw=_split(words[:t])
            content=_content(raw)
            if not content or content[0] in SPECIAL:
                continue
            if len(content)!=_indices(_frame(content))[2]:
                continue
            if 'section_topic' in self.place_rule_components(words[:t]):
                labels[t]='section_topic_place'
        return ids,labels,distributions
