"""Visible-history event linking across claims, prose, and source questions.

V10 retains v9's exact-prefix stream and vocabulary. A recent claim that has
not yet been mentioned in prose raises the probability of a narrative using
its speaker and object. Once linked prose appears, a later source question
more often asks about that object. All boosts are functions of visible,
completed clauses; no latent scheduled template is hidden from the oracle.
"""

from surface_mixture_v2 import PEOPLE, _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v9_source_bridge import SurfaceMixtureV9SourceBridge


def linked_state(history):
    claims = {}
    linked = set()
    answered = set()
    for raw in history:
        clause = _content(raw)
        if not clause:
            continue
        kind = clause[0]
        if kind == 'As':
            obj = clause[6] if clause[4]=='"' else clause[5]
            claims[obj] = clause[1]
        elif kind == 'Who':
            answered.add(clause[3])
        elif kind not in ('Where','How','Section','Detail','What','The'):
            frame = _frame(clause)
            subject = clause[_indices(frame)[0]]
            obj = clause[OBJECT_INDEX[frame]]
            if claims.get(obj) == subject:
                linked.add(obj)
    pending = [(obj,speaker) for obj,speaker in claims.items()
               if obj not in linked]
    queryable = [(obj,claims[obj]) for obj in linked if obj not in answered]
    # Dict insertion order makes the latest claim a deterministic target.
    order = {obj:i for i,obj in enumerate(claims)}
    queryable.sort(key=lambda pair:order[pair[0]])
    return dict(pending=pending[-1] if pending else None,
                queryable=queryable[-1] if queryable else None,
                claims=claims,linked=linked,answered=answered)


class SurfaceMixtureV10LinkedEvents(SurfaceMixtureV9SourceBridge):
    def __init__(self, seed=0, linked_subject_mass=.65,
                 linked_object_mass=.65, linked_question_mass=.30,
                 linked_query_key_mass=.65, **kwargs):
        super().__init__(seed=seed, **kwargs)
        for value in (linked_subject_mass, linked_object_mass,
                      linked_question_mass, linked_query_key_mass):
            assert 0 <= value < 1
        self.linked_subject_mass = linked_subject_mass
        self.linked_object_mass = linked_object_mass
        self.linked_question_mass = linked_question_mass
        self.linked_query_key_mass = linked_query_key_mass

    @staticmethod
    def _boost(out, token, mass, support=None):
        if mass == 0:
            return out
        support = tuple(out) if support is None else support
        total = sum(out.get(w,0.) for w in support)
        if total == 0:
            return out
        out = dict(out)
        for w in support:
            if w in out:
                out[w] *= 1-mass
        out[token] = out.get(token,0.) + total*mass
        assert abs(sum(out.values())-1)<1e-10
        return out

    def _start_distribution(self, history, after_newline):
        out = super()._start_distribution(history, after_newline)
        state = linked_state(history)
        if state['pending']:
            out = self._boost(out,state['pending'][1],
                              self.linked_subject_mass,PEOPLE)
        if state['queryable'] and out.get('Who',0)>0:
            support = tuple(w for w in out if w!='\n')
            out = self._boost(out,'Who',self.linked_question_mass,support)
        return out

    def _source_distribution(self, history, content):
        out = super()._source_distribution(history,content)
        if len(content)!=3:
            return out
        target = linked_state(history)['queryable']
        if target and target[0] in out:
            return self._boost(out,target[0],self.linked_query_key_mass)
        return out

    def next_distribution(self,prefix):
        out = super().next_distribution(prefix)
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if not content or content[0] in ('As','Who','Section','Detail',
                                          'What','Where','How'):
            return out
        frame = _frame(content)
        pending = linked_state(history)['pending']
        if not pending:
            return out
        subject_i = _indices(frame)[0]
        if len(content)==subject_i:
            return self._boost(out,pending[1],self.linked_subject_mass,PEOPLE)
        if (len(content)==OBJECT_INDEX[frame] and
                content[subject_i]==pending[1] and pending[0] in out):
            return self._boost(out,pending[0],self.linked_object_mass)
        return out

    def object_rule_components(self,prefix):
        """Named proposal masses at an ordinary narrative object site."""
        history,raw=_split(self._tokens(prefix))
        content=_content(raw)
        frame=_frame(content)
        assert len(content)==OBJECT_INDEX[frame]
        base=self.object_candidates(prefix)
        total=sum(self.object_weights[name] for name in base)
        source=self.source_object_candidates(prefix)
        source_mass=self.source_object_mass if source else 0.
        pending=linked_state(history)['pending']
        linked=(pending and content[_indices(frame)[0]]==pending[1]
                and pending[0] in source)
        link_mass=self.linked_object_mass if linked else 0.
        parts={name:dict(mass=(1-link_mass)*(1-source_mass)*
                              self.object_weights[name]/total,
                         distribution=dist)
               for name,dist in base.items()}
        if source:
            parts['source_fact']=dict(mass=(1-link_mass)*source_mass,
                                      distribution={obj:1/len(source)
                                                    for obj in source})
        if linked:
            parts['linked_claim']=dict(mass=link_mass,
                                      distribution={pending[0]:1.})
        assert abs(sum(part['mass'] for part in parts.values())-1)<1e-10
        return parts
