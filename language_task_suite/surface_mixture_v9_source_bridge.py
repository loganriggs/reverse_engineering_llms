"""Pilot: source facts join object competition; brackets join the same stream.

This is deliberately a partial DGP version. It retains v8's visible-prefix
oracle and sampled source questions, but lets the same source facts affect
ordinary narrative object choices. Optional brackets after ordinary narrative
phrases have a variable-length body and opener-specific closer. Section-to-
narrative coupling is still absent.
"""

from surface_mixture_v2 import (ADJECTIVES, OBJECTS, PEOPLE, _frame,
                                _indices, _split)
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v6 import source_facts
from surface_mixture_v8 import SurfaceMixtureV8
from surface_mixture_v7 import VOCAB_V7


VOCAB_V9 = VOCAB_V7 + tuple(token for token in ('(',')','[',']')
                            if token not in VOCAB_V7)
TOKEN2ID_V9 = {token:i for i,token in enumerate(VOCAB_V9)}
ID2TOKEN_V9 = {i:token for token,i in TOKEN2ID_V9.items()}


class SurfaceMixtureV9SourceBridge(SurfaceMixtureV8):
    def __init__(self, seed=0, source_object_mass=.25,
                 source_subject_mass=.25, bracket_rate=.12, **kwargs):
        super().__init__(seed=seed, **kwargs)
        assert 0 <= source_object_mass < 1
        assert 0 <= source_subject_mass < 1
        assert 0 <= bracket_rate < 1
        self.source_object_mass = source_object_mass
        self.source_subject_mass = source_subject_mass
        self.bracket_rate = bracket_rate
        self.vocab, self.token2id, self.id2token = (VOCAB_V9,TOKEN2ID_V9,
                                                     ID2TOKEN_V9)

    def _tokens(self, prefix):
        return [ID2TOKEN_V9[i] if isinstance(i,int) else i for i in prefix]

    @staticmethod
    def _bracket_distribution(suffix):
        opener = suffix[0]
        assert opener in ('(','[')
        closer = ')' if opener == '(' else ']'
        if suffix[-1] == closer:
            return {'.':1.}
        # 1–3 adjective/object pairs. The body distribution is deliberately
        # the same for both openers; the opener selects the closer.
        body = suffix[1:]
        if len(body)%2 == 0:
            if len(body) == 6:
                return {closer:1.}
            if len(body)>=2:
                out = {token:.7/len(ADJECTIVES) for token in ADJECTIVES}
                out[closer] = .3
                return out
            return {token:1/len(ADJECTIVES) for token in ADJECTIVES}
        return {token:1/len(OBJECTS) for token in OBJECTS}

    def _subject_distribution(self, history):
        base = super()._subject_distribution(history)
        speakers = sorted(set(source_facts(history).values()))
        if not speakers or self.source_subject_mass == 0:
            return base
        mass = self.source_subject_mass
        out = {name: (1-mass)*p for name,p in base.items()}
        for speaker in speakers:
            out[speaker] = out.get(speaker,0.) + mass/len(speakers)
        return out

    def _start_distribution(self, history, after_newline):
        out = super()._start_distribution(history, after_newline)
        speakers = sorted(set(source_facts(history).values()))
        if not speakers or self.source_subject_mass == 0:
            return out
        subject_mass = sum(out.get(name,0.) for name in PEOPLE)
        if subject_mass:
            factor = 1-self.source_subject_mass
            for name in PEOPLE:
                if name in out:
                    out[name] *= factor
            for speaker in speakers:
                out[speaker] = out.get(speaker,0.) + (subject_mass*
                     self.source_subject_mass/len(speakers))
        return out

    def source_object_candidates(self, prefix):
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if not content or content[0] in ('As','Who','Section','Detail',
                                          'What','Where','How'):
            return {}
        frame = _frame(content)
        if len(content) != OBJECT_INDEX[frame]:
            return {}
        speaker = content[_indices(frame)[0]]
        return {obj: owner for obj,owner in source_facts(history).items()
                if owner == speaker and obj in self.object_lexicon}

    def next_distribution(self, prefix):
        words = self._tokens(prefix)
        history, raw = _split(words)
        content = _content(raw)
        if content and content[0] not in ('As','Who','Section','Detail',
                                          'What','Where','How'):
            frame = _frame(content)
            suffix = content[_indices(frame)[3]:]
            if suffix and suffix[0] in ('(','['):
                return self._bracket_distribution(suffix)
        base = super().next_distribution(prefix)
        candidates = self.source_object_candidates(prefix)
        if candidates and self.source_object_mass:
            mass = self.source_object_mass
            base = {token: (1-mass)*p for token,p in base.items()}
            for obj in candidates:
                base[obj] = base.get(obj,0.) + mass/len(candidates)
        if (content and content[0] not in ('As','Who','Section','Detail',
                                           'What','Where','How') and
                len(content) == _indices(_frame(content))[3] and
                self.bracket_rate):
            base = {token:(1-self.bracket_rate)*p for token,p in base.items()}
            base['('] = self.bracket_rate/2
            base['['] = self.bracket_rate/2
        assert abs(sum(base.values())-1) < 1e-10
        return base

    def sample_sequence(self,length=128):
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
                elif content[0] not in ('Section','Detail','What','Who',
                                        'Where','How','As'):
                    frame = _frame(content)
                    if len(content) == _indices(frame)[1]:
                        label = 'mixed_verb'
                    elif len(content) == OBJECT_INDEX[frame]:
                        label = 'mixed_object'
                    elif content[_indices(frame)[3]:][:1] in (['('],['[']):
                        suffix = content[_indices(frame)[3]:]
                        if (')' if suffix[0]=='(' else ']') in q:
                            label = 'bracket_close'
            token = self.rng.choices(list(q),weights=list(q.values()))[0]
            labels.append(label)
            qs.append(q)
            tokens.append(token)
        return [TOKEN2ID_V9[token] for token in tokens], labels, qs
