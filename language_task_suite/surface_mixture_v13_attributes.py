"""V13: visible item attributes recur in ordinary descriptive prose.

Price and attribute facts share a sentence start. Attribute reuse enters the
ordinary adjective and object distributions, rather than an answer-only task.
All state is reconstructed from the visible prefix. This is an auditable
synthetic dependency, not a claim to reproduce web-text frequencies.
"""

from surface_mixture_v2 import _frame, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v12_price_faq import (
    PRODUCTS, SPECIAL, SurfaceMixtureV12PriceFAQ, price_state,
    VOCAB_V12,
)


ATTRIBUTES = ('clean', 'dusty', 'broken', 'valuable', 'rare', 'heavy')
VOCAB_V13 = tuple(dict.fromkeys(VOCAB_V12 + ('is',) + ATTRIBUTES))
TOKEN2ID_V13 = {token: i for i, token in enumerate(VOCAB_V13)}
ID2TOKEN_V13 = {i: token for token, i in TOKEN2ID_V13.items()}


def attribute_state(history):
    """Latest completed item attributes and completed descriptive mentions."""
    facts, mentioned = {}, set()
    for raw in history:
        content = _content(raw)
        if not content:
            continue
        if content[0] == 'The' and content[2:3] == ['is']:
            assert len(content) == 5 and content[-1] == '.'
            facts[content[1]] = content[3]
        elif content[0] not in SPECIAL:
            frame = _frame(content)
            object_i = OBJECT_INDEX[frame]
            item = content[object_i]
            if item in facts and content[object_i-1] == facts[item]:
                mentioned.add(item)
    return dict(facts=facts, mentioned=mentioned)


class SurfaceMixtureV13Attributes(SurfaceMixtureV12PriceFAQ):
    def __init__(self, seed=0, attribute_fact_rate=.10,
                 attribute_adjective_mass=.65, attribute_object_mass=.65,
                 **kwargs):
        super().__init__(seed=seed, **kwargs)
        for value in (attribute_fact_rate, attribute_adjective_mass,
                      attribute_object_mass):
            assert 0 <= value < 1
        self.attribute_fact_rate = attribute_fact_rate
        self.attribute_adjective_mass = attribute_adjective_mass
        self.attribute_object_mass = attribute_object_mass
        self.vocab, self.token2id, self.id2token = (
            VOCAB_V13, TOKEN2ID_V13, ID2TOKEN_V13)

    def _tokens(self, prefix):
        return [ID2TOKEN_V13[i] if isinstance(i, int) else i for i in prefix]

    def _fact_masses(self, history):
        prices = price_state(history)['facts']
        attributes = attribute_state(history)['facts']
        price_mass, _ = self._extra_masses(history)
        attribute_mass = (self.attribute_fact_rate
                          if len(attributes) < len(PRODUCTS) else 0.)
        # Price and attribute starts are independently offered; combining
        # them requires total mass < 1 before the V12 question family.
        return price_mass, attribute_mass, prices, attributes

    def _start_distribution(self, history, after_newline):
        out = super()._start_distribution(history, after_newline)
        _, attribute_mass, _, _ = self._fact_masses(history)
        if not attribute_mass:
            return out
        content_mass = 1. if after_newline else 1.-self.newline_rate
        out = {w:(p if w == '\n' else p*(1-attribute_mass))
               for w,p in out.items()}
        out['The'] = out.get('The',0.) + content_mass*attribute_mass
        assert abs(sum(out.values())-1) < 1e-10
        return out

    def _fact_distribution(self, history, content):
        pos = len(content)
        price_mass, attribute_mass, prices, attributes = self._fact_masses(history)
        # V13 scales V12's entire content distribution before adding the
        # attribute start. Therefore the V12 price start is scaled too.
        price_mass *= 1-attribute_mass
        available_price = [item for item in PRODUCTS if item not in prices]
        available_attribute = [item for item in PRODUCTS
                               if item not in attributes]
        if pos == 1:
            out = {item:0. for item in PRODUCTS}
            if price_mass:
                for item in available_price:
                    out[item] += price_mass/len(available_price)
            if attribute_mass:
                for item in available_attribute:
                    out[item] += attribute_mass/len(available_attribute)
            total = sum(out.values())
            assert total > 0
            return {item:p/total for item,p in out.items() if p}
        item = content[1]
        if pos == 2:
            p = price_mass/len(available_price) if item in available_price else 0.
            a = (attribute_mass/len(available_attribute)
                 if item in available_attribute else 0.)
            total = p+a
            assert total > 0
            return {word:mass/total for word,mass in
                    (('costs',p),('is',a)) if mass}
        if content[2] == 'costs':
            return super()._fact_distribution(history, content)
        assert content[2] == 'is'
        if pos == 3:
            remaining = [v for v in ATTRIBUTES
                         if v not in attributes.values()]
            assert remaining
            return {v:1/len(remaining) for v in remaining}
        assert pos == 4
        return {'.':1.}

    def _pending_attributes(self, history):
        state = attribute_state(history)
        return [(item, adjective) for item, adjective in state['facts'].items()
                if item not in state['mentioned']]

    def next_distribution(self, prefix):
        words = self._tokens(prefix)
        history, raw = _split(words)
        content = _content(raw)
        if content and content[0] == 'The':
            return self._fact_distribution(history, content)
        base = super().next_distribution(prefix)
        if not content or content[0] in SPECIAL:
            return base
        pending = self._pending_attributes(history)
        if not pending:
            return base
        frame = _frame(content)
        object_i = OBJECT_INDEX[frame]
        if len(content) == object_i-1:
            mass = self.attribute_adjective_mass
            out = {w:(1-mass)*p for w,p in base.items()}
            for _, adjective in pending:
                out[adjective] = out.get(adjective,0.) + mass/len(pending)
            assert abs(sum(out.values())-1)<1e-10
            return out
        if len(content) == object_i:
            matches = [item for item, adjective in pending
                       if content[-1] == adjective]
            if matches:
                assert len(matches) == 1
                return self._boost(base, matches[0],
                                   self.attribute_object_mass)
        return base

    def object_rule_components(self, prefix):
        """Object proposal masses, including the named attribute reuse."""
        parts = super().object_rule_components(prefix)
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        pending = self._pending_attributes(history)
        matches = [item for item, adjective in pending
                   if content[-1] == adjective]
        if not matches:
            return parts
        assert len(matches) == 1
        mass = self.attribute_object_mass
        parts = {name:dict(mass=(1-mass)*part['mass'],
                           distribution=part['distribution'])
                 for name,part in parts.items()}
        parts['attribute_fact'] = dict(mass=mass,
                                       distribution={matches[0]:1.})
        assert abs(sum(part['mass'] for part in parts.values())-1) < 1e-10
        return parts

    def sample_sequence(self, length=128):
        tokens, labels, distributions = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            history, raw = _split(tokens)
            content = _content(raw)
            label = 'frame'
            if content and content[0] == 'The':
                if len(content) == 2:
                    label = 'fact_kind'
                elif len(content) == 3:
                    label = ('attribute_fact_value' if content[2] == 'is'
                             else 'price_fact_value')
            elif content and content[0] not in SPECIAL:
                frame = _frame(content)
                object_i = OBJECT_INDEX[frame]
                pending = self._pending_attributes(history)
                if pending and len(content) == object_i-1:
                    label = 'attribute_adjective'
                elif (pending and len(content) == object_i and
                      any(content[-1] == adjective for _, adjective in pending)):
                    label = 'attribute_object'
            if label == 'frame':
                # Retain V12 task labels for all other sites using the
                # visible prefix; V12's sampler cannot be reused because
                # this version has two kinds of The-fact.
                label = self._base_label(tokens, content)
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            assert q[token] > 0 and abs(sum(q.values())-1) < 1e-9
            tokens.append(token)
            labels.append(label)
            distributions.append(q)
        return [TOKEN2ID_V13[w] for w in tokens], labels, distributions

    def _base_label(self, words, content):
        # Keep labels used by the existing V12 training/evaluation pipeline.
        if not content:
            return 'frame'
        if content[0] == 'How' and content[1:2] == ['much']:
            return ('price_query_key' if len(content) == 4 else
                    'price_answer' if len(content) == 9 else 'frame')
        if content[0] == 'What' and len(content) == 11:
            return 'scope_answer'
        if content[0] == 'Who' and len(content) == 9:
            return 'source_answer'
        if content[0] == 'Where' and len(content) == 8:
            return 'question_answer'
        if content[0] == 'How' and len(content) == 11:
            return 'count_answer'
        if content[0] in SPECIAL:
            return 'frame'
        frame = _frame(content)
        from surface_mixture_v2 import _indices
        if len(content) == _indices(frame)[1]:
            return 'mixed_verb'
        if len(content) == OBJECT_INDEX[frame]:
            return 'mixed_object'
        if content[_indices(frame)[3]:][:1] in (['('], ['[']):
            suffix = content[_indices(frame)[3]:]
            if (')' if suffix[0] == '(' else ']') in self.next_distribution(words):
                return 'bracket_close'
        return 'frame'
