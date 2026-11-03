"""V12: item/price facts recur in ordinary prose and later cost questions.

This extends the V10 shared stream rather than producing an answer-only
dataset. Each token is sampled from a normalized visible-prefix distribution.
The price state contains only completed facts, narrative mentions, and
completed questions; there is no latent episode schedule or hidden answer.
"""

from surface_mixture_v2 import OBJECTS, _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents
from surface_mixture_v9_source_bridge import VOCAB_V9


PRODUCTS = ('tablet', 'laptop', 'radio', 'bicycle', 'package', 'notebook')
PRICES = ('15', '20', '25', '30', '35', '40')
EXTRA = ('The', 'costs', 'cost', 'much', 'does', 'dollars') + PRICES
VOCAB_V12 = VOCAB_V9 + tuple(word for word in EXTRA if word not in VOCAB_V9)
TOKEN2ID_V12 = {word: i for i, word in enumerate(VOCAB_V12)}
ID2TOKEN_V12 = {i: word for word, i in TOKEN2ID_V12.items()}
SPECIAL = ('As', 'Who', 'Section', 'Detail', 'What', 'Where', 'How', 'The')


def price_state(history):
    """Visible completed item facts, prose mentions, and answered items."""
    facts, mentioned, answered = {}, set(), set()
    for raw in history:
        content = _content(raw)
        if not content:
            continue
        if content[0] == 'The':
            if content[2] == 'costs':
                assert (len(content) == 6 and
                        content[4:] == ['dollars', '.'])
                facts[content[1]] = content[3]
            else:
                assert (content[2] == 'is' and content[-1] == '.'
                        and (len(content) == 5 or
                             (len(content) == 10 and content[4:6] ==
                              [',', 'so'] and content[8] == 'it')))
        elif content[0] == 'How' and content[1:2] == ['much']:
            assert len(content) == 12 and content[5:9] == [
                'cost', '?', 'Answer', ':']
            answered.add(content[4])
        elif content[0] not in SPECIAL:
            frame = _frame(content)
            product = content[OBJECT_INDEX[frame]]
            if product in facts:
                mentioned.add(product)
    return dict(facts=facts, mentioned=mentioned, answered=answered)


class SurfaceMixtureV12PriceFAQ(SurfaceMixtureV10LinkedEvents):
    def __init__(self, seed=0, price_fact_rate=.08,
                 price_question_rate=.18, price_object_mass=.55,
                 price_query_mention_mass=.50, **kwargs):
        super().__init__(seed=seed, **kwargs)
        assert 0 <= price_fact_rate < 1
        assert 0 <= price_question_rate < 1
        assert 0 <= price_object_mass < 1
        assert 0 <= price_query_mention_mass < 1
        assert price_fact_rate+price_question_rate < 1
        assert set(PRODUCTS) <= set(self.object_lexicon)
        self.price_fact_rate = price_fact_rate
        self.price_question_rate = price_question_rate
        self.price_object_mass = price_object_mass
        self.price_query_mention_mass = price_query_mention_mass
        self.vocab, self.token2id, self.id2token = (
            VOCAB_V12, TOKEN2ID_V12, ID2TOKEN_V12)

    def _tokens(self, prefix):
        return [ID2TOKEN_V12[i] if isinstance(i, int) else i for i in prefix]

    def _extra_masses(self, history):
        state = price_state(history)
        fact = (self.price_fact_rate if len(state['facts']) < len(PRODUCTS)
                else 0.)
        query = (self.price_question_rate if len(state['facts']) >= 2 and
                 state['mentioned'] - state['answered'] else 0.)
        return fact, query

    def _start_distribution(self, history, after_newline):
        base = super()._start_distribution(history, after_newline)
        fact, query = self._extra_masses(history)
        content_mass = 1. if after_newline else 1.-self.newline_rate
        out = {token: mass if token == '\n' else mass*(1-fact-query)
               for token, mass in base.items()}
        if fact:
            out['The'] = content_mass*fact
        if query:
            out['How'] = out.get('How', 0.) + content_mass*query
        assert abs(sum(out.values())-1) < 1e-10
        return out

    def _fact_distribution(self, history, content):
        state = price_state(history)
        pos = len(content)
        if pos == 1:
            remaining = [p for p in PRODUCTS if p not in state['facts']]
            return {p:1/len(remaining) for p in remaining}
        if pos == 3:
            remaining = [p for p in PRICES if p not in state['facts'].values()]
            return {p:1/len(remaining) for p in remaining}
        fixed = {2:'costs', 4:'dollars', 5:'.'}
        return {fixed[pos]:1.}

    def _cost_distribution(self, history, content):
        pos = len(content)
        state = price_state(history)
        fixed = {2:'does', 3:'the', 5:'cost', 6:'?', 7:'Answer',
                 8:':', 10:'dollars', 11:'.'}
        if pos == 1:
            # The superclass's count query also begins "How". Reconstruct
            # the two start masses from visible history and line position.
            raise ValueError('Use next_distribution for the How fork')
        if pos == 4:
            choices = [p for p in state['facts'] if p not in state['answered']]
            out = {p:1/len(choices) for p in choices}
            mentioned = [p for p in choices if p in state['mentioned']]
            if mentioned:
                mass = self.price_query_mention_mass
                out = {p:(1-mass)*v for p,v in out.items()}
                for p in mentioned:
                    out[p] += mass/len(mentioned)
            return out
        if pos == 9:
            return {state['facts'][content[4]]:1.}
        return {fixed[pos]:1.}

    def next_distribution(self, prefix):
        words = self._tokens(prefix)
        history, raw = _split(words)
        content = _content(raw)
        if content and content[0] == 'The':
            return self._fact_distribution(history, content)
        if content and content[0] == 'How':
            if len(content) == 1:
                after_newline = raw[:1] == ['\n']
                parent = super()._start_distribution(history, after_newline)
                fact, query = self._extra_masses(history)
                old = parent.get('How', 0.)*(1-fact-query)
                new = (1. if after_newline else 1.-self.newline_rate)*query
                total = old+new
                assert total > 0
                out = {}
                if old: out['many'] = old/total
                if new: out['much'] = new/total
                return out
            if content[1] == 'much':
                return self._cost_distribution(history, content)
            return super().next_distribution(prefix)
        base = super().next_distribution(prefix)
        if not content or content[0] in SPECIAL:
            return base
        frame = _frame(content)
        if len(content) != OBJECT_INDEX[frame]:
            return base
        state = price_state(history)
        pending = [p for p in state['facts'] if p not in state['mentioned']]
        if not pending or self.price_object_mass == 0:
            return base
        target = pending[-1]
        if target not in base:
            return base
        return self._boost(base, target, self.price_object_mass)

    def object_rule_components(self, prefix):
        parts = super().object_rule_components(prefix)
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        assert content and content[0] not in SPECIAL
        state = price_state(history)
        pending = [p for p in state['facts'] if p not in state['mentioned']]
        target = pending[-1] if pending else None
        base = super().next_distribution(prefix)
        if target is None or target not in base or not self.price_object_mass:
            return parts
        mass = self.price_object_mass
        parts = {name:dict(mass=(1-mass)*part['mass'],
                           distribution=part['distribution'])
                 for name,part in parts.items()}
        parts['priced_item'] = dict(mass=mass, distribution={target:1.})
        assert abs(sum(part['mass'] for part in parts.values())-1)<1e-10
        return parts

    def sample_sequence(self, length=128):
        tokens, labels, distributions = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            history, raw = _split(tokens)
            content = _content(raw)
            label = 'frame'
            if content:
                if content[0] == 'The' and len(content) == 3:
                    label = 'price_fact_value'
                elif content[0] == 'How' and content[1:2] == ['much']:
                    if len(content) == 4: label = 'price_query_key'
                    elif len(content) == 9: label = 'price_answer'
                elif content[0] == 'What' and len(content) == 11:
                    label = 'scope_answer'
                elif content[0] == 'Who' and len(content) == 9:
                    label = 'source_answer'
                elif content[0] == 'Where' and len(content) == 8:
                    label = 'question_answer'
                elif content[0] == 'How' and len(content) == 11:
                    label = 'count_answer'
                elif content[0] not in SPECIAL:
                    frame = _frame(content)
                    if len(content) == _indices(frame)[1]:
                        label = 'mixed_verb'
                    elif len(content) == OBJECT_INDEX[frame]:
                        label = ('priced_item_object' if 'priced_item' in
                                 self.object_rule_components(tokens)
                                 else 'mixed_object')
                    elif content[_indices(frame)[3]:][:1] in (['('], ['[']):
                        suffix = content[_indices(frame)[3]:]
                        if (')' if suffix[0] == '(' else ']') in q:
                            label = 'bracket_close'
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            assert q[token] > 0 and abs(sum(q.values())-1) < 1e-9
            tokens.append(token)
            labels.append(label)
            distributions.append(q)
        return [TOKEN2ID_V12[token] for token in tokens], labels, distributions
