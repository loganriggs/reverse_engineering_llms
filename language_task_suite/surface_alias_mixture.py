"""Parenthetical aliases composed with the six-frame lexical/induction DGP.

A clause such as ``Alice (also called Ava).`` introduces an alias whose
referent is visible in the stream. Later Ava-subject clauses use Alice's
lexical distribution and Alice's prior-subject induction history. A matched
``Alice (a local colleague).`` aside adds parentheses without aliasing.

The oracle is a pure function of the visible prefix; no hidden state is
carried across calls. This is a diagnostic joint language, not a claim to
match FineWeb or natural alias frequency.
"""

from collections import defaultdict

from surface_mixture_v2 import (PEOPLE, TOKEN2ID as BASE_TOKEN2ID,
                                VOCAB as BASE_VOCAB, SurfaceMixtureV2,
                                _frame, _indices, _split)


ALIASES = ('Ava', 'Blair', 'Cleo', 'Dina', 'Eli', 'Faye', 'Gus', 'Hope',
           'Ivo', 'Jade', 'Kian', 'Luna', 'Moe', 'Nia', 'Omar', 'Pia')
EXTRA = ('(', ')', 'also', 'called', 'known', 'as', 'a', 'local', 'colleague')
VOCAB = BASE_VOCAB + tuple(x for x in ALIASES+EXTRA if x not in BASE_TOKEN2ID)
assert len(VOCAB) == len(set(VOCAB))
TOKEN2ID = {token: i for i, token in enumerate(VOCAB)}
ID2TOKEN = {i: token for token, i in TOKEN2ID.items()}
ALIAS_STYLES = (('also', 'called'), ('known', 'as'))


def _is_parenthetical(clause):
    return len(clause) >= 2 and clause[0] in PEOPLE and clause[1] == '('


def _alias_definition(clause):
    if (len(clause) == 7 and clause[0] in PEOPLE and clause[1] == '('
            and tuple(clause[2:4]) in ALIAS_STYLES
            and clause[4] in ALIASES and clause[5:] == [')', '.']):
        return clause[4], clause[0]
    return None


def alias_state(history):
    """Alias -> canonical person from completed visible clauses."""
    mapping = {}
    used_people = set()
    for clause in history:
        definition = _alias_definition(clause)
        if definition is not None:
            alias, person = definition
            assert alias not in mapping and person not in used_people
            mapping[alias] = person
            used_people.add(person)
    return mapping


class AliasMixtureLanguage(SurfaceMixtureV2):
    def __init__(self, seed=0, weights=None, parenthetical_rate=.55,
                 alias_fraction=.65, alias_use=.65,
                 definition_start_rate=.20, alias_reuse_rate=.25):
        super().__init__(seed=seed, weights=weights)
        assert 0 <= parenthetical_rate < 1
        assert 0 <= alias_fraction <= 1 and 0 <= alias_use <= 1
        assert 0 <= definition_start_rate < 1
        assert 0 <= alias_reuse_rate < 1
        self.parenthetical_rate = parenthetical_rate
        self.alias_fraction = alias_fraction
        self.alias_use = alias_use
        self.definition_start_rate = definition_start_rate
        self.alias_reuse_rate = alias_reuse_rate
        self.vocab, self.token2id, self.id2token = VOCAB, TOKEN2ID, ID2TOKEN

    def _tokens(self, prefix):
        return [ID2TOKEN[i] if isinstance(i, int) else i for i in prefix]

    def _canonical_prefix(self, history, clause):
        mapping = alias_state(history)
        narrative = [old for old in history if not _is_parenthetical(old)]
        flattened = [mapping.get(tok, tok) for old in narrative
                     for tok in old]
        flattened.extend(mapping.get(tok, tok) for tok in clause)
        return flattened

    def _alias_start_distribution(self, history, clause):
        """Distribution after the first person token of a potential aside."""
        mapping = alias_state(history)
        person = clause[0]
        eligible = (person not in mapping.values()
                    and len(mapping) < len(ALIASES))
        if not eligible:
            return {',': 1.0}
        return {',': 1-self.parenthetical_rate,
                '(': self.parenthetical_rate}

    def _parenthetical_distribution(self, history, clause):
        person = clause[0]
        assert person in PEOPLE and clause[1] == '('
        pos = len(clause)
        mapping = alias_state(history)
        if pos == 2:
            return {'also': self.alias_fraction/2,
                    'known': self.alias_fraction/2,
                    'a': 1-self.alias_fraction}
        if pos == 3:
            return {dict(also='called', known='as', a='local')[clause[2]]: 1.}
        if clause[2] == 'a':
            fixed = {4: 'colleague', 5: ')', 6: '.'}
            return {fixed[pos]: 1.}
        if pos == 4:
            available = [name for name in ALIASES if name not in mapping]
            return {name: 1/len(available) for name in available}
        fixed = {5: ')', 6: '.'}
        return {fixed[pos]: 1.}

    def _with_aliases(self, base, mapping):
        reverse = {person: alias for alias, person in mapping.items()}
        out = defaultdict(float)
        person_mass = sum(p for token, p in base.items() if token in PEOPLE)
        reuse = self.alias_reuse_rate if mapping else 0.
        for token, mass in base.items():
            if token in reverse:
                out[token] += mass*(1-reuse)*(1-self.alias_use)
                out[reverse[token]] += mass*(1-reuse)*self.alias_use
            elif token in PEOPLE:
                out[token] += mass*(1-reuse)
            else:
                out[token] += mass
        if mapping:
            for alias in mapping:
                out[alias] += reuse*person_mass/len(mapping)
        return {token: mass for token, mass in out.items() if mass > 0}

    def next_distribution(self, prefix):
        history, clause = _split(self._tokens(prefix))
        if _is_parenthetical(clause):
            return self._parenthetical_distribution(history, clause)
        if len(clause) == 1 and clause[0] in PEOPLE:
            return self._alias_start_distribution(history, clause)
        canonical = self._canonical_prefix(history, clause)
        base = super().next_distribution(canonical)
        if not clause:
            mapping = alias_state(history)
            available = [person for person in PEOPLE
                         if person not in mapping.values()]
            if available:
                base = {token: p*(1-self.definition_start_rate)
                        for token, p in base.items()}
                for person in available:
                    base[person] = (base.get(person, 0)
                                    + self.definition_start_rate/len(available))
        return self._with_aliases(base, alias_state(history))

    def candidates(self, prefix):
        history, clause = _split(self._tokens(prefix))
        if _is_parenthetical(clause):
            raise ValueError('No verb candidates in a parenthetical aside')
        return super().candidates(self._canonical_prefix(history, clause))

    def sample_sequence(self, length=128):
        tokens, labels, distributions = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            history, clause = _split(tokens)
            label = 'frame'
            if _is_parenthetical(clause):
                label = ('alias_name' if len(clause) == 4
                         and clause[2] in ('also', 'known') else 'parenthetical')
            else:
                canonical = self._canonical_prefix(history, clause)
                _, current = _split(canonical)
                if current:
                    frame = _frame(current)
                    if len(current) == _indices(frame)[1]:
                        label = 'mixed_verb'
            word = self.rng.choices(list(q), weights=list(q.values()))[0]
            tokens.append(word)
            labels.append(label)
            distributions.append(q)
        return [TOKEN2ID[word] for word in tokens], labels, distributions

    def sample_batch(self, n, length=128):
        rows = [self.sample_sequence(length) for _ in range(n)]
        return ([r[0] for r in rows], [r[1] for r in rows],
                [r[2] for r in rows])


def render(ids):
    text = ' '.join(ID2TOKEN[i] if isinstance(i, int) else i for i in ids)
    for mark in (',', '.', ')'):
        text = text.replace(' '+mark, mark)
    return text.replace('( ', '(')
