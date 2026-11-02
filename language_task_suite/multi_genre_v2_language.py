"""Versioned multi-genre stream with structured field-value retrieval.

Catalog records deliberately retain an exact visible-prefix oracle. Their
surface is inspired by structured records in a small inspected FineWeb panel;
the generator does not claim to reproduce natural web-text frequencies.
"""

import random

from multi_genre_language import (CONDITIONS, GENRES, MultiGenreLanguage,
                                  VOCAB, render as render_v1)
from surface_mixture_language import OBJECTS, PEOPLE


GENRES_V2 = GENRES + ('Catalog',)
GENRE_WEIGHTS_V2 = (.32, .16, .16, .16, .20)
CATALOG_OBJECTS = OBJECTS[:12]
CATALOG_PEOPLE = PEOPLE[:12]
CATALOG_STATUSES = CONDITIONS
CATALOG_FIELDS = ('owner', 'status')
EXTRA = ('Catalog', 'owner', 'status', 'of', 'Who', 'owns', 'What',
         'For', 'who', 'what', 'belongs', 'to', 'and', 'answer', '\n')
VOCAB_V2 = tuple(dict.fromkeys(VOCAB + EXTRA))
TOKEN2ID_V2 = {word: i for i, word in enumerate(VOCAB_V2)}
ID2TOKEN_V2 = {i: word for word, i in TOKEN2ID_V2.items()}


def catalog_state(body):
    """Parse completed visible Catalog clauses; ignore the open final clause."""
    clauses, clause = [], []
    for word in body:
        if word == '\n':
            continue
        clause.append(word)
        if word == '.':
            clauses.append(clause)
            clause = []
    records, asked = {}, set()
    for row in clauses:
        if row[0] == 'The':
            assert len(row) == 9 and row[1] in CATALOG_OBJECTS and row[-1] == '.'
            if row[2] == 'belongs':
                assert row[3] == 'to' and row[5:7] == ['and', 'is']
                values = {'owner': row[4], 'status': row[7]}
            else:
                assert row[2] == 'is' and row[4:7] == ['and', 'belongs', 'to']
                values = {'status': row[3], 'owner': row[7]}
            assert values['owner'] in CATALOG_PEOPLE
            assert values['status'] in CATALOG_STATUSES
            assert row[1] not in records
            records[row[1]] = values
        else:
            if row[0] == 'Who':
                assert len(row) == 10 and row[1:3] == ['owns', 'the']
                field, obj, answer = 'owner', row[3], row[8]
            elif row[0] == 'What':
                assert len(row) == 13 and row[1:6] == ['is', 'the', 'status', 'of', 'the']
                field, obj, answer = 'status', row[6], row[11]
            else:
                assert row[0] == 'For' and len(row) == 14
                field = 'owner' if row[4] == 'who' else 'status'
                obj, answer = row[2], row[12]
            assert obj in records and field in CATALOG_FIELDS
            assert answer == records[obj][field]
            asked.add((field, obj))
    return records, asked, clause


class MultiGenreV2Language:
    def __init__(self, seed=0, genre_weights=GENRE_WEIGHTS_V2,
                 record_rate=.60):
        self.rng = random.Random(seed)
        self.base = MultiGenreLanguage(seed=seed+31)
        self.genre_weights = tuple(genre_weights)
        assert len(self.genre_weights) == len(GENRES_V2)
        assert min(self.genre_weights) >= 0 and sum(self.genre_weights) > 0
        assert 0 <= record_rate <= 1
        self.record_rate = record_rate
        self.vocab = VOCAB_V2
        self.token2id = TOKEN2ID_V2
        self.id2token = ID2TOKEN_V2

    @staticmethod
    def _choices(records, asked, field):
        fresh = [obj for obj in records if (field, obj) not in asked]
        return fresh or (list(records) if len(records) >= 8 else [])

    def _catalog_start(self, records, asked):
        available = {field: self._choices(records, asked, field)
                     for field in CATALOG_FIELDS}
        can_record = len(records) < 8
        can_query = len(records) >= 2 and any(available.values())
        if not can_query:
            return {'The': 1.}, 'catalog_frame'
        if not can_record:
            record_mass = 0.
        else:
            record_mass = self.record_rate
        query_mass = 1.-record_mass
        forms = []
        if available['owner']:
            forms.append('Who')
        if available['status']:
            forms.append('What')
        forms.append('For')
        out = {word: query_mass/len(forms) for word in forms}
        if record_mass:
            out['The'] = record_mass
        return out, 'catalog_frame'

    def _catalog_record(self, records, clause):
        pos = len(clause)
        if pos == 1:
            choices = [obj for obj in CATALOG_OBJECTS if obj not in records]
            return {obj: 1/len(choices) for obj in choices}, 'catalog_record_object'
        if pos == 2:
            return {'belongs': .5, 'is': .5}, 'catalog_frame'
        first_owner = clause[2] == 'belongs'
        value_fields = ({4: 'owner', 7: 'status'} if first_owner else
                        {3: 'status', 7: 'owner'})
        if pos in value_fields:
            field = value_fields[pos]
            vocabulary = CATALOG_PEOPLE if field == 'owner' else CATALOG_STATUSES
            used = {record[field] for record in records.values()}
            choices = [value for value in vocabulary if value not in used]
            return {value: 1/len(choices) for value in choices}, f'catalog_{field}_value'
        fixed = ({3: 'to', 5: 'and', 6: 'is', 8: '.'} if first_owner else
                 {4: 'and', 5: 'belongs', 6: 'to', 8: '.'})
        if pos in fixed:
            return {fixed[pos]: 1.}, 'catalog_frame'
        raise ValueError(f'Bad Catalog record position {pos}')

    def _catalog_query(self, records, asked, clause):
        pos, form = len(clause), clause[0]
        if form == 'Who':
            field, object_at, answer_at = 'owner', 3, 8
            fixed = {1: 'owns', 2: 'the', 4: '?', 5: 'The', 6: 'answer',
                     7: 'is', 9: '.'}
        elif form == 'What':
            field, object_at, answer_at = 'status', 6, 11
            fixed = {1: 'is', 2: 'the', 3: 'status', 4: 'of', 5: 'the',
                     7: '?', 8: 'The', 9: 'answer', 10: 'is', 12: '.'}
        else:
            object_at, answer_at = 2, 12
            fixed = {1: 'the', 3: ',', 5: 'is', 6: 'the', 8: '?',
                     9: 'The', 10: 'answer', 11: 'is', 13: '.'}
            if pos == 4:
                options = []
                if clause[2] in self._choices(records, asked, 'owner'):
                    options.append('who')
                if clause[2] in self._choices(records, asked, 'status'):
                    options.append('what')
                return {word: 1/len(options) for word in options}, 'catalog_query_field'
            field = 'owner' if pos > 4 and clause[4] == 'who' else 'status'
            if pos == 7:
                return {field: 1.}, 'catalog_frame'
        if pos == object_at:
            choices = (self._choices(records, asked, field) if form != 'For'
                       else [obj for obj in records if any(
                           obj in self._choices(records, asked, f)
                           for f in CATALOG_FIELDS)])
            return {obj: 1/len(choices) for obj in choices}, 'catalog_query_object'
        if pos == answer_at:
            return {records[clause[object_at]][field]: 1.}, f'catalog_{field}_answer'
        if pos in fixed:
            return {fixed[pos]: 1.}, 'catalog_frame'
        raise ValueError(f'Bad Catalog query position {pos}')

    def next_distribution(self, prefix):
        words = [ID2TOKEN_V2[t] if isinstance(t, int) else t for t in prefix]
        if not words:
            total = sum(self.genre_weights)
            return ({genre: weight/total for genre, weight in
                     zip(GENRES_V2, self.genre_weights) if weight}, 'genre')
        if words[0] != 'Catalog':
            return self.base.next_distribution(words)
        if len(words) == 1:
            return {':': 1.}, 'header'
        assert words[1] == ':'
        body = words[2:]
        if not body or body[-1] == '.':
            return {'\n': 1.}, 'catalog_frame'
        records, asked, clause = catalog_state(body)
        if clause == []:
            return self._catalog_start(records, asked)
        if clause[0] == 'The':
            return self._catalog_record(records, clause)
        return self._catalog_query(records, asked, clause)

    def sample_sequence(self, length=72):
        words, labels, distributions = [], [], []
        for _ in range(length):
            q, label = self.next_distribution(words)
            word = self.rng.choices(list(q), weights=list(q.values()))[0]
            words.append(word)
            labels.append(label)
            distributions.append(q)
        return [TOKEN2ID_V2[word] for word in words], labels, distributions

    def sample_batch(self, n, length=72):
        rows = [self.sample_sequence(length) for _ in range(n)]
        return ([r[0] for r in rows], [r[1] for r in rows],
                [r[2] for r in rows])


def render(ids):
    words = [ID2TOKEN_V2[i] if isinstance(i, int) else i for i in ids]
    lines = []
    for line in ' '.join(words).split(' \n '):
        lines.append(render_v1(line.split()))
    return '\n'.join(lines)
