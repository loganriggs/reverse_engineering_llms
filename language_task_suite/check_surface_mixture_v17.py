"""Check sampled-prefix oracle, rule opportunities, and frequency control."""

from collections import Counter

from surface_mixture_v17_typed_lexicon import SurfaceMixtureV17


def check(exponent):
    generator = SurfaceMixtureV17(
        seed=20261103, frequency_exponent=exponent,
        subject_return_mass=.1)
    opportunities = Counter()
    for _ in range(200):
        ids, labels, distributions = generator.sample_sequence(100)
        tokens = [generator.id2token[index] for index in ids]
        for index, (token, label, distribution) in enumerate(
                zip(tokens, labels, distributions, strict=True)):
            assert abs(sum(distribution.values()) - 1) < 1e-9
            assert distribution[token] > 0
            assert generator.next_distribution(tokens[:index]) == distribution
            if label == 'mixed_verb':
                opportunities['mixed_verb'] += 1
                candidates = generator.candidates(tokens[:index])
                opportunities.update(candidates.keys())
                if len({next(iter(row)) for row in candidates.values()}) > 1:
                    opportunities['distinct_proposals'] += 1
    role = generator._dist('objects')
    opportunities['object_top10_mass'] = round(
        sum(sorted(role.values(), reverse=True)[:10]), 6)
    return dict(opportunities)


if __name__ == '__main__':
    for exponent in (0, .5, 1):
        print(exponent, check(exponent))
