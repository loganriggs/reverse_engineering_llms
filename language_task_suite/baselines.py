"""Simple lexical controls for the toy language split."""

from collections import Counter, defaultdict

from grammar import sample


def main():
    train = sample(12000, 42)
    counts = defaultdict(Counter)
    for ex in train:
        counts[ex.family][ex.target] += 1
    for variant, seed in ((0, 43), (1, 44)):
        test = sample(2400, seed, variant)
        score = defaultdict(lambda: [0, 0, 0])
        for ex in test:
            majority = counts[ex.family].most_common(1)[0][0]
            seen = [word for word in ex.prefix if word in ex.candidates]
            recent = seen[-1] if seen else majority
            row = score[ex.family]
            row[0] += int(majority == ex.target)
            row[1] += int(recent == ex.target)
            row[2] += 1
        print('variant', variant, 'majority',
              round(sum(v[0] for v in score.values())/len(test), 3),
              'recency', round(sum(v[1] for v in score.values())/len(test), 3))
        for family, (maj, rec, n) in sorted(score.items()):
            print(f'  {family:22s} majority={maj/n:.3f} recent={rec/n:.3f}')


if __name__ == '__main__':
    main()
