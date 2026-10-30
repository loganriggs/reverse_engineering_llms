"""Exact arithmetic negative control for single-coordinate intervention validation."""
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import hashlib
import json


def hidden(x):
    a, b = x[0] * x[1], x[2] * x[3]
    return [a, a, b, b]


def forward(h, k):
    return (h[0] + h[1]) * (h[2] + h[3]) / 4 + k * (h[0] - h[1]) * (h[2] - h[3])


def expanded(h, k):
    return ((F(1, 4) + k) * h[0] * h[2]
            + (F(1, 4) - k) * h[0] * h[3]
            + (F(1, 4) - k) * h[1] * h[2]
            + (F(1, 4) + k) * h[1] * h[3])


def intervene(h, replacements):
    result = h.copy()
    for i, value in replacements.items():
        result[i] = value
    return result


candidates = tuple(map(F, [-3, -2, -1, 0, 1, 2, 3, 4]))
discovery = [list(map(F, x)) for x in [(1, 2, 3, 4), (2, -1, 1, 3), (0, 2, 3, 4)]]
test_inputs = [list(map(F, x)) for x in [(3, 2, -1, 2), (-2, 3, 4, 1), (1, -3, -2, 2)]]
donor = hidden(list(map(F, [2, 2, -3, 1])))
checks = 0
for x in discovery + test_inputs:
    h = hidden(x)
    for k in candidates:
        assert forward(h, k) == x[0] * x[1] * x[2] * x[3]
        for mask in product([0, 1], repeat=4):
            modified = [v * m for v, m in zip(h, mask)]
            assert forward(modified, k) == expanded(modified, k)
            checks += 1
        for i in range(4):
            for replacement in [F(0), F(7, 3), donor[i], -2 * h[i]]:
                modified = intervene(h, {i: replacement})
                assert forward(modified, k) == forward(modified, F(0)) == expanded(modified, k)
                checks += 1

records = []
for truth in candidates:
    # Observations come through the literal target forward; filtering only uses outputs.
    viable = list(candidates)
    for x in discovery:
        h = hidden(x)
        for i in range(4):
            modified = intervene(h, {i: donor[i]})
            observation = forward(modified, truth)
            viable = [k for k in viable if expanded(modified, k) == observation]
    assert viable == list(candidates)
    base = hidden(discovery[0])
    pair_candidates = []
    for pair in combinations(range(4), 2):
        modified = intervene(base, {i: F(0) for i in pair})
        observed = forward(modified, truth)
        survivors = [k for k in viable if expanded(modified, k) == observed]
        pair_candidates.append(dict(pair=list(pair), survivors=len(survivors)))
    assert sum(row['survivors'] == 1 for row in pair_candidates) == 4
    modified = intervene(base, {0: F(0), 2: F(0)})
    observed = forward(modified, truth)
    after = [k for k in viable if expanded(modified, k) == observed]
    assert len(after) == 1  # Truth is not used for candidate selection.
    assert after[0] == truth
    zero = intervene(hidden(discovery[2]), {0: F(0), 2: F(0)})
    assert len({forward(zero, k) for k in candidates}) == 1
    before_sse = after_sse = F(0)
    count = 0
    for x in test_inputs:
        for pair in combinations(range(4), 2):
            modified = intervene(hidden(x), {i: donor[i] for i in pair})
            target = forward(modified, truth)
            predicted_before = sum(expanded(modified, k) for k in viable) / len(viable)
            predicted_after = expanded(modified, after[0])
            before_sse += (predicted_before - target) ** 2
            after_sse += (predicted_after - target) ** 2
            assert predicted_after == target
            count += 1
    records.append(dict(target=str(truth), after_single_probes=len(viable), after_cross_pair=len(after), random_pair_outcomes=pair_candidates, holdouts=count, before_mse=str(before_sse / count), after_mse=str(after_sse / count)))

out = Path('results/joint_intervention_blind_spot')
out.mkdir(exist_ok=False)
names = ['JOINT_INTERVENTION_BLIND_SPOT.md', 'check_joint_intervention_blind_spot.py']
result = dict(scope='Exact noiseless known-family assay; not learned-model interpretation', exact_forward_checks=checks, candidates=len(candidates), random_pair_identification_probability='2/3', records=records, sources={n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names})
(out / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(exact_forward_checks=checks, targets=len(records), heldout_predictions=sum(r['holdouts'] for r in records), all_after_mse_zero=all(r['after_mse']=='0' for r in records), single_probes_identify=False)), flush=True)
