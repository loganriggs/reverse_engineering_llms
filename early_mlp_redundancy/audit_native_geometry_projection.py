"""Independently check M32 post-result projections directly from native vectors."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

P = Path(__file__).resolve().parent / 'validation'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10), (a, b)
def centered_response(intact, changed, weight):
    # Centered raw logits equal centered log-probability responses exactly.
    delta = changed.numpy().astype(np.float64) - intact.numpy().astype(np.float64)
    return delta - np.dot(weight, delta)

rp = P / 'native_response_geometry.json'
cp = P / 'native_geometry_concentration.json'
r = json.loads(rp.read_text())
c = json.loads(cp.read_text())
assert c['result_sha256'] == sha(rp)
assert json.loads((P / 'native_response_geometry_audit.json').read_text())['result_sha256'] == sha(rp)
vp = Path(r['vectors_path'])
assert sha(vp) == r['vectors_sha256']
v = torch.load(vp, weights_only=True)
lookup = {(x['target_index'], x['context']): x for x in v['vectors']}
assert len(c['rows']) == 480
seen = set()
max_error = 0.
for row in c['rows']:
    key = row['target_index'], row['context']
    assert key not in seen
    seen.add(key)
    ref = lookup[row['target_index'], 'reference']
    ctx = lookup[key]
    a = v['nativecache'][ref['native_key']]
    b = v['nativecache'][ctx['native_key']]
    logits = a.numpy().astype(np.float64)
    weight = np.exp(logits - logits.max())
    weight /= weight.sum()
    u = centered_response(a, ref['changed'], weight)
    z = centered_response(b, ctx['changed'], weight)
    energy = np.dot(weight, u * u)
    if energy:
        gain = np.dot(weight, u * z) / energy
        parallel = (gain - 1) * u
        residual = z - gain * u
        close(gain, row['parallel_gain'])
    else:
        assert np.all(z == 0) and row['parallel_gain'] is None
        parallel = residual = np.zeros_like(u)
    pe = float(np.dot(weight, parallel * parallel))
    oe = float(np.dot(weight, residual * residual))
    close(pe, row['parallel_change_energy'])
    close(oe, row['orthogonal_change_energy'])
    close(float(np.dot(weight, parallel * residual)), 0.)
    close(pe + oe, row['centered_difference'])
    max_error = max(max_error, abs(pe-row['parallel_change_energy']), abs(oe-row['orthogonal_change_energy']))
assert seen == {(x['target_index'], x['context']) for x in r['comparisons']}
for s in c['summaries']:
    rows = [x for x in c['rows'] if all(x[k] == s[k] for k in ['kind', 'seed', 'context'])]
    total = sum(x['centered_difference'] for x in rows)
    for field, energy in [('orthogonal_fraction_of_centered_change', 'orthogonal_change_energy'), ('parallel_fraction_of_centered_change', 'parallel_change_energy')]:
        close(sum(x[energy] for x in rows)/total, s[field])
    families = {f: sum(x['centered_difference'] for x in rows if x['family'] == f) for f in {x['family'] for x in rows}}
    for f, energy in families.items(): close(energy, s['family_centered_change'][f])
    assert max(families, key=families.get) == s['largest_centered_change_family']
    close(max(families.values())/total, s['largest_family_share'])
    assert sum(x['parallel_gain'] is not None and x['parallel_gain'] < 0 for x in rows) == s['negative_gain_count']
    assert {x['omitted'] for x in s['leave_one_family_out']} == set(families)
    for omitted in s['leave_one_family_out']:
        keep = [x for x in rows if x['family'] != omitted['omitted']]
        close(sum(x['centered_difference'] for x in keep)/sum(x['total_difference'] for x in keep), omitted['centered_fraction'])
out = dict(status='passed', concentration_sha256=sha(cp), result_sha256=sha(rp), vector_projection_checks=len(seen), summary_checks=len(c['summaries']), max_energy_error=max_error, scope='Independent NumPy residual-vector projections from cached native logits; no transformer rerun or statistical significance test.')
(P / 'native_geometry_projection_audit.json').write_text(json.dumps(out, indent=2)+'\n')
print(out)
