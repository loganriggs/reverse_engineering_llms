"""Independent replay of standard-data 70260-update continuation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from audit_fashion_pilot import forward
from run_svhn_standard import digest, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regime', choices=['standard'], default='standard')
    parser.add_argument('--architecture', choices=['stack', 'tree'], required=True)
    parser.add_argument('--depth', type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).parent
    out = root / f'results/svhn_long_standard/d{args.depth}_{args.architecture}'
    origin = root / f'results/svhn_gpu/standard/d{args.depth}_{args.architecture}'
    protocol = json.loads((out / 'protocol.json').read_text())
    assert json.loads((out / 'status.json').read_text())['status'] == 'complete'
    assert digest(out / 'source.py') == protocol['source_sha256']
    for name, sha in protocol['dependencies'].items():
        assert digest(root / name) == sha
    arrays = {}
    for split, manifest in protocol['manifests'].items():
        for k in ['images', 'labels']:
            assert digest(Path(manifest[k]['path'])) == manifest[k]['sha256']
        arrays[split] = tuple(np.load(manifest[k]['path'], mmap_mode='r') for k in ['images', 'labels'])
    ids = np.load(out / 'splits.npz')
    expected = np.random.default_rng(200000).permutation(73257)
    np.testing.assert_array_equal(ids['training_standard'], expected[:68257])
    np.testing.assert_array_equal(ids['validation'], expected[68257:])
    hist = np.zeros((3, 256), dtype=np.int64)
    for start in range(0, 68257, 512):
        a = arrays['train'][0][expected[start:min(start + 512, 68257)]]
        for c in range(3):
            hist[c] += np.bincount(a[:, c].ravel(), minlength=256)
    v = np.arange(256, dtype=np.float64) / 255
    mean = hist @ v / hist.sum(1)
    std = np.sqrt(hist @ (v * v) / hist.sum(1) - mean * mean)
    np.testing.assert_allclose(mean, protocol['mean'], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(std, protocol['std'], atol=1e-12, rtol=1e-12)

    def normalize(a):
        a = a.astype(np.float32) / 255
        a -= mean.astype(np.float32)[None, :, None, None]
        a /= std.astype(np.float32)[None, :, None, None]
        return a.reshape(len(a), 3072)

    fixture = np.load(out / 'preprocessing.npz')
    np.testing.assert_array_equal(fixture['train_first'], normalize(arrays['train'][0][expected[:8]]))
    np.testing.assert_array_equal(fixture['val_first'], normalize(arrays['train'][0][expected[68257:68265]]))
    np.testing.assert_array_equal(fixture['labels_first'], arrays['train'][1][expected[:8]])
    if args.regime == 'extra':
        assert len(arrays['extra'][0]) == 531131
        np.testing.assert_array_equal(fixture['extra_first'], normalize(arrays['extra'][0][:8]))
    row = json.loads((out / 'result.json').read_text())
    row['rmsnorm'] = True
    n = 68257 + (531131 if args.regime == 'extra' else 0)
    assert protocol['ntrain'] == n and len(row['curves']) == 264
    assert row['curves'][:30] == json.loads((origin/'result.json').read_text())['curves']
    assert protocol['origin_checkpoint_sha256'] == digest(origin/'checkpoint.pt')
    for epoch, curve in enumerate(row['curves'], 1):
        updates = min(epoch * 267, 70260)
        assert curve['optimizer_updates'] == updates
        complete_epochs, remainder = divmod(updates, 267)
        assert curve['examples_seen'] == complete_epochs * n + remainder * 256
    records = []
    for checkpoint, reference, logit_file in [
        ('checkpoint.pt', row['curves'][-1], 'logits.npz'),
        ('matched_update.pt', json.loads((out / 'matched_update.json').read_text()), 'matched_update_logits.npz')
    ]:
        saved = torch.load(out / checkpoint, map_location='cpu', weights_only=True)
        state = {k: v.numpy() for k, v in saved['model'].items()}
        assert sum(a.size for a in state.values()) == row['parameters']
        assert saved['optimizer_updates'] == reference['optimizer_updates']
        assert saved['examples_seen'] == reference['examples_seen']
        if checkpoint == 'matched_update.pt':
            assert reference['optimizer_updates'] == 8010
            epochs, steps = divmod(8010, (n + 255) // 256)
            assert reference['examples_seen'] == epochs * n + steps * 256
        for split in (['train', 'validation'] if checkpoint == 'checkpoint.pt' else ['validation']):
            groups = [('train', expected[68257:])] if split == 'validation' else [('train', expected[:68257])]
            if split == 'train' and args.regime == 'extra':
                groups.append(('extra', np.arange(531131)))
            loss, correct, count, outputs = 0., 0, 0, []
            for source, selected in groups:
                for start in range(0, len(selected), 512):
                    batch = selected[start:start + 512]
                    z = forward(row, state, normalize(arrays[source][0][batch]))
                    yy = arrays[source][1][batch]
                    assert np.isfinite(z).all()
                    loss += (logsumexp(z.astype(np.float64), axis=1) - z[np.arange(len(yy)), yy]).sum()
                    correct += (z.argmax(1) == yy).sum()
                    count += len(batch)
                    if split == 'validation':
                        outputs.append(z)
            assert abs(loss / count - reference[split]['ce']) < 1e-4 * max(1, loss / count)
            assert abs(correct / count - reference[split]['accuracy']) <= 1 / count + 1e-7
            if split == 'validation':
                z = np.concatenate(outputs)
                ref = np.load(out / logit_file)['validation']
                error = float(np.max(abs(z - ref)) / max(1, np.max(abs(ref))))
                assert error < 1e-4
        records.append(dict(checkpoint=checkpoint, sha256=digest(out / checkpoint), relative_logit_error=error))
    if args.architecture == 'tree':
        sibling = out.parent / f'd{args.depth}_stack/initial.npy'
        np.testing.assert_allclose(np.load(out / 'initial.npy'), np.load(sibling), atol=1e-5, rtol=1e-5)
    write(out / 'audit.json', dict(status='passed', records=records, source_sha256=digest(Path(__file__)),
          result_sha256=digest(out / 'result.json'), full_training_replayed=True))
    print('Audited', out, records, flush=True)


if __name__ == '__main__':
    main()
