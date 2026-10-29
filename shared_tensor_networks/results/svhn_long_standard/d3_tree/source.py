"""Standard-data continuation to 70260 updates; restores audited model and Adam state."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_svhn_standard import digest, write
from vision_circuits import BilinearCircuit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regime', choices=['standard'], default='standard')
    p.add_argument('--architecture', choices=['stack', 'tree'], required=True)
    p.add_argument('--depth', type=int, choices=[2, 3, 4], default=2)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).parent
    out = root / f'results/svhn_long_standard/d{args.depth}_{args.architecture}'
    origin = root / f'results/svhn_gpu/standard/d{args.depth}_{args.architecture}'
    audit = json.loads((origin / 'audit.json').read_text())
    assert audit['status'] == 'passed'
    assert audit['result_sha256'] == digest(origin / 'result.json')
    assert audit['records'][0]['sha256'] == digest(origin / 'checkpoint.pt')
    out.mkdir(parents=True, exist_ok=True)
    cache = Path('/home/loganriggs/.cache/shared_tensor_networks/svhn')
    base = json.loads((root / 'results/svhn_standard/depth2/protocol.json').read_text())
    order = np.random.default_rng(200000).permutation(73257)
    train_ids, val_ids = order[:68257], order[68257:]
    mean = np.array(base['mean'], dtype=np.float32)[None, :, None, None]
    std = np.array(base['std'], dtype=np.float32)[None, :, None, None]
    arrays, manifests = {}, {}
    for split in ['train'] + (['extra'] if args.regime == 'extra' else []):
        manifest = json.loads((cache / f'{split}_array_manifest.json').read_text())
        for key in ['images', 'labels']:
            assert digest(Path(manifest[key]['path'])) == manifest[key]['sha256']
        arrays[split] = tuple(np.load(manifest[key]['path'], mmap_mode='r') for key in ['images', 'labels'])
        manifests[split] = manifest
    n = 68257 + (531131 if args.regime == 'extra' else 0)
    protocol = dict(regime=args.regime, architecture=args.architecture, depth=args.depth,
                    device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                    dtype='float32', tf32=False, ntrain=n, nvalidation=5000,
                    model_seed=200100, split_seed=200000, shuffle_seed_base=200200,
                    width=32, epochs=264, target_updates=70260, origin_checkpoint_sha256=digest(origin/'checkpoint.pt'), batch_size=256, lr=.001, optimizer='Adam', clip=10,
                    rmsnorm=True, mean=base['mean'], std=base['std'], official_test_used=False,
                    matched_update=8010, selection='fixed70260updates; no validation selection',
                    preprocessing='standard-train RGB statistics; NCHW; no augmentation',
                    source_sha256=digest(Path(__file__)),
                    dependencies={f: digest(root / f) for f in ['vision_circuits.py', 'run_svhn_standard.py']},
                    manifests=manifests)
    if (out / 'protocol.json').exists():
        assert json.loads((out / 'protocol.json').read_text()) == protocol
    else:
        write(out / 'protocol.json', protocol)
        (out / 'source.py').write_bytes(Path(__file__).read_bytes())
        np.savez_compressed(out / 'splits.npz', training_standard=train_ids, validation=val_ids)
    if (out / 'status.json').exists() and json.loads((out / 'status.json').read_text())['status'] == 'complete':
        print('Already complete', out, flush=True)
        return

    def normalized(a):
        a = a.astype(np.float32) / 255
        return torch.from_numpy(((a - mean) / std).reshape(len(a), 3072)).cuda()

    x = torch.empty((n, 3072), device='cuda')
    y = torch.empty(n, dtype=torch.long, device='cuda')
    for start in range(0, 68257, 4096):
        ids = train_ids[start:start + 4096]
        x[start:start + len(ids)] = normalized(arrays['train'][0][ids])
        y[start:start + len(ids)] = torch.from_numpy(np.array(arrays['train'][1][ids])).cuda()
    if args.regime == 'extra':
        for start in range(0, 531131, 4096):
            stop = min(start + 4096, 531131)
            x[68257 + start:68257 + stop] = normalized(arrays['extra'][0][start:stop])
            y[68257 + start:68257 + stop] = torch.from_numpy(np.array(arrays['extra'][1][start:stop])).cuda()
    vx = normalized(arrays['train'][0][val_ids])
    vy = torch.from_numpy(np.array(arrays['train'][1][val_ids])).cuda()
    # Persist a small preprocessing fixture for an independent replay.
    np.savez_compressed(out / 'preprocessing.npz', train_first=x[:8].cpu().numpy(),
                        val_first=vx[:8].cpu().numpy(), labels_first=y[:8].cpu().numpy(),
                        extra_first=x[68257:68265].cpu().numpy())
    model = BilinearCircuit(3072, 32, args.depth, 10, args.architecture == 'tree', True, 200100).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    with torch.no_grad():
        initial = model(vx[:128]).cpu().numpy()
    if not (out / 'initial.npy').exists():
        np.save(out / 'initial.npy', initial)
    cp = out / 'checkpoint.pt'
    if not cp.exists():
        shutil.copyfile(origin / 'checkpoint.pt', cp)
        for name in ['matched_update.pt', 'matched_update.json', 'matched_update_logits.npz']:
            shutil.copyfile(origin / name, out / name)
    curves = []
    if cp.exists():
        saved = torch.load(cp, map_location='cuda', weights_only=True)
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        curves = saved['curves']
    updates = saved['optimizer_updates']
    seen = saved['examples_seen']
    assert curves[:30] == json.loads((origin/'result.json').read_text())['curves']

    def metrics(xx, yy, keep=False):
        sums = torch.zeros(2, device='cuda', dtype=torch.float64)
        logits = []
        with torch.no_grad():
            for start in range(0, len(xx), 2048):
                z = model(xx[start:start + 2048])
                sums[0] += nn.functional.cross_entropy(z, yy[start:start + 2048], reduction='sum').double()
                sums[1] += (z.argmax(1) == yy[start:start + 2048]).sum()
                if keep:
                    logits.append(z.cpu())
        values = sums.cpu().numpy() / len(xx)
        assert np.isfinite(values).all()
        return dict(ce=float(values[0]), accuracy=float(values[1])), (torch.cat(logits).numpy() if keep else None)

    def save(path):
        tmp = path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), curves=curves,
                        optimizer_updates=updates, examples_seen=seen), tmp)
        tmp.replace(path)

    start_time = time.monotonic()
    write(out / 'status.json', dict(status='running', pid=os.getpid(), completed_epochs=len(curves)))
    for epoch in range(len(curves), 264):
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(200200 + epoch)).cuda()
        clipped = torch.zeros((), device='cuda', dtype=torch.long)
        for batch in indices.split(256):
            if updates >= 70260:
                break
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            clipped += norm > 10
            opt.step()
            updates += 1
            seen += len(batch)
            if updates == 8010:
                validation, logits = metrics(vx, vy, True)
                save(out / 'matched_update.pt')
                np.savez_compressed(out / 'matched_update_logits.npz', validation=logits)
                write(out / 'matched_update.json', dict(optimizer_updates=updates, examples_seen=seen,
                      epoch_index=epoch, validation=validation, recovery='restart from last complete epoch'))
        train, _ = metrics(x, y)
        validation, logits = metrics(vx, vy, True)
        curves.append(dict(epoch=epoch + 1, train=train, validation=validation,
                           optimizer_updates=updates, examples_seen=seen, clipped_steps=int(clipped)))
        save(cp)
        write(out / 'result.json', dict(architecture=args.architecture, depth=args.depth, width=32,
              parameters=sum(p.numel() for p in model.parameters()), curves=curves))
        np.savez_compressed(out / 'logits.npz', validation=logits)
        write(out / 'status.json', dict(status='running', pid=os.getpid(), completed_epochs=epoch + 1))
        print(args.regime, args.architecture, epoch + 1, train, validation, flush=True)
    write(out / 'status.json', dict(status='complete', epochs=len(curves), updates=updates,
          seconds_this_invocation=time.monotonic() - start_time, peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
