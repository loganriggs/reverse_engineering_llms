"""Make verified byte-preserving memory-mapped arrays outside the repository."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from svhn_data import load

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--splits',default='train,extra');args=parser.parse_args();root=Path('/home/loganriggs/.cache/shared_tensor_networks/svhn')
    for split in args.splits.split(','):
        assert (root/f'{split}_32x32.mat').exists(),'Download must finish and pass its checksum first'
        images,labels=load(root,split);ip=root/f'{split}_images.npy';lp=root/f'{split}_labels.npy'
        for path,a in [(ip,images),(lp,labels)]:
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f:np.save(f,a)
            tmp.replace(path);reloaded=np.load(path,mmap_mode='r');assert np.array_equal(reloaded,a)
        raw=json.loads((root/f'{split}_manifest.json').read_text());record=dict(raw=raw,images=dict(path=str(ip),sha256=digest(ip),shape=list(images.shape),dtype=str(images.dtype)),labels=dict(path=str(lp),sha256=digest(lp),shape=list(labels.shape),dtype=str(labels.dtype)))
        (root/f'{split}_array_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print('Verified memory-mapped arrays',split,images.shape,flush=True)
if __name__=='__main__':main()
