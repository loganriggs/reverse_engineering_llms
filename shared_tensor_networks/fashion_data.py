"""Official IDX files, publisher MD5 checks, no torchvision dependency."""
import gzip,hashlib,json,struct,urllib.request
from pathlib import Path
import numpy as np

CHECKSUMS={'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d','train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe',
           't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79','t10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}


def load(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);arrays={};records=[]
    for name,md5 in CHECKSUMS.items():
        p=root/name;url='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'+name
        if not p.exists():
            with urllib.request.urlopen(url,timeout=60) as r:p.write_bytes(r.read())
        data=p.read_bytes();assert hashlib.md5(data).hexdigest()==md5
        raw=gzip.decompress(data);magic,n=struct.unpack('>II',raw[:8])
        if 'images' in name:
            assert magic==2051 and struct.unpack('>II',raw[8:16])==(28,28)
            a=np.frombuffer(raw,dtype=np.uint8,offset=16).reshape(n,784).copy()
        else:
            assert magic==2049;a=np.frombuffer(raw,dtype=np.uint8,offset=8).copy();assert len(a)==n and a.max()==9
        assert n==(60000 if name.startswith('train') else 10000)
        arrays[name]=a;records.append(dict(file=name,url=url,md5=md5,sha256=hashlib.sha256(data).hexdigest()))
    (root/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    return arrays['train-images-idx3-ubyte.gz'],arrays['train-labels-idx1-ubyte.gz'],arrays['t10k-images-idx3-ubyte.gz'],arrays['t10k-labels-idx1-ubyte.gz']


if __name__=='__main__':
    data=load('/home/loganriggs/.cache/shared_tensor_networks/fashion-mnist');print([a.shape for a in data])
