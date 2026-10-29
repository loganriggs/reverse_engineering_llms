from pathlib import Path
import numpy as np

def load_data(out,seed):
    with np.load(Path(out)/f'{seed}_data.npz') as source:
        data={k:source[k] for k in source.files}
    chunks=[]
    for index in range(4):
        with np.load(Path(out)/f'{seed}_source_{index}.npz') as source:
            chunks.append({k:source[k] for k in source.files})
    for key in ['x','y','noise']:data[key]=np.concatenate([c[key] for c in chunks])
    return data
