"""Large short-context natural panel for outlier-robust unlabeled gauges."""

import json

import numpy as np
import torch

from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut, contexts, final_logits
from flagship_full_model_eval import GPT, GPTConfig
from flagship_matrix_free import CORPUS, SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART

SEED=20261104
LENGTH=5
NTRAIN=8192
NTEST=2048
NOOD=1024


def run():
    torch.set_num_threads(4)
    rng=np.random.default_rng(SEED)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted(set(layer for layer,_ in TARGETS.values()))}
    bases=np.load(ART/'later_head_rank_pilot_bases.npz')
    panels={'fineweb_train':contexts(CORPUS,rng,NTRAIN,range(0,16000),LENGTH),
            'fineweb_test':contexts(CORPUS,rng,NTEST,range(17500,20000),LENGTH)}
    for name,path in DATASETS.items():
        panels[name]=contexts(path,rng,NOOD,range(0,1200),LENGTH)
    arrays={}
    with torch.no_grad():
        for panel,ids in panels.items():
            collected={t:[] for t in TARGETS}
            for n,batch in enumerate(ids.split(32),1):
                final_logits(model,batch,stop_at=max(probes))
                for t,(layer,head) in TARGETS.items():
                    p=probes[layer];a=head*p.head_dim
                    basis=torch.tensor(bases[f'{t}__weight'][:,:4].copy(),dtype=torch.float32)
                    collected[t].append((p.last_input[:,-1,a:a+p.head_dim]@basis).numpy())
                if n%64==0:print(panel,n*32,flush=True)
            arrays[f'ids__{panel}']=ids.numpy()
            for t in TARGETS:
                arrays[f'{t}__{panel}']=np.concatenate(collected[t])
            print('complete',panel,len(ids),flush=True)
    np.savez(ART/'later_head_unlabeled_large.npz',**arrays)
    (ART/'later_head_unlabeled_large.json').write_text(json.dumps(dict(
        description='Large independent short-context natural panels in fixed rank-4 head subspaces',
        seed=SEED,length=LENGTH,ntrain=NTRAIN,ntest=NTEST,nood=NOOD,
        fineweb_train_docs=[0,16000],fineweb_test_docs=[17500,20000],
        arrays={k:list(v.shape) for k,v in arrays.items()}),indent=2)+'\n')


if __name__=='__main__':run()
