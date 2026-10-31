"""Check whether rank-256 shared-key reduction preserves input-edit responses."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_full_model import KeyDictionaryEdit
from flagship_transfer import ART


def records_from_reports(names):
    output=[];seen=set()
    for name in names:
        data=json.loads((ART/name).read_text())
        for row in data['records']:
            key=(row['corpus'],row['seed'],row['document_id'])
            if key in seen:continue
            seen.add(key)
            output.append(dict(source=name,**{k:row[k] for k in (
                'corpus','seed','document_id','start','key_position',
                'query_position','original_key_id','punctuation_id',
                'control_word_id')}))
    return output


def kl(p,q):
    a=F.log_softmax(torch.from_numpy(p).to(torch.float64),dim=-1)
    b=F.log_softmax(torch.from_numpy(q).to(torch.float64),dim=-1)
    return float(torch.sum(a.exp()*(a-b)))


def norm2(x):
    return float(np.sum(np.asarray(x,dtype=np.float64)**2))


def run(names,batch_size=2):
    torch.set_num_threads(4)
    rows=records_from_reports(names)
    model_config=json.loads((SNAPSHOT/'config.json').read_text())
    model_config.pop('step',None)
    model=GPT(GPTConfig(**model_config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    editor=KeyDictionaryEdit(model,128)
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')['basis']
    corpora={'fineweb_test':CORPUS,**DATASETS}
    output=[]
    with torch.no_grad():
        for offset in range(0,len(rows),batch_size):
            part=rows[offset:offset+batch_size]
            examples=[]
            for row in part:
                corpus=np.load(corpora[row['corpus']],mmap_mode='r')
                tokens=np.asarray(corpus[row['document_id'],
                          row['start']:row['start']+128],dtype=np.int64)
                assert len(tokens)==128 and tokens[row['key_position']]==row['original_key_id']
                variants=np.stack((tokens,tokens.copy(),tokens.copy()))
                variants[1,row['key_position']]=row['punctuation_id']
                variants[2,row['key_position']]=row['control_word_id']
                examples.append(variants)
            idx=torch.from_numpy(np.concatenate(examples).copy())
            editor.reset();native=logits(model,idx).numpy()
            editor.set_shared(basis);reduced=logits(model,idx).numpy()
            for local,row in enumerate(part):
                b=3*local;i=row['query_position']
                original_p=native[b+1,i]-native[b,i]
                original_w=native[b+2,i]-native[b,i]
                reduced_p=reduced[b+1,i]-reduced[b,i]
                reduced_w=reduced[b+2,i]-reduced[b,i]
                output.append(dict(**row,
                    base_query_kl=kl(native[b,i],reduced[b,i]),
                    punctuation_query_kl=kl(native[b+1,i],reduced[b+1,i]),
                    word_query_kl=kl(native[b+2,i],reduced[b+2,i]),
                    native_punctuation_response_energy=norm2(original_p),
                    native_word_response_energy=norm2(original_w),
                    punctuation_response_error_energy=norm2(original_p-reduced_p),
                    word_response_error_energy=norm2(original_w-reduced_w)))
            if (offset+batch_size)%16==0:print('evaluated',offset+len(part),flush=True)
    result=dict(source_reports=list(names),rank=256,layer=0,length=128,
        scope='native-versus-reduced full-model logits on base, punctuation, and word-control inputs and their input-edit responses',
        records=output)
    (ART/'all_head_shared_key_input_edit_fidelity_n96.json').write_text(
        json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('reports',nargs='+')
    p.add_argument('--batch-size',type=int,default=2)
    a=p.parse_args()
    run(a.reports,a.batch_size)
