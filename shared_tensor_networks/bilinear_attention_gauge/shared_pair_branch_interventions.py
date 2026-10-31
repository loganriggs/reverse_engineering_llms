"""Physical QK-branch edits versus ordinary and intervention-aware key cuts."""

import json
import numpy as np
import torch
from torch.nn import functional as F

from analyze_shared_pair_count import logits
from analyze_three_count_language import flatten
from analyze_toy_language import rms
from dual_joint_function_canonical import HERE
from train_shared_pair_count import SharedPairCountModel
from train_shared_pair_dense import examples_dense
from train_three_count_language import MAX_KEYS,PAD,examples


CONDITIONS=('original','qk1_constant','qk2_constant','both_constant')


def branch_tables(model,dose=1.):
    base=model.backbone
    embedding=base.embed.weight.detach().numpy()
    query=embedding[8:10];key=embedding[:8]
    width=base.qk_width
    total={name:np.zeros((2,8,3)) for name in CONDITIONS}
    for head in range(base.heads):
        sl=slice(head*width,(head+1)*width)
        def project(name,inputs):
            w=getattr(base,name).weight.detach().numpy()[sl]
            return rms(inputs@w.T)
        s1=project('q1',query)@project('k1',key).T/width
        s2=project('q2',query)@project('k2',key).T/width
        value=key@base.v.weight.detach().numpy()[head]
        read=base.read.weight.detach().numpy()[:,head]
        factor=value[None,:,None]*read[None,None,:]
        edited_s1=(1-dose)*s1+dose
        edited_s2=(1-dose)*s2+dose
        for name,score in (('original',s1*s2),('qk1_constant',edited_s1*s2),
                           ('qk2_constant',s1*edited_s2),
                           ('both_constant',edited_s1*edited_s2)):
            total[name]+=score[:,:,None]*factor
    return {name:flatten(table) for name,table in total.items()}


def physical_forward(model,tokens,condition,dose=1.):
    base=model.backbone
    z=base.embed(tokens)
    def project(name):
        raw=getattr(base,name)(z).reshape(len(tokens),MAX_KEYS+1,
                                          base.heads,base.qk_width)
        return F.rms_norm(raw,(base.qk_width,))
    q1,k1,q2,k2=[project(name) for name in ('q1','k1','q2','k2')]
    s1=torch.einsum('bhd,bshd->bsh',q1[:,-1],k1[:,:-1])/base.qk_width
    s2=torch.einsum('bhd,bshd->bsh',q2[:,-1],k2[:,:-1])/base.qk_width
    if condition in ('qk1_constant','both_constant'):s1=(1-dose)*s1+dose
    if condition in ('qk2_constant','both_constant'):s2=(1-dose)*s2+dose
    valid=(tokens[:,:-1]!=PAD).unsqueeze(-1)
    heads=(s1*s2*base.v(z)[:,:-1]*valid).sum(1)
    bits=base.read(heads)
    q=.5*(model.quadratic+model.quadratic.transpose(-1,-2))
    scaled=bits/model.logit_scale
    parity=torch.einsum('bi,pij,bj->bp',scaled,q,scaled)+model.parity_bias
    return torch.cat((bits,parity),dim=1)


def run(n_test=10000):
    torch.set_num_threads(1)
    rows=[]
    for coding,source,prefix in (('paired',examples,'shared_pair_count'),
                                  ('dense',examples_dense,'shared_pair_dense')):
        for seed in range(10):
            model=SharedPairCountModel().double()
            model.load_state_dict(torch.load(HERE/f'{prefix}_seed{seed}.pt',
                                             map_location='cpu',weights_only=True))
            model.eval()
            tables=branch_tables(model)
            bias=model.backbone.read.bias.detach().numpy()
            scale=model.logit_scale.detach().numpy()
            q=model.quadratic.detach().numpy()
            q=.5*(q+q.transpose(0,2,1))/scale[None,:,None]/scale[None,None,:]
            pb=model.parity_bias.detach().numpy()
            tokens,_,attrs=source(n_test,530000+seed,False)
            token_array=tokens.numpy()
            reference={};gates={}
            for condition in CONDITIONS:
                with torch.no_grad():direct=physical_forward(model,tokens[:128],condition).numpy()
                proxy=logits(tables[condition],bias,q,pb,token_array[:128])
                gates[condition]=float(np.max(np.abs(direct-proxy)))
                assert gates[condition]<1e-9,(coding,seed,condition,gates[condition])
                reference[condition]=logits(tables[condition],bias,q,pb,token_array)
            with torch.no_grad():
                native=model(tokens[:128]).numpy()
                replay=physical_forward(model,tokens[:128],'original').numpy()
            assert np.max(np.abs(native-replay))<1e-9
            for fit_name,seen in (('ordinary_only',('original',)),
                                  ('all_four',CONDITIONS)):
                stack=np.vstack([tables[k]/np.linalg.norm(tables[k]) for k in seen])
                _,singular,v=np.linalg.svd(stack,full_matrices=False)
                for rank in range(3,9):
                    projector=v[:rank].T@v[:rank]
                    condition_scores={}
                    for condition in CONDITIONS:
                        table=tables[condition]
                        reduced=table@projector
                        output=logits(reduced,bias,q,pb,token_array)
                        truth=reference[condition]
                        condition_scores[condition]=dict(
                            table_relative_mse=float(np.sum((reduced-table)**2)/
                                np.maximum(np.sum(table**2),1e-20)),
                            output_relative_mse=float(np.sum((output-truth)**2)/
                                np.maximum(np.sum(truth**2),1e-20)),
                            five_logit_sign_agreement=float(np.mean((output>0)==(truth>0))),
                            rare_output_relative_mse=float(np.sum((output[attrs['heldout']]-truth[attrs['heldout']])**2)/
                                np.maximum(np.sum(truth[attrs['heldout']]**2),1e-20)))
                    rows.append(dict(coding=coding,seed=seed,fit=fit_name,rank=rank,
                        max_exact_branch_gate=max(gates.values()),
                        singular_values=singular.tolist(),conditions=condition_scores))
            print(coding,seed,flush=True)
    result=dict(codings=['paired','dense'],conditions=list(CONDITIONS),
        fitting='SVD of individually unit-Frobenius exact branch-edit token tables; no sampled sentences or task labels',
        test=f'{n_test} independently generated sentences per model; compare 5-logit response to physically edited model',
        rows=rows)
    (HERE/'shared_pair_branch_interventions.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()
