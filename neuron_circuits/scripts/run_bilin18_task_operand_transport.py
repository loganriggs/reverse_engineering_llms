#!/usr/bin/env python3
"""H534: exact finite operand decomposition and discovery-only write transport."""
from pathlib import Path
import json,hashlib,time
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,forward,measure
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_task_operand_transport'
torch.set_num_threads(6)

@torch.inference_mode()
def main():
    start=time.time();source=ART/'bilin18_diverse_causal_tasks.json';j=json.loads(source.read_text());manifest=ART/'bilin18_diverse_causal_tasks_manifest.json';rows=json.loads(manifest.read_text())
    from transformers import AutoTokenizer
    from audit_bilin18_mlp_discriminators import TOKENIZER
    from run_bilin18_diverse_causal_tasks import groups,continuation_id
    tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    cfg={'experiment':'H534','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'selected_layers':j['selected_layers'],'selection':'H533 discovery only; no reselection','factorial_modes':['left','right','both','additive_without_cross','cross_only','identity'],'transport_families':['agreement','sentiment'],'transport_modes':['shared','shared_reversed','nuisance_norm_matched']+[f'random_{i:02}' for i in range(16)],'transport_fit':'Discovery mean of cf minus base last-position MLP output. Test held-out transfer in both directions. Nuisance mean and seeded Gaussian controls scaled to same Euclidean norm. Controls are not covariance-matched.','seed':20261022,'hypotheses':['A nontrivial finite operand cross term contributes beyond separately changing each operand.','A discovery-mean task write shifts transfer answers in the predicted direction beyond nuisance and isotropic random controls.'],'scope':'Reused H533 panel and frozen selected layers. No fresh superiority claim, no necessary circuit claim, no neuron ontology claim.'}
    (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');print('CONFIG LOCKED',flush=True)
    model=load_full_model('cuda')
    for mod in (model.embedding,model.blocks,model.lm_head):
        mod.float().eval()
        for p in mod.parameters():p.requires_grad_(False)
    assert model.config.bilinear and not model.config.gated
    examples={};cache={};native={};operand_replay=[]
    for r in rows:
        if r['family'] not in cfg['selected_layers'] or r['id'] in j['excluded_pairs']:continue
        for v,text in enumerate(r['prompts']+[r['nuisance']]):
            gs=[[continuation_id(tok,text,t) for t in g] for g in groups(r['family'],r['answers'])];gs=[[t for t in g if t is not None] for g in gs]
            examples[(r['id'],v)]={'row':r,'variant':v,'ids':tok.encode(text,add_special_tokens=False),'groups':gs}
    buckets=defaultdict(list)
    for key,e in examples.items():buckets[(cfg['selected_layers'][e['row']['family']],len(e['ids']))].append(key)
    for (layer,length),keys in sorted(buckets.items()):
        mlp=model.blocks[layer].mlp
        for begin in range(0,len(keys),12):
            kk=keys[begin:begin+12];captured={};handles=[]
            for name in ('Left','Right'):
                def hook(m,args,out,name=name):captured[name]=out[:,-1].clone()
                handles.append(getattr(mlp,name).register_forward_hook(hook))
            def whook(m,args,out):captured['write']=out[:,-1].clone()
            handles.append(mlp.register_forward_hook(whook))
            try:logits,_=forward(model,torch.tensor([examples[k]['ids'] for k in kk],device='cuda'))
            finally:
                for h in handles:h.remove()
            rec=mlp.Down(captured['Left']*captured['Right'])+mlp.Down_bias
            operand_replay.append(float((rec-captured['write']).abs().max()))
            for ix,key in enumerate(kk):
                cache[key]={n:v[ix].cpu() for n,v in captured.items()};native[key]=measure(logits[ix],examples[key]['groups'])
    # All native operands persisted before evaluating synthetic operand interventions.
    torch.save({'config':cfg,'cache':cache},ART/f'{STEM}_operands.pt')
    result={'config':cfg,'factorial':[],'transport':[],'geometry':[],'audits':{'operand_replay_max_abs':max(operand_replay)},'status':'running'}
    out=ART/f'{STEM}.json'
    def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
    jobs=[];deltas={}
    for r in rows:
        if (r['id'],0) not in cache:continue
        layer=cfg['selected_layers'][r['family']];down=model.blocks[layer].mlp.Down
        for rec,don in ((0,1),(1,0)):
            a=cache[(r['id'],rec)];b=cache[(r['id'],don)];dl=(b['Left']-a['Left']).cuda();dr=(b['Right']-a['Right']).cuda()
            left=down(dl*a['Right'].cuda()).cpu();right=down(a['Left'].cuda()*dr).cpu();cross=down(dl*dr).cpu();total=b['write']-a['write']
            err=float((left+right+cross-total).abs().max())
            result['geometry'].append({'id':r['id'],'family':r['family'],'split':r['split'],'recipient':rec,'identity_max_abs':err,'total_norm':float(total.norm()),'left_norm':float(left.norm()),'right_norm':float(right.norm()),'cross_norm':float(cross.norm())})
            dd={'left':left,'right':right,'both':total,'additive_without_cross':left+right,'cross_only':cross,'identity':torch.zeros_like(total)}
            for mode,d in dd.items():
                key=(r['id'],rec,mode);deltas[key]=a['write']+d;jobs.append((r,layer,rec,mode,key,'factorial'))
    # Shared task changes: fixed semantic orientation singular->plural and positive->negative.
    rng=torch.Generator().manual_seed(cfg['seed']);vectors={}
    for fam in cfg['transport_families']:
        rr=[r for r in rows if r['family']==fam and r['split']=='discovery' and (r['id'],0) in cache]
        task=torch.stack([cache[(r['id'],1)]['write']-cache[(r['id'],0)]['write'] for r in rr]).mean(0)
        nuisance=torch.stack([cache[(r['id'],2)]['write']-cache[(r['id'],0)]['write'] for r in rr]).mean(0)
        def scale(v):return v*(task.norm()/v.norm().clamp_min(1e-20))
        vv={'shared':task,'shared_reversed':-task,'nuisance_norm_matched':scale(nuisance)}
        for i in range(16):vv[f'random_{i:02}']=scale(torch.randn(task.shape,generator=rng))
        vectors[fam]=vv
        for r in rows:
            if r['family']!=fam or r['split']!='transfer' or (r['id'],0) not in cache:continue
            for rec in (0,1):
                for mode,v in vv.items():
                    key=(r['id'],rec,'transport_'+mode);deltas[key]=cache[(r['id'],rec)]['write']+v*(1 if rec==0 else -1)
                    jobs.append((r,cfg['selected_layers'][fam],rec,mode,key,'transport'))
    torch.save(vectors,ART/f'{STEM}_frozen_vectors.pt');save();print('FROZEN',len(jobs),'jobs',flush=True)
    buckets=defaultdict(list)
    for job in jobs:
        r,l,rec,mode,key,typ=job;buckets[(l,len(examples[(r['id'],rec)]['ids']))].append(job)
    for (layer,length),jj in sorted(buckets.items()):
        for begin in range(0,len(jj),12):
            batch=jj[begin:begin+12];ee=[examples[(r['id'],rec)] for r,l,rec,mode,key,typ in batch]
            ids=torch.tensor([e['ids'] for e in ee],device='cuda');replacement=torch.stack([deltas[key] for r,l,rec,mode,key,typ in batch])
            logits,_=forward(model,ids,patch=(layer,replacement))
            for ix,(r,l,rec,mode,key,typ) in enumerate(batch):
                m=measure(logits[ix],ee[ix]['groups']);delta=m['margin']-native[(r['id'],rec)]['margin']
                result[typ].append(dict(id=r['id'],family=r['family'],split=r['split'],layer=l,recipient=rec,mode=mode,margin=m['margin'],donor_directed_delta=delta*(1 if rec==1 else -1),top1=m['top1']))
        save();print('done layer/length',layer,length,flush=True)
    old={(c['id'],c['recipient']):c for c in j['causal'] if c['layer']==cfg['selected_layers'][c['family']] and c['recipient']!=c['donor'] and c['donor']<2}
    result['audits'].update({'factorial_identity_max_abs':max(g['identity_max_abs'] for g in result['geometry']),'self_margin_max_abs':max(abs(c['donor_directed_delta']) for c in result['factorial'] if c['mode']=='identity'),'h533_both_replay_margin_max_abs':max(abs(c['margin']-old[(c['id'],c['recipient'])]['margin']) for c in result['factorial'] if c['mode']=='both'),'finite':all(np.isfinite(c['margin']) for typ in ('factorial','transport') for c in result[typ]),'source_unchanged':cfg['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()})
    result['status']='complete';save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
