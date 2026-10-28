#!/usr/bin/env python3
"""H533: locked paired competence suite, followed by MLP-write causal discovery.
No prompt selection based on model outputs. Discovery/transfer item assignments and templates
differ, with some shared vocabulary. Full-vocabulary correctness is distinct from binary contrast accuracy.
"""
import json, hashlib, time, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_broad_task_competence import continuation_id, final_logits
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/bilin18_tensor_decomposition'
STEM='bilin18_diverse_causal_tasks'
torch.set_num_threads(6)

def suite():
    rows=[]
    def add(family, i, prompts, answers, nuisance):
        assert len(prompts)==len(answers)==2 and answers[0]!=answers[1]
        rows.append(dict(id=f'{family}-{i:02}',family=family,split='discovery' if i<8 else 'transfer',prompts=prompts,answers=answers,nuisance=nuisance))
    names=[('John','Mary'),('Alice','Bob'),('Carol','Dave'),('Sarah','Tom'),('Peter','Jane'),('James','Anna'),('David','Kate'),('Mark','Lucy'),('Paul','Lisa'),('George','Susan'),('Michael','Helen'),('Robert','Emily'),('Daniel','Laura'),('Jack','Emma'),('Henry','Rose'),('Thomas','Grace')]
    nouns=[('key','keys'),('dog','dogs'),('cat','cats'),('report','reports'),('picture','pictures'),('bottle','bottles'),('tree','trees'),('book','books'),('student','students'),('teacher','teachers'),('doctor','doctors'),('worker','workers'),('artist','artists'),('farmer','farmers'),('singer','singers'),('driver','drivers')]
    colors=['red','blue','green','yellow','black','white','orange','purple','gold','silver','pink','brown','gray','violet','cyan','tan']
    countries=[('France','Paris'),('Germany','Berlin'),('Italy','Rome'),('Japan','Tokyo'),('Spain','Madrid'),('China','Beijing'),('Russia','Moscow'),('England','London'),('Austria','Vienna'),('Greece','Athens'),('Ireland','Dublin'),('Portugal','Lisbon'),('Norway','Oslo'),('Sweden','Stockholm'),('Finland','Helsinki'),('Denmark','Copenhagen')]
    adjectives=[('wonderful','terrible'),('excellent','awful'),('delightful','dreadful'),('enjoyable','boring'),('great','bad'),('amazing','horrible'),('superb','poor'),('fantastic','disappointing'),('pleasant','unpleasant'),('brilliant','mediocre'),('entertaining','tedious'),('impressive','abysmal'),('lovely','ugly'),('satisfying','frustrating'),('outstanding','pathetic'),('perfect','lousy')]
    for i,((a,b),(sing,pl)) in enumerate(zip(names,nouns)):
        place,verb,obj=('store','gave','book') if i<8 else ('park','handed','bag')
        add('entity_binding',i,[f'{a} and {b} went to the {place}. {giver} {verb} the {obj} to' for giver in (a,b)],[b,a],f'{a} and {b} went to the {place}. {a} {verb} the {"pen" if i<8 else "box"} to')
        other=nouns[(i+3)%16][0]
        prep='near' if i<8 else 'beside'
        prompts=[f'The {n} {prep} the {other}' for n in (sing,pl)]
        add('agreement',i,prompts,['is','are'],f'The {sing} {prep} the {nouns[(i+5)%16][0]}')
        key=colors[i]; v,w=colors[(i+1)%16],colors[(i+4)%16]
        sep=';' if i<8 else ','
        add('sequence_copy',i,[f'{key} {x}{sep} {key} {x}{sep} {key}' for x in (v,w)],[v,w],f'{colors[(i+7)%16]} {v}{sep} {colors[(i+7)%16]} {v}{sep} {colors[(i+7)%16]}')
        country,capital=countries[i]; country2,capital2=countries[(i+1)%8+(0 if i<8 else 8)]
        template='The capital of {} is' if i<8 else '{} has its capital in'
        add('capital_recall',i,[template.format(c) for c in (country,country2)],[capital,capital2],'For reference, '+template.format(country))
        pos,neg=adjectives[i];thing='film' if i<8 else 'movie'
        prompts=[f'Review: The {thing} was {adj}. Sentiment:' for adj in (pos,neg)]
        add('sentiment',i,prompts,['positive','negative'],f'Review: The {thing} was {pos}. Overall sentiment:')
        add('negation',i,[f'Review: The {thing} was {prefix}{pos}. Sentiment:' for prefix in ('','not ')],['positive','negative'],f'Review: The {thing} was {pos}. Overall sentiment:')
        rel='taller' if i<8 else 'older'
        add('relational_order',i,[f'{a} is {rel} than {b}. Of the two, the {q} one is' for q in (rel,'shorter' if i<8 else 'younger')],[a,b],f'{a} is {rel} than {b}. Of these two people, the {rel} one is')
        # Controlled two-sense cue task; intended completions are associations,
        # not asserted unique answers. Full-vocab scores disclose ambiguity.
        objects=['bass','bat','bank','crane','pitch','seal','club','mouse','bass','bat','bank','crane','pitch','seal','club','mouse']
        senses=[('music','fish'),('animal','sport'),('money','river'),('bird','machine'),('music','sport'),('animal','stamp'),('sport','group'),('animal','computer')]
        word=objects[i];s1,s2=senses[i%8]
        template='In the context of {cue}, {word} refers to' if i<8 else 'When discussing {cue}, the word {word} refers to'
        add('context_association',i,[template.format(cue=c,word=word) for c in (s1,s2)],[s1,s2], 'Generally, '+template.format(cue=s1,word=word))
    return rows

def groups(family,answers):
    if family=='agreement':return [[' is',' was',' has',' does'],[' are',' were',' have',' do']]
    if family in ('sentiment','negation'):return [[' positive',' good',' Good',' great',' Great',' excellent',' favorable'],[' negative',' bad',' Bad',' poor',' Poor',' terrible',' unfavorable']]
    return [[' '+a] for a in answers]

def measure(logits, gs):
    scores=torch.stack([torch.logsumexp(logits[g],0) for g in gs])
    top=int(logits.argmax());lp=logits.log_softmax(0)
    return dict(margin=float(scores[0]-scores[1]),top1=[top in g for g in gs],ranks=[int((logits>logits[g].max()).sum())+1 for g in gs],mass=[float(lp[g].exp().sum()) for g in gs],top_token=top)

@torch.inference_mode()
def forward(model,ids, capture=False, patch=None):
    d=model.config.n_embd;x0=F.rms_norm(model.embedding(ids),(d,));x=x0;bus=None;cache=[]
    for layer,block in enumerate(model.blocks):
        hook=None
        if capture or (patch is not None and patch[0]==layer):
            def h(module,args,out):
                if capture:cache.append(out[:,-1].detach().cpu())
                if patch is not None and patch[0]==layer:
                    out=out.clone();out[:,-1]=patch[1].to(out)
                return out
            hook=block.mlp.register_forward_hook(h)
        try:x,bus=block(x,bus,x0)
        finally:
            if hook:hook.remove()
    logits=30*torch.tanh(model.lm_head(F.rms_norm(x,(d,))[:,-1]).float()/30)
    return logits,cache

def main():
    start=time.time(); tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    rows=suite();spec=ART/f'{STEM}_manifest.json'
    config={'experiment':'H533','dtype':'float32','task_pairs':128,'variants':['base','counterfactual','nuisance'],'discovery_pairs_per_family':8,'transfer_pairs_per_family':8,'gate':'On each split: >=4 token-valid pairs, binary correct on both endpoints >=0.75, pooled intended semantic full-vocab top1 >=0.5. No individual pair filtering.','causal_plan':'All 18 layers: swap last-position MLP write bidirectionally on every discovery pair in gate-passing families. Choose layer with largest mean donor-directed margin change on discovery, then evaluate every transfer pair there. Include same-answer nuisance donors. Whole writes localize only; no semantic unit claim.','full_vocab':'Exact retrieval tokens; fixed agreement and sentiment sets inherited from earlier redteam. Context association probes exploratory and ambiguous.','hypotheses':['Endpoint competence will vary across families and templates.','Task-changing donors should move answer contrast more than nuisance donors.','A discovery-selected MLP layer will transfer across held-out wording and lexicons.'],'fineweb_new_documents':0}
    (ROOT/f'configs/{STEM}.json').write_text(json.dumps(config,indent=2)+'\n')
    spec.write_text(json.dumps(rows,indent=2)+'\n')
    manifest_hash=hashlib.sha256(spec.read_bytes()).hexdigest()
    print('Locked manifest',manifest_hash,flush=True)
    examples=[];excluded=[]
    for row in rows:
        gs=groups(row['family'],row['answers']);texts=row['prompts']+[row['nuisance']];enc=[]
        for text in texts:
            gg=[[continuation_id(tok,text,c) for c in g] for g in gs]
            gg=[[v for v in g if v is not None] for g in gg]
            if any(not g for g in gg) or set(gg[0])&set(gg[1]):break
            enc.append(dict(text=text,ids=tok.encode(text,add_special_tokens=False),groups=gg))
        if len(enc)!=3:excluded.append(row['id']);continue
        for variant,e in enumerate(enc):examples.append(dict(**row,variant=variant,**e))
    model=load_full_model('cuda')
    for module in (model.embedding,model.blocks,model.lm_head):
        module.float().eval()
        for p in module.parameters():p.requires_grad_(False)
    bylen=defaultdict(list)
    for i,e in enumerate(examples):bylen[len(e['ids'])].append(i)
    native={};writes={};audit=0.
    for length,inds in sorted(bylen.items()):
        for begin in range(0,len(inds),12):
            ix=inds[begin:begin+12];ids=torch.tensor([examples[i]['ids'] for i in ix],device='cuda')
            logits,c=forward(model,ids,capture=True)
            if not native:audit=float((logits-final_logits(model,ids)).abs().max())
            for k,i in enumerate(ix):
                e=examples[i];key=(e['id'],e['variant']);m=measure(logits[k],e['groups']);m['predicted_text']=tok.decode([m['top_token']]);native[key]=m;writes[key]=torch.stack([z[k] for z in c])
        print('native length',length,'examples',len(inds),flush=True)
    index={(e['id'],e['variant']):e for e in examples};summ={}
    families=sorted(set(r['family'] for r in rows))
    for fam in families:
        summ[fam]={}
        for split in ('discovery','transfer'):
            rr=[r for r in rows if r['family']==fam and r['split']==split and (r['id'],0) in native]
            n=len(rr); both=[native[(r['id'],0)]['margin']>0 and native[(r['id'],1)]['margin']<0 for r in rr]
            top=[native[(r['id'],v)]['top1'][v] for r in rr for v in (0,1)]
            s=dict(pairs=n,both_endpoint_choice=float(np.mean(both)) if n else 0,full_vocab_top1=float(np.mean(top)) if n else 0,nuisance_choice=float(np.mean([native[(r['id'],2)]['margin']>0 for r in rr])) if n else 0,mean_answer_change=float(np.mean([native[(r['id'],0)]['margin']-native[(r['id'],1)]['margin'] for r in rr])) if n else 0)
            s['pass']=n>=4 and s['both_endpoint_choice']>=.75 and s['full_vocab_top1']>=.5;summ[fam][split]=s
    result=dict(config=config,manifest_sha256=manifest_hash,excluded_pairs=excluded,summary=summ,native=[dict(id=k[0],variant=k[1],**v) for k,v in native.items()],audits={'native_forward_max_abs':audit},causal=[],selected_layers={})
    out=ART/f'{STEM}.json'
    def save():
        result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
    save();print('SCREEN',json.dumps(summ),flush=True)
    # Selection on discovery only. Transfer gate is reported, never used to select layers.
    eligible=[f for f in families if summ[f]['discovery']['pass']]
    jobs=[]
    for fam in eligible:
        for r in rows:
            if r['family']==fam and r['split']=='discovery' and (r['id'],0) in native:
                for layer in range(18):
                    for recipient,donor in [(0,1),(1,0),(0,2),(0,0)]:
                        jobs.append((r,layer,recipient,donor))
    def execute(jobs):
        buckets=defaultdict(list)
        for job in jobs:
            r,l,rec,don=job;buckets[(l,len(index[(r['id'],rec)]['ids']))].append(job)
        for (layer,length),jj in sorted(buckets.items()):
            for begin in range(0,len(jj),12):
                batch=jj[begin:begin+12];ee=[index[(r['id'],rec)] for r,l,rec,don in batch]
                ids=torch.tensor([e['ids'] for e in ee],device='cuda');replacement=torch.stack([writes[(r['id'],don)][layer] for r,l,rec,don in batch])
                logits,_=forward(model,ids,patch=(layer,replacement))
                for k,(r,l,rec,don) in enumerate(batch):
                    m=measure(logits[k],ee[k]['groups']);base=native[(r['id'],rec)]['margin'];delta=m['margin']-base
                    result['causal'].append(dict(id=r['id'],family=r['family'],split=r['split'],layer=l,recipient=rec,donor=don,margin=m['margin'],delta=delta,donor_directed_delta=delta*(1 if rec==1 else -1),top1=m['top1']))
            save()
        print('causal jobs complete',len(jobs),flush=True)
    execute(jobs)
    for fam in eligible:
        scores=[np.mean([c['donor_directed_delta'] for c in result['causal'] if c['family']==fam and c['layer']==l and c['donor'] in (0,1) and c['recipient']!=c['donor']]) for l in range(18)]
        result['selected_layers'][fam]=int(np.argmax(scores))
    save();jobs=[]
    for r in rows:
        if r['family'] in eligible and r['split']=='transfer' and (r['id'],0) in native:
            for rec,don in [(0,1),(1,0),(0,2),(0,0)]:jobs.append((r,result['selected_layers'][r['family']],rec,don))
    execute(jobs)
    identity=[abs(c['delta']) for c in result['causal'] if c['recipient']==c['donor']]
    result['audits']['identity_max_margin_delta']=max(identity,default=0)
    result['audits']['all_finite']=all(np.isfinite(c['margin']) for c in result['causal'])
    result['audits']['manifest_unchanged']=manifest_hash==hashlib.sha256(spec.read_bytes()).hexdigest()
    result['status']='complete';save();print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
