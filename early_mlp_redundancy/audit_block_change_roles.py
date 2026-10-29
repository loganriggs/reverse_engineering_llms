"""Independent NumPy gate/decode and every saved final-response audit for M105."""
import json,sys
import numpy as np
import torch
from fit_pca_encoder import V,sha
from audit_encoded_factorial import lp,margin
from block_change_components import grouping


def main():
    torch.set_num_threads(1);cohort=sys.argv[1];assert cohort in ['train','source','whole'];rp=V/f'block_change_roles_{cohort}.json';r=json.loads(rp.read_text());assert r['status']=='completed' and r['source_sha256']==sha(V.parent/'probe_block_change_roles.py') and r['execution_sha256']==sha(V/'block_change_roles_execution.json')
    protocol=json.loads((V/'block_change_roles_protocol.json').read_text());stem={'train':'training_causal_coverage','source':'source_holdout_native','whole':'whole_task_native'}[cohort];n=json.loads((V/(stem+'.json')).read_text());assert sha(n['samples_path'])==n['samples_sha256'];native=torch.load(n['samples_path'],weights_only=True);targets={(x['pair_id'],x['endpoint'],x['region']):x for x in n['records'] if x['layer']==1};states={}
    for e in protocol['models']:
        assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True);st={k:v.double().numpy() for k,v in s['state'].items()};st['dictionary']/=np.linalg.norm(st['dictionary'],axis=0);states[e['seed'],e['arm']]=(st,s['mean'].double().numpy(),float(s['scale']))
    key=lambda x:(x['pair_id'],x['seed'],x['arm'],x['endpoint'],x['region'])
    rr={key(x)+(x['grouping'],x['group']):x for x in r['rows']};cc={key(x):x for x in r['controls']};ii={key(x)+(x['grouping'],):x for x in r['interactions']};assert len(rr)==len(r['rows']) and len(cc)==len(r['controls']) and len(ii)==len(r['interactions'])
    checks=0;vectors=0;decodes=0;maximum=0.;maxdecode=0.;calls=0;replays=0;missing=[]
    def metric(a,b):
        nonlocal checks,maximum
        assert np.isclose(a,b,rtol=1e-8,atol=1e-8),(a,b);maximum=max(maximum,abs(a-b));checks+=1
    def decode(a,b):
        nonlocal maxdecode,decodes
        err=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.));assert err<1e-5,err;maxdecode=max(maxdecode,err);decodes+=1
    for file in r['files']:
        assert sha(file['path'])==file['sha256'];s=torch.load(file['path'],weights_only=True);pid=s['pair_id'];assert pid==file['pair_id'];calls+=2;replays+=2
        for ep in [0,1]:assert torch.equal(s['baselines'][ep],native['baselines'][pid+f'|{ep}']['logits'])
        for ep in [0,1]:
            for region in ['changed_input_positions','final_position','identity']:
                target=targets[pid,ep,region]
                if target['status']=='evaluated':
                    delta=native['deltas'][target['sample_key']]['delta'];calls+=bool(torch.count_nonzero(delta));replays+=1
        for e in protocol['models']:
            mk=f"{e['seed']}:{e['arm']}";st,mu,scale=states[e['seed'],e['arm']];codes=[];recs=[];supports=[]
            for ep in [0,1]:
                rec=s['reconstructions'][mk+f'|{ep}'];w=s['writes'][ep].double().numpy();a=((w-mu)/scale)@st['encoder'].T+st['encoder_bias'];scores=a*a@st['membership'].T;chosen=np.argsort(scores,axis=-1)[...,-8:];g=np.zeros_like(scores);np.put_along_axis(g,chosen,1,axis=-1);q=g@st['membership']>0;assert np.array_equal(q,rec['support'].numpy());decode(a*q,rec['code'].numpy());code=rec['code'].double().numpy();decode((code@st['dictionary'].T+st['bias'])*scale+mu,rec['write'].numpy());codes.append(code);recs.append(rec['write'].numpy());supports.append(q)
            for ep in [0,1]:
                b=native['baselines'][pid+f'|{ep}'];base=b['logits'].numpy();bl=lp(base);prob=np.exp(bl);base_margin=margin(base,b['groups'],b['variant'])
                for region in ['changed_input_positions','final_position','identity']:
                    k=(pid,e['seed'],e['arm'],ep,region);target=targets[pid,ep,region]
                    if target['status']!='evaluated':missing.append(k);continue
                    pos=target['recipient_positions'];dp=target['donor_positions'];sk=mk+f'|{ep}|{region}';teacher=lp(native['changed'][target['sample_key']])-bl;joint=s['full'][sk].numpy();jr=lp(joint)-bl;jm=margin(joint,b['groups'],b['variant'])-base_margin;full=recs[1-ep][:,dp]-recs[ep][:,pos];control=cc[k]
                    assert control['noop']==(not np.any(full));calls+=not control['noop'];vectors+=1
                    if control['noop']:assert np.array_equal(joint,base)
                    metric(float((prob*jr**2).sum()),control['response_energy']);metric(float((prob*(jr-teacher)**2).sum()),control['teacher_error']);metric(jm,control['margin_change']);metric(float((full.astype(np.float64)**2).sum()),control['delta_energy']);metric(float((prob*teacher**2).sum()),control['teacher_response_energy'])
                    for shuffled in [False,True]:
                        name='shuffled' if shuffled else 'trained';groups=grouping(e['seed'],shuffled);dc=codes[1-ep][:,dp]-codes[ep][:,pos];total=np.zeros_like(jr);total_margin=0.;energy_sum=0.;local_sum=np.zeros_like(full,dtype=np.float64)
                        for group,atoms in enumerate(groups):
                            row=rr[k+(name,group)];delta=(dc[:,:,atoms]@st['dictionary'][:,atoms].T)*scale;local_sum+=delta;assert row['noop']==(not np.any(delta));assert row['active_coordinate_positions']==int((supports[ep][:,pos][:,:,atoms]|supports[1-ep][:,dp][:,:,atoms]).sum());assert row['positions']==len(pos) and row['local_sum_relative_error']<1e-5
                            assert np.isclose(float((delta**2).sum()),row['delta_energy'],rtol=1e-5,atol=1e-7)
                            out=s['changed'][sk+f'|{name}|{group}'].numpy();response=lp(out)-bl;effect=margin(out,b['groups'],b['variant'])-base_margin;energy=float((prob*response**2).sum());total+=response;total_margin+=effect;energy_sum+=energy
                            metric(energy,row['response_energy']);metric(float((prob*(response-teacher)**2).sum()),row['teacher_error']);metric(effect,row['margin_change']);calls+=not row['noop'];vectors+=1
                            if row['noop']:assert np.array_equal(out,base)
                        decode(local_sum,full);interaction=ii[k+(name,)]
                        for field,value in dict(sum_response_energy=float((prob*total**2).sum()),sum_individual_response_energies=energy_sum,joint_response_energy=float((prob*jr**2).sum()),nonadditive_response_energy=float((prob*(total-jr)**2).sum()),sum_margin_change=total_margin,joint_margin_change=jm,margin_nonadditivity=total_margin-jm).items():metric(value,interaction[field])
        print('AUDITED',cohort,pid,flush=True)
    assert set(missing)=={key(x) for x in r['ineligible']};assert calls==r['forward_count'] and replays==r['native_replays'];assert vectors==len(r['rows'])+len(r['controls']);assert len(r['interactions'])==2*len(r['controls'])
    out=dict(status='passed',source_sha256=sha(__file__),result_sha256=sha(rp),response_vectors=vectors,metric_checks=checks,decoder_checks=decodes,max_metric_absolute_error=maximum,max_decoder_relative_error=maxdecode,forward_count=calls,native_replays=replays,ineligible_assignments=len(missing),scope='All saved finalvectors independently rescored inNumPy; all endpoint gates and decodes independently reconstructed, allgroup energy/support/cost checked, group sum and downstream nonadditivity recomputed. Saved codes used for float64 groupdecode after independent encoder/gate verification. FullnativeTransformer not independently implemented.')
    dest=V/f'block_change_roles_{cohort}_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
