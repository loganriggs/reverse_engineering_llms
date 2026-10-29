"""Independent token/logit checks and Torch interpolation/finite-removal native decomposition."""
import json
import numpy as np
import torch,tiktoken
from fit_pca_encoder import V,sha


def main():
    torch.set_num_threads(1);rp=V/'repetition_cube_native.json';ap=V/'repetition_cube_analysis.json';pp=V/'repetition_cube_protocol.json';r=json.loads(rp.read_text());a=json.loads(ap.read_text());p=json.loads(pp.read_text());assert a['native_sha256']==sha(rp) and a['protocol_sha256']==sha(pp);assert sha(r['states_path'])==r['states_sha256'];saved=torch.load(r['states_path'],weights_only=True);wp=V/'matched_layer_operator_inputs.json';w=json.loads(wp.read_text());assert sha(w['path'])==w['sha256'];weights=torch.load(w['path'],weights_only=True)['states'];tok=tiktoken.get_encoding('gpt2');rows={x['id']:x for x in r['rows']};assert len(rows)==len(saved)==len(p['rows'])==80;maximum=0.;checks=0
    bits=torch.tensor([[bool(vertex&(1<<j)) for j in range(3)] for vertex in range(8)],dtype=torch.float64);signs=2*bits-1;H=torch.ones((8,8),dtype=torch.float64)
    for mask in range(8):
        for j in range(3):
            if mask&(1<<j):H[:,mask]*=signs[:,j]
    assert torch.equal(H.T@H,torch.eye(8,dtype=torch.float64)*8)
    def check(x,y):
        nonlocal maximum,checks
        x=np.asarray(x);y=np.asarray(y);err=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1.));assert err<1e-8,(err,x,y);maximum=max(maximum,err);checks+=1
    for m in p['rows']:
        key=m['id'];assert saved[key]['ids'][0].tolist()==m['token_ids'];z=saved[key]['logits'].double().numpy();scores=[]
        for group in m['answer_groups']:
            zz=z[group];scores.append(float(zz.max()+np.log(np.exp(zz-zz.max()).sum())))
        check(scores[1]-scores[0],rows[key]['variant_group_margin']);assert rows[key]['group_choice']==int(np.argmax(scores)) and rows[key]['top_token']==int(z.argmax());assert rows[key]['top1_value_group']==next((i for i,g in enumerate(m['answer_groups']) if int(z.argmax()) in g),None)
        if m['cohort']=='cube':
            sep='; ' if m['format']=='semicolon' else ', ';prompt=sep.join([m['key']+' '+m['values'][b] for b in m['bits']]+[m['key']]);assert prompt==m['prompt'] and tok.encode(prompt)==m['token_ids'];assert m['bits']==bits[m['vertex']].int().tolist()
            for value,answer in zip(m['values'],m['canonical_answer_tokens']):assert tok.encode(prompt+' '+value)==m['token_ids']+[answer]
            check(float(z[m['canonical_answer_tokens'][1]]-z[m['canonical_answer_tokens'][0]]),rows[key]['canonical_margin'])
    for b in a['behavior']:
        items=sorted([x for x in p['rows'] if x.get('case')==b['case'] and x.get('format')==b['format']],key=lambda x:x['vertex']);assert len(items)==8;assert len({len(x['token_ids']) for x in items})==1
        # Every cube edge changes exactly the token in one value occurrence.
        for vertex,m in enumerate(items):
            for bit in range(3):
                other=items[vertex^(1<<bit)];assert sum(x!=y for x,y in zip(m['token_ids'],other['token_ids']))==1
        y=torch.tensor([rows[x['id']]['canonical_margin'] for x in items],dtype=torch.float64);c=torch.linalg.solve(H,y);check(c.numpy(),b['coefficients']);check((H@c).numpy(),y.numpy());energies=[float(sum(c[k]**2 for k in range(8) if k.bit_count()==order)) for order in range(4)];check(energies,b['order_energies']);check(sum(energies),float(y.square().mean()));check(sum(energies[2:])/sum(energies[1:]),b['interaction_variation_fraction']);check(energies[3]/sum(energies[1:]),b['cubic_variation_fraction'])
        choices=(y>0).long();assert [int(choices[0])==0,int(choices[7])==1]==b['consistent_vertices_choose_repeated_value'];assert sum(rows[x['id']]['top1_value_group'] is not None for x in items)==b['top1_in_either_value']
        for name,pred in [('first',bits[:,0]),('last',bits[:,2]),('majority',bits.sum(1)>=2)]:
            check(float((choices==pred).double().mean()),b[name+'_agreement_all']);check(float((choices[1:7]==pred[1:7]).double().mean()),b[name+'_agreement_mixed'])
        for layer in [0,1,2]:
            row=next(x for x in a['layers'] if x['case']==b['case'] and x['format']==b['format'] and x['layer']==layer);Z=torch.stack([saved[x['id']]['layers'][layer]['input'][0,-1].double() for x in items]);L,R,D=[weights[layer][k].double() for k in ['L','R','D']];F=lambda x:((x@L.T)*(x@R.T))@D.T;Y=F(Z);zc=torch.linalg.solve(H,Z);yc=torch.linalg.solve(H,Y)
            ze=[float(sum(zc[k].square().sum() for k in range(8) if k.bit_count()==order)) for order in range(4)];ye=[float(sum(yc[k].square().sum() for k in range(8) if k.bit_count()==order)) for order in range(4)];check(ze,row['input_order_energies']);check(ye,row['write_order_energies']);check(ze[3]/sum(ze[1:]),row['input_cubic_variation_fraction']);check(ye[3]/sum(ye[1:]),row['write_cubic_variation_fraction'])
            # Remove the cubic input pattern and re-evaluate: its output cubic is the mixed part.
            no_cubic=Z-H[:,7,None]*zc[7];mixed=torch.linalg.solve(H,F(no_cubic))[7];carry=F(zc[0]+zc[7])-F(zc[0])-F(zc[7]);check((carry+mixed).numpy(),yc[7].numpy());check(float(carry.square().sum()),row['cubic_carry_energy']);check(float(mixed.square().sum()),row['cubic_mixed_energy']);check(float(2*(carry*mixed).sum()),row['cubic_cross_energy']);check(float(yc[7].square().sum()),row['cubic_target_energy'])
    for summary in a['layer_summaries']:
        rr=[x for x in a['layers'] if x['layer']==summary['layer']];ze=np.sum([x['input_order_energies'] for x in rr],0);ye=np.sum([x['write_order_energies'] for x in rr],0);check(ze[1:]/sum(ze[1:]),summary['input_variation_fractions']);check(ye[1:]/sum(ye[1:]),summary['write_variation_fractions']);den=sum(x['cubic_target_energy'] for x in rr)
        for field,src in [('cubic_carry_over_target','cubic_carry_energy'),('cubic_mixed_over_target','cubic_mixed_energy'),('cubic_cross_over_target','cubic_cross_energy')]:check(sum(x[src] for x in rr)/den,summary[field])
        assert summary['cubes_negative_cubic_cross']==sum(x['cubic_cross_energy']<0 for x in rr)
    assert len(a['behavior'])==8 and len(a['layers'])==24 and r['forward_count']==80 and r['historical_replays']==16
    out=dict(status='passed',source_sha256=sha(__file__),native_sha256=sha(rp),analysis_sha256=sha(ap),checks=checks,max_scaled_discrepancy=maximum,scope='All80 logits and64cube prompts/labels/edges; independentTorch linear-system interpolation versusNumPyWalsh summation. All24 nativeMLP decompositions verified by deleting cubicinput component anddirectly recomputingF, notXORconvolution. Aggregateandheuristicmetrics checked. No nativecausalpatch orTransformer reimplementation.')
    f=V/'repetition_cube_audit.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
