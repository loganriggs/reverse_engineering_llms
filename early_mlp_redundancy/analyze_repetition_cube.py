"""M108 observed position effects, exact cube interactions and native bilinear composition."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha
from repetition_cube_algebra import BITS,H,coefficients,reconstruct,order_energies,bilinear_coefficients


def main():
    torch.set_num_threads(1);rp=V/'repetition_cube_native.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(V.parent/'probe_repetition_cube.py');pp=V/'repetition_cube_protocol.json';p=json.loads(pp.read_text());assert r['protocol_sha256']==sha(pp);assert sha(r['states_path'])==r['states_sha256'];saved=torch.load(r['states_path'],weights_only=True)
    wp=V/'matched_layer_operator_inputs.json';w=json.loads(wp.read_text());assert sha(w['path'])==w['sha256'] and w['checkpoint_sha256']==r['checkpoint_sha256'];weights=torch.load(w['path'],weights_only=True)['states'];observed={x['id']:x for x in r['rows']};behavior=[];layers=[];maximum=0.;maxnative=0.
    for case in range(4):
        for fmt in ['semicolon','comma']:
            items=sorted([x for x in p['rows'] if x.get('case')==case and x.get('format')==fmt],key=lambda x:x['vertex']);assert len(items)==8 and [x['bits'] for x in items]==BITS.tolist();y=np.array([observed[x['id']]['canonical_margin'] for x in items]);c=coefficients(y);assert np.allclose(reconstruct(c),y,atol=1e-10);energy=order_energies(c);assert np.isclose(np.mean(y*y),sum(energy));var=sum(energy[1:]);choice=(y>0).astype(int);assert not np.any(y==0)
            row=dict(case=case,format=fmt,key=items[0]['key'],values=items[0]['values'],canonical_margins=y.tolist(),coefficients=c.tolist(),order_energies=energy,interaction_variation_fraction=sum(energy[2:])/var if var else None,cubic_variation_fraction=energy[3]/var if var else None,consistent_vertices_choose_repeated_value=[int(choice[0])==0,int(choice[7])==1],top1_in_either_value=sum(observed[x['id']]['top1_value_group'] is not None for x in items))
            for name,pred in [('first',BITS[:,0]),('last',BITS[:,2]),('majority',(BITS.sum(1)>=2).astype(int))]:
                row[name+'_agreement_all']=float(np.mean(choice==pred));row[name+'_agreement_mixed']=float(np.mean(choice[1:7]==pred[1:7]))
            behavior.append(row)
            for layer in [0,1,2]:
                z=np.stack([saved[x['id']]['layers'][layer]['input'][0,-1].double().numpy() for x in items]);native=np.stack([saved[x['id']]['layers'][layer]['write'][0,-1].double().numpy() for x in items]);L,R,D=[weights[layer][k].double().numpy() for k in ['L','R','D']];y64=((z@L.T)*(z@R.T))@D.T;err=float(np.linalg.norm(y64-native)/max(np.linalg.norm(native),1));maxnative=max(maxnative,err);assert err<1e-5
                zc=coefficients(z);yc=coefficients(y64);conv,carry,mixed=bilinear_coefficients(L,R,D,zc);err=float(np.linalg.norm(conv-yc)/max(np.linalg.norm(yc),1));maximum=max(maximum,err);assert err<1e-9;assert np.allclose(carry+mixed,yc[7],rtol=1e-9,atol=1e-7)
                zenergy=order_energies(zc);yenergy=order_energies(yc);cross=float(2*(carry*mixed).sum());target=float(np.square(yc[7]).sum());ce=float(np.square(carry).sum());me=float(np.square(mixed).sum());assert np.isclose(ce+me+cross,target,rtol=1e-9,atol=1e-5)
                layers.append(dict(case=case,format=fmt,layer=layer,input_order_energies=zenergy,write_order_energies=yenergy,input_cubic_variation_fraction=zenergy[3]/sum(zenergy[1:]),write_cubic_variation_fraction=yenergy[3]/sum(yenergy[1:]),cubic_carry_energy=ce,cubic_mixed_energy=me,cubic_cross_energy=cross,cubic_target_energy=target,native_float64_vs_float32_relative_error=float(np.linalg.norm(y64-native)/max(np.linalg.norm(native),1))))
    summaries=[]
    for layer in [0,1,2]:
        rr=[x for x in layers if x['layer']==layer];ze=np.sum([x['input_order_energies'] for x in rr],0);ye=np.sum([x['write_order_energies'] for x in rr],0);target=sum(x['cubic_target_energy'] for x in rr);summaries.append(dict(layer=layer,input_variation_fractions=(ze[1:]/ze[1:].sum()).tolist(),write_variation_fractions=(ye[1:]/ye[1:].sum()).tolist(),cubic_carry_over_target=sum(x['cubic_carry_energy'] for x in rr)/target,cubic_mixed_over_target=sum(x['cubic_mixed_energy'] for x in rr)/target,cubic_cross_over_target=sum(x['cubic_cross_energy'] for x in rr)/target,cubes_negative_cubic_cross=sum(x['cubic_cross_energy']<0 for x in rr)))
    out=dict(status='completed',source_sha256=sha(__file__),helper_sha256=sha(V.parent/'repetition_cube_algebra.py'),native_sha256=sha(rp),protocol_sha256=sha(pp),weights_receipt_sha256=sha(wp),behavior=behavior,layers=layers,layer_summaries=summaries,max_convolution_relative_error=maximum,max_native_float64_vs_float32_relative_error=maxnative,scope='Uniform3bit finiteinput-cube descriptive decomposition. Margin usescanonical rawlogitdifference, avoidinglog-softmax constant andcasevariant pooling nonlinearity. NativeMLPwrite spectra usefloat64 polynomial atcapturednativefloat32 normalizedinputs, checkedagainstnativefloat32 writes. Higherorder promptinteractions neednot belocalized toMLPs. Cubiccarry/mixedsplit isfunctionalinput-coefficient composition, notnativechannelcancellation. No fit orcomponent intervention.')
    dest=V/'repetition_cube_analysis.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('BEHAVIOR',[(x['case'],x['format'],x['first_agreement_mixed'],x['last_agreement_mixed'],x['majority_agreement_mixed'],x['interaction_variation_fraction']) for x in behavior]);print('LAYERS',summaries)
if __name__=='__main__':main()
