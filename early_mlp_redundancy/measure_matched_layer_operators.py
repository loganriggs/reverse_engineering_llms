"""M104 matched displacements and invariant restricted operators across early layers."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha
from coordinate_gauge_diagnostic import components,summaries
from measure_native_gauge_spectrum import spectrum


def main():
    torch.set_num_threads(1)
    pp=V/'matched_layer_operator_execution.json';execution=json.loads(pp.read_text());assert execution['source_sha256']==sha(__file__)
    for name,h in execution['inputs'].items():assert sha(V/name)==h
    receipt=json.loads((V/'matched_layer_operator_inputs.json').read_text());assert receipt['status']=='passed' and sha(receipt['path'])==receipt['sha256'];states=torch.load(receipt['path'],weights_only=True)['states']
    gauge=json.loads((V/'native_coordinate_gauge.json').read_text());cases=[x for x in gauge['rows'] if x['basis']=='PCA'];previous=json.loads((V/'native_gauge_spectrum.json').read_text())
    rows=[];aggregates=[];maxima={};oracle_checks=0
    def check(name,a,b):
        err=float(np.linalg.norm(a-b)/(1+np.linalg.norm(b)));maxima[name]=max(maxima.get(name,0.),err);assert err<1e-9,(name,err)
    for layer,state in states.items():
        Z=state['z'].double().cuda();B=state['basis'].cuda();L=state['L'].double().cuda();R=state['R'].double().cuda();D=state['D'].double().cuda();total=np.zeros((64,64))
        F=lambda x:((x@L.T)*(x@R.T))@D.T
        J=lambda h,v:((h@L.T)*(v@R.T)+(h@R.T)*(v@L.T))@D.T
        for old in cases:
            case=old['case'];z=Z[old['recipient_index']];delta=Z[old['donor_index']]-z;parts=(delta@B)[:,None]*B.T;d=parts.sum(0);h=z+d/2
            mid=J(h,parts);single=J(z,parts)+F(parts);response=F(z+d)-F(z);interaction=F(d)-F(parts).sum(0)
            arrays=[x.cpu().numpy() for x in [parts,mid,single,response,interaction]];stats=summaries(*arrays)
            check('midpoint_finite',arrays[1].sum(0),arrays[3]);check('individual_finite',arrays[2],(F(z+parts)-F(z)).cpu().numpy());check('interaction',arrays[2].sum(0)+arrays[4],arrays[3])
            K=J(h,B.T);gram=(K@K.T).cpu().numpy();total+=gram;spec=spectrum(gram);sv=torch.linalg.svdvals(K).square().cpu().numpy();check('spectrum',np.array(spec['eigenvalues']),sv)
            if case in [0,63]:
                oracle=components(state['L'].double().numpy(),state['R'].double().numpy(),state['D'].double().numpy(),z.cpu().numpy(),delta.cpu().numpy(),B.cpu().numpy())
                for a,b in zip(arrays,oracle):check('numpy_components',a,b)
                oracle_checks+=1
            if layer==1:
                for name,value in stats.items():
                    if value is not None:check('MLP1_metric_replay',np.array(value),np.array(old[name]))
                check('MLP1_spectrum_replay',np.array(spec['eigenvalues']),np.array(previous['rows'][case]['eigenvalues']))
            rows.append(dict(layer=layer,case=case,recipient_index=old['recipient_index'],donor_index=old['donor_index'],**stats,operator_effective_dimension=spec['effective_rank'],operator_eigenvalues=spec['eigenvalues']))
        rr=[x for x in rows if x['layer']==layer];pooled=spectrum(total/len(cases));r=next(x for x in receipt['rows'] if x['layer']==layer)
        aggregates.append(dict(layer=layer,centered_variance_captured=r['centered_variance_captured'],input_effective_dimension=r['input_effective_dimension'],input_participation_mean=float(np.mean([x['input_participation'] for x in rr])),midpoint_participation_mean=float(np.mean([x['midpoint_participation'] for x in rr])),local_operator_dimension_mean=float(np.mean([x['operator_effective_dimension'] for x in rr])),local_operator_dimension_min=min(x['operator_effective_dimension'] for x in rr),local_operator_dimension_max=max(x['operator_effective_dimension'] for x in rr),pooled_operator=pooled))
        print('LAYER',layer,'DONE',flush=True)
    out=dict(status='passed',source_sha256=sha(__file__),execution_sha256=sha(pp),rows=rows,aggregates=aggregates,max_scaled_errors=maxima,numpy_oracle_conditions=oracle_checks,scope='192cases across3layers. Same token-index differences, layer-specific normalized inputs and PCA64 spans; not matched variance or perturbation energy. All local spectra independently checked by NumPy eigenvalues/TorchSVD. Six prescribed NumPy factor-oracle conditions; not independent full replay. MLP1metrics and spectra reproduce M103. No downstream causal fidelity or semantic tests.')
    f=V/'matched_layer_operators.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n')
    for row in aggregates:print({k:v for k,v in row.items() if k!='pooled_operator'},'POOLED',row['pooled_operator']['effective_rank'],row['pooled_operator']['remaining_energy'],flush=True)
if __name__=='__main__':main()
