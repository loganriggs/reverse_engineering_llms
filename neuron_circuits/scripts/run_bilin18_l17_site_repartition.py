from pathlib import Path
import json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_site_repartition'
@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['source_sha256'];responses=torch.tensor(np.load(path),device='cuda').reshape(18,4096,1152);rng=np.random.default_rng(cfg['partition_seed']);results=[];partitions=[]
    for rep in range(cfg['replicates']):
        perm=rng.permutation(4096);partitions.append(perm.tolist());assert np.array_equal(np.sort(perm),np.arange(4096));covs=[]
        for b in range(4):
            y=responses[:,torch.tensor(perm[b*1024:(b+1)*1024],device='cuda')].double();covs.append(y.transpose(1,2)@y)
        covs=torch.stack(covs);values={str(k):np.zeros((4,4,18)) for k in cfg['output_ranks']}
        for fit in range(4):
            for g in range(18):
                _,U=torch.linalg.eigh(covs[fit,g]);U=U.flip(1)
                for k in cfg['output_ranks']:
                    v=U[:,:k]
                    for test in range(4):values[str(k)][fit,test,g]=float(torch.einsum('ik,ij,jk->',v,covs[test,g],v)/covs[test,g].trace())
        results.append({k:v.tolist() for k,v in values.items()});print('replicate',rep,'complete',flush=True)
    old=json.loads((ART/'bilin18_l17_output_block_transfer.json').read_text())['results'];summary={}
    for k in cfg['output_ranks']:
        obs=np.array([[np.mean(v) for v in old[f'block{i}']['specialist'][str(k)]] for i in range(4)]);sim=[]
        for rep in results:
            m=np.array(rep[str(k)]).mean(2);inside=float(np.diag(m).mean());outside=float(m[~np.eye(4,dtype=bool)].mean());sim.append({'within':inside,'cross':outside,'gap':inside-outside})
        summary[str(k)]={'observed_within':float(np.diag(obs).mean()),'observed_cross':float(obs[~np.eye(4,dtype=bool)].mean()),'observed_gap':float(np.diag(obs).mean()-obs[~np.eye(4,dtype=bool)].mean()),'repartitions':sim}
    whole=json.loads((ART/'bilin18_l17_document_repartition.json').read_text())['summary']; effects={k:float(np.mean([z['cross'] for z in v['repartitions']])-np.mean([z['cross'] for z in whole[k]['repartitions']])) for k,v in summary.items()}; gates={'document_dependence_gain':all(v>=.05 for v in effects.values())}; summary['site_minus_document_transfer']=effects;STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'partitions':partitions,'results':results,'summary':summary,'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
