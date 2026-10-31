"""Whole-edit-type holdouts for the faithful part of the 36-head scan."""

import json

import numpy as np

from flagship_joint_edit_geometry import (ART, axis_match, concentration,
                                          covariances, fit, normalize)


def top_axis_fraction(cov,rotation):
    transformed=rotation.T@cov@rotation
    return float(np.max(np.diag(transformed))/np.trace(transformed))


def run():
    saved=json.loads((ART/'all_head_joint_edit_geometry.json').read_text())
    factors=np.load(ART/'all_head_joint_edit_geometry_factors.npz')
    data=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    train_types=np.array([r['type'] for r in meta['panels']['train']])
    val_types=np.array([r['type'] for r in meta['panels']['validation']])
    names=saved['types']
    # Selection uses the fitting panel's response coverage, not validation
    # labels or the path/answer results for any head.
    eligible=[r for r in saved['rows'] if r['mode']=='equal_type' and
              r['rank4_response_coverage']['train']>.75]
    heads=[(r['layer'],r['head']) for r in eligible]
    print('train-coverage-selected heads',heads,flush=True)
    rows=[]
    for layer,head in heads:
        for mode_index,mode in enumerate(('equal_type','raw_energy')):
            prefix=f'l{layer}_h{head}_{mode}'
            basis=factors[prefix+'_rank4_basis']
            full=factors[prefix+'_rotation']
            train=data[f'layer{layer}__train'][:,head,:].astype(np.float64)@basis
            val=data[f'layer{layer}__validation'][:,head,:].astype(np.float64)@basis
            train_cov=covariances(train,train_types,names)
            val_cov=covariances(val,val_types,names)
            for i,name in enumerate(names):
                remaining=[j for j in range(len(names)) if j!=i]
                fitted,_=fit(normalize(train_cov[remaining],mode),
                             20261118+layer*10000+head*100+mode_index*10+i)
                heldout=val_cov[i]/max(np.trace(val_cov[i]),1e-30)
                rows.append(dict(layer=layer,head=head,mode=mode,
                    heldout_type=name,
                    leaveout_to_full_axis_alignment=axis_match(fitted,full),
                    heldout_type_oracle_axis_fraction=top_axis_fraction(heldout,fitted),
                    full_fit_type_oracle_axis_fraction=top_axis_fraction(heldout,full),
                    pca_type_oracle_axis_fraction=top_axis_fraction(heldout,np.eye(4))))
    result=dict(description='Input-type holdout from orthogonal joint diagonalization, restricted to heads selected by >75% train-panel rank-four response-energy coverage. Oracle-axis fractions use validation edit-type labels only after fit; they do not show a type-blind axis selection rule.',
        selection_rule='rank-four train response coverage >0.75',heads=heads,
        types=names,rows=rows)
    (ART/'all_head_joint_edit_type_holdout.json').write_text(json.dumps(result,indent=2)+'\n')
    for mode in ('equal_type','raw_energy'):
        part=[r for r in rows if r['mode']==mode]
        print(mode,'axis alignment',np.mean([r['leaveout_to_full_axis_alignment']
                                             for r in part]),
              'heldout fraction',np.mean([r['heldout_type_oracle_axis_fraction']
                                           for r in part]),
              'pca fraction',np.mean([r['pca_type_oracle_axis_fraction']
                                       for r in part]))


if __name__=='__main__':run()
