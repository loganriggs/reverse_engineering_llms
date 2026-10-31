"""Few-shot whole-task transfer from label-blind internal-head dictionaries."""

import json
from itertools import combinations

import numpy as np

from analyze_dense_weak_rule_graph import RAW
from dual_joint_function_canonical import HERE
from train_dense_four_count_weak import SIGNS


MEASURES=('weight_only','empirical_usage')
PRIORS=('svd','all_pair_edit_l1','no_direct_D_flip_edit_l1')
PAIRS=tuple(combinations(range(4),2))
N_EPISODES=20
N_TRAIN=64
N_TEST=20000


def raw_targets(keys):
    counts=SIGNS[keys].sum(axis=1)
    primitives=counts>0
    parity=np.stack([primitives[:,a]!=primitives[:,b] for a,b in PAIRS],axis=1)
    return primitives,parity


def candidates(dictionary,keys):
    centered=dictionary-dictionary.mean(axis=1,keepdims=True)
    activations=centered[:,keys].sum(axis=2).T
    primitive=activations>0
    parity=np.stack([(activations[:,a]*activations[:,b])>0
                     for a,b in PAIRS],axis=1)
    return primitive,parity


def select_and_score(train_candidates,test_agreement,train_truth):
    """Use only train labels for candidate axis/pair and polarity selection."""
    scores=[]
    for task in range(train_truth.shape[1]):
        agreements=np.mean(train_candidates==train_truth[:,task,None],axis=0)
        choices=np.maximum(agreements,1-agreements)
        chosen=int(np.argmax(choices))
        polarity=bool(agreements[chosen]>=.5)
        scores.append(dict(task=task,selected_candidate=chosen,
            train_accuracy=float(choices[chosen]),
            test_accuracy=float(test_agreement[chosen,task] if polarity
                                else 1-test_agreement[chosen,task])))
    return scores


def random_rotation(dictionary,rng):
    raw=rng.standard_normal((4,4))
    q,r=np.linalg.qr(raw)
    return (q*np.sign(np.diag(r))[None,:])@dictionary


def run():
    factors=np.load(HERE/'dense_weak_unlabeled_head_factors.npz')
    rows=[];coverage_by_seed_measure={}
    for seed in range(10):
        rng=np.random.default_rng(20261110+seed)
        test_keys=rng.integers(0,16,size=(N_TEST,7))
        test_primitive,test_parity=raw_targets(test_keys)
        train_keys=[rng.integers(0,16,size=(N_TRAIN,7))
                    for _ in range(N_EPISODES)]
        train_truth=[raw_targets(keys) for keys in train_keys]
        for measure in MEASURES:
            bases={prior:factors[f's{seed}_{measure}_{prior}_dictionary']
                   for prior in PRIORS}
            baseline=bases['svd']
            for draw in range(32):
                bases[f'random_rotation_{draw}']=random_rotation(baseline,rng)
            projection=baseline.T@np.linalg.pinv(baseline@baseline.T)@baseline
            coverage=np.diag(RAW@projection@RAW.T)
            coverage_by_seed_measure[f's{seed}_{measure}']=coverage.tolist()
            for prior,dictionary in bases.items():
                test_candidates=candidates(dictionary,test_keys)
                test_agreements=tuple(np.mean(c[:, :, None]==y[:, None, :],axis=0)
                                      for c,y in zip(test_candidates,
                                                     (test_primitive,test_parity)))
                for episode,keys in enumerate(train_keys):
                    train_candidates=candidates(dictionary,keys)
                    for family,train_c,test_agreement,train_y in zip(
                        ('primitive_one_axis','xor_one_pair'),
                        train_candidates,test_agreements,train_truth[episode]):
                        scores=select_and_score(train_c,test_agreement,train_y)
                        rows.append(dict(seed=seed,measure=measure,
                            prior=prior,episode=episode,family=family,
                            selected_candidates=[x['selected_candidate'] for x in scores],
                            train_accuracies=[x['train_accuracy'] for x in scores],
                            test_accuracies=[x['test_accuracy'] for x in scores]))
        print('seed',seed,'complete',flush=True)
    summary={}
    for measure in MEASURES:
        summary[measure]={}
        for prior in (*PRIORS,'random_rotation'):
            part=[row for row in rows if row['measure']==measure and
                  (row['prior'].startswith('random_rotation') if prior=='random_rotation'
                   else row['prior']==prior)]
            summary[measure][prior]={family:{
                'mean_test_accuracy':float(np.mean([row['test_accuracies'] for row in part
                                                  if row['family']==family])),
                'mean_train_accuracy':float(np.mean([row['train_accuracies'] for row in part
                                                   if row['family']==family]))}
                for family in ('primitive_one_axis','xor_one_pair')}
            summary[measure][prior]['mean_raw_bit_subspace_coverage']=float(np.mean([
                np.mean(coverage_by_seed_measure[f's{seed}_{measure}'])
                for seed in range(10)]))
    result=dict(description='Twenty 64-label episodes per seed on four unseen raw-bit majority tasks and six unseen XOR-of-majorities tasks; atom/pair and polarity selected on training labels only, tested on 20,000 fresh sequences. Internal head dictionaries were fitted without these task labels or output readout.',
        n_episodes=N_EPISODES,n_train=N_TRAIN,n_test=N_TEST,
        coverage_by_seed_measure=coverage_by_seed_measure,rows=rows,summary=summary)
    (HERE/'dense_weak_new_task_transfer.json').write_text(
        json.dumps(result,separators=(',',':'))+'\n')
    for measure,priors in summary.items():
        for prior,results in priors.items():
            print(measure,prior,'one',round(results['primitive_one_axis'][
                'mean_test_accuracy'],4),'xor',round(results['xor_one_pair'][
                'mean_test_accuracy'],4),'coverage',round(results[
                'mean_raw_bit_subspace_coverage'],5))


if __name__=='__main__':run()
