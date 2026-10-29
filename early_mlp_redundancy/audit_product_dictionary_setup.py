"""M95 state, optimizer, split and schedule gates before any calibration or fit."""
import json
import torch
from fit_pca_encoder import A, V, sha
from product_dictionary_featurizer import ProductDictionaryFeaturizer


def read(name):
    return json.loads((V / name).read_text())


def main():
    torch.set_num_threads(1)
    design = read('product_dictionary_featurizer_design.json')
    for name, digest in design['source_hashes'].items():
        assert sha(V / name) == digest
    manifest = read('product_dictionary_initial_states.json')
    assert manifest['design_sha256'] == sha(V / 'product_dictionary_featurizer_design.json')
    assert manifest['runtime_sha256'] == sha(V.parent / 'product_dictionary_featurizer.py')
    init = read('product_dictionary_initialization_audit.json')
    native = read('product_dictionary_native_checks.json')
    gradient = read('product_dictionary_gradient_checks.json')
    diagnostic = read('product_dictionary_gradient_diagnostic.json')
    assert init['status'] == native['status'] == diagnostic['status'] == 'passed'
    assert init['manifest_sha256'] == native['manifest_sha256'] == sha(V / 'product_dictionary_initial_states.json')
    for result, source in [(init, 'audit_product_dictionary_initialization.py'),
                           (native, 'check_product_dictionary_native.py'),
                           (gradient, 'check_product_dictionary_gradients.py'),
                           (diagnostic, 'diagnose_product_dictionary_gradients.py')]:
        assert result['source_sha256'] == sha(V.parent / source)
        if 'runtime_sha256' in result:
            assert result['runtime_sha256'] == sha(V.parent / 'product_dictionary_featurizer.py')
        if 'loss_source_sha256' in result:
            assert result['loss_source_sha256'] == sha(V.parent / 'bilinear_input_causal_loss.py')
    assert {(r['seed'], r['scope']) for r in gradient['rows'] if not r['passed']} == {(71, 'task_parameters'), (72, 'task_parameters')}
    assert len(diagnostic['records']) == 2
    assert all(all(r['modes_pass'].values()) and r['analytic_pullback_relative_error'] <= 1e-5 for r in diagnostic['records'])
    for name, digest in read('product_dictionary_gradient_diagnostic_protocol.json')['references'].items():
        assert sha(V / name) == digest
    inventory = read('input_training_inventory.json')
    assert sha(inventory['path']) == inventory['sha256']
    inputs = torch.load(inventory['path'], weights_only=True, map_location='cpu')['inputs']
    pairs = {k.rsplit('|', 1)[0] for k in inputs}
    families = {x['family'] for x in inputs.values()}
    assert len(inputs) == 140 and len(pairs) == 70 and len(families) == 11
    excluded = set(read('diverse_supervision_split.json')['evaluation_pair_ids'])
    whole = read('whole_task_native_protocol.json')['pairs']
    assert len(excluded) == 14 and not pairs & excluded
    assert not families & {p['family'] for p in whole}
    train_prompts = {tuple(x['ids'].reshape(-1).tolist()) for x in inputs.values()}
    assert not train_prompts & {tuple(ids) for p in whole for ids in p['token_ids']}
    protocol = read('balanced_response_training_protocol.json')
    old = read('bilinear_input_joint_training_protocol.json')
    assert old['learning_rate'] == 1e-4 and old['steps'] == 132
    rows = []
    for entry in manifest['models']:
        assert sha(entry['path']) == entry['sha256']
        state = torch.load(entry['path'], weights_only=True, map_location='cpu')['state']
        f = ProductDictionaryFeaturizer(state)
        assert set(dict(f.named_parameters())) == {'coefficients'}
        assert set(dict(f.named_buffers())) == {'left', 'right', 'rms', 'dictionary', 'offset'}
        assert all(torch.equal(v, state[k]) for k, v in f.state_dict().items())
        optimizer = torch.optim.Adam(f.parameters(), lr=1e-4)
        group = optimizer.param_groups[0]
        assert len(optimizer.state) == 0
        assert len(group['params']) == 1 and group['params'][0] is f.coefficients
        assert group['betas'] == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0
        assert not group['amsgrad'] and not group['maximize']
        schedule = protocol['schedules'][str(entry['seed'])]
        assert len(schedule['natural_documents']) == len(schedule['task_updates']) == 66
        assert all(0 <= d < 32 for d in schedule['natural_documents'])
        for assignment in schedule['task_updates']:
            assert assignment['pair_id'] in pairs and assignment['pair_id'] not in excluded
            assert inputs[assignment['pair_id'] + '|0']['family'] == assignment['family']
        rows.append({'seed': entry['seed'], 'parameters': f.coefficients.numel(),
                     'optimizer_defaults': {k: v for k, v in group.items() if k != 'params'},
                     'natural_updates': 66, 'task_updates': 66,
                     'scheduled_unique_pairs': len({x['pair_id'] for x in schedule['task_updates']})})
    out = {'status': 'passed', 'source_sha256': sha(__file__),
           'references': {n: sha(V / n) for n in ['product_dictionary_featurizer_design.json',
               'product_dictionary_initial_states.json', 'product_dictionary_initialization_audit.json',
               'product_dictionary_native_checks.json', 'product_dictionary_gradient_checks.json',
               'product_dictionary_gradient_diagnostic.json', 'balanced_response_training_protocol.json',
               'input_training_inventory.json', 'diverse_supervision_split.json', 'whole_task_native_protocol.json']},
           'rows': rows, 'optimizer_updates': 0,
           'scope': 'Initial state/frozen buffer inventory, fresh Adam defaults matching M87, exact M60 schedules and training exclusions. Original two small-step failures preserved with bounded same-direction diagnostic. Actual training execution must still be audited after fitting.'}
    path = V / 'product_dictionary_setup_audit.json'
    assert not path.exists()
    path.write_text(json.dumps(out, indent=2) + '\n')
    print('SETUP PASSED', rows, flush=True)


if __name__ == '__main__':
    main()
