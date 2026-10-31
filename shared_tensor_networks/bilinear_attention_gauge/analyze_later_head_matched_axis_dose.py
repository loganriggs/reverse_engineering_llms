"""Case-level paired bootstrap for frozen later-head matched-dose interventions."""

import collections
import json

import numpy as np

from flagship_transfer import ART


def run(seed=302031, draws=10000):
    raw = json.loads((ART / 'later_head_matched_axis_dose.json').read_text())
    records = raw['records']
    assert len(records) == 38 * 4 * 8
    grouped = collections.defaultdict(list)
    for row in records:
        assert 0 < row['candidate_fraction'] <= 1 + 1e-6
        assert 0 < row['random_fraction'] <= 1 + 1e-6
        assert row['target_post_c_proj_l2'] > 1
        grouped[(row['panel'], row['family'], row['variant'],
                 row['base_prompt'])].append(row)
    cases = []
    for (panel, family, variant, prompt), rows in grouped.items():
        assert len(rows) == 8 and len({r['random_axis'] for r in rows}) == 8
        candidate = np.array([r['candidate_signed_per_dose'] for r in rows])
        control = np.array([r['random_signed_per_dose'] for r in rows])
        cases.append(dict(panel=panel, family=family, variant=variant,
            base_prompt=prompt, candidate_mean=float(candidate.mean()),
            random_mean=float(control.mean()),
            mean_difference=float((candidate-control).mean()),
            exceeds_all_eight=bool(np.all(candidate > control)),
            median_target_post_c_proj_l2=float(np.median([
                r['target_post_c_proj_l2'] for r in rows]))))
    rng = np.random.default_rng(seed)
    summary = {}
    for family in ('bracket_closure', 'subject_verb_agreement'):
        for variant in raw['variants']:
            chosen = [c for c in cases if c['family'] == family and
                      c['variant'] == variant]
            assert len(chosen) == (18 if family == 'bracket_closure' else 20)
            by_panel = {p: [c for c in chosen if c['panel'] == p]
                        for p in ('lexical_holdout', 'format_shift')}
            boot = np.zeros(draws)
            for panel, subset in by_panel.items():
                values = np.array([c['mean_difference'] for c in subset])
                indices = rng.integers(len(values), size=(draws, len(values)))
                boot += values[indices].sum(axis=1)
            boot /= len(chosen)
            candidate_mean = float(np.mean([c['candidate_mean'] for c in chosen]))
            random_mean = float(np.mean([c['random_mean'] for c in chosen]))
            summary[f'{family}__{variant}'] = dict(
                n_cases=len(chosen), n_lexical=len(by_panel['lexical_holdout']),
                n_format=len(by_panel['format_shift']),
                candidate_signed_margin_per_residual_l2=candidate_mean,
                random_signed_margin_per_residual_l2=random_mean,
                candidate_over_random_ratio=candidate_mean/random_mean,
                mean_case_difference=float(np.mean([
                    c['mean_difference'] for c in chosen])),
                paired_case_bootstrap_95=np.quantile(boot, [.025, .975]).tolist(),
                candidate_above_random_mean=sum(c['mean_difference'] > 0
                                                for c in chosen),
                candidate_above_all_eight=sum(c['exceeds_all_eight']
                                              for c in chosen),
                median_matched_post_c_proj_l2=float(np.median([
                    c['median_target_post_c_proj_l2'] for c in chosen])))
    result = dict(description='Finite full-model correct-minus-foil margin response divided by exactly matched post-c_proj residual dose. Prompt is bootstrap unit; resample separately within lexical and format panels. Intervals condition on this single trained model and on templated prompt families.', seed=seed,
                  draws=draws, cases=cases, summary=summary)
    path = ART / 'later_head_matched_axis_dose_analysis.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    for name, row in summary.items():
        print(name, 'ratio', round(row['candidate_over_random_ratio'], 3),
              'CI', [round(v, 7) for v in row['paired_case_bootstrap_95']],
              'wins', row['candidate_above_random_mean'],
              'maxwins', row['candidate_above_all_eight'])
    return result


if __name__ == '__main__':
    run()
