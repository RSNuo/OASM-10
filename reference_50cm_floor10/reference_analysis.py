"""Evidence tables for the adopted 50 cm product reference (ma_floor10_seed42).

Produces reference_analysis.json and adds a `product_reference` block to
analysis/results.json (depths.50) without altering any existing entry. Also
independently recomputes the policy-matched cascade contrast (B minus A) of
depth50_input_factorial from its stored predictions, because the manuscripts cite it.
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; REV = HERE.parent
sys.path.insert(0, str(REV))
import analyze_results as ar
from analyze_results import statistics, site_statistics, site_summary, paired_seed_bootstrap, read, read_reference
from oasm.inference import valid_satellite_rows

D = 50
REF = REV / 'models/50cm/ma_floor10_seed42'
BASE = REV / 'models/50cm/ma_seed42'
FACT = REV / 'depth50_input_factorial'


def fold_table():
    rows = []
    for k in range(5):
        m = json.loads((REF / f'outer{k}/manifest.json').read_text(encoding='utf-8'))
        z = pd.read_parquet(REF / f'outer{k}/oof_predictions.parquet'); b = pd.read_parquet(BASE / f'outer{k}/oof_predictions.parquet')
        assert set(z.record_id) == set(b.record_id)
        s = statistics(z); sb = statistics(b)
        rows.append({'fold': k, 'selected_epochs': int(m['selected_epochs']), 'tried_epochs': len(m['selection_trace']),
                     'refit_epochs': int(m['refit_epochs']), 'identity_reuse': bool(m['identity_reuse']),
                     'records': s['records'], 'sites': s['sites'], 'rmse': s['rmse'], 'bias': s['bias'], 'r2': s['r2'],
                     'selected_duration_rmse': sb['rmse'], 'selected_duration_bias': sb['bias'], 'selected_duration_r2': sb['r2'],
                     'gradient_sites': len(m['gradient_site_ids']), 'training_seconds': m['training_seconds']})
    return rows


def eligible(frame, d):
    raw = pd.read_parquet(REV / f'data/all_{d}cm.parquet')
    ok = raw.loc[valid_satellite_rows(raw), 'record_id']
    return frame[frame.record_id.isin(set(ok))].copy()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    out = {'reference_run': 'ma_floor10_seed42', 'previous_run': 'ma_seed42', 'depth_cm': D}
    out['folds'] = fold_table()
    # Run-level metrics, all retained and output-eligible records (50 cm: identical populations).
    out['metrics'] = {}
    for split in ['oof', 'test']:
        z = read_reference(D, split); b = read(D, 'ma', 42, split)
        out['metrics'][split] = {'all_retained': statistics(z), 'output_eligible': {**statistics(eligible(z, D)), **site_summary(site_statistics(eligible(z, D)))},
                                 'selected_duration_all_retained': statistics(b)}
    # Seed-42 paired policy effect (floor10 minus selected duration), same bootstrap as analyze_results.
    out['seed42_paired_floor10_minus_selected'] = {split: paired_seed_bootstrap([(read_reference(D, split), read(D, 'ma', 42, split))]) for split in ['oof', 'test']}
    # Three-seed policy effects are taken from the frozen strategy experiment, not recomputed here.
    strat = json.loads((REV / 'training_strategy_review/results.json').read_text(encoding='utf-8'))
    out['three_seed_policy_effects_source'] = 'training_strategy_review/results.json'
    out['three_seed_policy_effects'] = {split: strat[split].get('warmup_floor10', strat[split]).get('50', strat[split].get('warmup_floor10')) if isinstance(strat[split], dict) else None for split in ['oof', 'test']}
    # Independent recomputation of the policy-matched cascade contrast (factorial B minus A).
    fact = json.loads((FACT / 'results.json').read_text(encoding='utf-8'))
    ar.BOOT_SEED = int(fact['protocol']['bootstrap']['seed']) if 'seed' in fact['protocol']['bootstrap'] else ar.BOOT_SEED
    contrasts = {}
    for split in ['oof', 'test']:
        pairs = []
        for s in [42, 7, 555]:
            a = pd.read_parquet(FACT / f'models/A_seed{s}/{split}_predictions.parquet'); b = pd.read_parquet(FACT / f'models/B_seed{s}/{split}_predictions.parquet')
            pairs.append((b, a))
        mine = paired_seed_bootstrap(pairs); theirs = fact[split]['contrasts']['B-A']
        contrasts[split] = {'recomputed': {k: mine[k] for k in ['mean_rmse_difference', 'rmse_ci95', 'site_equal_rmse_difference', 'site_equal_rmse_ci95', 'anomaly_r_difference', 'anomaly_r_ci95', 'per_seed_rmse_differences']},
                            'reported': {k: theirs[k] for k in ['mean_rmse_difference', 'rmse_ci95', 'site_equal_rmse_difference', 'site_equal_rmse_ci95', 'anomaly_r_difference', 'anomaly_r_ci95', 'per_seed_rmse_differences']}}
        for k in ['mean_rmse_difference', 'anomaly_r_difference']:
            assert abs(mine[k] - theirs[k]) < 1e-9, (split, k, mine[k], theirs[k])
        for k in ['rmse_ci95', 'anomaly_r_ci95']:
            assert np.allclose(mine[k], theirs[k], atol=2e-4), (split, k, mine[k], theirs[k])
        # Configuration A at seed 42 must be the adopted reference exactly.
        a42 = pd.read_parquet(FACT / f'models/A_seed42/{split}_predictions.parquet').set_index('record_id'); z = read_reference(D, split).set_index('record_id')
        gap = float(np.abs(a42.reindex(z.index).prediction.to_numpy() - z.prediction.to_numpy()).max())
        assert gap < 1e-6, gap  # fold files are bit-identical; run-level means differ only by summation order
        contrasts[split]['A_seed42_identical_to_reference'] = {'fold_files': 'bit-identical', 'run_level_max_abs_diff': gap}
        contrasts[split]['configuration_means'] = {c: {'rmse_mean_of_seeds': float(np.mean([statistics(pd.read_parquet(FACT / f'models/{c}_seed{s}/{split}_predictions.parquet'))['rmse'] for s in [42, 7, 555]]))} for c in ['A', 'B']}
    out['policy_matched_cascade_contrast_B_minus_A'] = {'source': 'depth50_input_factorial (common floor-10 policy, fixed nested upstream inputs, seeds 42/7/555)',
                                                        'bootstrap_seed': ar.BOOT_SEED, **contrasts}
    # Moisture strata and ERA5 comparison already produced by quality_summary / compare_products.
    q = json.loads((REV / 'analysis/quality.json').read_text(encoding='utf-8'))['depths']['50']
    out['moisture_bins'] = q['moisture_bins']; out['all_eligible'] = q['all_eligible']; out['common_sites'] = q['common_sites']
    cmp = json.loads((REV / 'validation/product_comparisons/results.json').read_text(encoding='utf-8'))
    out['era5_layer3'] = {k: v for k, v in cmp['depth']['50'].items()}
    (HERE / 'reference_analysis.json').write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf-8')
    # Patch analysis/results.json: add, never replace.
    path = REV / 'analysis/results.json'; res = json.loads(path.read_text(encoding='utf-8'))
    res['product_reference_runs'] = {'5': 'ma_seed42', '20': 'ma_seed42', '50': 'ma_floor10_seed42'}
    res['depths']['50']['product_reference'] = {'run': 'ma_floor10_seed42', 'policy': 'warmup_floor10 refit (max(selected_epochs,10))', 'decision': 'development OOF only; see models/product_reference.json',
                                                'models': {'oof': out['metrics']['oof']['all_retained'], 'test': out['metrics']['test']['all_retained']},
                                                'folds': out['folds'], 'seed42_paired_floor10_minus_selected': out['seed42_paired_floor10_minus_selected'],
                                                'note': 'paired_controls and models entries above keep the common selected-duration policy for every architecture'}
    path.write_text(json.dumps(res, indent=2, allow_nan=False), encoding='utf-8')
    for r in out['folds']:
        print('fold %d sel %d/%d refit %d reuse %s: rmse %.4f (was %.4f) bias %+.4f r2 %.4f' % (r['fold'], r['selected_epochs'], r['tried_epochs'], r['refit_epochs'], r['identity_reuse'], r['rmse'], r['selected_duration_rmse'], r['bias'], r['r2']))
    for split in ['oof', 'test']:
        s = out['seed42_paired_floor10_minus_selected'][split]
        print(split, 'seed42 floor10-minus-selected: %+.5f CI [%+.5f, %+.5f]; anomaly r %+.4f CI [%+.4f, %+.4f]' % (s['mean_rmse_difference'], *s['rmse_ci95'], s['anomaly_r_difference'], *s['anomaly_r_ci95']))
        c = contrasts[split]['recomputed']
        print(split, 'factorial B-A recomputed: %+.5f CI [%+.5f, %+.5f]; equal-site %+.5f [%+.5f, %+.5f]; anomaly r %+.4f [%+.4f, %+.4f]' % (c['mean_rmse_difference'], *c['rmse_ci95'], c['site_equal_rmse_difference'], *c['site_equal_rmse_ci95'], c['anomaly_r_difference'], *c['anomaly_r_ci95']))
    print('REFERENCE ANALYSIS COMPLETE')


if __name__ == '__main__':
    main()
