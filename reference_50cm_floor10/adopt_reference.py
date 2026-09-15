"""Adopt the minimum-duration (floor-10) seed-42 fit as the archived 50 cm product reference.

Decision basis (development evidence only): in training_strategy_review the three-seed paired
outer-OOF RMSE difference floor10 minus selected-duration at 50 cm is -0.01197 m3/m3
[-0.01840, -0.00549]; at 5 and 20 cm the test intervals include zero, so those references are
unchanged. Test metrics of every policy had already been computed and reported (SciData v13 /
ISPRS v11) before this adoption; they are disclosed, not the selection criterion.

Nothing is retrained here. Folds whose original selection already reached ten epochs reuse the
identical seed-42 weights; the two short folds (outer2: 2 epochs, outer3: 7 epochs) take the
verified floor-10 refits. The nested upstream cascade inputs and all architecture controls are
unchanged. Weights are replayed on the archived fold inputs before anything is written.
"""
from pathlib import Path
import hashlib, json, shutil, sys
from datetime import datetime, timezone
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; REV = HERE.parent
sys.path.insert(0, str(REV))
from oasm.training import load_model, predict_tensors, tensors

SRC = REV / 'training_strategy_review/models/50cm/warmup_floor10_seed42'
BASE = REV / 'models/50cm/ma_seed42'
DEST = REV / 'models/50cm/ma_floor10_seed42'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def stats(z):
    e = z.prediction - z.y
    return {'n': int(len(z)), 'sites': int(z.physical_site_id.nunique()), 'rmse': float(np.sqrt(np.mean(e ** 2))),
            'bias': float(e.mean()), 'ubrmse': float(np.std(e)), 'r2': float(1 - np.sum(e ** 2) / np.sum((z.y - z.y.mean()) ** 2))}


def frozen_hash(protocol, rel):
    hashes = protocol['sources_sha256']
    for key in (rel, rel.replace('/', '\\')):
        if key in hashes:
            return hashes[key]
    raise KeyError(rel)


def main():
    sys.stdout.reconfigure(encoding='utf-8'); torch.set_num_threads(8); torch.set_float32_matmul_precision('high')
    # Frozen-source guard: the strategy experiment's protocol hashes must still describe the baseline.
    protocol = json.loads((REV / 'training_strategy_review/protocol.json').read_text(encoding='utf-8'))
    for k in range(5):
        for name in ['manifest.json', 'model.pt', 'preprocessor.joblib', 'oof_predictions.parquet', 'test_predictions.parquet']:
            rel = f'models/50cm/ma_seed42/outer{k}/{name}'
            assert sha(BASE / f'outer{k}' / name) == frozen_hash(protocol, rel), ('baseline artifact changed since the strategy freeze', rel)
    folds = []; replay = []
    for k in range(5):
        sman = json.loads((SRC / f'outer{k}/manifest.json').read_text(encoding='utf-8'))
        bman = json.loads((BASE / f'outer{k}/manifest.json').read_text(encoding='utf-8'))
        assert sman['strategy'] == 'warmup_floor10' and sman['selected_epochs'] == bman['selected_epochs']
        weights = BASE / f'outer{k}' if sman['identity_reuse'] else SRC / f'outer{k}'
        dest = DEST / f'outer{k}'; dest.mkdir(parents=True, exist_ok=True)
        for name in ['model.pt', 'preprocessor.joblib']:
            shutil.copy2(weights / name, dest / name)
        for name in ['oof_predictions.parquet', 'test_predictions.parquet']:
            shutil.copy2(SRC / f'outer{k}/{name}', dest / name)
        meta = dict(bman)
        extra = float(sman.get('additional_training_seconds', 0.0))
        meta.update({
            'product_role': 'archived 50 cm product reference (adopted 2026-09-14)',
            'refit_policy': 'warmup_floor10: refit on all permitted sites for max(selected_epochs,10) epochs with the original preprocessing, initialization and 150-epoch schedule',
            'selection_policy_unchanged': 'internal 15% GroupShuffleSplit site holdout; selected_epochs is the recorded internal-validation minimum',
            'identity_reuse': bool(sman['identity_reuse']), 'refit_epochs': int(sman['fitted_epochs']),
            'source_selected_duration_fit': f'models/50cm/ma_seed42/outer{k}',
            'source_strategy_fit': f'training_strategy_review/models/50cm/warmup_floor10_seed42/outer{k}',
            'additional_training_seconds': extra, 'training_seconds': float(bman['training_seconds']) + extra,
            'decision_basis': 'development OOF only: three-seed paired floor10-minus-selected-duration outer-OOF RMSE difference -0.01197 m3/m3 [-0.01840, -0.00549] (training_strategy_review/results.json); test outcomes disclosed, not used as the criterion',
            'weights_sha256': {'model.pt': sha(dest / 'model.pt'), 'preprocessor.joblib': sha(dest / 'preprocessor.joblib')}})
        if sman['identity_reuse']:
            assert meta['weights_sha256']['model.pt'] == sha(BASE / f'outer{k}/model.pt')
            assert meta['weights_sha256']['preprocessor.joblib'] == sha(BASE / f'outer{k}/preprocessor.joblib')
            meta['refit_trace_note'] = 'identical to the selected-duration fit; selected epochs already >= 10'
        else:
            assert set(sman['gradient_site_ids']) == set(bman['gradient_site_ids'])
            assert set(sman['early_stopping_site_ids']) == set(bman['early_stopping_site_ids'])
            assert set(sman['forbidden_site_ids']) == set(bman['forbidden_site_ids'])
            meta['refit_trace'] = sman['trace']; meta['training_records'] = sman['training_records']; meta['permitted_records'] = sman['permitted_records']
            assert meta['weights_sha256']['model.pt'] != sha(BASE / f'outer{k}/model.pt')
        (dest / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        # Replay the copied weights on the archived fold inputs and compare with the stored predictions.
        model, prep, _ = load_model(dest)
        casc = REV / f'data/cascade/50cm/outer{k}'
        rep = {'fold': k, 'identity_reuse': bool(sman['identity_reuse']), 'refit_epochs': int(sman['fitted_epochs'])}
        for split in ['oof', 'test']:
            frame = pd.read_parquet(casc / f'{split}.parquet')
            stored = pd.read_parquet(dest / f'{split}_predictions.parquet').set_index('record_id')
            assert set(frame.physical_site_id).isdisjoint(set(prep.fitted_site_ids)), 'evaluation site inside fitted preprocessing'
            pred = predict_tensors(model, tensors(prep, frame))
            diff = np.abs(pred - stored.loc[frame.record_id, 'prediction'].to_numpy())
            rep[f'{split}_replay_max_abs_diff'] = float(diff.max()); rep[f'{split}_records'] = int(len(frame))
        del model; torch.cuda.empty_cache(); replay.append(rep)
        folds.append({'fold': k, 'selected_epochs': int(bman['selected_epochs']), 'refit_epochs': int(sman['fitted_epochs']),
                      'identity_reuse': bool(sman['identity_reuse']), 'weights_sha256': meta['weights_sha256']})
        print('FOLD', rep, flush=True)
    assert all(r['oof_replay_max_abs_diff'] < 1e-4 and r['test_replay_max_abs_diff'] < 1e-4 for r in replay), replay
    # Run-level files in the same layout as run_campaign.summarize().
    vals = [pd.read_parquet(DEST / f'outer{k}/oof_predictions.parquet') for k in range(5)]
    tests = [pd.read_parquet(DEST / f'outer{k}/test_predictions.parquet').set_index('record_id') for k in range(5)]
    oof = pd.concat(vals, ignore_index=True); assert not oof.record_id.duplicated().any()
    base_oof = pd.read_parquet(BASE / 'oof_predictions.parquet'); assert set(oof.record_id) == set(base_oof.record_id)
    test = tests[0].copy(); test['prediction'] = np.mean([z.reindex(test.index).prediction.to_numpy() for z in tests], axis=0); test = test.reset_index()
    for split, z in [('oof', oof), ('test', test)]:
        ref = pd.read_parquet(SRC / f'{split}_predictions.parquet').set_index('record_id').reindex(z.record_id)
        assert np.allclose(ref.prediction.to_numpy(), z.prediction.to_numpy(), atol=1e-9, rtol=0)
        assert np.array_equal(ref.y.to_numpy(), z.y.to_numpy())
        z.to_parquet(DEST / f'{split}_predictions.parquet', index=False)
    metrics = {'oof': stats(oof), 'test': stats(test)}
    (DEST / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    basemetrics = json.loads((BASE / 'metrics.json').read_text())
    spec = {'revision': 'oasm10_public_release', 'adopted_utc': datetime.now(timezone.utc).isoformat(),
            'runs': {'5cm': 'ma_seed42', '20cm': 'ma_seed42', '50cm': 'ma_floor10_seed42'},
            'unchanged': ['5 cm reference', '20 cm reference', 'nested upstream cascade inputs (data/cascade)',
                          'all architecture and input controls', 'shared split, site registry and records'],
            'policy_50cm': 'internal-holdout epoch selection unchanged; final refit on all permitted sites for max(selected_epochs,10) epochs (warmup floor); folds outer0/1/4 reuse the identical seed-42 weights, outer2/outer3 use the verified floor-10 refits',
            'decision_basis': 'development OOF evidence only (training_strategy_review): three-seed paired floor10-minus-selected-duration outer-OOF RMSE difference at 50 cm -0.01197 m3/m3 [-0.01840, -0.00549]; at 5 cm -0.00008 [-0.00060, +0.00044] and at 20 cm -0.00224 [-0.00438, -0.00049] with test intervals including zero, so no change there',
            'test_disclosure': 'Test metrics of all policies were computed after the OOF freeze and reported in SciData v13 / ISPRS v11 before this adoption; the reported test outcome of the adopted policy is therefore not an untouched prospective validation',
            'retained_checkpoint_not_adopted': 'smaller OOF gain (-0.00723) and lower anomaly correlation (-0.0068 [-0.0111, -0.0021]); it also changes the gradient-training population',
            'previous_reference': 'models/50cm/ma_seed42 (selected-duration refit); retained as the common-policy anchor of the architecture comparisons',
            'metrics': {'ma_floor10_seed42': metrics, 'ma_seed42': basemetrics}, 'folds': folds, 'replay': replay,
            'inference_rule': 'oasm.inference.predict_product reads this file; deep fold k receives the surface prediction of 5cm/ma_seed42/outer{k}'}
    (REV / 'models/product_reference.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    (HERE / 'adoption.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    print('ADOPTED 50 cm reference: OOF %.4f -> %.4f, test %.4f -> %.4f' % (
        basemetrics['oof']['rmse'], metrics['oof']['rmse'], basemetrics['test']['rmse'], metrics['test']['rmse']), flush=True)


if __name__ == '__main__':
    main()
