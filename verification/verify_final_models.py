import sys as _sys
from pathlib import Path as _Path
REV = _Path(__file__).resolve().parents[1]   # repository root
HERE = _Path(__file__).resolve().parent
_sys.path.insert(0, str(REV))
"""Hard release gate for ancestor exclusion, predictions and fitted transforms."""
from pathlib import Path
import json
import joblib,numpy as np,pandas as pd
from freeze_protocol import main as integrity

REV=_Path(__file__).resolve().parents[1]

def expected_runs():
    runs=[]
    for d in [5,20,50]:
        variants=['ma','concat','ft','no_modality','static_forcing'] if d==5 else ['ma','concat','ft','no_cascade']
        for v in variants:
            for s in [42,7,555]:runs.append((d,v,s))
        for v in ['rf','xgb']+(['no_static','static_only'] if d==5 else []):runs.append((d,v,42))
    return runs

def main():
    integrity();frames={d:pd.read_parquet(REV/f'data/all_{d}cm.parquet') for d in [5,20,50]}
    external=set().union(*(set(f.loc[f.split=='test','physical_site_id']) for f in frames.values()))
    allroles=pd.concat([f[['physical_site_id','split','outer_fold']] for f in frames.values()]).drop_duplicates()
    assert not allroles.physical_site_id.duplicated().any(),'Cross-depth site roles disagree'
    checked=[]
    for d,v,s in expected_runs():
        expected_test=set(frames[d].loc[frames[d].split=='test','record_id']);oofs=[]
        for k in range(5):
            path=REV/f'models/{d}cm/{v}_seed{s}/outer{k}';m=json.loads((path/'manifest.json').read_text())
            outer=set(allroles.loc[(allroles.split=='development')&(allroles.outer_fold==k),'physical_site_id'])
            forbidden=external|outer
            # A surface fitting manifest lists sites represented at that depth;
            # absent deeper-only IDs cannot contribute an input or a label.
            represented_forbidden=forbidden & set(frames[d].physical_site_id)
            assert represented_forbidden<=set(m['forbidden_site_ids'])
            assert forbidden.isdisjoint(m['gradient_site_ids']) and forbidden.isdisjoint(m['early_stopping_site_ids'])
            prep=joblib.load(path/'model.joblib')['preprocessor'] if v in ['rf','xgb'] else joblib.load(path/'preprocessor.joblib')
            assert set(prep.fitted_site_ids)==set(m['gradient_site_ids'])
            assert forbidden.isdisjoint(prep.fitted_site_ids)
            for split in ['oof','test']:
                z=pd.read_parquet(path/f'{split}_predictions.parquet')
                assert not z.record_id.duplicated().any() and np.isfinite(z.prediction).all()
                expected=expected_test if split=='test' else set(frames[d].loc[(frames[d].split=='development')&(frames[d].outer_fold==k),'record_id'])
                assert set(z.record_id)==expected
                target=frames[d].set_index('record_id').reindex(z.record_id)
                assert np.allclose(z.y.to_numpy(),target.y.to_numpy(),atol=1e-10,rtol=0)
                if split=='oof':oofs.extend(z.record_id)
            checked.append(str(path.relative_to(REV)))
        assert len(oofs)==len(set(oofs))==sum(frames[d].split=='development')
    for k in range(5):
        outer=set(allroles.loc[(allroles.split=='development')&(allroles.outer_fold==k),'physical_site_id'])
        for j in range(5):
            path=REV/f'models/upstream_nested/outer{k}/inner{j}';m=json.loads((path/'manifest.json').read_text())
            z=pd.read_parquet(path/'cascade_training_predictions.parquet')
            forbidden=external|outer|set(z.physical_site_id)
            assert forbidden.isdisjoint(m['gradient_site_ids']) and forbidden.isdisjoint(m['early_stopping_site_ids'])
            prep=joblib.load(path/'preprocessor.joblib');assert forbidden.isdisjoint(prep.fitted_site_ids)
            checked.append(str(path.relative_to(REV)))
    result={'status':'passed','checked_fits':len(checked),'external_sites':len(external),'checks':
        ['Training input/source hashes unchanged','Same cross-depth site roles','Outer and test sites absent from gradient/early stopping/preprocessing',
         'Nested receiving sites excluded from corresponding upstream fit','Every expected OOF/test record present once per appropriate fold','Finite predictions and exact target alignment'],
        'fits':checked}
    (HERE/'verification_final_models.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('V6 FINAL MODEL VERIFICATION PASSED',len(checked),flush=True)

if __name__=='__main__':main()
