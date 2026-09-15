"""OOF/development-only check of matched rows, data coverage, and upstream roles."""
import sys
import numpy as np
import pandas as pd
import joblib
from run_factorial import HERE,REV,CONFIGS,SEEDS,get_data,numeric,write,sha,read,now

def main():
    folds=[get_data(k) for k in range(5)]
    dev=pd.concat([o for _,o,_ in folds],ignore_index=True).set_index('record_id').sort_index()
    assert dev.index.is_unique
    rows=[];upstream=[]
    for k,(tr,oo,forbidden) in enumerate(folds):
        a=tr.set_index('record_id');b=oo.set_index('record_id')
        assert set(a.index).isdisjoint(b.index) and set(a.index)|set(b.index)==set(dev.index)
        for frame in [a,b]:
            ref=dev.loc[frame.index]
            assert frame.physical_site_id.equals(ref.physical_site_id)
            for c in numeric('D')+['y']:
                assert np.array_equal(frame[c].to_numpy(),ref[c].to_numpy(),equal_nan=True),(k,c)
        paths=[REV/f'models/5cm/ma_seed42/outer{k}']+[REV/f'models/upstream_nested/outer{k}/inner{j}' for j in range(5)]
        for p in paths:
            m=read(p/'manifest.json');prep=joblib.load(p/'preprocessor.joblib')
            for q in ['gradient_site_ids','early_stopping_site_ids','selection_gradient_site_ids']:
                assert forbidden.isdisjoint(m[q]),(p,q)
            assert forbidden.isdisjoint(prep.fitted_site_ids)
            upstream.append({'path':str(p.relative_to(REV)),'manifest_sha256':sha(p/'manifest.json'),'checks_passed':True})
        # Every injected development prediction corresponds to a held-out inner group.
        nested=[]
        for j in range(5):
            p=REV/f'models/upstream_nested/outer{k}/inner{j}'
            m=read(p/'manifest.json');n=pd.read_parquet(p/'cascade_training_predictions.parquet')
            n=n[n.record_id.isin(a.index)]
            assert set(n.physical_site_id).isdisjoint(m['gradient_site_ids'])
            assert set(n.physical_site_id).isdisjoint(m['early_stopping_site_ids'])
            nested.append(n[['record_id','prediction']])
        n=pd.concat(nested).set_index('record_id').sort_index()
        assert n.index.is_unique and set(n.index)==set(a.index)
        assert np.array_equal(n.loc[a.index,'prediction'].to_numpy(),a.soil_moisture_5cm.to_numpy())
        rows.append({'fold':k,'train_records':len(a),'outer_oof_records':len(b),'matched_non_cascade_inputs':True,'nested_training_inputs_verified':True})
    result={'utc':now(),'passed':True,'development_records':len(dev),'development_physical_sites':dev.physical_site_id.nunique(),
        'numeric_feature_counts':{c:len(numeric(c)) for c in CONFIGS},
        'development_precipitation':{c:{'missing_records':int(dev[c].isna().sum()),'valid_records':int(dev[c].notna().sum()),'stored_encoding':'log1p(cumulative mm)'} for c in ['precip_7d','precip_30d','precip_90d']},
        'folds':rows,'upstream_model_checks':upstream,'test_records_loaded':False}
    write(HERE/'design_check.json',result)
    print({k:v for k,v in result.items() if k not in ['folds','upstream_model_checks']},flush=True)

if __name__=='__main__':main()
