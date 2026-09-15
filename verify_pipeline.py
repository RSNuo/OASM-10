from pathlib import Path
import json,sys,time
import numpy as np
import pandas as pd
import torch
from oasm.features import feature_names,derive_features
from oasm.preprocessing import Preprocessor,assert_excluded
from oasm.training import fit_neural

REV=Path(__file__).resolve().parent
def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    f=pd.read_parquet(REV/'data/all_5cm.parquet')
    train=f[(f.split=='development')&(f.outer_fold!=0)].copy()
    val=f[(f.split=='development')&(f.outer_fold==0)].copy()
    forbidden=set(f.loc[f.split=='test','physical_site_id'])|set(val.physical_site_id)
    cols,cats=feature_names(5)
    prep=Preprocessor(cols,cats).fit(train);assert_excluded(prep,forbidden)
    x,c=prep.transform(val)
    assert np.isfinite(x).all()
    mutant=val.head(2).copy();mutant['LandCover']='NEVER_SEEN_CLASS'
    assert (prep.transform(mutant)[1][:,-1]==0).all()
    for d in [5,20,50]:
        z=pd.read_parquet(REV/f'data/all_{d}cm.parquet').head(1000)
        rebuilt=derive_features(z)
        for name in ['s2_lag','landsat_lag','TWI_proxy','NDVI_Best','NDMI_Best','Day_sin','Day_cos']:
            assert np.allclose(z[name],rebuilt[name],rtol=1e-6,atol=1e-6,equal_nan=True),name
    meta=fit_neural(train,{'oof':val},REV/'smoke/reference_outer0',5,'ma',42,forbidden,max_epochs=3)
    assert set(meta['gradient_site_ids']).isdisjoint(meta['evaluation_site_ids']['oof'])
    assert set(meta['early_stopping_site_ids']).isdisjoint(meta['forbidden_site_ids'])
    report={'schema_finite':True,'unknown_category_reserved_zero':True,'shared_feature_reconstruction':True,'outer_early_stopping_exclusion':True,'smoke_train_seconds':meta['training_seconds'],'smoke_selected_epochs':meta['selected_epochs']}
    (REV/'verification_initial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
