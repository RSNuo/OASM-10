"""Freeze complete OOF evidence before test inference; report every declared contrast."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,shutil
import numpy as np,pandas as pd,torch,joblib
HERE=Path(__file__).resolve().parent;REV=HERE.parent;sys.path.insert(0,str(REV))
from oasm.training import load_model,predict_tensors,tensors
from analyze_results import statistics,site_statistics,site_summary,align
import analyze_results as metrics
metrics.BOOT_SEED=20260913
from run_strategies import write,sha,directory
SEEDS=[42,7,555];STRATEGIES=['baseline','warmup_floor10','retain_internal_best']
def aggregate(d,s,strategy,split):
    if strategy=='baseline':return pd.read_parquet(REV/f'models/{d}cm/ma_seed{s}/{split}_predictions.parquet')
    root=HERE/f'models/{d}cm/{strategy}_seed{s}'
    frames=[pd.read_parquet(root/f'outer{k}/{split}_predictions.parquet') for k in range(5)]
    if split=='oof':
        out=pd.concat(frames,ignore_index=True);assert not out.record_id.duplicated().any()
    else:
        out=frames[0].copy()
        out['prediction']=np.mean([align(out,z)[1].set_index('record_id').loc[out.record_id,'prediction'].to_numpy() for z in frames],axis=0)
    baseline=pd.read_parquet(REV/f'models/{d}cm/ma_seed{s}/{split}_predictions.parquet')
    align(out,baseline);out.to_parquet(root/f'{split}_predictions.parquet',index=False)
    return out
def analyze(split):
    result={}
    for d in [5,20,50]:
        row={'models':{},'paired':{}}
        dfs={}
        for strategy in STRATEGIES:
            row['models'][strategy]={}
            for s in SEEDS:
                z=aggregate(d,s,strategy,split);dfs[(strategy,s)]=z
                sites=site_statistics(z);sites.to_csv(HERE/f'{d}cm_{strategy}_seed{s}_{split}_sites.csv')
                row['models'][strategy][s]={**statistics(z),**site_summary(sites)}
        for strategy in STRATEGIES[1:]:
            pairs=[(dfs[(strategy,s)],dfs[('baseline',s)]) for s in SEEDS]
            paired=metrics.paired_seed_bootstrap(pairs)
            paired['per_seed']={s:metrics.paired_seed_bootstrap([pair]) for s,pair in zip(SEEDS,pairs)}
            row['paired'][strategy+'_minus_baseline']=paired
        result[d]=row
    return result
def test_inference():
    assert (HERE/'oof_freeze.json').exists()
    frozen=json.loads((HERE/'oof_freeze.json').read_text())
    for name,hashcode in frozen['fitted_assets_sha256'].items():assert sha(HERE/name)==hashcode,name
    assert sha(HERE/'legacy_cohort_plan.json')==frozen['legacy_cohort_plan_sha256']
    for d in [5,20,50]:
        for k in range(5):
            if d==5:
                f=pd.read_parquet(REV/'data/all_5cm.parquet');test=f[f.split=='test'].copy()
            else:test=pd.read_parquet(REV/f'data/cascade/{d}cm/outer{k}/test.parquet')
            for strategy in STRATEGIES[1:]:
                for s in SEEDS:
                    p=directory(d,s,k,strategy);dest=p/'test_predictions.parquet'
                    if dest.exists():continue
                    meta=json.loads((p/'manifest.json').read_text())
                    if meta['identity_reuse']:
                        shutil.copy2(REV/meta['source_reference']/'test_predictions.parquet',dest);continue
                    model,prep,config=load_model(p)
                    assert not set(test.physical_site_id)&set(prep.fitted_site_ids)
                    out=test[['record_id','source_station','physical_site_id','target_time','y']].copy()
                    out['prediction']=predict_tensors(model,tensors(prep,test));out.to_parquet(dest,index=False)
                    del model;torch.cuda.empty_cache()
            print(f'TEST_INFERENCE depth={d} fold={k}',flush=True)
def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    assert (HERE/'fit_completed.json').exists()
    manifests=list((HERE/'models').rglob('manifest.json'));assert len(manifests)==90
    for p in manifests:
        m=json.loads(p.read_text());base=json.loads((REV/m['source_reference']/'manifest.json').read_text())
        if not m['identity_reuse']:
            prep=joblib.load(p.parent/'preprocessor.joblib')
            assert not set(prep.fitted_site_ids)&set(m['forbidden_site_ids'])
            assert set(prep.fitted_site_ids)==set(m['gradient_site_ids'])
            if m['strategy']=='retain_internal_best':assert set(m['gradient_site_ids']).isdisjoint(m['early_stopping_site_ids'])
        else:assert m['selected_epochs']>=10
    oof=analyze('oof');write(HERE/'oof_results.json',oof)
    frozen={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(HERE/'protocol.json'),'oof_sha256':sha(HERE/'oof_results.json'),'models':{str(p.parent.relative_to(HERE)):sha(p.parent/'oof_predictions.parquet') for p in manifests},'test_usage':'New strategy test predictions have not been read to select a protocol; all declared strategies will be evaluated and reported.'}
    frozen['fitted_assets_sha256']={str(asset.relative_to(HERE)):sha(asset) for p in manifests for asset in [p,p.parent/'model.pt',p.parent/'preprocessor.joblib'] if asset.exists()}
    frozen['legacy_cohort_plan_sha256']=sha(HERE/'legacy_cohort_plan.json')
    path=HERE/'oof_freeze.json'
    if path.exists():
        prev=json.loads(path.read_text());assert prev['models']==frozen['models'] and prev['oof_sha256']==frozen['oof_sha256'] and prev['fitted_assets_sha256']==frozen['fitted_assets_sha256']
    else:write(path,frozen)
    print('OOF_EVIDENCE_FROZEN',flush=True)
    test_inference();test=analyze('test')
    fold_rows=[]
    for p in manifests:
        m=json.loads(p.read_text());src=REV/m['source_reference'];base=json.loads((src/'manifest.json').read_text())
        item={k:m[k] for k in ['strategy','depth','seed','fold','selected_epochs','fitted_epochs','identity_reuse','additional_training_seconds']}
        item['gradient_sites']=len(base['gradient_site_ids'] if m['identity_reuse'] else m['gradient_site_ids'])
        item['permitted_sites']=len(base['gradient_site_ids'])
        for split in ['oof','test']:
            z=pd.read_parquet(p.parent/f'{split}_predictions.parquet');original=pd.read_parquet(src/f'{split}_predictions.parquet')
            item[split]=statistics(z);item[split+'_baseline']=statistics(original)
        fold_rows.append(item)
    write(HERE/'results.json',{'protocol':json.loads((HERE/'protocol.json').read_text()),'oof':oof,'test':test,'folds':fold_rows})
    flat=[]
    for row in fold_rows:
        v={k:t for k,t in row.items() if not isinstance(t,dict)}
        for key in ['oof','test','oof_baseline','test_baseline']:
            v.update({key+'_'+k:value for k,value in row[key].items()})
        flat.append(v)
    pd.DataFrame(flat).to_csv(HERE/'fold_quality.csv',index=False)
    write(HERE/'analysis_completed.json',{'utc':datetime.now(timezone.utc).isoformat(),'results_sha256':sha(HERE/'results.json'),'variant_folds':90,'new_fits':sum(not json.loads(p.read_text())['identity_reuse'] for p in manifests)})
    for split,results in [('OOF',oof),('TEST',test)]:
        for d,row in results.items():
            for strategy,entry in row['paired'].items():print(f'{split} depth={d} {strategy} delta={entry["mean_rmse_difference"]:+.6f} CI={entry["rmse_ci95"]} anomaly={entry["anomaly_r_difference"]:+.6f}',flush=True)
    print('STRATEGY_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
