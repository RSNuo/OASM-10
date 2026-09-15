"""Independent sufficient-statistic checks and saved-checkpoint replay."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd,torch,joblib
H=Path(__file__).resolve().parent;R=H.parent;sys.path.insert(0,str(R))
from oasm.models import SoilTransformer
from run_strategies import write,sha
SEEDS=[42,7,555];POLICIES=['warmup_floor10','retain_internal_best']
def ss(frame):
    z=frame.copy();y=z.y.to_numpy(dtype=float);p=z.prediction.to_numpy(dtype=float)
    z['error']=p-y
    series=z.groupby(['physical_site_id','source_station'])[['y','prediction']].transform('mean')
    z['yc']=y-series.y.to_numpy(dtype=float);z['pc']=p-series.prediction.to_numpy(dtype=float)
    z['e2']=z.error**2;z['yy']=z.yc**2;z['pp']=z.pc**2;z['xy']=z.yc*z.pc
    g=z.groupby('physical_site_id',sort=True)
    t=g.agg(n=('y','size'),sse=('e2','sum'),sum_e=('error','sum'),yy=('yy','sum'),pp=('pp','sum'),xy=('xy','sum'))
    t['rmse']=np.sqrt(t.sse/t.n);return t
def number(s):
    return {'rmse':float(np.sqrt(s.sse.sum()/s.n.sum())),'bias':float(s.sum_e.sum()/s.n.sum()),'anomaly':float(s.xy.sum()/np.sqrt(s.yy.sum()*s.pp.sum())),'equal_rmse':float(s.rmse.mean())}
def bootstrap(tables):
    n=len(tables[0][0]);rng=np.random.default_rng(20260913);w=rng.multinomial(n,np.full(n,1/n),size=2000)
    diffs=[];equals=[];rs=[];points=[];ep=[];ap=[]
    for a,b in tables:
        assert a.index.equals(b.index)
        def rad(x):return (w@x.xy)/np.sqrt((w@x.yy)*(w@x.pp))
        diffs.append(np.sqrt((w@a.sse)/(w@a.n))-np.sqrt((w@b.sse)/(w@b.n)))
        equals.append(w@(a.rmse-b.rmse)/n);rs.append(rad(a)-rad(b))
        aa=number(a);bb=number(b);points.append(aa['rmse']-bb['rmse']);ep.append(aa['equal_rmse']-bb['equal_rmse']);ap.append(aa['anomaly']-bb['anomaly'])
    return {'mean_rmse_difference':float(np.mean(points)),'rmse_ci95':np.quantile(np.mean(diffs,axis=0),[.025,.975]).tolist(),'site_equal_rmse_difference':float(np.mean(ep)),'site_equal_rmse_ci95':np.quantile(np.mean(equals,axis=0),[.025,.975]).tolist(),'anomaly_r_difference':float(np.mean(ap)),'anomaly_r_ci95':np.quantile(np.mean(rs,axis=0),[.025,.975]).tolist()}
def close(a,b):
    assert np.allclose(a,b,atol=2e-10,rtol=0),(a,b)

def same_state(a,b,path='preprocessor'):
    """Serialization memo layout can differ on redump; every learned value must match."""
    assert type(a) is type(b),(path,type(a),type(b))
    if isinstance(a,np.ndarray):
        assert a.dtype==b.dtype and a.shape==b.shape,path
        assert np.array_equal(a,b,equal_nan=True),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:same_state(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(aa,bb) in enumerate(zip(a,b)):same_state(aa,bb,path+f'[{i}]')
    elif hasattr(a,'__dict__'):same_state(vars(a),vars(b),path)
    else:assert a==b,path
def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    result=json.loads((H/'results.json').read_text());freeze=json.loads((H/'oof_freeze.json').read_text())
    checks=[];replays=[]
    assert sha(H/'protocol.json')==freeze['protocol_sha256']
    for name,hashcode in result['protocol']['sources_sha256'].items():assert sha(R/name)==hashcode,name
    for name,hashcode in freeze['models'].items():assert sha(H/name/'oof_predictions.parquet')==hashcode
    for name,hashcode in freeze['fitted_assets_sha256'].items():assert sha(H/name)==hashcode,name
    assert sha(H/'legacy_cohort_plan.json')==freeze['legacy_cohort_plan_sha256']
    for split in ['oof','test']:
        for d in [5,20,50]:
            for policy in POLICIES:
                tables=[]
                for s in SEEDS:
                    p=H/f'models/{d}cm/{policy}_seed{s}'
                    z=pd.read_parquet(p/f'{split}_predictions.parquet').set_index('record_id').sort_index()
                    base=pd.read_parquet(R/f'models/{d}cm/ma_seed{s}/{split}_predictions.parquet').set_index('record_id').sort_index()
                    assert z.index.equals(base.index);assert np.array_equal(z.y,base.y);assert z.physical_site_id.equals(base.physical_site_id)
                    a=ss(z.reset_index());b=ss(base.reset_index());tables.append((a,b));nums=number(a)
                    target=result[split][str(d)]['models'][policy][str(s)]
                    close(nums['rmse'],target['rmse']);close(nums['bias'],target['bias']);close(nums['anomaly'],target['pooled_within_site_anomaly_r']);close(nums['equal_rmse'],target['site_equal_rmse'])
                    # Rebuild aggregation from the five stored components, by key.
                    fs=[pd.read_parquet(p/f'outer{k}/{split}_predictions.parquet').set_index('record_id') for k in range(5)]
                    agg=pd.concat(fs).sort_index() if split=='oof' else fs[0].assign(prediction=np.mean([f.reindex(fs[0].index).prediction for f in fs],axis=0)).sort_index()
                    assert agg.index.equals(z.index);close(agg.prediction,z.prediction)
                got=bootstrap(tables);expected=result[split][str(d)]['paired'][policy+'_minus_baseline']
                for key,value in got.items():close(value,expected[key])
                checks.append({'depth':d,'split':split,'policy':policy,'three_seed_bootstrap_matches':True})
    for d in [5,20,50]:
        for k in range(5):
            f=pd.read_parquet(R/'data/all_5cm.parquet').query("split == 'test'") if d==5 else pd.read_parquet(R/f'data/cascade/{d}cm/outer{k}/test.parquet')
            # Same deterministic records for every seed and policy at this fold.
            sample=f.sort_values('record_id').iloc[:64].copy()
            for policy in POLICIES:
                for s in SEEDS:
                    p=H/f'models/{d}cm/{policy}_seed{s}/outer{k}';m=json.loads((p/'manifest.json').read_text());base=R/m['source_reference']
                    if m['identity_reuse']:
                        for split in ['oof','test']:assert sha(p/f'{split}_predictions.parquet')==sha(base/f'{split}_predictions.parquet')
                        replays.append({'depth':d,'fold':k,'seed':s,'policy':policy,'identity_reuse_exact':True});continue
                    prep=joblib.load(p/'preprocessor.joblib');assert set(prep.fitted_site_ids)==set(m['gradient_site_ids']);assert not set(prep.fitted_site_ids)&set(m['forbidden_site_ids'])
                    if policy=='warmup_floor10':
                        assert m['fitted_epochs']==max(10,m['selected_epochs'])
                        old_manifest=json.loads((base/'manifest.json').read_text())
                        assert len(old_manifest['refit_trace'])==m['selected_epochs']
                        # The extension must reproduce the old prefix, not merely
                        # share nominal hyperparameters and preprocessing.
                        for actual,original in zip(m['trace'],old_manifest['refit_trace']):
                            assert actual['epoch']==original['epoch']
                            close(actual['lr'],original['lr']);close(actual['delta'],original['delta'])
                            assert abs(actual['loss']-original['loss'])<2e-7,(p,actual,original)
                        same_state(prep,joblib.load(base/'preprocessor.joblib'))
                        # In addition to inspecting every learned attribute, test the
                        # transform on all source training and OOF records, bit for bit.
                        from run_strategies import get_data
                        train,valid,_=get_data(d,k);oldprep=joblib.load(base/'preprocessor.joblib')
                        for frame in [train,valid]:
                            aa=prep.transform(frame);bb=oldprep.transform(frame)
                            assert all(np.array_equal(x,y) for x,y in zip(aa,bb))
                    else:
                        assert m['fitted_epochs']==m['selected_epochs'];assert not set(m['gradient_site_ids'])&set(m['early_stopping_site_ids'])
                        assert m['selection_replay_max_rmse_difference']<2e-5
                    pack=torch.load(p/'model.pt',map_location='cpu',weights_only=False);c=pack['config']
                    model=SoilTransformer(c['numeric'],c['categorical'],pack['cat_dims'],variant=c['architecture'],**c['model']).cuda().eval();model.load_state_dict(pack['state_dict'],strict=True)
                    x,cats=prep.transform(sample)
                    with torch.no_grad():pred=model(torch.from_numpy(x).cuda(),torch.from_numpy(cats).cuda()).cpu().numpy()
                    expected=pd.read_parquet(p/'test_predictions.parquet').set_index('record_id').loc[sample.record_id,'prediction'].to_numpy()
                    error=float(np.max(np.abs(pred-expected)))
                    detail={'depth':d,'fold':k,'seed':s,'policy':policy,'replayed_records':len(sample),'small_batch_max_abs_error':error,'small_batch_check_threshold':1e-4}
                    if error>=1e-4:
                        # Investigate reduced-precision kernel changes, without
                        # relaxing the saved-prediction reproduction tolerance.
                        def forward_frame(frame,batch=8192):
                            xx,cc=prep.transform(frame);values=[]
                            with torch.no_grad():
                                for start in range(0,len(frame),batch):values.append(model(torch.from_numpy(xx[start:start+batch]).cuda(),torch.from_numpy(cc[start:start+batch]).cuda()).cpu().numpy())
                            return np.concatenate(values)
                        full=forward_frame(f)
                        saved=pd.read_parquet(p/'test_predictions.parquet').set_index('record_id').loc[f.record_id,'prediction'].to_numpy()
                        exact_error=float(np.max(np.abs(full-saved)));assert exact_error<1e-6,(p,'Original-batch reproduction',exact_error)
                        torch.set_float32_matmul_precision('highest')
                        precise_small=forward_frame(sample,batch=64);precise_full=forward_frame(f)
                        precise_selected=pd.Series(precise_full,index=f.record_id).loc[sample.record_id].to_numpy()
                        invariant_error=float(np.max(np.abs(precise_small-precise_selected)))
                        torch.set_float32_matmul_precision('high')
                        assert invariant_error<2e-6,(p,'Full-precision batch consistency',invariant_error)
                        detail.update({'replayed_records':len(f),'original_batch_size':8192,'original_batch_max_abs_error':exact_error,'original_batch_tolerance':1e-6,'highest_precision_batch_consistency_error':invariant_error,'explanation':'The high matmul setting can choose different reduced-precision kernels across batch sizes. Original-batch predictions reproduce; highest precision removes the large batch discrepancy. Frozen scientific predictions are unchanged.'})
                    else:detail['max_abs_error']=error
                    replays.append(detail)
                    del model;torch.cuda.empty_cache()
            print('REPLAY_CHECKED',d,k,flush=True)
    write(H/'independent_verification.json',{'three_seed_comparisons':checks,'checkpoint_checks':replays,'source_hashes_unchanged':True,'oof_freeze_unchanged':True,'results_sha256':sha(H/'results.json')})
    print('STRATEGY_VERIFICATION_COMPLETE',flush=True)
if __name__=='__main__':main()
