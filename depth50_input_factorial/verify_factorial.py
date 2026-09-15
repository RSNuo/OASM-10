"""Independent metric implementation, cluster inference, and saved-weight replay.

Does not import analysis summary/metric functions. Uses training code only to
reconstruct saved model inference. Does not write original scientific artifacts.
"""
from pathlib import Path
import sys, json, hashlib
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

HERE=Path(__file__).resolve().parent;REV=HERE.parent;sys.path.insert(0,str(REV))
from oasm.training import load_model, tensors, predict_tensors
from oasm.preprocessing import Preprocessor
from oasm.features import BASE,SOIL,PRECIP,CATS
SEEDS=[42,7,555];CONFIGS='ABCD';TOL=2e-10

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def check(x,y,tol=TOL):
    if x is None or y is None:assert x is None and y is None;return
    assert np.allclose(x,y,rtol=0,atol=tol,equal_nan=True),(x,y)

def group_sums(z):
    z=z.copy();z[['y','prediction']]=z[['y','prediction']].astype(np.float64)
    residual=z.prediction-z.y
    ym=z.groupby(['physical_site_id','source_station']).y.transform('mean')
    pm=z.groupby(['physical_site_id','source_station']).prediction.transform('mean')
    yy=z.y-ym;pp=z.prediction-pm
    v=pd.DataFrame({'physical_site_id':z.physical_site_id,'n':1,'se':residual**2,'e':residual,
                    'xy':yy*pp,'yy':yy**2,'pp':pp**2})
    return v.groupby('physical_site_id',sort=True).sum()

def calc(z):
    y=z.y.to_numpy(dtype=float);p=z.prediction.to_numpy(dtype=float);e=p-y;ss=group_sums(z)
    rm=np.sqrt(ss.se/ss.n);bias=ss.e/ss.n
    return {'rmse':float(np.linalg.norm(e)/np.sqrt(len(e))),'mse':float(e@e/len(e)),
        'bias':float(np.mean(e)),'ubrmse':float(np.linalg.norm(e-e.mean())/np.sqrt(len(e))),
        'r2':float(1-e@e/((y-y.mean())@(y-y.mean()))),
        'site_equal_rmse':float(rm.mean()),'site_equal_ubrmse':float(np.sqrt(np.maximum(ss.se/ss.n-bias**2,0)).mean()),
        'pooled_within_site_anomaly_r':float(ss.xy.sum()/np.sqrt(ss.yy.sum()*ss.pp.sum())),
        'site_bias_mse_share':float((ss.n*bias**2).sum()/ss.se.sum())}

def boot(pairs):
    first=group_sums(pairs[0][0]);n=len(first)
    w=np.random.default_rng(20260914).multinomial(n,np.repeat(1/n,n),size=2000).astype(float)
    draws=[];points=[];equald=[];equalp=[];anomd=[];anomp=[]
    def corr(st,weights):
        return (weights@st.xy)/np.sqrt((weights@st.yy)*(weights@st.pp))
    for a,b in pairs:
        sa=group_sums(a);sb=group_sums(b);assert sa.index.equals(first.index) and sb.index.equals(first.index)
        r1=np.sqrt((w@sa.se)/(w@sa.n));r2=np.sqrt((w@sb.se)/(w@sb.n))
        draws.append(r1-r2)
        va=np.sqrt(sa.se/sa.n);vb=np.sqrt(sb.se/sb.n)
        equald.append(w@(va-vb)/n);equalp.append(float((va-vb).mean()))
        anomd.append(corr(sa,w)-corr(sb,w))
        points.append(calc(a)['rmse']-calc(b)['rmse'])
        anomp.append(float(corr(sa,np.ones(n))-corr(sb,np.ones(n))))
    return {'mean_rmse_difference':float(np.mean(points)),'rmse_ci95':np.quantile(np.mean(draws,axis=0),[.025,.975]),
        'site_equal_rmse_difference':float(np.mean(equalp)),'site_equal_rmse_ci95':np.quantile(np.mean(equald,axis=0),[.025,.975]),
        'anomaly_r_difference':float(np.mean(anomp)),'anomaly_r_ci95':np.quantile(np.mean(anomd,axis=0),[.025,.975])}

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    protocol=read(HERE/'protocol.json');result=read(HERE/'results.json');frozen=read(HERE/'oof_freeze.json')
    for name,digest in protocol['sources_sha256'].items():assert sha(REV/name)==digest,name
    for name,digest in frozen['fitted_assets_sha256'].items():assert sha(REV/name)==digest,name
    for name,digest in frozen['analysis_code_sha256'].items():assert sha(HERE/name)==digest,name
    numerics={};frames={};count=0
    for split in ['oof','test']:
        for c in CONFIGS:
            for s in SEEDS:
                root=HERE/f'models/{c}_seed{s}'
                z=pd.read_parquet(root/f'{split}_predictions.parquet').set_index('record_id').sort_index()
                pieces=[pd.read_parquet(root/f'outer{k}/{split}_predictions.parquet').set_index('record_id') for k in range(5)]
                if split=='oof':
                    expected=pd.concat(pieces).sort_index();assert expected.index.is_unique
                    assert expected.index.equals(z.index)
                    check(expected.prediction,z.prediction,0)
                else:
                    for f in pieces:assert set(f.index)==set(z.index)
                    check(sum(f.reindex(z.index).prediction.to_numpy(float) for f in pieces)/5,z.prediction,1e-15)
                z=z.reset_index();assert np.isfinite(z[['y','prediction']]).all().all()
                if (split,'A',42) in frames:
                    ref=frames[split,'A',42]
                    assert ref.record_id.equals(z.record_id) and ref.physical_site_id.equals(z.physical_site_id)
                    check(ref.y,z.y,0)
                frames[split,c,s]=z
                got=calc(z);reported=result[split]['configurations'][c]['per_seed'][str(s)]
                for k,v in got.items():check(v,reported[k],2e-9)
                numerics[f'{split}/{c}/{s}']=got
        for key,reported in result[split]['contrasts'].items():
            a,b=key.split('-');pairs=[(frames[split,a,s],frames[split,b,s]) for s in SEEDS]
            for k,v in boot(pairs).items():check(v,reported[k])
            for s,pair in zip(SEEDS,pairs):
                for k,v in boot([pair]).items():check(v,reported['per_seed'][str(s)][k])
            count+=1
        n=len(group_sums(frames[split,'A',42]));w=np.random.default_rng(20260914).multinomial(n,np.ones(n)/n,size=2000)
        draws=[]
        for s in SEEDS:
            ss={c:group_sums(frames[split,c,s]) for c in CONFIGS}
            rr={c:np.sqrt((w@v.se)/(w@v.n)) for c,v in ss.items()}
            draws.append(rr['C']-rr['A']-rr['D']+rr['B'])
        check(np.quantile(np.mean(draws,axis=0),[.025,.975]),result[split]['rmse_interaction']['ci95'])
    ranking=sorted(CONFIGS,key=lambda c:(np.mean([numerics[f'oof/{c}/{s}']['rmse'] for s in SEEDS]),c))
    assert ranking==frozen['ranking']==result['oof']['ranking_by_mean_seed_rmse']
    print('INDEPENDENT_METRICS_AND_BOOTSTRAPS_PASS',flush=True)
    # Evaluate external/OOF upstream isolation using model and preprocessor artifacts.
    upstream=[];folds=[]
    for k in range(5):
        p=REV/f'data/cascade/50cm/outer{k}'
        tr=pd.read_parquet(p/'train.parquet');oo=pd.read_parquet(p/'oof.parquet');te=pd.read_parquet(p/'test.parquet')
        forbidden=set(read(p/'provenance.json')['forbidden_sites'])
        assert set(tr.physical_site_id).isdisjoint(forbidden)
        assert set(oo.physical_site_id).issubset(forbidden) and set(te.physical_site_id).issubset(forbidden)
        upstream_paths=[REV/f'models/5cm/ma_seed42/outer{k}']+[REV/f'models/upstream_nested/outer{k}/inner{j}' for j in range(5)]
        for up in upstream_paths:
            um=read(up/'manifest.json');pr=joblib.load(up/'preprocessor.joblib')
            for key in ['gradient_site_ids','early_stopping_site_ids','selection_gradient_site_ids']:
                assert forbidden.isdisjoint(um[key]),(up,key)
            assert forbidden.isdisjoint(pr.fitted_site_ids)
            upstream.append({'directory':str(up.relative_to(REV)),'manifest_sha256':sha(up/'manifest.json'),'preprocessor_sha256':sha(up/'preprocessor.joblib'),'external_and_outer_sites_excluded':True})
        for c in CONFIGS:
            for s in SEEDS:
                dest=HERE/f'models/{c}_seed{s}/outer{k}';m=read(dest/'manifest.json');wp=REV/m['weight_directory']
                cfg=m['config'];expected=BASE+SOIL+(PRECIP if c in 'CD' else [])+(['soil_moisture_5cm'] if c in 'AC' else [])
                assert cfg['numeric']==expected and cfg['categorical']==CATS
                assert cfg['model']=={'d_model':168,'nhead':8,'layers':2,'ff':512,'dropout':.35,'static_dropout':.55}
                assert m['fitted_epochs']==max(10,m['selected_epochs'])
                assert len(m['refit_trace'])==m['fitted_epochs']
                selected=min(m['selection_trace'],key=lambda v:v['internal_rmse'])['epoch']
                assert selected==m['selected_epochs']
                a,b=next(GroupShuffleSplit(1,test_size=.15,random_state=s+1000).split(tr,groups=tr.physical_site_id))
                assert set(m['selection_gradient_site_ids'])==set(tr.iloc[a].physical_site_id)
                assert set(m['early_stopping_site_ids'])==set(tr.iloc[b].physical_site_id)
                assert forbidden.isdisjoint(m['gradient_site_ids']) and forbidden.isdisjoint(m['early_stopping_site_ids'])
                model,prep,_=load_model(wp)
                assert set(prep.fitted_site_ids)==set(tr.physical_site_id)==set(m['gradient_site_ids'])
                again=Preprocessor(expected,CATS).fit(tr)
                # Equal transformed values are stronger than joblib byte identity.
                check(prep.fill,again.fill,0)
                for f in [tr,oo]:
                    x,cat=prep.transform(f);xx,cc=again.transform(f)
                    check(x,xx,0);assert np.array_equal(cat,cc)
                err={}
                for split,frame in [('oof',oo),('test',te)]:
                    old=pd.read_parquet(dest/f'{split}_predictions.parquet')
                    assert old.record_id.tolist()==frame.record_id.tolist()
                    predicted=predict_tensors(model,tensors(prep,frame),batch=8192)
                    delta=float(np.max(np.abs(predicted.astype(float)-old.prediction.to_numpy(float))))
                    assert delta<2e-6,(c,s,k,split,delta)
                    err[split+'_max_prediction_difference']=delta
                folds.append({'config':c,'seed':s,'fold':k,'epochs':m['fitted_epochs'],'reused':m['identity_reuse'],**err})
                del model,prep,again;torch.cuda.empty_cache()
        print(f'WEIGHT_REPLAY_AND_SITE_CHECKS_PASS fold={k}',flush=True)
    # Recheck immutability after independent reconstruction.
    for name,digest in protocol['sources_sha256'].items():assert sha(REV/name)==digest,name
    output={'utc':datetime.now(timezone.utc).isoformat(),'passed':True,'records':{split:len(frames[split,'A',42]) for split in ['oof','test']},
        'sites':{split:frames[split,'A',42].physical_site_id.nunique() for split in ['oof','test']},
        'numerical_implementations':'Raw float64 residuals and grouped sums independent of analyze_results.py; same-declared cluster bootstrap draws.',
        'verified_contrasts':count,'per_seed_ci_checks':count*3,'ranking':ranking,'source_hashes_unchanged':True,
        'numeric_metrics':numerics,'folds':folds,'upstream_exclusion_checks':upstream,
        'full_frame_checkpoint_replays':len(folds)*2,'replay_batch_size':8192,'float32_matmul_precision':'high',
        'scope':'Software and arithmetic verification using saved data; not an independent new field validation or independent reviewer.'}
    (HERE/'independent_verification.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print('ALL_FACTORIAL_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
