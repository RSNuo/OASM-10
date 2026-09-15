"""Post-audit training-strategy experiment, with a frozen source and no test access during fitting.

Three matched seeds at all depths; all original artifacts remain immutable.
Deep cascade inputs are held fixed to the source v6 nested predictions. This
isolates the downstream fitting policy, not an end-to-end upstream revision.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,shutil
from datetime import datetime,timezone
import numpy as np,pandas as pd,torch,joblib
from sklearn.model_selection import GroupShuffleSplit

HERE=Path(__file__).resolve().parent;REV=HERE.parent
sys.path.insert(0,str(REV))
from oasm.training import seed_everything,new_model,fit_loop,tensors,predict_tensors
from oasm.preprocessing import Preprocessor,assert_excluded

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def freeze():
    files=[REV/f'oasm/{n}.py' for n in ['training','preprocessing','models','features']]+[Path(__file__)]+[REV/f'data/all_{d}cm.parquet' for d in [5,20,50]]
    for d in [5,20,50]:
        for s in [42,7,555]:
            for k in range(5):
                p=REV/f'models/{d}cm/ma_seed{s}/outer{k}'
                files += [p/n for n in ['manifest.json','model.pt','preprocessor.joblib','oof_predictions.parquet','test_predictions.parquet']]
        if d>5:
            for k in range(5):files += [REV/f'data/cascade/{d}cm/outer{k}/{n}' for n in ['train.parquet','oof.parquet','test.parquet','provenance.json']]
    hashes={str(p.relative_to(REV)):sha(p) for p in files}
    path=HERE/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['sources_sha256']==hashes,'Frozen source changed';return old
    obj={'created_utc':datetime.now(timezone.utc).isoformat(),'source_release':'v6_site_isolated','scope':'Post-audit strategy sensitivity; prior test examination and the observed short-fit problem motivate this experiment. Not prospective preregistration.',
         'depths':[5,20,50],'seeds':[42,7,555],'outer_folds':[0,1,2,3,4],
         'strategies':{'baseline':'Existing internal group-holdout epoch selection then all-permitted-site refit',
                       'warmup_floor10':'Same saved internal epoch choice, preprocessing, initialization and refit schedule; refit for max(selected_epochs,10); fits >=10 epochs reuse identical baseline weights',
                       'retain_internal_best':'Repeat exactly the internal 15% group holdout selection and retain its minimum-RMSE checkpoint and selection-only preprocessor; fewer gradient-training sites are a disclosed tradeoff'},
         'fixed_controls':'All model dimensions, loss, optimizer, 150-epoch horizon, source rows, folds and deep per-outer-fold upstream features remain fixed. No model or fold dropped.',
         'training_metrics':'OOF RMSE, bias, ubRMSE, equal-site RMSE, within-source-station demeaned anomaly correlation, seed-wise deltas and common-site bootstrap',
         'test_gate':'No test parquet loaded by the fitting stage. Test inference begins only after all 90 variant folds have OOF predictions and an OOF evidence freeze is saved.',
         'product_rule':'Sensitivity experiments do not silently replace the archived reference or its matched architecture controls. A modified downstream fit using fixed upstream inputs is not labeled a retrained end-to-end product.',
         'bootstrap':{'resamples':2000,'confidence':0.95,'method':'percentile whole physical-site clusters with multiplicity; shared resamples across seeds','seed':20260913},
         'sources_sha256':hashes}
    write(path,obj);return obj
def get_data(depth,fold):
    if depth==5:
        f=pd.read_parquet(REV/'data/all_5cm.parquet')
        train=f[(f.split=='development')&(f.outer_fold!=fold)].copy()
        oof=f[(f.split=='development')&(f.outer_fold==fold)].copy()
        forbidden=set(f.loc[(f.split=='test')|(f.outer_fold==fold),'physical_site_id'])
    else:
        p=REV/f'data/cascade/{depth}cm/outer{fold}'
        train=pd.read_parquet(p/'train.parquet');oof=pd.read_parquet(p/'oof.parquet')
        forbidden=set(json.loads((p/'provenance.json').read_text())['forbidden_sites'])
    return train,oof,forbidden
def save_prediction(path,model,prep,frame):
    out=frame[['record_id','source_station','physical_site_id','target_time','y']].copy()
    out['prediction']=predict_tensors(model,tensors(prep,frame));out.to_parquet(path,index=False)
def directory(d,s,k,strategy):return HERE/f'models/{d}cm/{strategy}_seed{s}/outer{k}'
def fit_one(d,s,k,strategy):
    dest=directory(d,s,k,strategy);dest.mkdir(parents=True,exist_ok=True)
    source=REV/f'models/{d}cm/ma_seed{s}/outer{k}'
    base=json.loads((source/'manifest.json').read_text());selected=int(base['selected_epochs'])
    if (dest/'manifest.json').exists() and (dest/'oof_predictions.parquet').exists():
        print(f'RESUME depth={d} seed={s} fold={k} strategy={strategy}',flush=True);return
    if strategy=='warmup_floor10' and selected>=10:
        shutil.copy2(source/'oof_predictions.parquet',dest/'oof_predictions.parquet')
        write(dest/'manifest.json',{'strategy':strategy,'depth':d,'seed':s,'fold':k,'source_reference':str(source.relative_to(REV)),'identity_reuse':True,'selected_epochs':selected,'fitted_epochs':selected,'additional_training_seconds':0})
        print(f'IDENTICAL depth={d} seed={s} fold={k} floor10 existing_epochs={selected}',flush=True);return
    train,oof,forbidden=get_data(d,k)
    assert set(train.physical_site_id).isdisjoint(forbidden)
    start=time.perf_counter();num=base['numeric'];cats=base['categorical'];config={q:base[q] for q in ['depth','variant','architecture','seed','model','numeric','categorical','max_epochs','patience','batch_size']}
    replay_error=None
    if strategy=='warmup_floor10':
        fitdf=train;prep=joblib.load(source/'preprocessor.joblib');assert_excluded(prep,forbidden)
        seed_everything(s);model=new_model(config,prep)
        _,trace,_=fit_loop(model,tensors(prep,fitdf,True),s,10,150)
        epochs=10;selection_sites=base['early_stopping_site_ids']
    else:
        a,b=next(GroupShuffleSplit(1,test_size=.15,random_state=s+1000).split(train,groups=train.physical_site_id))
        fitdf=train.iloc[a];monitor=train.iloc[b]
        assert set(fitdf.physical_site_id)==set(base['selection_gradient_site_ids'])
        assert set(monitor.physical_site_id)==set(base['early_stopping_site_ids'])
        prep=Preprocessor(num,cats).fit(fitdf);assert_excluded(prep,forbidden|set(monitor.physical_site_id))
        seed_everything(s);model=new_model(config,prep)
        epochs,trace,best=fit_loop(model,tensors(prep,fitdf,True),s,150,150,tensors(prep,monitor,True))
        assert best is not None
        replay_error=max(abs(float(a['internal_rmse'])-float(b['internal_rmse'])) for a,b in zip(trace,base['selection_trace']))
        # Stop on non-reproducible selection rather than silently compare another random fit.
        assert epochs==selected and replay_error<2e-5,(d,s,k,epochs,selected,replay_error)
        model.load_state_dict(best,strict=True);selection_sites=sorted(monitor.physical_site_id.unique())
    joblib.dump(prep,dest/'preprocessor.joblib')
    torch.save({'state_dict':{key:val.detach().cpu() for key,val in model.state_dict().items()},'config':config,'cat_dims':prep.cat_dims},dest/'model.pt')
    save_prediction(dest/'oof_predictions.parquet',model,prep,oof)
    elapsed=time.perf_counter()-start
    meta={'strategy':strategy,'depth':d,'seed':s,'fold':k,'source_reference':str(source.relative_to(REV)),'identity_reuse':False,'selected_epochs':selected,'fitted_epochs':epochs,'additional_training_seconds':elapsed,'gradient_site_ids':sorted(fitdf.physical_site_id.unique()),'early_stopping_site_ids':list(selection_sites),'forbidden_site_ids':sorted(forbidden),'training_records':len(fitdf),'permitted_records':len(train),'trace':trace,'selection_replay_max_rmse_difference':replay_error,'source_model_parameters':base['model']}
    write(dest/'manifest.json',meta)
    del model;torch.cuda.empty_cache()
    print(f'FIT_COMPLETE depth={d} seed={s} fold={k} strategy={strategy} epochs={epochs} minutes={elapsed/60:.2f}',flush=True)
def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['freeze','fit'],default='fit');args=parser.parse_args()
    protocol=freeze()
    if args.stage=='freeze':print('PROTOCOL_FROZEN',flush=True);return
    # Fixed order prioritizes the duration-only control, then the checkpoint policy.
    for strategy in ['warmup_floor10','retain_internal_best']:
        for d in [5,20,50]:
            for s in [42,7,555]:
                for k in range(5):fit_one(d,s,k,strategy)
    write(HERE/'fit_completed.json',{'utc':datetime.now(timezone.utc).isoformat(),'variant_folds':90,'protocol_sha256':sha(HERE/'protocol.json')})
    print('ALL_STRATEGY_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
