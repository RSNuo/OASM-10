"""50 cm input factorial; immutable v6 sources and a common floor-10 refit policy.

Fitting never loads test tables. All four configurations are frozen before the
new fits, and complete OOF results precede any new test inference.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil, sys, time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
REV = HERE.parent
H = REV / 'training_strategy_review'
sys.path.insert(0, str(REV))
from oasm.features import BASE, SOIL, PRECIP, CATS
from oasm.preprocessing import Preprocessor, assert_excluded
from oasm.training import seed_everything, new_model, fit_loop, tensors, predict_tensors

SEEDS = [42, 7, 555]
CONFIGS = {'A': {'cascade': True, 'precip': False},
           'B': {'cascade': False, 'precip': False},
           'C': {'cascade': True, 'precip': True},
           'D': {'cascade': False, 'precip': True}}

def now(): return datetime.now(timezone.utc).isoformat()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def destination(c, s, k): return HERE / f'models/{c}_seed{s}/outer{k}'
def source(s, k, cascade=True): return REV / f'models/50cm/{"ma" if cascade else "no_cascade"}_seed{s}/outer{k}'

def get_data(k):
    p = REV / f'data/cascade/50cm/outer{k}'
    tr = pd.read_parquet(p / 'train.parquet'); oo = pd.read_parquet(p / 'oof.parquet')
    forbidden = set(read(p / 'provenance.json')['forbidden_sites'])
    assert set(tr.physical_site_id).isdisjoint(forbidden)
    assert set(tr.physical_site_id).isdisjoint(oo.physical_site_id)
    assert not tr.record_id.duplicated().any() and not oo.record_id.duplicated().any()
    return tr, oo, forbidden

def numeric(c):
    x = BASE + SOIL + (PRECIP if CONFIGS[c]['precip'] else [])
    return x + (['soil_moisture_5cm'] if CONFIGS[c]['cascade'] else [])

def model_config(c, s, k):
    base = read(source(s,k) / 'manifest.json')
    cfg = {q: base[q] for q in ['depth','architecture','seed','model','max_epochs','patience','batch_size']}
    cfg.update(variant=f'factorial_{c}', numeric=numeric(c), categorical=CATS,
               epoch_selection='Internal 15% site holdout minimum RMSE; fresh full-permitted refit for max(selected_epochs,10)')
    return cfg

def freeze():
    paths = [Path(__file__), REV/'data/all_50cm.parquet', H/'protocol.json']
    paths += [REV/f'oasm/{n}.py' for n in ['training','features','models','preprocessing']]
    for k in range(5):
        paths += [REV/f'data/cascade/50cm/outer{k}/{n}' for n in ['train.parquet','oof.parquet','test.parquet','provenance.json']]
        for s in SEEDS:
            for yes in [True, False]:
                p = source(s,k,yes)
                paths += [p/n for n in ['model.pt','preprocessor.joblib','manifest.json','oof_predictions.parquet']]
            hp = H/f'models/50cm/warmup_floor10_seed{s}/outer{k}'
            paths += [hp/n for n in ['manifest.json','model.pt','preprocessor.joblib','oof_predictions.parquet'] if (hp/n).exists()]
    hashes = {str(p.relative_to(REV)):sha(p) for p in paths}
    if (HERE/'protocol.json').exists():
        old = read(HERE/'protocol.json'); assert old['sources_sha256'] == hashes, 'Frozen source changed'
        return old
    # Confirm that reused floor-10 results still refer to the verified inputs.
    oldh = read(H/'protocol.json')['sources_sha256']
    for name,digest in hashes.items():
        if name in oldh: assert digest == oldh[name], name
    rows = []
    for k in range(5):
        tr,oo,forbidden = get_data(k)
        assert all(n in tr and n in oo for n in numeric('C')+CATS)
        rows.append({'fold':k,'train_records':len(tr),'oof_records':len(oo),
                     'train_sites':tr.physical_site_id.nunique(),'oof_sites':oo.physical_site_id.nunique(),
                     'precip_missing_train':{c:int(tr[c].isna().sum()) for c in PRECIP}})
        for s in SEEDS:
            a,b=next(GroupShuffleSplit(1,test_size=.15,random_state=s+1000).split(tr,groups=tr.physical_site_id))
            for yes in [True,False]:
                m=read(source(s,k,yes)/'manifest.json')
                assert set(m['gradient_site_ids']) == set(tr.physical_site_id)
                assert set(m['selection_gradient_site_ids']) == set(tr.iloc[a].physical_site_id)
                assert set(m['early_stopping_site_ids']) == set(tr.iloc[b].physical_site_id)
                assert set(m['forbidden_site_ids']) == forbidden
                assert m['model'] == model_config('A',s,k)['model']
    protocol={'created_utc':now(),'configurations':CONFIGS,'seeds':SEEDS,'outer_folds':list(range(5)),
              'policy':'All configurations: internal 15% site holdout chooses epoch, fresh full-permitted refit max(epoch,10), original 150-epoch LR/Huber schedule; no architecture or loss search.',
              'upstream':'Fixed v6 nested 5 cm inputs, indexed by outer fold; conditional downstream input experiment, not upstream retraining. Outer OOF and external sites remain excluded from upstream gradient/selection/preprocessing. Inner target-depth epoch selection uses these fixed upstream features.',
              'source_history':'Post-hoc hypothesis motivated by prior repeated examination of this test set; new OOF freeze is not prospective untouched-test validation.',
              'reuse':'A reuses verified floor-10 weights; B reuses original no-cascade fits only where selected epochs >=10, otherwise same preprocessor and initialisation refit 10. C and D perform their own internal epoch selection and floor-10 refit.',
              'ranking':'Lowest mean of the three per-seed OOF pooled RMSE values; exact tie resolved A,B,C,D. Rank all four before any new test inference. Report equal-site RMSE and anomaly correlation separately; no automatic product promotion.',
              'contrasts':['B-A','C-A','D-A','C-B','D-B','D-C'],
              'bootstrap':{'resamples':2000,'confidence':.95,'method':'Percentile resampling of whole physical-site clusters with all records and multiplicity; identical site draws across seeds and configurations; pointwise conditional intervals, no multiplicity adjustment.','seed':20260914},
              'test_gate':'Fit stage reads train and OOF only; freeze all fitted weights, preprocessors and OOF predictions plus OOF ranking before test inference for every configuration.',
              'product_and_manuscripts':'Remain unchanged during this input experiment.',
              'numeric_features':{c:numeric(c) for c in CONFIGS},'fold_records':rows,'sources_sha256':hashes}
    write(HERE/'protocol.json',protocol)
    return protocol

def save_pred(path,model,prep,df):
    out=df[['record_id','source_station','physical_site_id','target_time','y']].copy()
    out['prediction']=predict_tensors(model,tensors(prep,df))
    assert np.isfinite(out.prediction).all()
    out.to_parquet(path,index=False)

def fit_one(c,s,k):
    dest=destination(c,s,k);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():
        m=read(dest/'manifest.json')
        assert (dest/'oof_predictions.parquet').exists()
        assert sha(dest/'oof_predictions.parquet') == m['oof_sha256']
        print(f'RESUME {c} seed{s} fold{k}',flush=True);return
    start=time.perf_counter();tr,oo,forbidden=get_data(k);cfg=model_config(c,s,k)
    base=read(source(s,k,c!='B')/'manifest.json')
    a,b=next(GroupShuffleSplit(1,test_size=.15,random_state=s+1000).split(tr,groups=tr.physical_site_id))
    grad,monitor=tr.iloc[a],tr.iloc[b]
    selection_sites=sorted(monitor.physical_site_id.unique())
    m={'config_id':c,'config':cfg,'seed':s,'fold':k,'started_utc':now(),
       'gradient_site_ids':sorted(tr.physical_site_id.unique()),'selection_gradient_site_ids':sorted(grad.physical_site_id.unique()),
       'early_stopping_site_ids':selection_sites,'forbidden_site_ids':sorted(forbidden),
       'train_records':len(tr),'oof_records':len(oo),'policy':'floor10'}
    reuse_path=None
    if c=='A':
        hp=H/f'models/50cm/warmup_floor10_seed{s}/outer{k}';hm=read(hp/'manifest.json')
        reuse_path=REV/hm['source_reference'] if hm['identity_reuse'] else hp
        selected=int(base['selected_epochs']);epochs=int(hm['fitted_epochs'])
        assert epochs==max(selected,10)
        shutil.copy2(hp/'oof_predictions.parquet',dest/'oof_predictions.parquet')
        refit=base['refit_trace'] if hm['identity_reuse'] else hm['trace']
    elif c=='B':
        selected=int(base['selected_epochs']);epochs=max(selected,10)
        if selected>=10:
            reuse_path=source(s,k,False)
            shutil.copy2(reuse_path/'oof_predictions.parquet',dest/'oof_predictions.parquet')
            refit=base['refit_trace']
    if reuse_path is not None:
        pack=torch.load(reuse_path/'model.pt',map_location='cpu',weights_only=False)
        for key in ['numeric','categorical','model','architecture','seed']:
            assert pack['config'][key]==cfg[key], (key,reuse_path)
        prep=joblib.load(reuse_path/'preprocessor.joblib');assert_excluded(prep,forbidden)
        assert set(prep.fitted_site_ids)==set(tr.physical_site_id)
        m.update(identity_reuse=True,weight_directory=str(reuse_path.relative_to(REV)),parameter_count=sum(v.numel() for v in pack['state_dict'].values()))
        selection_trace=base['selection_trace'];del pack
    else:
        selection_trace=base['selection_trace'] if c=='B' else None
        if c=='B':
            prep=joblib.load(source(s,k,False)/'preprocessor.joblib')
        else:
            ps=Preprocessor(cfg['numeric'],CATS).fit(grad);assert_excluded(ps,forbidden|set(selection_sites))
            seed_everything(s);model=new_model(cfg,ps)
            selected,selection_trace,_=fit_loop(model,tensors(ps,grad,True),s,150,150,tensors(ps,monitor,True))
            epochs=max(selected,10);del model,ps;torch.cuda.empty_cache()
            prep=Preprocessor(cfg['numeric'],CATS).fit(tr)
        assert_excluded(prep,forbidden)
        seed_everything(s);model=new_model(cfg,prep)
        _,refit,_=fit_loop(model,tensors(prep,tr,True),s,epochs,150)
        if c=='B':
            prefix=max(abs(u['loss']-v['loss']) for u,v in zip(refit,base['refit_trace']))
            assert prefix<1e-7,('Refit prefix differs',prefix)
            for u,v in zip(refit,base['refit_trace']): assert u['lr']==v['lr'] and u['delta']==v['delta']
            m['reference_refit_prefix_max_loss_difference']=prefix
        joblib.dump(prep,dest/'preprocessor.joblib')
        torch.save({'state_dict':{q:v.detach().cpu() for q,v in model.state_dict().items()},'config':cfg,'cat_dims':prep.cat_dims},dest/'model.pt')
        save_pred(dest/'oof_predictions.parquet',model,prep,oo)
        m.update(identity_reuse=False,weight_directory=str(dest.relative_to(REV)),parameter_count=sum(p.numel() for p in model.parameters()))
        del model;torch.cuda.empty_cache()
    m.update(selected_epochs=selected,fitted_epochs=epochs,selection_trace=selection_trace,refit_trace=refit,
             completed_utc=now(),additional_seconds=time.perf_counter()-start,
             hardware={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'float32_matmul_precision':torch.get_float32_matmul_precision()},
             oof_sha256=sha(dest/'oof_predictions.parquet'))
    wp=REV/m['weight_directory']
    m['weights_sha256']=sha(wp/'model.pt');m['preprocessor_sha256']=sha(wp/'preprocessor.joblib')
    write(dest/'manifest.json',m)
    print(f'FOLD_COMPLETE {c} seed{s} fold{k} selected={selected} fitted={epochs} reuse={m["identity_reuse"]} minutes={m["additional_seconds"]/60:.2f}',flush=True)

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['freeze','fit'],default='fit');p.add_argument('--configs',default='ABCD');p.add_argument('--one',action='store_true');args=p.parse_args()
    freeze()
    if args.stage=='freeze':print('PROTOCOL_FROZEN',flush=True);return
    for c in args.configs:
        assert c in CONFIGS
        for s in SEEDS:
            for k in range(5):
                fit_one(c,s,k)
                if args.one:return
    write(HERE/f'fit_{args.configs}_complete.json',{'utc':now(),'configs':list(args.configs),'protocol_sha256':sha(HERE/'protocol.json')})
    print(f'FITS_COMPLETE {args.configs}',flush=True)

if __name__=='__main__':main()
