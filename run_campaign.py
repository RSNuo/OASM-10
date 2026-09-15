"""Resume-safe, nested cross-depth evaluation; all paths are package-relative."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from oasm.features import soil_for_depth
from oasm.training import fit_neural,fit_tree,predict_frame

REV=Path(__file__).resolve().parent;DATA=REV/'data';MODELS=REV/'models'

def load_data():return {d:pd.read_parquet(DATA/f'all_{d}cm.parquet') for d in [5,20,50]}
def outer_path(d,v,s,k):return MODELS/f'{d}cm/{v}_seed{s}/outer{k}'

def fit_surface(frames,variant='ma',seed=42,epochs=150):
    f=frames[5];test=f[f.split=='test'];dev=f[f.split=='development']
    for k in range(5):
        train=dev[dev.outer_fold!=k];va=dev[dev.outer_fold==k]
        forbidden=set(test.physical_site_id)|set(va.physical_site_id)
        func=fit_tree if variant in ['rf','xgb'] else fit_neural
        kwargs={} if variant in ['rf','xgb'] else {'max_epochs':epochs}
        func(train,{'oof':va,'test':test},outer_path(5,variant,seed,k),5,variant,seed,forbidden,**kwargs)

def nested_inputs(frames,epochs=150):
    """Every upstream model's ancestors exclude all outer OOF/external sites."""
    allrows=pd.concat([f[['physical_site_id','split','outer_fold']] for f in frames.values()],ignore_index=True)
    external=set(allrows.loc[allrows.split=='test','physical_site_id'])
    for k in range(5):
        permitted=allrows[(allrows.split=='development')&(allrows.outer_fold!=k)].reset_index(drop=True)
        prohibited=external|set(allrows.loc[(allrows.split=='development')&(allrows.outer_fold==k),'physical_site_id'])
        inner={}
        for j,(_,vi) in enumerate(GroupKFold(5,shuffle=True,random_state=4200+k).split(permitted,groups=permitted.physical_site_id)):
            inner.update({g:j for g in permitted.iloc[vi].physical_site_id.unique()})
        train5=frames[5][(frames[5].split=='development')&(frames[5].outer_fold!=k)].copy()
        train5['inner_fold']=train5.physical_site_id.map(inner)
        deep_dev=pd.concat([soil_for_depth(f[(f.split=='development')&(f.outer_fold!=k)],5) for d,f in frames.items() if d>5],ignore_index=True)
        deep_dev['inner_fold']=deep_dev.physical_site_id.map(inner)
        for j in range(5):
            train=train5[train5.inner_fold!=j];va=deep_dev[deep_dev.inner_fold==j]
            excluded=prohibited|{g for g,h in inner.items() if h==j}
            path=MODELS/f'upstream_nested/outer{k}/inner{j}'
            fit_neural(train,{'cascade_training':va},path,5,'ma',42,excluded,max_epochs=epochs)
        nested=pd.concat([pd.read_parquet(MODELS/f'upstream_nested/outer{k}/inner{j}/cascade_training_predictions.parquet') for j in range(5)],ignore_index=True)
        assert not nested.record_id.duplicated().any()
        for d in [20,50]:
            out=DATA/f'cascade/{d}cm/outer{k}';out.mkdir(parents=True,exist_ok=True)
            f=frames[d]
            train=f[(f.split=='development')&(f.outer_fold!=k)].copy()
            train=train.merge(nested[['record_id','prediction']],on='record_id',validate='one_to_one')
            train=train.rename(columns={'prediction':'soil_moisture_5cm'})
            assert train.soil_moisture_5cm.notna().all()
            train.to_parquet(out/'train.parquet',index=False)
            for name,mask in [('oof',(f.split=='development')&(f.outer_fold==k)),('test',f.split=='test')]:
                z=f[mask].copy();z['soil_moisture_5cm']=predict_frame(outer_path(5,'ma',42,k),soil_for_depth(z,5))
                z.to_parquet(out/f'{name}.parquet',index=False)
            (out/'provenance.json').write_text(json.dumps({'outer_fold':k,'inner_fold_seed':4200+k,'upstream_seed':42,'forbidden_sites':sorted(prohibited),'training_input':'one nested held-site-out upstream model per inner fold','evaluation_input':'paired outer-fold upstream reference, without averaging before downstream inference'},indent=2),encoding='utf-8')
        print(f'NESTED INPUTS READY outer={k}',flush=True)

def fit_depth(d,variant,seed,epochs=150):
    for k in range(5):
        p=DATA/f'cascade/{d}cm/outer{k}'
        train=pd.read_parquet(p/'train.parquet');evals={n:pd.read_parquet(p/f'{n}.parquet') for n in ['oof','test']}
        forbidden=set(json.loads((p/'provenance.json').read_text())['forbidden_sites'])
        func=fit_tree if variant in ['rf','xgb'] else fit_neural
        kwargs={} if variant in ['rf','xgb'] else {'max_epochs':epochs}
        func(train,evals,outer_path(d,variant,seed,k),d,variant,seed,forbidden,**kwargs)

def summarize():
    for depth in [5,20,50]:
        for run in (MODELS/f'{depth}cm').glob('*'):
            if not all((run/f'outer{k}/manifest.json').exists() for k in range(5)):continue
            vals=[];tests=[]
            for k in range(5):
                vals.append(pd.read_parquet(run/f'outer{k}/oof_predictions.parquet'))
                t=pd.read_parquet(run/f'outer{k}/test_predictions.parquet').set_index('record_id');tests.append(t)
            oof=pd.concat(vals,ignore_index=True);assert not oof.record_id.duplicated().any()
            test=tests[0].copy();test['prediction']=np.mean([z.reindex(test.index).prediction.to_numpy() for z in tests],axis=0)
            test=test.reset_index();oof.to_parquet(run/'oof_predictions.parquet',index=False);test.to_parquet(run/'test_predictions.parquet',index=False)
            metrics={}
            for name,z in [('oof',oof),('test',test)]:
                e=z.prediction-z.y;metrics[name]={'n':len(z),'sites':z.physical_site_id.nunique(),'rmse':float(np.sqrt(np.mean(e**2))),'bias':float(e.mean()),'ubrmse':float(np.std(e)),'r2':float(1-np.sum(e**2)/np.sum((z.y-z.y.mean())**2))}
            (run/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
            print(f'RESULT depth={depth} run={run.name} OOF={metrics["oof"]["rmse"]:.5f} Test={metrics["test"]["rmse"]:.5f}',flush=True)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['surface','nested','depth','summarize','all'],default='all');parser.add_argument('--variant',default=None);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--depth',type=int,default=20);parser.add_argument('--epochs',type=int,default=150)
    args=parser.parse_args();frames=load_data()
    if args.stage=='surface':fit_surface(frames,args.variant or 'ma',args.seed,args.epochs)
    elif args.stage=='nested':nested_inputs(frames,args.epochs)
    elif args.stage=='depth':fit_depth(args.depth,args.variant or 'ma',args.seed,args.epochs)
    elif args.stage=='summarize':summarize()
    else:
        # Fixed suite, declared before revised outcomes. No metric controls this list.
        fit_surface(frames,'ma',42,args.epochs)
        nested_inputs(frames,args.epochs)
        for d in [20,50]:
            for variant in ['ma','concat','ft','no_cascade']:
                for seed in [42,7,555]:fit_depth(d,variant,seed,args.epochs)
            for variant in ['rf','xgb']:fit_depth(d,variant,42,args.epochs)
        for variant in ['ma','concat','ft','no_modality','static_forcing']:
            for seed in [42,7,555]:fit_surface(frames,variant,seed,args.epochs)
        for variant in ['no_static','static_only','rf','xgb']:fit_surface(frames,variant,42,args.epochs)
        summarize();print('V6 CORE CAMPAIGN COMPLETE',flush=True)

if __name__=='__main__':main()
