"""Training with internal epoch selection and refitting on allowed sites.

Neither outer OOF sites nor external test sites may fit transforms, select an
epoch, or contribute gradients. The manifests expose each set for verification.
"""
from pathlib import Path
import copy,hashlib,json,random,time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from .features import feature_names
from .preprocessing import Preprocessor,assert_excluded
from .models import SoilTransformer,model_config

EPOCHS=150
PATIENCE=20
BATCH=1024
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def weights(y,uniform=False):
    if uniform:return np.ones_like(y)
    lo,hi=np.quantile(y,[.1,.9]);w=np.ones_like(y)
    a=y<lo;b=y>hi
    if a.any():w[a]=1.5-.5*(y[a]-y[a].min())/max(lo-y[a].min(),1e-8)
    if b.any():w[b]=1+.5*(y[b]-hi)/max(y[b].max()-hi,1e-8)
    return w/w.mean()

def fingerprint(train,evals,config):
    h=hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for frame in [train]+list(evals.values()):
        h.update('|'.join(frame.record_id.astype(str)).encode())
        # Include all model input values and labels to reject stale caches.
        cols=[c for c in config['numeric'] if c in frame]
        h.update(frame[cols].to_numpy(dtype=np.float64).tobytes())
    h.update(train.y.to_numpy(dtype=np.float64).tobytes())
    return h.hexdigest()

def tensors(prep,frame,include_y=False):
    x,c=prep.transform(frame)
    out=(torch.from_numpy(x).to(DEVICE),torch.from_numpy(c).to(DEVICE))
    if include_y:out=out+(torch.tensor(frame.y.to_numpy(dtype=np.float32),device=DEVICE),)
    return out

@torch.no_grad()
def predict_tensors(model,data,batch=8192):
    model.eval();x,c=data[:2]
    return torch.cat([model(x[i:i+batch],c[i:i+batch]) for i in range(0,len(x),batch)]).float().cpu().numpy()

def new_model(config,prep):
    return SoilTransformer(config['numeric'],config['categorical'],prep.cat_dims,variant=config['architecture'],**config['model']).to(DEVICE)

def fit_loop(model,data,seed,epochs,max_epochs,monitor=None,patience=PATIENCE):
    seed_everything(seed)
    x,c,y=data;w=torch.tensor(weights(y.cpu().numpy()),device=DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.02)
    best=float('inf');selected=epochs;best_state=None;stale=0;trace=[]
    for epoch in range(epochs):
        if epoch<10:lr=(epoch+1)/10
        else:lr=.1+.9*.5*(1+np.cos(np.pi*(epoch-10)/max(1,max_epochs-10)))
        for group in opt.param_groups:group['lr']=2e-4*lr
        delta=.02+.5*(.5-.02)*(1+np.cos(np.pi*epoch/max_epochs))
        model.train();perm=torch.randperm(len(x),device=DEVICE)
        loss_sum=0.
        # All allowed records participate; the final partial batch is retained.
        for start in range(0,len(x),BATCH):
            idx=perm[start:start+BATCH]
            opt.zero_grad(set_to_none=True)
            err=model(x[idx],c[idx])-y[idx];ae=err.abs()
            loss=(torch.where(ae<=delta,.5*err.square(),delta*(ae-.5*delta))*w[idx]).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Training loss is not finite')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();loss_sum+=float(loss.detach())*len(idx)
        record={'epoch':epoch+1,'loss':loss_sum/len(x),'lr':lr*2e-4,'delta':float(delta)}
        if monitor is not None:
            pred=predict_tensors(model,monitor)
            score=float(np.sqrt(np.mean((pred-monitor[2].cpu().numpy())**2)))
            record['internal_rmse']=score
            if score<best:
                best=score;selected=epoch+1;stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if stale>=patience:
                trace.append(record);break
        trace.append(record)
        if (epoch+1)%20==0:print(f'    epoch {epoch+1}'+(f' internal RMSE={score:.5f}' if monitor is not None else ' refit'),flush=True)
    return selected,trace,best_state

def fit_neural(train,evals,directory,depth,variant,seed,forbidden_sites,max_epochs=EPOCHS,min_refit_epochs=0):
    if max_epochs < 1 or not 0 <= min_refit_epochs <= max_epochs:
        raise ValueError('Require 0 <= min_refit_epochs <= max_epochs and max_epochs >= 1')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    num,cat=feature_names(depth,variant)
    arch=variant if variant in ['concat','ft','no_modality','no_static','static_only'] else 'ma'
    config={'depth':depth,'variant':variant,'architecture':arch,'seed':seed,'model':model_config(depth),'numeric':num,'categorical':cat,'max_epochs':max_epochs,'patience':PATIENCE,'batch_size':BATCH,'epoch_selection':'internal group holdout, then refit all permitted sites'}
    if min_refit_epochs:
        config['min_refit_epochs']=min_refit_epochs
    digest=fingerprint(train,evals,config);manifest=directory/'manifest.json'
    expected=['model.pt','preprocessor.joblib']+[f'{name}_predictions.parquet' for name in evals]
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        if existing['input_sha256']!=digest:raise RuntimeError(f'Stale model cache: {directory}')
        if all((directory/n).exists() for n in expected):return existing
    allowed=set(train.physical_site_id)
    assert allowed.isdisjoint(forbidden_sites),'Forbidden site in permitted training records'
    for name,df in evals.items():assert allowed.isdisjoint(df.physical_site_id),f'{name}: model evaluation overlaps training groups'
    select_tr,select_va=next(GroupShuffleSplit(1,test_size=.15,random_state=seed+1000).split(train,groups=train.physical_site_id))
    fitdf=train.iloc[select_tr];mondf=train.iloc[select_va]
    start=time.perf_counter()
    prep_select=Preprocessor(num,cat).fit(fitdf)
    assert_excluded(prep_select,forbidden_sites|set(mondf.physical_site_id))
    seed_everything(seed);model=new_model(config,prep_select)
    selected,trace,_=fit_loop(model,tensors(prep_select,fitdf,True),seed,max_epochs,max_epochs,tensors(prep_select,mondf,True))
    del model;torch.cuda.empty_cache()
    # Refit the same fixed architecture using all allowed records for the selected epoch count.
    prep=Preprocessor(num,cat).fit(train);assert_excluded(prep,forbidden_sites)
    seed_everything(seed);model=new_model(config,prep)
    refit_epochs=max(selected,min_refit_epochs)
    _,refit_trace,_=fit_loop(model,tensors(prep,train,True),seed,refit_epochs,max_epochs)
    training_seconds=time.perf_counter()-start
    joblib.dump(prep,directory/'preprocessor.joblib')
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':config,'cat_dims':prep.cat_dims},directory/'model.pt')
    for name,df in evals.items():
        out=df[['record_id','source_station','physical_site_id','target_time','y']].copy()
        out['prediction']=predict_tensors(model,tensors(prep,df));out.to_parquet(directory/f'{name}_predictions.parquet',index=False)
    meta={**config,'input_sha256':digest,'selected_epochs':selected,'training_seconds':training_seconds,'parameter_count':sum(p.numel() for p in model.parameters()),'gradient_site_ids':sorted(allowed),'early_stopping_site_ids':sorted(set(mondf.physical_site_id)),'selection_gradient_site_ids':sorted(set(fitdf.physical_site_id)),'forbidden_site_ids':sorted(forbidden_sites),'evaluation_site_ids':{n:sorted(set(f.physical_site_id)) for n,f in evals.items()},'selection_trace':trace,'refit_trace':refit_trace,'hardware':{'device':DEVICE,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch':torch.__version__,'float32_matmul_precision':torch.get_float32_matmul_precision()}}
    meta['refit_epochs']=refit_epochs
    manifest.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    del model;torch.cuda.empty_cache()
    print(f'COMPLETE depth={depth} variant={variant} seed={seed} epochs={selected} elapsed={training_seconds/60:.1f}min {directory}',flush=True)
    return meta

def load_model(directory):
    directory=Path(directory)
    pack=torch.load(directory/'model.pt',map_location=DEVICE,weights_only=False)
    prep=joblib.load(directory/'preprocessor.joblib');model=new_model(pack['config'],prep)
    model.load_state_dict(pack['state_dict'],strict=True);model.eval()
    return model,prep,pack['config']

def predict_frame(directory,frame):
    model,prep,_=load_model(directory)
    pred=predict_tensors(model,tensors(prep,frame));del model;torch.cuda.empty_cache()
    return pred

def tree_matrix(prep,df):
    x,c=prep.transform(df)
    # One-hot columns are defined by permitted training vocabularies only.
    blocks=[x]
    for i,n in enumerate(prep.cat_dims):blocks.append(np.eye(n,dtype=np.float32)[c[:,i]])
    return np.concatenate(blocks,axis=1)

def fit_tree(train,evals,directory,depth,variant,seed,forbidden_sites):
    from sklearn.ensemble import RandomForestRegressor
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    num,cat=feature_names(depth,'ma')
    config={'depth':depth,'variant':variant,'seed':seed,'numeric':num,'categorical':cat,'epoch_selection':'internal group holdout for XGBoost, followed by refit'}
    digest=fingerprint(train,evals,config);manifest=directory/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['input_sha256']!=digest:raise RuntimeError(f'Stale tree cache: {directory}')
        return old
    assert set(train.physical_site_id).isdisjoint(forbidden_sites)
    for n,f in evals.items():assert set(train.physical_site_id).isdisjoint(f.physical_site_id)
    start=time.perf_counter();selected=300;early=[]
    if variant=='xgb':
        from xgboost import XGBRegressor
        a,b=next(GroupShuffleSplit(1,test_size=.15,random_state=seed+1000).split(train,groups=train.physical_site_id))
        grad,monitor=train.iloc[a],train.iloc[b];early=sorted(monitor.physical_site_id.unique())
        p=Preprocessor(num,cat).fit(grad);assert_excluded(p,forbidden_sites|set(early))
        params=dict(learning_rate=.05,max_depth=8,subsample=.8,colsample_bytree=.8,min_child_weight=5,tree_method='hist',device='cpu',random_state=seed,n_jobs=8,eval_metric='rmse')
        provisional=XGBRegressor(n_estimators=3000,early_stopping_rounds=100,**params)
        provisional.fit(tree_matrix(p,grad),grad.y.to_numpy(),eval_set=[(tree_matrix(p,monitor),monitor.y.to_numpy())],verbose=False)
        selected=provisional.best_iteration+1;del provisional
        model=XGBRegressor(n_estimators=selected,**params)
    else:model=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,max_features=.5,n_jobs=8,random_state=seed)
    prep=Preprocessor(num,cat).fit(train);assert_excluded(prep,forbidden_sites)
    model.fit(tree_matrix(prep,train),train.y.to_numpy())
    duration=time.perf_counter()-start
    joblib.dump({'model':model,'preprocessor':prep,'config':config},directory/'model.joblib')
    for name,df in evals.items():
        out=df[['record_id','source_station','physical_site_id','target_time','y']].copy();out['prediction']=model.predict(tree_matrix(prep,df))
        out.to_parquet(directory/f'{name}_predictions.parquet',index=False)
    meta={**config,'input_sha256':digest,'training_seconds':duration,'selected_trees':selected,'gradient_site_ids':sorted(train.physical_site_id.unique()),'early_stopping_site_ids':early,'forbidden_site_ids':sorted(forbidden_sites)}
    manifest.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'COMPLETE depth={depth} {variant} seed={seed} {duration/60:.1f}min {directory}',flush=True)
    return meta
