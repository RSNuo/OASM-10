"""Measured local inference cost; run after GPU training has finished."""
from pathlib import Path
import gc,json,platform,subprocess,time
import joblib,numpy as np,pandas as pd,torch
from oasm.training import load_model,tensors,predict_tensors,tree_matrix,DEVICE
from oasm.features import soil_for_depth

REV=Path(__file__).resolve().parent;N=1024;REPEATS=20

def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()

def bank(depth,variant,seed=42):
    ans=[];load_times=[];meta=[];sizes=[]
    for k in range(5):
        path=REV/f'models/{depth}cm/{variant}_seed{seed}/outer{k}'
        m=json.loads((path/'manifest.json').read_text());meta.append(m)
        sync();start=time.perf_counter()
        if variant in ['rf','xgb']:
            pack=joblib.load(path/'model.joblib');obj=(pack['model'],pack['preprocessor'],pack['config'])
            size=(path/'model.joblib').stat().st_size
        else:
            obj=load_model(path);size=(path/'model.pt').stat().st_size+(path/'preprocessor.joblib').stat().st_size
        sync();load_times.append(time.perf_counter()-start);ans.append(obj);sizes.append(size)
    return ans,{'load_seconds':sum(load_times),'disk_bytes':sum(sizes),
         'training_seconds_including_internal_selection_and_refit':sum(m['training_seconds'] for m in meta),
         'fold_parameters':[m.get('parameter_count') for m in meta],
         'fold_tree_counts':[m.get('selected_trees') for m in meta],
         'fold_selected_epochs':[m.get('selected_epochs') for m in meta]}

def pred(obj,f,tree=False):
    model,prep,_=obj
    if tree:return model.predict(tree_matrix(prep,f))
    return predict_tensors(model,tensors(prep,f),batch=N)

def measure(call):
    for _ in range(3):call()
    samples=[];reference=None
    for _ in range(REPEATS):
        sync();t=time.perf_counter();y=call();sync();samples.append(time.perf_counter()-t)
        if reference is None:reference=y
        if not np.allclose(reference,y,atol=1e-6,rtol=1e-6):raise ValueError('Inference not deterministic in evaluation mode')
    return {'repeats':REPEATS,'batch_records':N,'median_seconds':float(np.median(samples)),
            'p10_seconds':float(np.quantile(samples,.1)),'p90_seconds':float(np.quantile(samples,.9))}

def main():
    torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    hardware={'system':platform.platform(),'processor':platform.processor(),'python':platform.python_version(),
      'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'torch_threads':torch.get_num_threads(),
      'device':DEVICE,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:hardware['gpu_state']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):pass
    result={'hardware':hardware,'timing_boundary':'Resident models: fitted preprocessing, CPU-to-device inputs, prediction and device-to-CPU outputs. Excludes disk model loading, GEE retrieval and GeoTIFF writing. Depth product includes all five paired upstream and downstream evaluations. No competing training process is launched by this script.',
      'batch_selection':'First1024 test record IDs sorted lexicographically at each depth, independent of errors; same records across model families.','models':{}}
    for d in [5,20,50]:
        f=pd.read_parquet(REV/f'data/all_{d}cm.parquet');f=f[f.split=='test'].sort_values('record_id').head(N).copy()
        up,upmeta=bank(5,'ma') if d>5 else (None,None)
        for variant in ['ma','concat','ft','rf','xgb']+(['static_forcing'] if d==5 else []):
            fitted,metadata=bank(d,variant);tree=variant in ['rf','xgb']
            def call():
                outputs=[]
                for k,obj in enumerate(fitted):
                    z=soil_for_depth(f,d)
                    if d>5:z['soil_moisture_5cm']=pred(up[k],soil_for_depth(f,5))
                    outputs.append(pred(obj,z,tree))
                return np.mean(outputs,axis=0)
            metadata['inference']=measure(call);metadata['model_evaluations']=10 if d>5 else 5
            if d>5:metadata['shared_upstream']=upmeta
            result['models'][f'{d}cm_{variant}']=metadata
            del fitted;gc.collect();torch.cuda.empty_cache()
            print('COST COMPLETE',d,variant,metadata['inference']['median_seconds'],flush=True)
        if up is not None:del up;gc.collect();torch.cuda.empty_cache()
    # Secondary5cm ensembles use exactly the fitted OOF-selected weights.
    sel=json.loads((REV/'models/5cm/ma_oof_selected/selection.json').read_text())
    f=pd.read_parquet(REV/'data/all_5cm.parquet');f=f[f.split=='test'].sort_values('record_id').head(N)
    slots=sel['selected_slots'];banks={};metas={}
    for seed in sorted(set(slots)):banks[seed],metas[seed]=bank(5,'ma',seed)
    def ensemble_call():
        parts={s:np.mean([pred(o,soil_for_depth(f,5)) for o in fitted],axis=0) for s,fitted in banks.items()}
        return np.mean([parts[s] for s in slots],axis=0)
    result['models']['5cm_oof_ensemble']={'inference':measure(ensemble_call),'selected_slots':slots,
        'model_evaluations':5*len(banks),'disk_bytes':sum(m['disk_bytes'] for m in metas.values()),
        'load_seconds':sum(m['load_seconds'] for m in metas.values()),
        'training_seconds_including_internal_selection_and_refit':sum(m['training_seconds_including_internal_selection_and_refit'] for m in metas.values()),
        'interpretation':'Repeated greedy seats are weights; each distinct seed/fold model is evaluated once. Candidate-search training cost is additional.'}
    nested=[json.loads(p.read_text()) for p in (REV/'models/upstream_nested').rglob('manifest.json')]
    result['shared_nested_cascade_training']={'fits':len(nested),'seconds':sum(m['training_seconds'] for m in nested),
        'interpretation':'One-time upstream input-generation cost reused by all deeper controls; not charged again per downstream architecture.'}
    (REV/'analysis/cost.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('V6 COST COMPLETE',flush=True)

if __name__=='__main__':main()
