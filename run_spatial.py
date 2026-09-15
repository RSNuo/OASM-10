"""A fixed 5-degree block diagnostic on the same revised 5-cm development pool."""
from pathlib import Path
import itertools,json,sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from oasm.training import fit_neural,fit_tree
from analyze_results import statistics,paired_seed_bootstrap,read

REV=Path(__file__).resolve().parent

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    f=pd.read_parquet(REV/'data/all_5cm.parquet')
    # One representative coordinate per spatial group avoids aliases straddling a
    # numeric boundary. Grid cells are not assumed ecologically homogeneous.
    xy=f.groupby('physical_site_id')[['longitude','latitude']].median()
    xy['block']=np.floor(xy.longitude/5).astype(int).astype(str)+'_'+np.floor(xy.latitude/5).astype(int).astype(str)
    f=f.merge(xy[['block']],left_on='physical_site_id',right_index=True,validate='many_to_one')
    dev=f[f.split=='development'].copy();test=f[f.split=='test'];folds=[]
    for k,(a,b) in enumerate(GroupKFold(5,shuffle=True,random_state=42).split(dev,groups=dev.block)):
        folds.append((k,dev.iloc[a],dev.iloc[b]))
    out=REV/'models/spatial5cm';out.mkdir(exist_ok=True)
    dev[['record_id','physical_site_id','block']].to_parquet(out/'groups.parquet',index=False)
    preds={}
    for variant in ['ma','ft','rf','xgb']:
        vals=[]
        for k,train,val in folds:
            forbidden=set(val.physical_site_id)|set(test.physical_site_id)
            func=fit_tree if variant in ['rf','xgb'] else fit_neural
            func(train,{'oof':val,'test':test},out/variant/f'outer{k}',5,variant,42,forbidden)
            p=pd.read_parquet(out/variant/f'outer{k}/oof_predictions.parquet')
            vals.append(p.merge(val[['record_id','block']],on='record_id',validate='one_to_one'))
        z=pd.concat(vals,ignore_index=True);z.to_parquet(out/variant/'oof_predictions.parquet',index=False);preds[variant]=z
    results={'development_records':len(dev),'blocks':dev.block.nunique(),'folds':5,'block_rule':'floor(longitude/5), floor(latitude/5) at per-site median coordinates','models':{},'paired_block_oof':{}}
    for v,z in preds.items():results['models'][v]={'block_oof':statistics(z),'site_oof':statistics(read(5,v,42,'oof'))}
    for a,b in itertools.combinations(preds,2):
        results['paired_block_oof'][a+'_minus_'+b]=paired_seed_bootstrap([(preds[a],preds[b])],cluster='block')
    (REV/'analysis/spatial.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('V6 SPATIAL DIAGNOSTIC COMPLETE',flush=True)

if __name__=='__main__':main()
