"""Recheck the requested floor10/no-cascade/direct-precipitation contrast.

Reuses the already fitted exact experiment; replays seed42 candidate weights.
Does not retrain, change the deployed reference, or edit manuscripts.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,sys
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
REV=HERE.parent
sys.path.insert(0,str(REV))
from oasm.training import load_model,tensors,predict_tensors
from oasm.features import BASE,SOIL,PRECIP,CATS
from verify_factorial import calc,boot

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def native(v):
    if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
    if isinstance(v,list):return [native(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v
def frame(p):
    z=pd.read_parquet(p).set_index('record_id').sort_index()
    assert z.index.is_unique and np.isfinite(z[['y','prediction']]).all().all()
    return z.reset_index()

def main():
    torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    out=HERE/'current_reference_check_20260914'
    if out.exists():out=HERE/('current_reference_check_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    refspec=REV/'models/product_reference.json';refhash=sha(refspec)
    assert read(refspec)['runs']['50cm']=='ma_floor10_seed42'
    prod=REV/'models/50cm/ma_floor10_seed42'
    frozen=read(HERE/'oof_freeze.json');protocol=read(HERE/'protocol.json')
    for name,digest in protocol['sources_sha256'].items():assert sha(REV/name)==digest,name
    for name,digest in frozen['fitted_assets_sha256'].items():assert sha(REV/name)==digest,name
    rows=[];data={};metrics={};current={};aggregation={}
    for split in ['oof','test']:
        current[split]=frame(prod/f'{split}_predictions.parquet')
        for c in ['A','D']:
            for seed in [42,7,555]:
                z=frame(HERE/f'models/{c}_seed{seed}/{split}_predictions.parquet')
                ref=current[split]
                assert z.record_id.equals(ref.record_id)
                assert z.physical_site_id.equals(ref.physical_site_id) and np.array_equal(z.y,ref.y)
                if c=='A' and seed==42:
                    delta=float(np.max(np.abs(z.prediction.to_numpy(float)-ref.prediction.to_numpy(float))))
                    aggregation[split]=delta
                    assert delta<6e-8,('aggregate rounding difference',split,delta)
                    # Compare against the actual deployed aggregate; the older
                    # factorial summary averaged float64 instead of float32.
                    if split=='test':
                        pieces=[pd.read_parquet(prod/f'outer{k}/test_predictions.parquet').set_index('record_id').loc[ref.record_id].prediction.to_numpy() for k in range(5)]
                        raw32=np.mean(pieces,axis=0);raw64=np.mean([x.astype(float) for x in pieces],axis=0)
                        assert np.array_equal(raw32,ref.prediction.to_numpy())
                        assert np.max(np.abs(raw64-z.prediction.to_numpy(float)))<1e-15
                    z=ref.copy()
                data[split,c,seed]=z
                metrics[f'{split}/{c}/{seed}']=calc(z)
                rows.append({'split':split,'configuration':c,'seed':seed,**metrics[f'{split}/{c}/{seed}']})
    report={'checked_utc':datetime.now(timezone.utc).isoformat(),'scope':'Existing trained floor10 experiment, fresh seed42 saved-weight replay and prediction-based comparison; not a new training run or untouched test.',
            'reference':'models/50cm/ma_floor10_seed42','candidate':'depth50_input_factorial/models/D_seed42',
            'same_data':{s:{'records':len(z),'sites':z.physical_site_id.nunique()} for s,z in current.items()},
            'per_seed_metrics':metrics,'D_minus_A':{},'seed42_D_replay':[],
            'A_aggregate_rounding_max_difference':aggregation,
            'aggregate_note':'Compare D with the actual current product. Factorial A test previously averaged float64; product averages float32. Same per-fold weights; rounding difference verified by recreating both means.'}
    for split in ['oof','test']:
        report['D_minus_A'][split]={
            'seed42':boot([(data[split,'D',42],data[split,'A',42])]),
            'three_seed_mean':boot([(data[split,'D',s],data[split,'A',s]) for s in [42,7,555]])}
    print('PAIRED_METRICS_RECOMPUTED',flush=True)
    # Replay all candidate seed42 records with the cascade column absent.
    for k in range(5):
        d=HERE/f'models/D_seed42/outer{k}';m=read(d/'manifest.json');cfg=m['config']
        assert cfg['numeric']==BASE+SOIL+PRECIP and cfg['categorical']==CATS
        assert cfg['model']=={'d_model':168,'nhead':8,'layers':2,'ff':512,'dropout':.35,'static_dropout':.55}
        assert m['fitted_epochs']==max(m['selected_epochs'],10)
        a=read(HERE/f'models/A_seed42/outer{k}/manifest.json')
        wp=REV/a['weight_directory']
        for name in ['model.pt','preprocessor.joblib']:assert sha(wp/name)==sha(prod/f'outer{k}'/name)
        tr=pd.read_parquet(REV/f'data/cascade/50cm/outer{k}/train.parquet')
        forbidden=set(read(REV/f'data/cascade/50cm/outer{k}/provenance.json')['forbidden_sites'])
        for key in ['gradient_site_ids','early_stopping_site_ids','selection_gradient_site_ids']:assert forbidden.isdisjoint(m[key])
        model,prep,_=load_model(REV/m['weight_directory'])
        assert set(prep.fitted_site_ids)==set(tr.physical_site_id)
        record={'fold':k,'selected_epochs':m['selected_epochs'],'fitted_epochs':m['fitted_epochs'],'precipitation_columns':PRECIP,'cascade_input':False}
        for split in ['oof','test']:
            x=pd.read_parquet(REV/f'data/cascade/50cm/outer{k}/{split}.parquet').drop(columns=['soil_moisture_5cm'])
            old=pd.read_parquet(d/f'{split}_predictions.parquet')
            assert x.record_id.equals(old.record_id)
            p=predict_tensors(model,tensors(prep,x),batch=8192)
            err=float(np.max(np.abs(p.astype(float)-old.prediction.to_numpy(float))))
            assert err==0.0,(k,split,err)
            record[split+'_max_prediction_difference']=err
        report['seed42_D_replay'].append(record)
        print(f'D_SEED42_REPLAY_PASS outer{k} epochs={m["fitted_epochs"]}',flush=True)
        del model,prep;torch.cuda.empty_cache()
    report['means']={s:{c:{name:float(np.mean([metrics[f'{s}/{c}/{seed}'][name] for seed in [42,7,555]])) for name in ['rmse','r2','site_equal_rmse','pooled_within_site_anomaly_r']} for c in ['A','D']} for s in ['oof','test']}
    assert sha(refspec)==refhash
    report['product_reference_unchanged']=True
    report=native(report)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    pd.DataFrame(rows).to_csv(out/'per_seed_metrics.csv',index=False)
    text=['# 50 cm：当前 floor10 参考与无级联＋直接降水对照','',
          'A 为当前产品：5 cm 预测级联、不直接输入降水；D 去掉级联，直接输入 ln(1+P7/P30/P90)，保留卫星、DOY、土壤/地形等其余输入。两者使用相同 floor10 策略、数据和相应种子。本次核验了此前已训练完成的对应配置，重新从 D 的 seed42 五折权重推理，没有重复训练。','',
          '|配置（seed42）|OOF RMSE|OOF R²|Test RMSE|Test R²|','|---|---:|---:|---:|---:|']
    for c,label in [('A','当前参考 A'),('D','无级联＋直接降水 D')]:
        o=metrics[f'oof/{c}/42'];t=metrics[f'test/{c}/42']
        text.append(f'|{label}|{o["rmse"]:.5f}|{o["r2"]:.4f}|{t["rmse"]:.5f}|{t["r2"]:.4f}|')
    text+=['','RMSE 单位：m³/m³。差值方向均为 D−A，负 RMSE 差有利于 D。','',
           '|统计范围|OOF RMSE 差及 95% CI|Test RMSE 差及 95% CI|','|---|---|---|']
    for scope,label in [('seed42','单 seed42'),('three_seed_mean','三种子成对差均值')]:
        def val(split):
            v=report['D_minus_A'][split][scope];lo,hi=v['rmse_ci95'];return f'{v["mean_rmse_difference"]:+.5f} [{lo:+.5f}, {hi:+.5f}]'
        text.append(f'|{label}|{val("oof")}|{val("test")}|')
    text+=['','三种子为 42/7/555；均值指三个模型分别计算指标后取平均，不是预测集成。95% CI 使用 2,000 次物理站点整群百分位重采样，同一抽样同时作用于所有种子，保留站内全部记录及抽样重复次数。区间条件于已拟合模型与当前基准，未作多重比较校正。','',
           'D 的 seed42 五折在移除级联列后的完整 OOF/test 输入上重新推理，10 次回放的最大绝对误差均为 0.0。所有接收外层验证/测试站均被排除在预处理、梯度拟合和 epoch 选择之外。当前产品 A 与已完成 factorial A seed42 的权重及预处理器完全相同；此前 factorial 聚合采用 float64，而产品聚合采用 float32，微小浮点舍入差已经用两种均值重建核实。本次配对使用实际产品预测。','',
           '解释：D 在三种子平均上呈现较低 RMSE 的趋势，但 D−A 的 OOF/test RMSE 区间均包含零，不能说已证明其优于当前参考。单 seed42 的 test RMSE 略升，也不支持将这次试验描述成 seed42 测试精度提升。当前产品 selector、原模型及两篇手稿保持不变。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({'output':str(out),'seed42':{k:v for k,v in metrics.items() if k.endswith('/42')},'three_seed_means':report['means'],'paired':report['D_minus_A']},indent=2),flush=True)

if __name__=='__main__':main()
