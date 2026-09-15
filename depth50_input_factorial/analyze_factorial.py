"""Complete OOF ranking and fitted-asset freeze, then evaluate all test outputs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_factorial import HERE, REV, SEEDS, CONFIGS, destination, read, write, sha, now, save_pred
sys.path.insert(0,str(REV))
import analyze_results as metrics
from oasm.training import load_model
metrics.BOOT_SEED=20260914

def aggregate(c,s,split):
    root=HERE/f'models/{c}_seed{s}'
    frames=[pd.read_parquet(root/f'outer{k}/{split}_predictions.parquet') for k in range(5)]
    if split=='oof':
        z=pd.concat(frames,ignore_index=True);assert not z.record_id.duplicated().any()
    else:
        z=frames[0].set_index('record_id').sort_index()
        for f in frames:metrics.align(z.reset_index(),f)
        z['prediction']=np.mean([f.set_index('record_id').loc[z.index,'prediction'].to_numpy(dtype=float) for f in frames],axis=0)
        z=z.reset_index()
    z[['y','prediction']]=z[['y','prediction']].astype('float64')
    z=z.sort_values('record_id').reset_index(drop=True)
    z.to_parquet(root/f'{split}_predictions.parquet',index=False)
    return z

def summary(split):
    result={'configurations':{},'contrasts':{}};frames={};sites={}
    for c in CONFIGS:
        per={}
        for s in SEEDS:
            z=aggregate(c,s,split)
            if frames:metrics.align(next(iter(frames.values())),z)
            frames[c,s]=z;st=metrics.site_statistics(z);sites[c,s]=st
            st.to_csv(HERE/f'{split}_{c}_seed{s}_sites.csv')
            sm={**metrics.statistics(z),**metrics.site_summary(st)}
            sm['site_bias_mse_share']=float((st.n*st.bias**2).sum()/st.sse.sum())
            per[str(s)]=sm
        keys=['rmse','ubrmse','bias','r2','site_equal_rmse','site_equal_ubrmse','pooled_within_site_anomaly_r','site_bias_mse_share']
        means={k:float(np.mean([per[str(s)][k] for s in SEEDS])) for k in keys}
        means['seed_sd_rmse']=float(np.std([per[str(s)]['rmse'] for s in SEEDS],ddof=1))
        result['configurations'][c]={'inputs':CONFIGS[c],'per_seed':per,'mean_of_seed_metrics':means}
    for a,b in [('B','A'),('C','A'),('D','A'),('C','B'),('D','B'),('D','C')]:
        pairs=[(frames[a,s],frames[b,s]) for s in SEEDS]
        out=metrics.paired_seed_bootstrap(pairs)
        out['per_seed']={str(s):metrics.paired_seed_bootstrap([p]) for s,p in zip(SEEDS,pairs)}
        out['fraction_sites_with_lower_rmse_mean_across_seeds']=float(np.mean([(sites[a,s].rmse<sites[b,s].rmse).mean() for s in SEEDS]))
        out['fraction_sites_with_lower_ubrmse_mean_across_seeds']=float(np.mean([(sites[a,s].ubrmse<sites[b,s].ubrmse).mean() for s in SEEDS]))
        result['contrasts'][a+'-'+b]=out
    # Interaction of input effects on pooled RMSE; common cluster draws across all four models.
    common=sites['A',42].index;n=len(common)
    w=np.random.default_rng(20260914).multinomial(n,np.ones(n)/n,size=2000).astype(float)
    draws=[];points=[]
    for s in SEEDS:
        rr={c:np.sqrt((w@sites[c,s].sse)/(w@sites[c,s].n)) for c in CONFIGS}
        draws.append((rr['C']-rr['A'])-(rr['D']-rr['B']))
        r={c:result['configurations'][c]['per_seed'][str(s)]['rmse'] for c in CONFIGS}
        points.append((r['C']-r['A'])-(r['D']-r['B']))
    result['rmse_interaction']={'definition':'(C-A)-(D-B): direct-precipitation effect with cascade minus effect without cascade',
        'mean_difference':float(np.mean(points)),'per_seed':dict(zip(map(str,SEEDS),points)),
        'ci95':np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist(),
        'interpretation':'Contrast of predictive errors, not a physical causal interaction.'}
    result['ranking_by_mean_seed_rmse']=sorted(CONFIGS,key=lambda c:(result['configurations'][c]['mean_of_seed_metrics']['rmse'],c))
    return result

def freeze_oof():
    assert all((destination(c,s,k)/'manifest.json').exists() for c in CONFIGS for s in SEEDS for k in range(5))
    sources=read(HERE/'protocol.json')['sources_sha256']
    for name,digest in sources.items():assert sha(REV/name)==digest,name
    assets={};metas=[]
    for c in CONFIGS:
        for s in SEEDS:
            for k in range(5):
                p=destination(c,s,k);m=read(p/'manifest.json');metas.append(m)
                wp=REV/m['weight_directory']
                assert sha(wp/'model.pt')==m['weights_sha256']
                assert sha(wp/'preprocessor.joblib')==m['preprocessor_sha256']
                assert sha(p/'oof_predictions.parquet')==m['oof_sha256']
                for f in [p/'manifest.json',p/'oof_predictions.parquet',wp/'model.pt',wp/'preprocessor.joblib']:
                    assets[str(f.relative_to(REV))]=sha(f)
    oof=summary('oof');write(HERE/'oof_results.json',oof)
    obj={'created_utc':now(),'protocol_sha256':sha(HERE/'protocol.json'),'oof_results_sha256':sha(HERE/'oof_results.json'),
        'analysis_code_sha256':{n:sha(HERE/n) for n in ['analyze_factorial.py','verify_factorial.py','check_design.py']},
        'fitted_assets_sha256':assets,'ranking':oof['ranking_by_mean_seed_rmse'],
        'selected_config_for_reporting':oof['ranking_by_mean_seed_rmse'][0],
        'selection_rule':read(HERE/'protocol.json')['ranking'],
        'test_status':'No new factorial test inference at first creation; historic test results have been examined. All four configurations will be reported.'}
    if (HERE/'oof_freeze.json').exists():
        prev=read(HERE/'oof_freeze.json')
        for key in ['fitted_assets_sha256','ranking','protocol_sha256','oof_results_sha256','analysis_code_sha256']:assert prev[key]==obj[key],key
    else:write(HERE/'oof_freeze.json',obj)
    write(HERE/'fold_training_summary.json',{'folds':metas,'new_fits':sum(not m['identity_reuse'] for m in metas),'reused_fits':sum(m['identity_reuse'] for m in metas)})
    print('OOF_FROZEN ranking='+str(obj['ranking']),flush=True)
    return oof

def test_inference():
    frozen=read(HERE/'oof_freeze.json')
    for name,digest in frozen['fitted_assets_sha256'].items():assert sha(REV/name)==digest,name
    for name,digest in frozen['analysis_code_sha256'].items():assert sha(HERE/name)==digest,name
    for k in range(5):
        f=pd.read_parquet(REV/f'data/cascade/50cm/outer{k}/test.parquet')
        for c in CONFIGS:
            for s in SEEDS:
                p=destination(c,s,k);dest=p/'test_predictions.parquet'
                if dest.exists():continue
                m=read(p/'manifest.json');model,prep,cfg=load_model(REV/m['weight_directory'])
                assert set(prep.fitted_site_ids).isdisjoint(f.physical_site_id)
                assert cfg['numeric']==m['config']['numeric']
                save_pred(dest,model,prep,f)
                del model;torch.cuda.empty_cache()
        print(f'TEST_FOLD_COMPLETE {k}',flush=True)

def report(r):
    names={'A':'5 cm级联；无直接降水','B':'无级联；无直接降水','C':'5 cm级联＋直接降水','D':'无级联；直接降水'}
    out=['# 50 cm 输入组合对照结果','',
         '所有配置使用相同的三个种子（42、7、555）、物理站点划分、完整记录和至少10轮的重拟合策略。级联配置复用同一套按外层折配对的嵌套5 cm预测；5 cm模型本身已经使用降水，所以“无直接降水”只指没有给50 cm模型额外输入降水列。', '',
         '|配置|输入|OOF RMSE（三种子均值）|测试 RMSE（三种子均值）|测试 RMSE（seed 42）|测试站点等权 RMSE|测试异常相关|',
         '|---|---|---:|---:|---:|---:|---:|']
    for c in CONFIGS:
        oo=r['oof']['configurations'][c]['mean_of_seed_metrics'];te=r['test']['configurations'][c]['mean_of_seed_metrics']
        seed42=r['test']['configurations'][c]['per_seed']['42']['rmse']
        out.append(f'|{c}|{names[c]}|{oo["rmse"]:.5f}|{te["rmse"]:.5f}|{seed42:.5f}|{te["site_equal_rmse"]:.5f}|{te["pooled_within_site_anomaly_r"]:.4f}|')
    out+=['','表中是三个模型分别计算指标后取平均，不是三个种子预测集成的误差。RMSE单位为 m³/m³。站点等权RMSE为每个物理站点RMSE的算术平均；异常相关先在每个物理站点内部按原始观测序列去均值，再汇总协方差。', '',
          'OOF冻结排序：'+' → '.join(r['oof']['ranking_by_mean_seed_rmse'])+'。测试阶段不改变此排序或据此发布新参考模型。','',
          '|比较（前者−后者）|OOF RMSE差及95% CI|测试 RMSE差及95% CI|测试异常相关差及95% CI|','|---|---|---|---|']
    for key in r['test']['contrasts']:
        oo=r['oof']['contrasts'][key];te=r['test']['contrasts'][key]
        def val(x,k,ci):return f'{x[k]:+.5f} [{x[ci][0]:+.5f}, {x[ci][1]:+.5f}]'
        out.append('|'+key+'|'+val(oo,'mean_rmse_difference','rmse_ci95')+'|'+val(te,'mean_rmse_difference','rmse_ci95')+'|'+val(te,'anomaly_r_difference','anomaly_r_ci95')+'|')
    out+=['','所有区间为2,000次物理站点整群重采样的95%百分位区间，同一次重采样用于所有配置和种子；保留被抽中站点的全部记录及重复次数。区间条件于这些已拟合模型，未作多重比较校正。种子标准差另存于results.json，不能作为显著性阈值。', '',
          '这是既有测试集上提出并验证的后续敏感性实验；该测试集已被多次查看，因此结果不能被称为全新的前瞻性独立验证。OOF排序也属于四配置选择所用的开发证据。', '',
          '本轮只改变50 cm的直接输入，保持上游拟合固定；未更改原模型、数据、手稿或发布产品。是否采用新配置，需要同时考虑OOF、测试稳定性和时间变化恢复能力，不能只挑测试RMSE最低的配置。','',
          '复现文件：protocol.json（协议与源文件哈希）、run_factorial.py（训练）、oof_freeze.json（测试前冻结）、results.json（完整数值）、fold_training_summary.json（逐折选轮/拟合轮数）、independent_verification.json（另一实现的数值核对与模型回放）。']
    (HERE/'RESULTS_ZH.md').write_text('\n'.join(out)+'\n',encoding='utf-8')

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    oof=freeze_oof();test_inference();test=summary('test')
    inputs=pd.read_parquet(REV/'data/cascade/50cm/outer0/test.parquet')
    coverage={'development':read(HERE/'design_check.json')['development_precipitation'],
              'test':{c:{'valid_records':int(inputs[c].notna().sum()),'missing_records':int(inputs[c].isna().sum())} for c in ['precip_7d','precip_30d','precip_90d']}}
    r={'created_utc':now(),'protocol':read(HERE/'protocol.json'),'input_coverage':coverage,'oof':oof,'test':test}
    write(HERE/'results.json',r);report(r)
    write(HERE/'analysis_complete.json',{'utc':now(),'results_sha256':sha(HERE/'results.json')})
    for c in CONFIGS:print(c,r['test']['configurations'][c]['mean_of_seed_metrics'],flush=True)
    print('FACTORIAL_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
