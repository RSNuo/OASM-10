"""Paired spatial-site inference, distinct from variation across fitted seeds."""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd

REV=Path(__file__).resolve().parent;OUT=REV/'analysis';OUT.mkdir(exist_ok=True)
SEEDS=[42,7,555];B=2000;BOOT_SEED=20260912
MIN_SD=1e-6  # Numerical constancy threshold, in m3/m3; not a skill threshold.

def statistics(frame):
    y=frame.y.to_numpy(float);p=frame.prediction.to_numpy(float);e=p-y
    mse=np.mean(e**2);bias=e.mean();v=np.sum((y-y.mean())**2)
    return {'records':len(y),'sites':frame.physical_site_id.nunique(),'rmse':float(np.sqrt(mse)),
            'mse':float(mse),'bias':float(bias),'ubrmse':float(np.std(e)),
            'r2':float(1-np.sum(e**2)/v) if v>0 else None,
            'range_violation_fraction':float(np.mean((p<0)|(p>1)))}

def site_statistics(frame):
    rows=[]
    for s,g in frame.groupby('physical_site_id',sort=True):
        y=g.y.to_numpy(float);p=g.prediction.to_numpy(float);e=p-y
        # Spatial groups can contain different co-located sensors or aliases.
        # Remove each original series mean so a between-sensor offset is not
        # mistaken for temporal variability. Bootstrap clusters remain spatial.
        key='_anomaly_group' if '_anomaly_group' in g else 'source_station' if 'source_station' in g else None
        if key:
            means=g.groupby(key)[['y','prediction']].transform('mean')
            yc=y-means.y.to_numpy();pc=p-means.prediction.to_numpy()
        else:yc=y-y.mean();pc=p-p.mean()
        vx=np.sum(yc**2);vp=np.sum(pc**2)
        rows.append({'physical_site_id':s,'n':len(g),'rmse':float(np.sqrt(np.mean(e**2))),
            'bias':float(e.mean()),'ubrmse':float(np.std(e)),
            'observed_sd':float(np.sqrt(vx/len(g))),'predicted_sd':float(np.sqrt(vp/len(g))),
            'within_series_ubrmse':float(np.sqrt(np.mean((pc-yc)**2))),
            'amplitude_ratio':float(np.sqrt(vp/vx)) if vx>len(g)*MIN_SD**2 else np.nan,
            'r':float(np.sum(yc*pc)/np.sqrt(vx*vp)) if len(g)>=10 and vx>len(g)*MIN_SD**2 and vp>len(g)*MIN_SD**2 else np.nan,
            'sse':float(np.sum(e**2)),'sum_e':float(e.sum()),'anom_xy':float(np.sum(yc*pc)),
            'anom_yy':float(vx),'anom_pp':float(vp)})
    return pd.DataFrame(rows).set_index('physical_site_id')

def site_summary(s):
    def mean(c):return float(s[c].mean()) if s[c].notna().any() else None
    xy=s.anom_xy.sum();yy=s.anom_yy.sum();pp=s.anom_pp.sum()
    return {'site_equal_rmse':mean('rmse'),'site_equal_ubrmse':mean('ubrmse'),'site_equal_bias':mean('bias'),
            'median_site_r':float(s.r.median()) if s.r.notna().any() else None,'site_r_count':int(s.r.notna().sum()),
            'median_amplitude_ratio':float(s.amplitude_ratio.median()) if s.amplitude_ratio.notna().any() else None,
            'pooled_within_site_anomaly_r':float(xy/np.sqrt(yy*pp)) if yy>s.n.sum()*MIN_SD**2 and pp>s.n.sum()*MIN_SD**2 else None}

def align(a,b):
    if a.record_id.duplicated().any() or b.record_id.duplicated().any():raise ValueError('Duplicate comparison record keys')
    if set(a.record_id)!=set(b.record_id):raise ValueError('Different comparison record sets')
    a=a.set_index('record_id').sort_index();b=b.set_index('record_id').reindex(a.index)
    if b.y.isna().any() or len(a)!=len(b):raise ValueError('Different comparison records')
    if not np.allclose(a.y,b.y,rtol=0,atol=1e-7) or not a.physical_site_id.equals(b.physical_site_id):raise ValueError('Reference mismatch')
    return a.reset_index(),b.reset_index()

def paired_seed_bootstrap(pairs,cluster='physical_site_id'):
    """Mean of same-seed metric differences; same site draws for every seed.

    CIs condition on these fitted models. They do not substitute for additional
    random-seed replication, new spatial domains or prospective observations.
    """
    stats=[];points=[];common=None
    for a,b in pairs:
        a,b=align(a,b)
        if cluster!='physical_site_id':
            a=a.copy();b=b.copy()
            for frame in [a,b]:
                frame['_anomaly_group']=frame.physical_site_id.astype(str)+('|'+frame.source_station.astype(str) if 'source_station' in frame else '')
                frame['physical_site_id']=frame[cluster]
        sa=site_statistics(a);sb=site_statistics(b).reindex(sa.index)
        if common is None:common=sa.index
        if not common.equals(sa.index):raise ValueError('Seed site sets differ')
        stats.append((sa,sb));points.append(statistics(a)['rmse']-statistics(b)['rmse'])
    n=len(common);rng=np.random.default_rng(BOOT_SEED)
    w=rng.multinomial(n,np.ones(n)/n,size=B).astype(float)
    rmse_draw=[];anom_draw=[];equal_draw=[]
    for a,b in stats:
        rmse_draw.append(np.sqrt((w@a.sse)/(w@a.n))-np.sqrt((w@b.sse)/(w@b.n)))
        equal_draw.append((w@(a.rmse-b.rmse))/n)
        def ar(s):
            yy=w@s.anom_yy;pp=w@s.anom_pp;nn=w@s.n;denom=np.sqrt(yy*pp)
            return np.divide(w@s.anom_xy,denom,out=np.full(B,np.nan),where=(yy>nn*MIN_SD**2)&(pp>nn*MIN_SD**2))
        anom_draw.append(ar(a)-ar(b))
    def ci(x):
        x=np.asarray(x);count=np.isfinite(x).sum(axis=0)
        v=np.divide(np.nansum(x,axis=0),count,out=np.full(x.shape[1],np.nan),where=count>0)
        return [float(z) for z in np.nanquantile(v,[.025,.975])] if np.isfinite(v).any() else None
    pointanom=[]
    for a,b in stats:
        x=site_summary(a)['pooled_within_site_anomaly_r'];y=site_summary(b)['pooled_within_site_anomaly_r']
        pointanom.append(x-y if x is not None and y is not None else np.nan)
    return {'difference_direction':'first minus second','seeds':len(pairs),'clusters':n,
            'per_seed_rmse_differences':points,'mean_rmse_difference':float(np.mean(points)),
            'sd_of_paired_seed_differences':float(np.std(points,ddof=1)) if len(points)>1 else None,
            'rmse_ci95':ci(rmse_draw),'site_equal_rmse_difference':float(np.mean([(a.rmse-b.rmse).mean() for a,b in stats])),
            'site_equal_rmse_ci95':ci(equal_draw),'anomaly_r_difference':float(np.nanmean(pointanom)) if np.isfinite(pointanom).any() else None,
            'anomaly_r_ci95':ci(anom_draw),'bootstrap_method':'2000 percentile spatial-site cluster resamples; all records retained with sampled-site multiplicity; identical draws across models and seeds'}

def read(d,variant,seed=42,split='test'):
    return pd.read_parquet(REV/f'models/{d}cm/{variant}_seed{seed}/{split}_predictions.parquet')

def reference_run(d):
    path=REV/'models/product_reference.json'
    spec=json.loads(path.read_text(encoding='utf-8')).get('runs',{}) if path.exists() else {}
    return spec.get(f'{d}cm','ma_seed42')

def read_reference(d,split='test'):
    """Deployed product reference at depth d (models/product_reference.json); controls keep read()."""
    return pd.read_parquet(REV/f'models/{d}cm/{reference_run(d)}/{split}_predictions.parquet')

def greedy(d):
    oofs=[read(d,'ma',s,'oof').set_index('record_id').sort_index() for s in SEEDS]
    tests=[read(d,'ma',s,'test').set_index('record_id').sort_index() for s in SEEDS]
    assert all(x.index.equals(oofs[0].index) for x in oofs)
    p=np.stack([x.prediction.to_numpy() for x in oofs]);y=oofs[0].y.to_numpy()
    total=np.zeros(len(y));slots=[];best=np.inf
    for k in range(50):
        scores=np.sqrt(np.mean(((total[None,:]+p)/(k+1)-y[None,:])**2,axis=1));i=int(scores.argmin())
        if best-scores[i]<1e-6:break
        best=float(scores[i]);total+=p[i];slots.append(i)
    folder=REV/f'models/{d}cm/ma_oof_selected';folder.mkdir(exist_ok=True)
    result={'candidate_seeds':SEEDS,'selected_slots':[SEEDS[i] for i in slots],'selection':'greedy development OOF RMSE only; improvement at least 1e-6; maximum 50 slots','metrics':{}}
    for split,frames in [('oof',oofs),('test',tests)]:
        z=frames[0].copy();z['prediction']=np.mean([frames[i].prediction.to_numpy() for i in slots],axis=0);z=z.reset_index()
        z.to_parquet(folder/f'{split}_predictions.parquet',index=False);result['metrics'][split]=statistics(z)
        eq=z.copy();eq['prediction']=np.mean([x.prediction.to_numpy() for x in frames],axis=0);result['metrics'][split+'_equal']=statistics(eq)
    (folder/'selection.json').write_text(json.dumps(result,indent=2));return result

def analyze_depth(d):
    result={'models':{},'paired_controls':{},'ensemble':greedy(d)}
    for folder in (REV/f'models/{d}cm').glob('*_seed*'):
        entry={}
        for split in ['oof','test']:
            p=folder/f'{split}_predictions.parquet'
            if not p.exists():continue
            z=pd.read_parquet(p);s=site_statistics(z);s.to_csv(OUT/f'{d}cm_{folder.name}_{split}_sites.csv')
            entry[split]={**statistics(z),**site_summary(s)}
        if entry:result['models'][folder.name]=entry
    controls=['concat','ft','no_modality','static_forcing'] if d==5 else ['concat','ft','no_cascade']
    for v in controls:
        for split in ['oof','test']:
            pairs=[(read(d,v,s,split),read(d,'ma',s,split)) for s in SEEDS]
            summary=paired_seed_bootstrap(pairs)
            summary['per_seed_details']={str(seed):paired_seed_bootstrap([pair]) for seed,pair in zip(SEEDS,pairs)}
            result['paired_controls'][v+'_minus_ma_'+split]=summary
    for v in ['rf','xgb']+(['no_static','static_only'] if d==5 else []):
        for split in ['oof','test']:
            a,b=read(d,'ma',42,split),read(d,v,42,split)
            result['paired_controls']['ma_minus_'+v+'_'+split]=paired_seed_bootstrap([(a,b)])
        a,b=align(read(d,'ma'),read(d,v));sa=site_statistics(a);sb=site_statistics(b)
        result['paired_controls']['ma_minus_'+v+'_test'].update({'fraction_sites_rmse_lower':float((sa.rmse<sb.rmse).mean()),
            'fraction_sites_ubrmse_lower':float((sa.ubrmse<sb.ubrmse).mean()),'valid_r_comparison_sites':int((sa.r.notna()&sb.r.notna()).sum()),
            'fraction_valid_sites_r_higher':float((sa.loc[sa.r.notna()&sb.r.notna(),'r']>sb.loc[sa.r.notna()&sb.r.notna(),'r']).mean()) if (sa.r.notna()&sb.r.notna()).any() else None})
        ma=statistics(a);tree=statistics(b);gap=tree['mse']-ma['mse'];biasgap=tree['bias']**2-ma['bias']**2
        result['paired_controls']['ma_minus_'+v+'_test']['mse_decomposition']={'total_reduction':gap,'bias_squared_reduction':biasgap,
            'centered_mse_reduction':tree['ubrmse']**2-ma['ubrmse']**2,'bias_share':biasgap/gap if abs(gap)>1e-10 else None}
    # Fixed observed-moisture bins; do not choose thresholds from test error.
    bins=[-np.inf,.1,.2,.3,.4,np.inf]
    z=read(d,'ma');z['moisture_bin']=pd.cut(z.y,bins=bins,labels=['<0.1','0.1-0.2','0.2-0.3','0.3-0.4','>=0.4'],right=False)
    result['moisture_strata']={str(k):statistics(g) for k,g in z.groupby('moisture_bin',observed=True)}
    return result

def lag_analysis():
    f=pd.read_parquet(REV/'data/all_5cm.parquet');p=read(5,'ma')
    z=p.merge(f[['record_id','s2_lag','landsat_lag']],on='record_id',validate='one_to_one')
    z['abs_error']=abs(z.prediction-z.y);z['squared_error']=(z.prediction-z.y)**2
    date=pd.to_datetime(z.target_time);z['year_month']=date.dt.to_period('M').astype(str)
    out={'unadjusted_bins':{}}
    z['lag_bin']=pd.cut(z.s2_lag,[0,3,7,14],include_lowest=True,labels=['0-3','>3-7','>7-14'])
    for label,g in z.groupby('lag_bin',observed=True):out['unadjusted_bins'][str(label)]=statistics(g)
    # Exact site-year-month comparison; groups without within-period lag variation
    # contribute zero. This remains observational and does not isolate cloud effects.
    for response in ['abs_error','squared_error']:
        groups=z.groupby(['physical_site_id','year_month'])
        dx=z.s2_lag-groups.s2_lag.transform('mean');dy=z[response]-groups[response].transform('mean')
        ss=pd.DataFrame({'site':z.physical_site_id,'xy':dx*dy,'xx':dx**2}).groupby('site').sum()
        valid=ss.xx>1e-10;ss=ss[valid];n=len(ss)
        rng=np.random.default_rng(BOOT_SEED);w=rng.multinomial(n,np.ones(n)/n,size=B)
        slopes=(w@ss.xy)/(w@ss.xx)
        out[response]={'site_year_month_slope':float(ss.xy.sum()/ss.xx.sum()),'ci95':np.quantile(slopes,[.025,.975]).tolist(),
                       'contributing_sites':n,'eligible_records':int((dx.abs()>1e-8).sum())}
    out['interpretation']='Association within site and calendar year-month; not a causal acquisition-lag effect; no lag-selection or error-driven exclusions.'
    return out

def main():
    from run_campaign import summarize
    summarize()
    result={'revision':'oasm10_public_release','resampling':{'confidence':.95,'method':'percentile','resamples':B,'seed':BOOT_SEED},
            'temporal_centering':'Means removed within original source_station within physical site; spatial-site bootstrap retains all co-located series. Different sensor offsets are not counted as temporal variability.',
            'depths':{str(d):analyze_depth(d) for d in [5,20,50]},'lag':lag_analysis()}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('V6 ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':main()
