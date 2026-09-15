"""Product quality on fixed record/site populations, with outcome-blind examples."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from oasm.inference import valid_satellite_rows
from analyze_results import statistics,site_statistics,site_summary

REV=Path(__file__).resolve().parent;OUT=REV/'analysis';OUT.mkdir(exist_ok=True)

def selection():
    path=OUT/'quality_selection.json'
    if path.exists():return json.loads(path.read_text())
    frames={d:pd.read_parquet(REV/f'data/all_{d}cm.parquet') for d in [5,20,50]}
    candidates={};per_depth={}
    for d,f in frames.items():
        z=f[(f.split=='test')&valid_satellite_rows(f)].copy()
        counts=z.groupby('physical_site_id').size();candidates[d]=set(counts[counts>=24].index)
        per_depth[d]=z
    common=sorted(set.intersection(*candidates.values()))
    if not common:raise ValueError('No common eligible site for the prespecified quality summary')
    # Highest minimum annual observation count across the depths, then a hash
    # tie-break. Only availability and timestamps enter this selection.
    ranks=[]
    for site in common:
        annual=[]
        for d,z in per_depth.items():
            g=z[z.physical_site_id==site]
            c=g.groupby(pd.to_datetime(g.target_time).dt.year).size()
            annual.append(c.rename(str(d)))
        a=pd.concat(annual,axis=1).fillna(0)
        for year,row in a.iterrows():
            ranks.append((int(row.min()),int(row.sum()),hashlib.sha256(f'{site}|{year}'.encode()).hexdigest(),site,int(year)))
    best=sorted(ranks,key=lambda x:(-x[0],-x[1],x[2]))[0]
    chosen={}
    for d,z in per_depth.items():
        g=z[(z.physical_site_id==best[3])&(pd.to_datetime(z.target_time).dt.year==best[4])]
        counts=g.groupby('source_station').size().sort_values(ascending=False)
        names=sorted(counts[counts==counts.max()].index,key=lambda s:hashlib.sha256(s.encode()).hexdigest())
        chosen[str(d)]={'source_station':names[0],'records':int(counts[names[0]])}
    result={'common_sites':common,'common_site_count':len(common),'minimum_records_per_depth':24,
      'temporal_example':{'physical_site_id':best[3],'year':best[4],'minimum_annual_depth_count':best[0],'sources':chosen},
      'selection_rule':'Eligibility before prediction: common external sites >=24 valid-satellite records per depth; example site/year maximizes the minimum annual depth count, then total count, then deterministic hash; source alias chosen by count/hash per depth. No y or prediction value used.'}
    path.write_text(json.dumps(result,indent=2),encoding='utf-8');return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--select-only',action='store_true');args=parser.parse_args()
    sel=selection()
    if args.select_only:print(json.dumps(sel,indent=2));return
    from compare_products import product_frame
    result={**sel,'depths':{}};common_rows=[]
    for d in [5,20,50]:
        z=product_frame(d);raw=pd.read_parquet(REV/f'data/all_{d}cm.parquet')
        z=z[z.output_eligible].copy();s=site_statistics(z)
        coords=raw.groupby('physical_site_id')[['longitude','latitude']].median()
        s=s.join(coords);s['depth_cm']=d;s.to_csv(OUT/f'product_{d}cm_site_quality.csv')
        common=z[z.physical_site_id.isin(sel['common_sites'])];sc=site_statistics(common)
        sc['depth_cm']=d;common_rows.append(sc.reset_index())
        result['depths'][str(d)]={'all_eligible':{**statistics(z),**site_summary(s)},
            'common_sites':{**statistics(common),**site_summary(sc)},'moisture_bins':{}}
        z['bin']=pd.cut(z.y,[-np.inf,.1,.2,.3,.4,np.inf],labels=['<0.1','0.1-0.2','0.2-0.3','0.3-0.4','>=0.4'],right=False)
        for key,g in z.groupby('bin',observed=True):result['depths'][str(d)]['moisture_bins'][str(key)]=statistics(g)
        ex=sel['temporal_example'];chosen=ex['sources'][str(d)]['source_station']
        series=z[(z.physical_site_id==ex['physical_site_id'])&(z.source_station==chosen)&(pd.to_datetime(z.target_time).dt.year==ex['year'])]
        # Multiple orbit records remain in the archive. Display daily averages of
        # the same observed/predicted records to avoid double vertical line segments.
        series[['record_id','source_station','physical_site_id','target_time','y','prediction']].to_parquet(OUT/f'temporal_example_{d}cm.parquet',index=False)
        result['depths'][str(d)]['temporal_example']=statistics(series)
    pd.concat(common_rows,ignore_index=True).to_csv(OUT/'common_site_quality.csv',index=False)
    result['correlation_rule']='Site r requires >=10 records and SD of each series >1e-6 m3/m3; no informative correlation is assigned to constant predictions.'
    result['interpretation']='Available records differ by depth; common spatial location does not imply identical sensors, synchronous observations, vertical mass balance or lag reproduction.'
    (OUT/'quality.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('V6 QUALITY SUMMARY COMPLETE',flush=True)

if __name__=='__main__':main()
