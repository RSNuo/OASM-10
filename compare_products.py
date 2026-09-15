"""Matched reference-model product comparisons with explicit spatial/temporal support."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
from analyze_results import read,read_reference,reference_run,statistics,site_statistics,site_summary,paired_seed_bootstrap
from oasm.inference import valid_satellite_rows

REV=Path(__file__).resolve().parent;VAL=REV/'validation';OUT=VAL/'product_comparisons';OUT.mkdir(exist_ok=True)

def product_frame(d,split='test'):
    p=read_reference(d,split);raw=pd.read_parquet(REV/f'data/all_{d}cm.parquet')
    keep=['record_id','LandCover','longitude','latitude','s2_lag','landsat_lag']
    raw['output_eligible']=valid_satellite_rows(raw)
    f=p.merge(raw[keep+['output_eligible']],on='record_id',validate='one_to_one')
    f['date']=pd.to_datetime(f.target_time).dt.normalize();return f

def comparison(z,other,label,extra=None):
    z=z[np.isfinite(z.prediction)&np.isfinite(z[other])].copy()
    z=z[z.output_eligible]
    if z.record_id.duplicated().any():raise ValueError('Benchmark matching multiplied records')
    ref=z[['record_id','physical_site_id','target_time','y','prediction']].copy()
    baseline=ref.copy();baseline.prediction=z[other].to_numpy()
    z.to_parquet(OUT/(label+'_matched.parquet'),index=False)
    result={'oasm':{**statistics(ref),**site_summary(site_statistics(ref))},'benchmark':statistics(baseline),
            'oasm_minus_benchmark':paired_seed_bootstrap([(ref,baseline)]),
            'eligibility':'same satellite/land-cover output mask as public predictor; missing forcing or soil retains training-fitted imputation'}
    if extra:result.update(extra)
    return result

def surface_benchmarks():
    p=product_frame(5);registry=pd.read_csv(REV/'data/site_registry.csv')
    reg=registry[['source_station','physical_site_id']].drop_duplicates()
    unique=reg.groupby('source_station').physical_site_id.nunique();reg=reg[reg.source_station.isin(unique[unique==1].index)]
    hb=pd.read_csv(VAL/'extracted_SMAPHB_sm.csv');hb['date']=pd.to_datetime(hb.date).dt.normalize()
    hb=hb.merge(reg,left_on='station',right_on='source_station',validate='many_to_one')
    # This file contains repeated extraction requests. Only equal-valued duplicates
    # may be collapsed; a disagreement must be resolved from the source raster.
    spread=hb.groupby(['source_station','date']).soil_moisture.agg(['min','max'])
    if ((spread['max']-spread['min']).abs()>1e-6).any():raise ValueError('Conflicting SMAP-HB values on the same site-date')
    hb=hb.drop_duplicates(['source_station','date']).rename(columns={'soil_moisture':'smap_hb'})
    m=p.merge(hb[['source_station','date','smap_hb']],on=['source_station','date'],validate='many_to_one')
    results={'smap_hb':comparison(m,'smap_hb','smap_hb',{'support':'nearest archived 30m SMAP-HydroBlocks pixel; daily-date match'})}
    s=pd.read_csv(VAL/'s2mp_extracted_pixels.csv')
    source_count=len(s)
    # Provider README: MV_[S1A/S1B]_[site]_[time] identifies moisture;
    # zero denotes no estimate, even if a TIFF omits its nodata metadata.
    s=s[s.tiff_file.str.match(r'^MV_S1[AB]_')].copy();moisture_file_count=len(s)
    s=s[s.s2mp.gt(0)&s.s2mp.le(1)];positive_count=len(s)
    s=s.merge(reg,left_on='station',right_on='source_station',validate='many_to_one')
    stamps=s.tiff_file.str.extract(r'(\d{8}T\d{6})',expand=False)
    s['acquisition_time']=pd.to_datetime(stamps,format='%Y%m%dT%H%M%S',errors='coerce')
    if s.acquisition_time.isna().any():raise ValueError('S2MP acquisition time could not be parsed')
    s=s[np.isfinite(s.s2mp)&s.s2mp.between(0,1)].drop_duplicates(['source_station','acquisition_time','s2mp'])
    if s.duplicated(['source_station','acquisition_time']).any():raise ValueError('Conflicting S2MP acquisitions')
    m=pd.merge_asof(p.sort_values('target_time'),s[['source_station','acquisition_time','s2mp']].sort_values('acquisition_time'),
                    left_on='target_time',right_on='acquisition_time',by='source_station',direction='nearest',tolerance=pd.Timedelta('24h'))
    m=m[m.s2mp.notna()]
    results['s2mp']=comparison(m,'s2mp','s2mp',{'support':'MV_S1A/S1B moisture products only; zeros excluded per provider README; nearest acquisition within24h; timestamp parsed from source filename',
                    'filter_counts':{'all_csv_rows':source_count,'moisture_product_rows':moisture_file_count,'valid_positive_moisture_rows':positive_count},
                    'provider_readme':'https://www.theia-land.fr/wp-content/uploads/2019/12/readme_soil_moisture_general_20122019.pdf',
                    'median_time_difference_hours':float((m.target_time-m.acquisition_time).abs().dt.total_seconds().median()/3600)})
    return results

def main():
    result={'surface':surface_benchmarks(),'depth':{},'product_reference':{},'reference_runs':{str(d):reference_run(d) for d in [5,20,50]}}
    era=pd.read_parquet(VAL/'era5_depth_test_native_grid.parquet')
    for d in [20,50]:
        p=product_frame(d);m=p.merge(era,on=['physical_site_id','date'],validate='many_to_one')
        result['depth'][str(d)]=comparison(m,f'era5_{d}cm',f'era5_{d}cm',
            {'support':'ERA5-Land native0.1-degree grid daily mean; layer2 7-28cm or layer3 28-100cm versus nominal-depth point observation',
             'interpretation':'paired empirical benchmark across unequal vertical/spatial support; does not validate10m effective resolution or layer equivalence'})
    for d in [5,20,50]:
        result['product_reference'][str(d)]={}
        for split in ['oof','test']:
            p=product_frame(d,split);eligible=p[p.output_eligible]
            result['product_reference'][str(d)][split]={'all_retained_records':statistics(p),'output_eligible':{**statistics(eligible),**site_summary(site_statistics(eligible))},
                       'output_eligible_fraction':len(eligible)/len(p)}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('V6 PRODUCT COMPARISONS COMPLETE',flush=True)

if __name__=='__main__':main()
