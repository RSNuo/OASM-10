"""Archive ERA5-Land layer means for the revised, shared deep test sites."""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json,sys,time,shutil
import pandas as pd
import ee

REV=Path(__file__).resolve().parent;OUT=REV/'validation';PIECES=OUT/'era5_site_pieces'
BANDS=['volumetric_soil_water_layer_2','volumetric_soil_water_layer_3']

def main():
    sys.stdout.reconfigure(encoding='utf-8');PIECES.mkdir(parents=True,exist_ok=True)
    ee.Initialize();ee.data.setDeadline(120000)
    frames=[pd.read_parquet(REV/f'data/all_{d}cm.parquet') for d in [20,50]]
    sites=pd.concat([f.loc[f.split=='test',['physical_site_id','latitude','longitude']] for f in frames]).sort_values(['physical_site_id','latitude','longitude']).drop_duplicates('physical_site_id')
    sites.to_csv(OUT/'era5_requested_sites.csv',index=False)
    col=ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate('2016-01-01','2026-01-01').select(BANDS)
    projection=ee.Image(col.first()).select(BANDS[0]).projection().getInfo()
    metadata={'collection':'ECMWF/ERA5_LAND/DAILY_AGGR','bands':BANDS,'period':['2016-01-01','2026-01-01 (exclusive)'],'sampling':'nearest value on the native source affine grid','projection':projection,'query_utc':datetime.now(timezone.utc).isoformat(),'sites':len(sites)}
    (OUT/'era5_query_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    def extract(row):
        site=row.physical_site_id;p=PIECES/f'{site}.parquet'
        if p.exists():return site,'cached'
        point=ee.Geometry.Point([float(row.longitude),float(row.latitude)])
        error=None
        for attempt in range(3):
            try:
                table=col.getRegion(point,crs=projection['crs'],crsTransform=projection['transform']).getInfo()
                df=pd.DataFrame(table[1:],columns=table[0])
                df['date']=pd.to_datetime(df.time,unit='ms').dt.normalize();df['physical_site_id']=site
                df=df[['physical_site_id','date']+BANDS].rename(columns={BANDS[0]:'era5_20cm',BANDS[1]:'era5_50cm'})
                if df.date.max()<pd.Timestamp('2025-12-31'):raise ValueError('ERA5 series does not cover the requested end date')
                assert not df.duplicated(['physical_site_id','date']).any()
                df.to_parquet(p,index=False);return site,'saved'
            except Exception as exc:
                error=str(exc);time.sleep(3*(attempt+1))
        return site,error
    errors={}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(extract,row) for row in sites.itertuples(index=False)]
        for n,future in enumerate(as_completed(futures),1):
            site,result=future.result()
            if result not in ['saved','cached']:errors[site]=result;print('FAILED',site,result[:160],flush=True)
            if n%10==0 or n==len(sites):print(f'ERA5 {n}/{len(sites)} sites, {len(errors)} failures',flush=True)
    if errors:
        (OUT/'era5_failures.json').write_text(json.dumps(errors,indent=2),encoding='utf-8');raise RuntimeError(f'Unfinished ERA5 sites: {len(errors)}')
    result=pd.concat([pd.read_parquet(PIECES/f'{s}.parquet') for s in sites.physical_site_id],ignore_index=True)
    result.to_parquet(OUT/'era5_depth_test_native_grid.parquet',index=False)
    for name in ['extracted_SMAPHB_sm.csv','s2mp_extracted_pixels.csv']:
        source=Path(os.environ.get('OASM10_LEGACY_ARCHIVE', 'legacy_archive'))/'validation'/name
        shutil.copy2(source,OUT/name)
    print('ERA5 BENCHMARK EXTRACTION COMPLETE',len(result),flush=True)

if __name__=='__main__':main()
