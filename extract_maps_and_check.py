"""Freeze agricultural display windows and model-blind holdout checks, then extract."""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,sys,traceback
import numpy as np
import pandas as pd
from oasm.gee import fetch_feature_grid,ANGULAR_STEP

REV=Path(__file__).resolve().parent;OUT=REV/'validation/map_pipeline';OUT.mkdir(exist_ok=True,parents=True)
REGIONS=[('US_drought','US Corn Belt','2023-10-18'),('EastEuropeanPlain','East European Plain','2024-08-14'),
         ('China','North China Plain','2024-10-27'),('Australia','Murray-Darling Basin','2024-12-19'),
         ('India','Indo-Gangetic Plain','2024-11-03'),('SouthAfrica','Western Cape','2025-01-30'),
         ('Pampas','The Pampas','2024-07-01'),('centralvalley','US Central Valley','2023-07-17')]

def pick_regions():
    path=OUT/'region_requests.json'
    if path.exists():return json.loads(path.read_text())
    requests=[]
    for key,label,date in REGIONS:
        source=Path(os.environ.get('OASM10_LEGACY_ARCHIVE', 'legacy_archive'))/f'application/{key}/{key}_{date.replace("-","")}_for_pred.pkl'
        f=pd.read_pickle(source)[['longitude','latitude','LandCover']].iloc[::4].copy()
        west,east=f.longitude.min(),f.longitude.max();south,north=f.latitude.min(),f.latitude.max()
        # A fixed 400 x 280 nominal-grid-cell window; maximize only WorldCover
        # cropland fraction. Prediction values and errors do not enter selection.
        sx=400*ANGULAR_STEP;sy=280*ANGULAR_STEP
        best=(-1,None)
        for x in np.linspace(west,max(west,east-sx),21):
            for y in np.linspace(south,max(south,north-sy),21):
                z=f[f.longitude.between(x,x+sx)&f.latitude.between(y,y+sy)]
                share=float((pd.to_numeric(z.LandCover,errors='coerce')==40).mean()) if len(z) else -1
                if share>best[0]:best=(share,[float(x),float(y),float(x+sx),float(y+sy)])
        requests.append({'key':key,'label':label,'target_time':date+'T00:00:00','bounds':best[1],
            'selection_cropland_fraction':best[0],'selection_basis':'fixed 400x280-cell window maximizing WorldCover class40 fraction on a fixed21x21 search of every fourth source-grid record; no prediction-based selection',
            'selection_source':str(source)})
        print('REGION FROZEN',key,best[0],flush=True)
    path.write_text(json.dumps(requests,indent=2));return requests

def pick_checks():
    path=OUT/'check_requests.json'
    if path.exists():return json.loads(path.read_text())
    requests=[]
    for d in [5,20,50]:
        f=pd.read_parquet(REV/f'data/all_{d}cm.parquet');f=f[f.split=='test'].copy()
        count=f.groupby('physical_site_id').size();eligible=count[count>=24].index
        # Hash-based fixed selection and six temporal quantiles: reproducible,
        # independent of model predictions, model errors and optical availability.
        chosen=sorted(eligible,key=lambda x:hashlib.sha256(('mapcheck-v6|'+x).encode()).hexdigest())[:12]
        for site in chosen:
            g=f[f.physical_site_id==site].sort_values(['target_time','record_id'])
            g=g.iloc[np.unique(np.linspace(0,len(g)-1,6).round().astype(int))]
            for _,r in g.iterrows():
                x,y=float(r.longitude),float(r.latitude);delta=ANGULAR_STEP/2
                requests.append({'key':f'{d}cm_{r.record_id}','record_id':r.record_id,'depth_cm':d,'physical_site_id':site,
                    'source_station':r.source_station,'target_time':pd.Timestamp(r.target_time).isoformat(),
                    'bounds':[x-delta,y-delta,x+delta,y+delta],'longitude':x,'latitude':y,
                    'selection_basis':'12 hash-ranked holdout sites with >=24 records per depth; six time quantiles per site, before map-output validity was known'})
    path.write_text(json.dumps(requests,indent=2));return requests

def extract(r,kind):
    dest=OUT/kind/r['key']
    try:
        f,grid=fetch_feature_grid(r['bounds'],r['target_time'],dest)
        if kind=='checks':
            # Floating extent arithmetic may generate two rows or columns;
            # retain the center nearest to the actual station coordinate.
            distance=(f.longitude-r['longitude'])**2+(f.latitude-r['latitude'])**2
            z=f.loc[[distance.idxmin()]].copy()
            for c in ['record_id','physical_site_id','source_station','depth_cm']:z[c]=r[c]
            z.to_parquet(dest/'station_pixel.parquet',index=False)
        return {'key':r['key'],'kind':kind,'status':'complete','rows':len(f)}
    except Exception as exc:
        dest.mkdir(parents=True,exist_ok=True);(dest/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        return {'key':r['key'],'kind':kind,'status':'failed','reason':str(exc)}

def main():
    sys.stdout.reconfigure(encoding='utf-8');regions=pick_regions();checks=pick_checks();done=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(extract,r,k) for k,rr in [('regions',regions),('checks',checks)] for r in rr]
        for i,task in enumerate(as_completed(futures),1):
            result=task.result();done.append(result)
            (OUT/'extraction_status.json').write_text(json.dumps(done,indent=2),encoding='utf-8')
            print(f'EXTRACT {i}/{len(futures)} {result["kind"]}/{result["key"]}: {result["status"]}',flush=True)
    print('MAP EXTRACTIONS FINISHED; failures',sum(x['status']=='failed' for x in done),flush=True)

if __name__=='__main__':main()
