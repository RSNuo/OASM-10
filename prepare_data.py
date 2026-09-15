"""Build raw-value tables and shared site-isolated splits without fitting on outcomes.

The legacy z-score metadata are inverted, never used as a revised fitted transform.
Missing ancillary values remain missing until the downstream training-only fit.
"""
import os
from pathlib import Path
from collections import defaultdict
import hashlib, json, re, shutil, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
REV=Path(__file__).resolve().parent
DATA=REV/'data'
PCOLS=['precip_7d','precip_30d','precip_90d']
SGPROPS=['clay','sand','soc','bdod']
LAYERS={5:'0-5cm',20:'15-30cm',50:'30-60cm'}

def write_json(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def norm(s):
    return re.sub(r'[^a-z0-9]','',str(s).casefold())

def backup():
    out=REV/'originals_20260912'
    files=[ROOT/'training_5cm.ipynb',ROOT/'prediction_5cm.ipynb',ROOT/'prediction_multidepth.ipynb',ROOT/'README.md',ROOT/'README_archive_v5.txt']
    files+=list((ROOT/'manuscript/scientific_data').glob('*.docx'))
    files+=list((ROOT/'manuscript/ISPRS').glob('*.docx'))
    for p in files:
        dest=out/p.relative_to(ROOT)
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)

def load_inputs():
    frames={}; metadata={}
    for d in (5,20,50):
        meta_path=ROOT/'5cm/data/standardizer_meta.json' if d==5 else ROOT/f'{d}cm/model_{d}cm/standardizer_meta.json'
        m=json.loads(meta_path.read_text());metadata[d]=m
        parts=[]
        for split in ['train','test']:
            f=pd.read_pickle(ROOT/f'{d}cm/data/{split}_set.pkl').copy()
            for c in m['standardized_cols']:
                f[c]=f[c].astype(float)*m['scale_'][c]+m['mean_'][c]
            f['depth_cm']=d;f['legacy_split']=split
            f['source_station']=f.station.astype(str)
            f['observation_time']=pd.to_datetime(f.datetime)
            f['target_time']=f.observation_time
            f['s1_time']=pd.to_datetime(f['s1_datetime']) if 's1_datetime' in f else pd.NaT
            target='soil_moisture' if d==5 else f'soil_moisture_{d}cm'
            f['y']=f[target].astype(float)
            f['record_id']=[hashlib.sha256(f'{d}|{s}|{t.isoformat()}'.encode()).hexdigest()[:24] for s,t in zip(f.source_station,f.observation_time)]
            assert not f.record_id.duplicated().any()
            # Time-derived values are rebuilt from retained timestamps, not z-scores.
            f['s2_lag']=(f.target_time-pd.to_datetime(f.s2_closest_datetime)).abs().dt.total_seconds()/86400
            f['landsat_lag']=(f.target_time-pd.to_datetime(f.Landsat_closest_datetime)).abs().dt.total_seconds()/86400
            doy=f.target_time.dt.dayofyear
            f['Day_sin']=np.sin(2*np.pi*doy/365.0);f['Day_cos']=np.cos(2*np.pi*doy/365.0)
            f['NDVI_Best']=(f.Landsat_B5-f.Landsat_B4)/(f.Landsat_B5+f.Landsat_B4)
            f['NDMI_Best']=(f.Landsat_B5-f.Landsat_B6)/(f.Landsat_B5+f.Landsat_B6)
            f['TWI_proxy']=-np.log(np.tan(np.deg2rad(f.Slope))+0.001)
            f['VH_minus_VV']=f.VH-f.VV
            # Drop stale cascade/ancillary features; they are regenerated below.
            f=f.drop(columns=PCOLS+[f'sg_{p}' for p in SGPROPS]+['soil_moisture_5cm','soil_moisture_20cm'],errors='ignore')
            parts.append(f)
        frames[d]=pd.concat(parts,ignore_index=True)
        assert not frames[d].record_id.duplicated().any()
    return frames,metadata

def site_registry(frames):
    allrows=pd.concat([f[['source_station','latitude','longitude','depth_cm','legacy_split']] for f in frames.values()],ignore_index=True)
    sites=allrows[['source_station','latitude','longitude']].drop_duplicates().reset_index(drop=True)
    assert sites[['latitude','longitude']].notna().all().all()
    parent=np.arange(len(sites))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    loc=defaultdict(list);names=defaultdict(list);audit=[]
    for i,r in sites.iterrows():
        loc[(round(r.latitude,6),round(r.longitude,6))].append(i)
        names[norm(r.source_station)].append(i)
    for xy,idx in loc.items():
        for j in idx[1:]:union(idx[0],j)
    # Name matching alone never merges geographically distinct sites.
    for key,idx in names.items():
        for pos,a in enumerate(idx):
            for b in idx[pos+1:]:
                ra,rb=sites.iloc[a],sites.iloc[b]
                dx=(ra.longitude-rb.longitude)*111320*np.cos(np.deg2rad((ra.latitude+rb.latitude)/2))
                dy=(ra.latitude-rb.latitude)*111320
                dist=float(np.hypot(dx,dy))
                if dist<=30:
                    union(a,b)
                    if ra.source_station!=rb.source_station:
                        audit.append({'station_a':ra.source_station,'station_b':rb.source_station,'distance_m':dist,'basis':'normalized name and distance <=30 m'})
    comps=defaultdict(list)
    for i in range(len(sites)):comps[find(i)].append(i)
    for idx in comps.values():
        members=sites.iloc[idx]
        xy=sorted(zip(members.latitude.round(6),members.longitude.round(6)))[0]
        gid='site_'+hashlib.sha256(f'{xy[0]:.6f},{xy[1]:.6f}'.encode()).hexdigest()[:16]
        sites.loc[idx,'physical_site_id']=gid
    allrows=allrows.merge(sites,on=['source_station','latitude','longitude'],validate='many_to_one')
    heldout=set(allrows.loc[allrows.legacy_split=='test','physical_site_id'])
    roles=allrows.groupby('physical_site_id').size().rename('records').reset_index()
    roles['split']=np.where(roles.physical_site_id.isin(heldout),'test','development')
    devrows=allrows[~allrows.physical_site_id.isin(heldout)].reset_index(drop=True)
    foldmap={}
    for k,(_,va) in enumerate(GroupKFold(5,shuffle=True,random_state=42).split(devrows,groups=devrows.physical_site_id)):
        for g in devrows.iloc[va].physical_site_id.unique():foldmap[g]=k
    roles['outer_fold']=roles.physical_site_id.map(foldmap).fillna(-1).astype(int)
    sites=sites.merge(roles,on='physical_site_id',validate='many_to_one')
    for d,f in frames.items():
        frames[d]=f.merge(sites.drop(columns='records'),on=['source_station','latitude','longitude'],validate='many_to_one')
        assert not ((frames[d].legacy_split=='test') & (frames[d].split!='test')).any()
    return sites,pd.DataFrame(audit),roles

def ancillary(frames,sites):
    archive=DATA/'ancillary';archive.mkdir(exist_ok=True,parents=True)
    legacy=Path(os.environ.get('OASM10_LEGACY_ARCHIVE', 'legacy_archive'))/'precip'
    for name in ['era5land_precip_daily_all_stations.parquet','soilgrids_all_stations.parquet']:
        dest=archive/name
        if not dest.exists():shutil.copy2(legacy/name,dest)
    source=sites[['source_station','physical_site_id']].drop_duplicates()
    ambiguous=source.groupby('source_station').physical_site_id.nunique()
    if (ambiguous>1).any():
        write_json(DATA/'ambiguous_ancillary_names.json',ambiguous[ambiguous>1].to_dict())
    # An ancillary name referring to multiple distinct locations must not be guessed.
    source=source[~source.source_station.isin(ambiguous[ambiguous>1].index)]
    pr=pd.read_parquet(archive/'era5land_precip_daily_all_stations.parquet')
    pr['station']=pr.station.astype(str);pr['date']=pd.to_datetime(pr.date)
    pr=pr.merge(source,left_on='station',right_on='source_station',validate='many_to_one')
    count=pr.groupby('station').size();pr['priority']=pr.station.map(count)
    pr=pr.sort_values(['physical_site_id','date','priority','station'],ascending=[True,True,False,True])
    pr=pr.drop_duplicates(['physical_site_id','date'],keep='first')
    pr['precip_mm']=pr.precip.astype(float).clip(lower=0)*1000
    pr[['physical_site_id','date','precip_mm']].to_parquet(archive/'precip_daily_by_site.parquet',index=False)
    accum=[]
    for site,g in pr.groupby('physical_site_id',sort=False):
        s=g.set_index('date').precip_mm.sort_index().asfreq('D')
        sums=pd.DataFrame({f'precip_{n}d':np.log1p(s.rolling(n,min_periods=n).sum()) for n in [7,30,90]})
        sums['physical_site_id']=site;accum.append(sums.reset_index())
    feat=pd.concat(accum,ignore_index=True)
    sg=pd.read_parquet(archive/'soilgrids_all_stations.parquet')
    sg['station']=sg.station.astype(str)
    sg=sg.merge(source,left_on='station',right_on='source_station',validate='many_to_one')
    sg=sg.sort_values('station').groupby('physical_site_id',as_index=False).first()
    sg.to_parquet(archive/'soilgrids_by_site.parquet',index=False)
    for d,f in frames.items():
        f['date']=f.target_time.dt.normalize()
        f=f.merge(feat,on=['physical_site_id','date'],how='left',validate='many_to_one')
        # Keep all layers for upstream predictions at deep observation records.
        for layer in LAYERS.values():
            cols=[f'{p}_{layer}_mean' for p in SGPROPS]
            f=f.merge(sg[['physical_site_id']+cols],on='physical_site_id',how='left',validate='many_to_one')
        for p in SGPROPS:f[f'sg_{p}']=f[f'{p}_{LAYERS[d]}_mean']
        for c in PCOLS+[f'sg_{p}' for p in SGPROPS]:f[f'{c}_missing']=f[c].isna()
        f['s1_time_available']=f.s1_time.notna()
        frames[d]=f

def main():
    sys.stdout.reconfigure(encoding='utf-8');DATA.mkdir(parents=True,exist_ok=True)
    backup();frames,metadata=load_inputs();sites,aliases,roles=site_registry(frames)
    sites.to_csv(DATA/'site_registry.csv',index=False)
    aliases.to_csv(DATA/'station_aliases.csv',index=False);roles.to_csv(DATA/'shared_split.csv',index=False)
    ancillary(frames,sites)
    counts={}
    for d,f in frames.items():
        f.to_parquet(DATA/f'all_{d}cm.parquet',index=False)
        counts[d]={}
        for split,g in f.groupby('split'):
            counts[d][split]={'records':len(g),'sites':g.physical_site_id.nunique(),'source_station_names':g.source_station.nunique(),'start':str(g.target_time.min()),'end':str(g.target_time.max()),'precip_missing_fraction':g[PCOLS].isna().mean().to_dict(),'soil_missing_fraction':float(g.sg_clay.isna().mean()),'outer_fold_counts':g.outer_fold.value_counts().sort_index().to_dict()}
        assert set(f.loc[f.split=='test','physical_site_id']).isdisjoint(f.loc[f.split=='development','physical_site_id'])
    write_json(DATA/'counts.json',counts)
    write_json(DATA/'feature_schema.json',{'version':'oasm10_public_release','base_numeric':metadata[5]['standardized_cols'],'precipitation_columns':PCOLS,'soil_properties':SGPROPS,'soil_layers':LAYERS,'categorical':['BeckKG_band1','Soil_Texture_USDA','LandCover'],'input_scale':'before legacy standardization; fitted transforms are training-only','index_definition':'Landsat DN normalized differences; historical storage names NDVI_Best and NDMI_Best','wetness_epsilon':0.001,'precipitation_encoding':'ln(1+P_mm), complete rolling 7/30/90 calendar days ending on target_time date','target_time':'retained matched observation timestamp; S1 time stored separately where available','record_key':'record_id (depth, source station, observation timestamp)','site_split':'union of all legacy test spatial groups; shared outer group folds, seed 42','spatial_groups':'coordinates rounded to 6 decimals, plus normalized station names within 30 m; used conservatively to group co-located sensors, not to identify identical measurements'})
    print(json.dumps(counts,indent=2),flush=True);print('V6 DATA READY',flush=True)

if __name__=='__main__':main()
