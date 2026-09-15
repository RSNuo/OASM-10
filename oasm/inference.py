"""The public product predictor uses matched upstream/downstream fold pairs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .features import BASE,PRECIP,SOIL,CATS,soil_for_depth,derive_features
from .training import predict_frame

def valid_satellite_rows(frame):
    x=frame[BASE].to_numpy(dtype=float)
    valid=np.isfinite(x).all(axis=1)
    valid&=frame.s2_lag.between(0,14).to_numpy()&frame.landsat_lag.between(0,14).to_numpy()
    valid&=pd.to_numeric(frame.LandCover,errors='coerce').isin([10,20,30,40,60,70,90,95,100]).to_numpy()
    return valid

def reference_runs(model_root):
    """Run directory per depth for the deployed product; models/product_reference.json overrides the seed-42 default."""
    runs={5:'ma_seed42',20:'ma_seed42',50:'ma_seed42'};path=Path(model_root)/'product_reference.json'
    if path.exists():
        spec=json.loads(path.read_text(encoding='utf-8')).get('runs',{})
        for d in runs:runs[d]=spec.get(f'{d}cm',runs[d])
    return runs

def validate_exclusion(directory,site_ids):
    meta=json.loads((Path(directory)/'manifest.json').read_text())
    seen=set(meta['gradient_site_ids'])|set(meta['early_stopping_site_ids'])
    overlap=seen&set(site_ids)
    if overlap:raise ValueError(f'Evaluation sites entered training/early stopping: {sorted(overlap)[:3]}')

def predict_product(frame,model_root,depths=(5,20,50),strict_holdout=False,batch_rows=65536):
    """Input contains raw-value features, not standardized station-table values.

    Missing forcing/soil uses each fold's training-only imputation values. No
    ROI-dependent fitting or nearest-value replacement takes place here.
    """
    model_root=Path(model_root);f=derive_features(frame)
    required=BASE+PRECIP+CATS+[f'{p}_{layer}_mean' for p in ['clay','sand','soc','bdod'] for layer in ['0-5cm','15-30cm','30-60cm']]
    missing=[c for c in required if c not in f]
    if missing:raise ValueError(f'Missing required raw predictors: {missing}')
    if strict_holdout and 'physical_site_id' not in f:raise ValueError('Holdout validation requires physical_site_id')
    valid=valid_satellite_rows(f);idx=np.flatnonzero(valid)
    result=pd.DataFrame(index=f.index);result['valid_satellite_output']=valid
    result['forcing_missing']=f[PRECIP].isna().any(axis=1)
    for d in depths:
        result[f'sm_{d}cm']=np.nan
        result[f'outside_physical_range_{d}cm']=False
        layer={5:'0-5cm',20:'15-30cm',50:'30-60cm'}[d]
        result[f'soil_missing_{d}cm']=f[[f'{p}_{layer}_mean' for p in ['clay','sand','soc','bdod']]].isna().any(axis=1)
    if not len(idx):return result
    accum={d:np.zeros(len(idx),dtype=float) for d in depths};runs=reference_runs(model_root)
    for k in range(5):
        upstream=model_root/f'5cm/{runs[5]}/outer{k}'
        if strict_holdout:validate_exclusion(upstream,f.iloc[idx].physical_site_id)
        for start in range(0,len(idx),batch_rows):
            chosen=idx[start:start+batch_rows];z=f.iloc[chosen].copy()
            p5=predict_frame(upstream,soil_for_depth(z,5))
            if 5 in accum:accum[5][start:start+len(chosen)]+=p5/5
            z['soil_moisture_5cm']=p5
            for d in depths:
                if d==5:continue
                downstream=model_root/f'{d}cm/{runs[d]}/outer{k}'
                if strict_holdout:validate_exclusion(downstream,z.physical_site_id)
                accum[d][start:start+len(chosen)]+=predict_frame(downstream,soil_for_depth(z,d))/5
    for d,values in accum.items():
        result.iloc[idx,result.columns.get_loc(f'sm_{d}cm')]=values
        result[f'outside_physical_range_{d}cm']=result[f'sm_{d}cm'].notna()&~result[f'sm_{d}cm'].between(0,1)
    return result

def grid_transform_from_centers(lons,lats):
    from rasterio.transform import from_origin
    lons=np.asarray(lons,dtype=float);lats=np.asarray(lats,dtype=float)
    if len(lons)<2 or len(lats)<2:raise ValueError('Provide the source affine transform for a single-row/column raster')
    dx=np.diff(lons);dy=-np.diff(lats)
    if not (np.all(dx>0) and np.all(dy>0)):raise ValueError('Longitude must ascend and latitude descend')
    if not (np.allclose(dx,np.median(dx),rtol=1e-5,atol=1e-10) and np.allclose(dy,np.median(dy),rtol=1e-5,atol=1e-10)):raise ValueError('Nonuniform center grid; preserve the source transform instead')
    xres=float(np.median(dx));yres=float(np.median(dy))
    return from_origin(lons[0]-xres/2,lats[0]+yres/2,xres,yres)

def write_geotiff(path,values,transform,crs='EPSG:4326',tags=None):
    import rasterio
    a=np.asarray(values,dtype=np.float32)
    with rasterio.open(path,'w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype='float32',crs=crs,transform=transform,nodata=np.nan,compress='deflate',tiled=True) as dst:
        dst.write(a,1);dst.update_tags(**(tags or {}))
