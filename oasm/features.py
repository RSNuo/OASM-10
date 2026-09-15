"""One feature specification for station training and gridded prediction."""
import numpy as np
import pandas as pd

BASE=['angle','VV','VH','VH_minus_VV','s2_lag','landsat_lag','DSM','Slope','TWI_proxy','Aspect_sin','Aspect_cos','Sentinel2_B2','Sentinel2_B3','Sentinel2_B4','Sentinel2_B5','Sentinel2_B6','Sentinel2_B7','Sentinel2_B8','Sentinel2_B8A','Sentinel2_B11','Sentinel2_B12','Landsat_B2','Landsat_B3','Landsat_B4','Landsat_B5','Landsat_B6','Landsat_B7','Landsat_B10','NDVI_Best','NDMI_Best','Day_sin','Day_cos']
SAR={'angle','VV','VH','VH_minus_VV'}
OPT={c for c in BASE if c.startswith('Sentinel2_') or c.startswith('Landsat_')}|{'NDVI_Best','NDMI_Best'}
PRECIP=['precip_7d','precip_30d','precip_90d']
TIME={'s2_lag','landsat_lag','Day_sin','Day_cos',*PRECIP}
SOIL=['sg_clay','sg_sand','sg_soc','sg_bdod']
STATIC=['DSM','Slope','TWI_proxy','Aspect_sin','Aspect_cos']+SOIL
CATS=['BeckKG_band1','Soil_Texture_USDA','LandCover']
LAYERS={5:'0-5cm',20:'15-30cm',50:'30-60cm'}

def feature_names(depth,variant='ma'):
    cols=BASE+SOIL+(PRECIP if depth==5 or variant=='add_precip' else [])
    if depth>5 and variant!='no_cascade':cols=cols+['soil_moisture_5cm']
    if variant=='static_forcing':cols=STATIC+['Day_sin','Day_cos']+PRECIP
    if variant=='static_only':cols=STATIC
    if variant=='no_static':cols=[c for c in cols if c not in STATIC]
    if variant=='no_lag':cols=[c for c in cols if c not in {'s2_lag','landsat_lag'}]
    return cols,([] if variant=='no_static' else CATS)

def modality_indices(cols):
    groups={k:[] for k in ['sar','optical_thermal','temporal','surface_sm','static']}
    for i,c in enumerate(cols):
        name='sar' if c in SAR else 'optical_thermal' if c in OPT else 'temporal' if c in TIME else 'surface_sm' if c.startswith('soil_moisture_') else 'static'
        groups[name].append(i)
    return groups

def normalize_category(v):
    if pd.isna(v):return '__MISSING__'
    try:
        f=float(v)
        if np.isfinite(f) and f==int(f):return str(int(f))
    except (ValueError,TypeError):pass
    return str(v)

def soil_for_depth(frame,depth):
    f=frame.copy()
    for prop in ['clay','sand','soc','bdod']:
        source=f'{prop}_{LAYERS[depth]}_mean'
        if source not in f:raise ValueError(f'Missing soil layer: {source}')
        f[f'sg_{prop}']=f[source]
    return f

def derive_features(frame,target_time=None):
    """Inputs are scaled DN/reflectance and raw dB; no indices are multiplied again."""
    f=frame.copy()
    if target_time is not None:f['target_time']=pd.to_datetime(target_time)
    t=pd.to_datetime(f.target_time)
    s2='s2_closest_datetime' if 's2_closest_datetime' in f else 'Sentinel2_closest_datetime'
    f['s2_lag']=(t-pd.to_datetime(f[s2])).abs().dt.total_seconds()/86400
    f['landsat_lag']=(t-pd.to_datetime(f.Landsat_closest_datetime)).abs().dt.total_seconds()/86400
    f['Day_sin']=np.sin(2*np.pi*t.dt.dayofyear/365.0)
    f['Day_cos']=np.cos(2*np.pi*t.dt.dayofyear/365.0)
    f['VH_minus_VV']=f.VH-f.VV
    if 'Aspect' in f:
        f['Aspect_sin']=np.sin(np.deg2rad(f.Aspect));f['Aspect_cos']=np.cos(np.deg2rad(f.Aspect))
    f['TWI_proxy']=-np.log(np.tan(np.deg2rad(f.Slope))+0.001)
    for c,a,b in [('NDVI_Best','Landsat_B5','Landsat_B4'),('NDMI_Best','Landsat_B5','Landsat_B6')]:
        denom=(f[a]+f[b]).replace(0,np.nan);f[c]=(f[a]-f[b])/denom
    return f
