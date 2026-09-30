"""Read-only GEE feature extraction on an explicit, reproducible output grid.

No model, imputation or regional statistics are fitted during map extraction.
The symmetric optical window defines a retrospective product.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,time,urllib.request
import numpy as np
import pandas as pd
import ee
from .features import derive_features

REV=Path(__file__).resolve().parents[1]
ANGULAR_STEP=10/111319.49079327358
NODATA=-999999.
S2B=['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
LSB=['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10']
LSN=['Landsat_B2','Landsat_B3','Landsat_B4','Landsat_B5','Landsat_B6','Landsat_B7','Landsat_B10']

def initialize(project=None):
    ee.Initialize(project=project);ee.data.setDeadline(120000)

def _stamp(img,name):
    return ee.Image.constant(ee.Number(img.get('system:time_start')).divide(1000)).rename(name).toDouble()

def build_image(bounds,target_time,s1_window_hours=24,project=None):
    """Select one S1 orbit around the requested observation time; retain provenance."""
    initialize(project);roi=ee.Geometry.Rectangle(list(bounds),geodesic=False)
    target=pd.Timestamp(target_time)
    if target.tzinfo is not None:target=target.tz_convert('UTC').tz_localize(None)
    date=ee.Date(target.isoformat());start=date.advance(-14,'day');end=date.advance(14,'day').advance(1,'second')
    s1=(ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi)
        .filterDate(date.advance(-s1_window_hours,'hour'),date.advance(s1_window_hours,'hour'))
        .filter(ee.Filter.eq('instrumentMode','IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH')))
    s1=s1.map(lambda im:im.set('distance',ee.Number(im.get('system:time_start')).subtract(date.millis()).abs())).sort('distance')
    if not s1.size().getInfo():raise ValueError('No dual-polarization S1 IW scene within the declared 24-hour window')
    anchor=ee.Image(s1.first());orbit=anchor.get('relativeOrbitNumber_start');track=anchor.get('orbitProperties_pass')
    s1=s1.filter(ee.Filter.eq('relativeOrbitNumber_start',orbit)).filter(ee.Filter.eq('orbitProperties_pass',track))
    def prep1(im):
        z=im.select(['VV','VH','angle']);return z.addBands(_stamp(im,'s1_seconds')).updateMask(z.mask().reduce(ee.Reducer.min())).copyProperties(im,['system:time_start','distance'])
    radar=s1.sort('distance',False).map(prep1).mosaic()
    s2=(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate(start,end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',60)))
    ls=(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
        .filterBounds(roi).filterDate(start,end))
    if not s2.size().getInfo() or not ls.size().getInfo():raise ValueError('Both optical collections are required within +/-14 days')
    def prep2(im):
        qa=im.select('QA60');clear=qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
        z=im.select(S2B,['Sentinel2_'+b for b in S2B]).multiply(1e-4).resample('bilinear')
        mask=z.mask().reduce(ee.Reducer.min()).And(clear)
        score=ee.Image.constant(ee.Number(im.get('system:time_start')).subtract(date.millis()).abs().multiply(-1)).rename('nearest').toDouble()
        return z.addBands(_stamp(im,'s2_seconds')).addBands(score).updateMask(mask)
    def prepls(im):
        qa=im.select('QA_PIXEL');clear=qa.bitwiseAnd(1<<3).eq(0).And(qa.bitwiseAnd(1<<4).eq(0))
        z=im.select(LSB,LSN).multiply(1e-4).resample('bilinear')
        mask=z.mask().reduce(ee.Reducer.min()).And(clear)
        score=ee.Image.constant(ee.Number(im.get('system:time_start')).subtract(date.millis()).abs().multiply(-1)).rename('nearest').toDouble()
        return z.addBands(_stamp(im,'ls_seconds')).addBands(score).updateMask(mask)
    optical=s2.map(prep2).qualityMosaic('nearest').select(['Sentinel2_'+b for b in S2B]+['s2_seconds'])
    landsat=ls.map(prepls).qualityMosaic('nearest').select(LSN+['ls_seconds'])
    dem=(ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().rename('elevation')
         .unmask(ee.ImageCollection('JAXA/ALOS/AW3D30/V4_1').select('DSM').mosaic().rename('elevation'))
         .setDefaultProjection(crs='EPSG:4326',scale=30))
    terrain=ee.Terrain.products(dem).select(['elevation','slope','aspect'],['DSM','Slope','Aspect']).resample('bilinear')
    texture=ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02').select('b0').rename('Soil_Texture_USDA')
    cover=ee.Image(ee.ImageCollection('ESA/WorldCover/v200').first()).select('Map').rename('LandCover')
    daily=ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').select('total_precipitation_sum')
    # A daily sum includes the whole target calendar day, including hours after
    # an observation earlier that day. The product is explicitly retrospective.
    tomorrow=ee.Date(target.normalize().isoformat()).advance(1,'day')
    forcing=[]
    for n in [7,30,90]:
        c=daily.filterDate(tomorrow.advance(-n,'day'),tomorrow)
        z=c.map(lambda im:im.max(0).multiply(1000)).sum().add(1).log()
        forcing.append(z.updateMask(c.count().eq(n)).rename(f'precip_{n}d'))
    soils=[]
    for prop in ['clay','sand','soc','bdod']:
        names=[f'{prop}_{layer}_mean' for layer in ['0-5cm','15-30cm','30-60cm']]
        soils.append(ee.Image(f'projects/soilgrids-isric/{prop}_mean').select(names))
    composite=ee.Image.cat([radar,optical,landsat,terrain,texture,cover,*forcing,*soils]).clip(roi).toDouble()
    meta={'query_utc':datetime.now(timezone.utc).isoformat(),'target_time_utc':target.isoformat(),
          's1_window_hours':s1_window_hours,'s1_relative_orbit':orbit.getInfo(),'s1_pass':track.getInfo(),
          's1_assets':s1.aggregate_array('system:index').getInfo(),'s2_candidate_assets':s2.aggregate_array('system:index').getInfo(),
          'landsat_candidate_assets':ls.aggregate_array('system:index').getInfo(),
          's2_cloud_threshold':'<60% with QA60 bits 10/11 clear; masked QA60 produces missing output',
          'optical_choice':'nearest clear valid pixel within +/-14 days, both sensors required',
          'precipitation':'ln(1+mm), full 7/30/90 calendar days ending on target day; missing windows remain missing',
          'soil_missing':'retained as NaN for fold-training imputation; no ROI fill',
          'landsat_indices':'DN-based normalized differences computed locally; no sensor switching',
          'epsilon':0.001,'continuous_resampling':'bilinear for optical and terrain; nearest for precipitation and SoilGrids',
          'categorical_resampling':'nearest','protocol':'oasm-feature-grid-1'}
    return composite,meta

def fetch_feature_grid(bounds,target_time,cache_path,tile_size=192,climate_raster=None,project=None):
    import rasterio
    from rasterio.transform import Affine
    out=Path(cache_path);out.mkdir(parents=True,exist_ok=True)
    west,south,east,north=map(float,bounds)
    if not (-180<=west<east<=180 and -90<=south<north<=90):raise ValueError('Invalid bounds')
    width=math.ceil((east-west)/ANGULAR_STEP);height=math.ceil((north-south)/ANGULAR_STEP)
    if width*height>4_000_000:raise ValueError('Use smaller regions (at most 4 million grid cells per request)')
    transform=Affine(ANGULAR_STEP,0,west,0,-ANGULAR_STEP,north)
    beck=Path(climate_raster) if climate_raster is not None else REV/'data/ancillary/Beck_KG_V1_present_0p0083.tif'
    if not beck.is_absolute():beck=REV/beck
    if not beck.is_file():
        raise FileNotFoundError('Missing Beck climate raster. See ZENODO_CONTENTS.md or pass climate_raster explicitly.')
    with beck.open('rb') as src:climate_sha256=hashlib.file_digest(src,'sha256').hexdigest()
    request={'bounds':list(bounds),'target_time':str(target_time),'step':ANGULAR_STEP,'protocol':'oasm-feature-grid-2','climate_sha256':climate_sha256}
    fingerprint=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
    record=out/'grid.json';parquet=out/'features.parquet'
    if record.exists():
        old=json.loads(record.read_text())
        if old.get('fingerprint')!=fingerprint:raise ValueError('Cache belongs to a different extraction request')
        if parquet.exists():return pd.read_parquet(parquet),old
    im,meta=build_image(bounds,target_time,project=project);names=im.bandNames().getInfo()
    arr=np.full((len(names),height,width),np.nan,dtype=np.float64)
    for row in range(0,height,tile_size):
        for col in range(0,width,tile_size):
            h=min(tile_size,height-row);w=min(tile_size,width-col)
            tr=transform*Affine.translation(col,row);tile=out/f'tile_{row}_{col}.tif'
            if not tile.exists():
                for attempt in range(3):
                    try:
                        url=im.unmask(NODATA,False).getDownloadURL({'crs':'EPSG:4326','crs_transform':list(tr)[:6],'dimensions':[w,h],'format':'GEO_TIFF','filePerBand':False})
                        partial=tile.with_suffix('.part')
                        with urllib.request.urlopen(url,timeout=120) as src,partial.open('wb') as dst:dst.write(src.read())
                        with rasterio.open(partial) as ds:assert ds.shape==(h,w) and ds.count==len(names)
                        partial.replace(tile);break
                    except Exception as exc:
                        if isinstance(exc,urllib.error.HTTPError):
                            detail=exc.read().decode('utf-8',errors='replace')
                            if attempt==2:raise RuntimeError(f'GEE grid request failed: {detail}') from exc
                        if attempt==2:raise
                        time.sleep(3*(attempt+1))
            with rasterio.open(tile) as ds:
                if not np.allclose(list(ds.transform)[:6],list(tr)[:6],rtol=0,atol=1e-10):raise ValueError('Downloaded grid differs from requested affine')
                z=ds.read();z[z==NODATA]=np.nan;arr[:,row:row+h,col:col+w]=z
    f=pd.DataFrame(arr.reshape(len(names),-1).T,columns=names)
    rr,cc=np.indices((height,width));f['longitude'],f['latitude']=transform*(cc.ravel()+.5,rr.ravel()+.5)
    f['target_time']=pd.Timestamp(target_time)
    for old,new in [('s1_seconds','s1_acquisition_time'),('s2_seconds','s2_closest_datetime'),('ls_seconds','Landsat_closest_datetime')]:
        f[new]=pd.to_datetime(f.pop(old),unit='s',errors='coerce')
    with rasterio.open(beck) as ds:
        if ds.crs.to_epsg()!=4326:raise ValueError('Expected WGS84 Beck classification raster')
        f['BeckKG_band1']=[v[0] for v in ds.sample(zip(f.longitude,f.latitude))]
        if ds.nodata is not None:f.loc[f.BeckKG_band1==ds.nodata,'BeckKG_band1']=np.nan
    f=derive_features(f)
    meta.update(request);meta.update({'fingerprint':fingerprint,'width':width,'height':height,'transform':list(transform)[:6],
         'crs':'EPSG:4326','angular_spacing_degrees':ANGULAR_STEP,'nominal_equatorial_spacing_m':10,
         'longitudinal_spacing_m_at_center':10*math.cos(math.radians((north+south)/2)),
         'band_names':names,'s2_display':'B4/B3/B2 from the same nearest-clear composite as predictors'})
    f.to_parquet(parquet,index=False);record.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return f,meta
