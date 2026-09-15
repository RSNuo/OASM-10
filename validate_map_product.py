"""All three depths: actual map extraction, complete fold-paired cascade, fixed holdout."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from rasterio.transform import Affine
import torch
from oasm.inference import predict_product,valid_satellite_rows,write_geotiff,reference_runs
from analyze_results import statistics,paired_seed_bootstrap,read_reference

REV=Path(__file__).resolve().parent;EX=REV/'validation/map_pipeline';OUT=REV/'product';OUT.mkdir(exist_ok=True)

def checks():
    requests=json.loads((EX/'check_requests.json').read_text());summaries={};allmatched=[];details=[]
    for d in [5,20,50]:
        chosen=[r for r in requests if r['depth_cm']==d];valid=[]
        for r in chosen:
            path=EX/f'checks/{r["key"]}/station_pixel.parquet'
            if path.exists():valid.append(pd.read_parquet(path))
            else:details.append({**r,'status':'extraction_failed'})
        f=pd.concat(valid,ignore_index=True) if valid else pd.DataFrame()
        raw=pd.read_parquet(REV/f'data/all_{d}cm.parquet')
        if len(f):
            pred=predict_product(f,REV/'models',depths=(d,),strict_holdout=True,batch_rows=1024)
            f['prediction']=pred[f'sm_{d}cm'];f['valid_map_output']=pred.valid_satellite_output
            f=f.merge(raw[['record_id','y','split']],on='record_id',validate='one_to_one')
            assert (f.split=='test').all()
            for _,r in f.iterrows():details.append({'record_id':r.record_id,'depth_cm':d,'status':'valid_output' if r.valid_map_output else 'masked_or_missing_predictors'})
            z=f[f.prediction.notna()].copy()
        else:z=pd.DataFrame()
        summary={'requested_records':len(chosen),'requested_sites':len(set(r['physical_site_id'] for r in chosen)),
                 'extraction_completed_records':len(f),'valid_records':len(z),'valid_output_retention':len(z)/len(chosen)}
        if len(z):
            table=read_reference(d).set_index('record_id').reindex(z.record_id).reset_index()
            assert np.allclose(table.y,z.y,atol=1e-7,rtol=0)
            dif=z.prediction.to_numpy()-table.prediction.to_numpy()
            summary.update({'valid_sites':z.physical_site_id.nunique(),'map_pathway':statistics(z),'station_table_pathway':statistics(table),
                'pathway_agreement_rmse':float(np.sqrt(np.mean(dif**2))),'pathway_agreement_mae':float(np.abs(dif).mean()),
                'map_minus_station_error':paired_seed_bootstrap([(z,table)])})
            z['station_table_prediction']=table.prediction.to_numpy();z['depth_cm']=d
            z.to_parquet(EX/f'matched_{d}cm.parquet',index=False);allmatched.append(z)
        summaries[str(d)]=summary
    result={'selection_protocol':'12 hash-ranked external sites with >=24 records, six time quantiles per depth; selection frozen before extraction',
            'scope':'actual gridded predictor extraction and full paired upstream/downstream reference models at all three depths','reference_runs':{str(d):r for d,r in reference_runs(REV/'models').items()},
            'fold_exclusion':'all requested physical site IDs checked against gradient and epoch-selection manifests at every upstream and downstream outer fold',
            'summaries':summaries}
    (EX/'validation_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (EX/'record_status.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    return result

def regions():
    req=json.loads((EX/'region_requests.json').read_text());manifest=[]
    for r in req:
        source=EX/'regions'/r['key'];f=pd.read_parquet(source/'features.parquet');grid=json.loads((source/'grid.json').read_text())
        pred=predict_product(f,REV/'models',depths=(5,20,50),batch_rows=4096)
        dest=OUT/'example_geotiffs'/r['key'];dest.mkdir(parents=True,exist_ok=True)
        shape=(grid['height'],grid['width']);tr=Affine(*grid['transform']);record={**r,**grid,'files':[]}
        for d in [5,20,50]:
            name=f'OASM10_{d}cm_{r["target_time"][:10].replace("-","")}.tif';path=dest/name
            values=pred[f'sm_{d}cm'].to_numpy().reshape(shape)
            write_geotiff(path,values,tr,tags={'units':'m3 m-3','depth_cm':str(d),'model_revision':'oasm10_public_release',
                'target_time_utc':r['target_time'],'output_spacing':'nominal 10 m at equator; EPSG4326 angular grid',
                'prediction_rule':'mean of five paired upstream/downstream outer models','reference_run':reference_runs(REV/'models')[d],'clipping':'none; outside [0,1] flagged in quality table'})
            observed=values[np.isfinite(values)]
            record['files'].append({'name':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'valid_pixel_fraction':float(np.isfinite(values).mean()),
                    'out_of_range_fraction_of_valid':float(np.mean((observed<0)|(observed>1))) if len(observed) else None})
        pred.to_parquet(dest/'quality_and_predictions.parquet',index=False)
        # Archive reflectances rather than a contrast-stretched screenshot.
        import rasterio
        rgb=np.stack([f[c].to_numpy().reshape(shape) for c in ['Sentinel2_B4','Sentinel2_B3','Sentinel2_B2']]).astype('float32')
        with rasterio.open(dest/'S2_true_color_reflectance.tif','w',driver='GTiff',height=shape[0],width=shape[1],count=3,dtype='float32',crs='EPSG:4326',transform=tr,nodata=np.nan,compress='deflate') as dst:
            dst.write(rgb);dst.update_tags(band_order='B4 B3 B2',units='reflectance',composite='nearest clear pixel within +/-14 days',target_time_utc=r['target_time'])
        record['files'].append({'name':'S2_true_color_reflectance.tif',
            'sha256':hashlib.sha256((dest/'S2_true_color_reflectance.tif').read_bytes()).hexdigest(),
            'valid_pixel_fraction':float(np.isfinite(rgb).all(axis=0).mean())})
        (dest/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8');manifest.append(record)
        print('PRODUCT GRID COMPLETE',r['key'],flush=True)
    (OUT/'example_geotiffs/manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    checks();regions();print('V6 MAP PRODUCT VALIDATION COMPLETE',flush=True)

if __name__=='__main__':main()
