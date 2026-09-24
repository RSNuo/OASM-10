import sys as _sys
from pathlib import Path as _Path
REV = _Path(__file__).resolve().parents[1]   # repository root
HERE = _Path(__file__).resolve().parent
_sys.path.insert(0, str(REV))
"""Numerical checks of the paired-cluster statistic and raster pixel geometry."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_results import paired_seed_bootstrap,statistics,site_statistics,site_summary,align
from oasm.inference import grid_transform_from_centers,predict_product

def main():
    rng=np.random.default_rng(11);n=240
    y=rng.uniform(.05,.5,n)
    ref=pd.DataFrame({'record_id':[str(i) for i in range(n)],'physical_site_id':np.repeat([f's{i}' for i in range(12)],20),
                      'y':y,'prediction':y+.02,'target_time':pd.date_range('2020-01-01',periods=n)})
    pairs=[]
    for offset in [.01,.02,.03]:
        alt=ref.copy();alt.prediction=alt.prediction+offset;pairs.append((alt,ref))
    result=paired_seed_bootstrap(pairs)
    assert np.allclose(result['per_seed_rmse_differences'],[.01,.02,.03])
    assert np.allclose(result['rmse_ci95'],[.02,.02])
    assert abs(result['anomaly_r_difference'])<1e-12
    z=ref.copy();z.prediction=z.y+.01*np.sin(np.arange(n))
    s=statistics(z);assert abs(s['rmse']**2-s['ubrmse']**2-s['bias']**2)<1e-12
    x=np.array([-121.,-120.9999]);y=np.array([38.,37.9999]);tr=grid_transform_from_centers(x,y)
    assert np.allclose(tr*(.5,.5),(x[0],y[0]),rtol=0,atol=1e-10)
    assert np.allclose(tr*(1.5,1.5),(x[1],y[1]),rtol=0,atol=1e-10)
    # This checks membership pairing, not arbitrary CSV row order.
    shuffled=ref.sample(frac=1,random_state=4)
    identity=paired_seed_bootstrap([(ref,shuffled)])
    assert identity['rmse_ci95']==[0.,0.]
    try:
        align(ref,ref.iloc[:-1])
        raise AssertionError('A missing comparison key was accepted')
    except ValueError:pass
    # Co-located but offset sensor series must not acquire spurious temporal
    # correlation from a model which predicts a separate constant per sensor.
    twins=pd.concat([ref.assign(source_station='a'),ref.assign(source_station='b')],ignore_index=True)
    twins.loc[twins.source_station=='b','y']+=.2
    twins['prediction']=twins.groupby(['physical_site_id','source_station']).y.transform('mean')
    s=site_statistics(twins)
    assert s.r.isna().all() and site_summary(s)['pooled_within_site_anomaly_r'] is None
    # A fully masked valid query has a stable output schema without loading models.
    root=_Path(__file__).resolve().parents[1]
    source=pd.read_parquet(root/'data/all_5cm.parquet').head(3).copy();source['LandCover']=80
    masked=predict_product(source,root/'deliberately_absent_models')
    assert not masked.valid_satellite_output.any()
    for d in [5,20,50]:
        assert masked[f'sm_{d}cm'].isna().all() and not masked[f'outside_physical_range_{d}cm'].any()
    (HERE/'verification_analysis.json').write_text(json.dumps({'paired_known_offsets':result,'row_order_invariance':identity,'mse_identity':True,'pixel_center_geometry':True,
        'co_located_constant_series_undefined_correlation':True,'fully_masked_product_schema':True,'missing_pair_key_rejected':True},indent=2))
    print('ANALYSIS AND GRID CHECKS PASS')

if __name__=='__main__':main()
