"""Development-only arithmetic cross-check while the new fits are running."""
import pandas as pd
from run_factorial import HERE,SEEDS,write,now
from verify_factorial import calc,boot,check
from analyze_factorial import metrics

def main():
    pairs=[]
    for s in SEEDS:
        frames=[]
        for c in ['B','A']:
            z=pd.concat([pd.read_parquet(HERE/f'models/{c}_seed{s}/outer{k}/oof_predictions.parquet') for k in range(5)],ignore_index=True)
            z[['y','prediction']]=z[['y','prediction']].astype(float)
            assert z.record_id.is_unique
            ref={**metrics.statistics(z),**metrics.site_summary(metrics.site_statistics(z))}
            for key,value in calc(z).items():
                if key!='site_bias_mse_share':check(value,ref[key],2e-9)
            frames.append(z)
        pairs.append(tuple(frames))
    a=boot(pairs);b=metrics.paired_seed_bootstrap(pairs)
    for key,value in a.items():check(value,b[key])
    identical=boot([(pairs[0][0],pairs[0][0])])
    for value in identical.values():check(value,0)
    write(HERE/'metric_code_check.json',{'utc':now(),'passed':True,'scope':'OOF B/A only; absolute metrics, matched-seed bootstrap and zero contrast identities; no test data loaded.'})
    print('METRIC_CODE_CHECK_PASS')

if __name__=='__main__':main()
