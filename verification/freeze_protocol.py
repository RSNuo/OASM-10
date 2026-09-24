import sys as _sys
from pathlib import Path as _Path
REV = _Path(__file__).resolve().parents[1]   # repository root
HERE = _Path(__file__).resolve().parent
_sys.path.insert(0, str(REV))
"""Record the already-declared corrected protocol and its immutable training inputs."""
from pathlib import Path
import hashlib,json,sys
import pandas as pd

REV=_Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    records={}
    for p in [REV/f'data/all_{d}cm.parquet' for d in [5,20,50]]+[REV/f'oasm/{n}.py' for n in ['features','models','preprocessing','training']]+[REV/'run_campaign.py']:
        records[str(p.relative_to(REV))]=sha(p)
    path=HERE/'training_input_integrity.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['sha256']==records,'Training source or input tables changed since freezing'
    else:
        path.write_text(json.dumps({'revision':'oasm10_public_release','sha256':records,'suite_declared_in':'WORKLOG.md',
            'definition':'Post-audit refit under a fixed protocol; earlier benchmark use is disclosed, not presented as a prospective untouched holdout.',
            'seeds':[42,7,555],'product_reference_seed':42,'architecture_selection':'none using revised test metrics',
            'main_controls':'MA, concatenation fusion, FT-style single stream; plus site context and forcing at5cm and no cascade at20/50cm',
            'tree_benchmarks':'RF and XGBoost; native squared-error objectives; same input records and isolation',
            'ensemble':'secondary greedy OOF-only combination of the3predeclared MA seeds; not used to replace the fixed reference product'},indent=2),encoding='utf-8')
    annual=[]
    for d in [5,20,50]:
        f=pd.read_parquet(REV/f'data/all_{d}cm.parquet');f['year']=pd.to_datetime(f.target_time).dt.year
        g=f.groupby(['year','split']).agg(records=('record_id','size'),sites=('physical_site_id','nunique')).reset_index();g['depth_cm']=d;annual.append(g)
    pd.concat(annual,ignore_index=True).to_csv(REV/'data/annual_record_counts.csv',index=False)
    print('TRAINING INPUT INTEGRITY VERIFIED')

if __name__=='__main__':main()
