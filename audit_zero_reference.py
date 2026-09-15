"""Trace the availability-selected zero-valued example to original sources."""
import os
from pathlib import Path
import json,hashlib,datetime,shutil
import pandas as pd
R=Path(__file__).resolve().parent
out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':'Fixed record availability, independent of prediction error','interpretation':'Flags transcribed only; no inferred network M semantics and no retrospective record exclusion','depths':{}}
for d in [20,50]:
    paths=list(Path(os.environ.get('OASM10_ISMN_ROOT', 'ISMN'))/'HOAL/Hoal-19'.glob(f'*sm_0.{d}0000_0.{d}0000*'))
    assert len(paths)==1,paths
    path=paths[0];lines=[x for x in path.read_text().splitlines() if x.startswith('2018/')]
    archived=R/'analysis/reference_sources'/path.name
    archived.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,archived)
    fields=[x.split() for x in lines]
    f=pd.read_parquet(R/f'analysis/temporal_example_{d}cm.parquet')
    assert len(f)==78 and (f.y==0).all()
    observed={x[0]+' '+x[1]:x for x in fields}
    retained_flags=[]
    for t in pd.to_datetime(f.target_time):
        k=t.strftime('%Y/%m/%d %H:%M');assert k in observed and float(observed[k][2])==0,(d,k)
        retained_flags.append(tuple(observed[k][3:]))
    out['depths'][str(d)]={'source_file':str(path),'archived_source':str(archived.relative_to(R)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_2018_rows':len(fields),'source_2018_zero_rows':sum(float(x[2])==0 for x in fields),
        'source_flag_pairs':sorted({tuple(x[3:]) for x in fields}),'retained_example_rows':len(f),
        'retained_flag_pair_counts':{str(pair):retained_flags.count(pair) for pair in sorted(set(retained_flags))},
        'all_retained_targets_exact_zero':True,'all_retained_timestamps_verified_in_raw_source':True,'representative_lines':lines[:4]}
(R/'analysis/reference_zero_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
