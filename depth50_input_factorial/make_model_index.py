"""List exact model locations, including the explicitly reused checkpoints."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;REV=HERE.parent

def main():
    entries=[]
    for p in sorted((HERE/'models').glob('*_seed*/outer*/manifest.json')):
        m=json.loads(p.read_text(encoding='utf-8'));wp=REV/m['weight_directory']
        entries.append({'configuration':m['config_id'],'seed':m['seed'],'outer_fold':m['fold'],
            'newly_trained':not m['identity_reuse'],'selected_epochs':m['selected_epochs'],'refitted_epochs':m['fitted_epochs'],
            'numeric_predictors':len(m['config']['numeric']),'model':str(wp/'model.pt'),
            'preprocessor':str(wp/'preprocessor.joblib'),'experiment_manifest':str(p),
            'model_sha256':m['weights_sha256']})
    assert len(entries)==60
    with (HERE/'MODEL_INDEX.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
    print('MODEL_INDEX_COMPLETE',len(entries))

if __name__=='__main__':main()
