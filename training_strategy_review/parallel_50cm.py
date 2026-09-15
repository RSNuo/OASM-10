"""Execute independent 50 cm fits with the identical frozen function/configuration.

The main process is still in the 5 cm retained stage. These small jobs finish
before its 20 cm stage; its resume check then reuses completed 50 cm artifacts.
No outcomes are used for scheduling or model choices.
"""
import sys,torch
from run_strategies import fit_one,write,HERE
from datetime import datetime,timezone
sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
write(HERE/'parallel_execution.json',{'utc':datetime.now(timezone.utc).isoformat(),'scope':'Execution scheduling only; same frozen fit_one function and all 15 declared 50 cm retained fits. No policy, data or seed changes; main process resumes completed artifacts.','test_access':False})
for seed in [42,7,555]:
    for fold in range(5):fit_one(50,seed,fold,'retain_internal_best')
write(HERE/'parallel_50cm_complete.json',{'utc':datetime.now(timezone.utc).isoformat(),'fits':15})
print('PARALLEL_50CM_COMPLETE',flush=True)
