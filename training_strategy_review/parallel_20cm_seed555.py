"""Schedule one independent later batch after the parallel 50 cm worker ends."""
import sys,time,torch
from datetime import datetime,timezone
from run_strategies import HERE,fit_one,write
sys.stdout.reconfigure(encoding='utf-8')
while not (HERE/'parallel_50cm_complete.json').exists():time.sleep(20)
torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
write(HERE/'parallel_20cm_execution.json',{'utc':datetime.now(timezone.utc).isoformat(),'scope':'Execution-only scheduling of the same five frozen 20 cm seed555 retained fits; upstream inputs and all scientific choices unchanged.'})
for fold in range(5):fit_one(20,555,fold,'retain_internal_best')
write(HERE/'parallel_20cm_complete.json',{'utc':datetime.now(timezone.utc).isoformat(),'fits':5})
print('PARALLEL_20CM_COMPLETE',flush=True)
