"""Detach the requested full CPU suite and persist its PID and source snapshot."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

root=Path(__file__).resolve().parent
out=root/'results/test_exact_cpu'
if (out/'launch.json').exists():
    raise SystemExit('Already launched. Use status.py to inspect or the documented suite.py command to resume.')
for folder in [out,root/'results/smoke_cpu']:
    snapshot=folder/'source_snapshot'
    snapshot.mkdir(parents=True,exist_ok=True)
    for path in list(root.glob('*.py')) + [root/n for n in ['models.json','README.md','requirements.lock.txt','source_revisions.json']]:
        shutil.copy2(path,snapshot/path.name)
command=['nice','-n','10',sys.executable,'-u',str(root/'suite.py'),'--models','all',
         '--split','test','--condition','exact','--device','cpu','--threads','2','--output',str(out)]
with (root/'logs/test_exact_cpu.log').open('a') as log:
    proc=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                          start_new_session=True,close_fds=True)
record={'pid':proc.pid,'started_at_utc':datetime.now(timezone.utc).isoformat(),'command':command,
        'log':str(root/'logs/test_exact_cpu.log')}
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
