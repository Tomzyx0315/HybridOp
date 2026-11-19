"""Run a frozen manifest in order, waiting for idle GPUs before every job."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-dir',type=Path,required=True)
args=parser.parse_args()
ROOT=args.experiment_dir.resolve()
MANIFEST=json.loads((ROOT/'manifest.json').read_text())
SOURCE=ROOT/'source'
JOBS=MANIFEST['jobs']
completed=[]
current=None
last_step=0

def utc():
    return datetime.now(timezone.utc).isoformat()

def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def event(kind,**fields):
    record=dict(event=kind,utc=utc(),**fields)
    line=json.dumps(record,allow_nan=False)
    with (ROOT/'events.jsonl').open('a') as handle:
        handle.write(line+'\n')
    print(line,flush=True)

def status(state,**fields):
    atomic(ROOT/'status.json',dict(status=state,current=current,last_step=last_step,completed=completed,
        queued=[j['name'] for j in JOBS if j['name'] not in completed and j['name']!=current],updated_utc=utc(),**fields))

def main():
    global current,last_step
    for job in JOBS:
        current,last_step=job['name'],job['start_step']
        case=ROOT/current
        case.mkdir(exist_ok=False)
        command=job['command']+['--output-dir',str(case/'run')]
        atomic(case/'command.json',command)
        guard=MANIFEST['gpu_queue']
        guarded_command=[sys.executable,'-m','hybrid.wait_for_gpus',
            '--gpu-indices',','.join(map(str,guard['indices'])),
            '--poll-seconds',str(guard['poll_seconds']),
            '--idle-checks',str(guard['idle_checks']),
            '--max-memory-mib',str(guard['max_memory_mib']),
            '--max-utilization',str(guard['max_utilization_percent']),
            '--state-file',str(case/'gpu_wait.json'),'--lock-file',guard['lock_file'],
            '--',*command]
        atomic(case/'gpu_guard_command.json',guarded_command)
        env=dict(os.environ,HYBRID_STALL_SECONDS='180',HYBRID_DIAGNOSTICS_DIR=str(case/'diagnostics'))
        event('queued',variant=current,command=command)
        with (case/'console.log').open('w',buffering=1) as log:
            proc=subprocess.Popen(guarded_command,cwd=SOURCE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            status('waiting_for_gpus',launcher_pid=proc.pid)
            for line in proc.stdout:
                log.write(line)
                try:
                    row=json.loads(line)
                except (ValueError,TypeError):
                    continue
                if not isinstance(row,dict):
                    continue
                if row.get('event')=='gpu_wait':
                    status('waiting_for_gpus',launcher_pid=proc.pid,gpu_check=row)
                elif row.get('event')=='gpu_ready':
                    status('starting',launcher_pid=proc.pid,gpu_check=row)
                    event('start',variant=current,command=command)
                elif row.get('event')=='train':
                    last_step=row['step']
                    status('running',launcher_pid=proc.pid,train_loss=row['train_loss'],step_seconds=row['step_seconds'])
                    if last_step==job['start_step']+1 or last_step%250==0:
                        event('progress',variant=current,metric=row)
                elif row.get('event')=='validation':
                    event('validation',variant=current,metric=row)
            code=proc.wait()
        (case/'exit_code').write_text(str(code)+'\n')
        if code or last_step!=19523:
            raise RuntimeError(f'{current}: exit={code}, last_step={last_step}')
        run=case/'run'
        expected=['model_019523.pt','meta_019523.json']+[f'{kind}_019523_rank{rank}.pt' for kind in ['optim','training'] for rank in range(8)]
        missing=[name for name in expected if not (run/name).is_file() or (run/name).stat().st_size==0]
        if missing:
            raise RuntimeError(f'{current}: missing checkpoint files {missing}')
        completed.append(current)
        event('complete',variant=current,final_step=last_step)
    current=None
    final={}
    for job in JOBS:
        records=[json.loads(line) for line in (ROOT/job['name']/'run/metrics.jsonl').read_text().splitlines()]
        final[job['name']]=[r for r in records if r['event']=='validation'][-1]
    atomic(ROOT/'final_comparison.json',final)
    status('complete')
    event('queue_complete',variants=completed)

if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        status('failed',error=str(exc))
        event('failed',variant=current,error=str(exc))
        traceback.print_exc()
        raise
