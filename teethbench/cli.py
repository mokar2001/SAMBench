"""User-facing commands; model imports happen only in worker processes."""
import argparse
from datetime import datetime,timezone
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from .config import FAMILIES,SCHEMA,build_plan,read,write,verify_plan

ROOT=Path(__file__).resolve().parents[1]


def positive(text):
    value=int(text)
    if value<1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def nonnegative(text):
    value=int(text)
    if value<0:
        raise argparse.ArgumentTypeError('must be zero or a positive integer')
    return value


def probability(text):
    value=float(text)
    if not 0<=value<=1:
        raise argparse.ArgumentTypeError('must be between 0 and 1')
    return value


def parser():
    p=argparse.ArgumentParser(description='Benchmark dental tooth masks with automatic SAM masks, box prompts, or both.',
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--version',action='version',version=SCHEMA)
    sub=p.add_subparsers(dest='command',required=True)
    models=sub.add_parser('models',help='List valid model IDs, families and sizes')
    models.add_argument('--root',type=Path,default=ROOT)
    doctor=sub.add_parser('doctor',help='Check Python, CUDA, dataset and checkpoints')
    doctor.add_argument('--root',type=Path,default=ROOT)
    run=sub.add_parser('run',help='Run or resume an immutable experiment',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run.add_argument('--root',type=Path,default=ROOT)
    run.add_argument('--mode',choices=['auto','bbox','both'],required=True,help='auto: no GT prompts; bbox: one GT box per tooth; both: two separate leaderboards')
    selection=run.add_mutually_exclusive_group(required=True)
    selection.add_argument('--models',nargs='+',metavar='ID',help='Explicit IDs, all (including gated SAM 3), or ungated (SAM 1/2/2.1)')
    selection.add_argument('--family',choices=list(FAMILIES),help='Select one family, together with --size')
    run.add_argument('--size',choices=['tiny','small','base','base_plus','large','huge','default'],help='Model size; optional for SAM 3, which has one checkpoint')
    run.add_argument('--samples','--limit',type=nonnegative,default=0,help='Exact number of sampled images, not teeth; 0 uses the full selected split')
    run.add_argument('--seed',type=nonnegative,default=20260909,help='Deterministic image sampling and box jitter seed')
    run.add_argument('--split',choices=['dev','test','all'],default='test')
    run.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run.add_argument('--threads',type=positive,default=2,help='PyTorch CPU threads per sequential job')
    run.add_argument('--output',type=Path,help='Run directory; reusing the same arguments and directory resumes completed images')
    run.add_argument('--background',action='store_true',help='Detach; save process ID and suite.log in the output directory')
    run.add_argument('--dry-run',action='store_true',help='Validate model/cohort selection and display a plan without inference or output writes')
    run.add_argument('--bootstrap',type=positive,default=5000,help='Image-group bootstrap replicates')
    bbox=run.add_argument_group('box-prompt options')
    bbox.add_argument('--box-condition',choices=['exact','pad5','pad10','jitter5','jitter10'],default='exact')
    auto=run.add_argument_group('automatic mask generation and matching')
    auto.add_argument('--points-per-side',type=positive,default=32,help='Uniform grid width: 32 means 1,024 points per full image')
    auto.add_argument('--points-per-batch',type=positive,default=8,help='Point batch size; higher values use more memory')
    auto.add_argument('--pred-iou-thresh',type=probability,default=0.88,help='Model-predicted quality filter, independent of GT')
    auto.add_argument('--stability-thresh',type=probability,default=0.95)
    auto.add_argument('--nms-thresh',type=probability,default=0.7)
    auto.add_argument('--crop-layers',type=nonnegative,default=0,help='Additional automatic image-crop layers')
    auto.add_argument('--match-iou',type=probability,default=0.5,help='Minimum IoU for one-to-one automatic evaluation matches; must be > 0')
    for name,help_text in [('status','Show completed images, failures and live process state'),
                           ('report','Rebuild separate leaderboards and CSV/JSON reports'),
                           ('resume','Resume an existing run using its saved arguments')]:
        command=sub.add_parser(name,help=help_text)
        command.add_argument('--output',type=Path,required=True)
        if name=='resume':
            command.add_argument('--background',action='store_true')
    return p


def show_models(root):
    registry=read(root/'models.json')
    print(f'{"MODEL ID":20} {"FAMILY":9} {"SIZE":11} CHECKPOINT')
    for family,sizes in FAMILIES.items():
        for size,name in sizes.items():
            ready=(root/'checkpoints'/registry[name]['checkpoint']).is_file()
            print(f'{name:20} {family:9} {size:11} {"ready" if ready else "missing"}')


def status(output):
    plan=read(output/'plan.json')
    if plan.get('schema')!=SCHEMA:
        raise ValueError('Use the legacy status.py for legacy suites.')
    with (output/'execution.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            running=False
        except BlockingIOError:
            running=True
    print(f'{"Running" if running else "Not running"}: {output}')
    print(f'{plan["image_count"]} images / {plan["image_group_count"]} groups / {plan["tooth_count"]} teeth; {plan["device"]}')
    for mode in plan['modes']:
        for model in plan['models']:
            job=output/mode/model
            count=sum((job/f'images/{im["id"]:04d}.json').exists() for im in plan['images'])
            failed=sum(1 for _ in (job/'errors').glob('*.json'))
            state='complete' if count==plan['image_count'] else read(job/'status.json').get('state','pending') if (job/'status.json').exists() else 'queued'
            if not running and state in ['loading','warmup','running']:
                state='interrupted; resumable'
            print(f'{mode:5} {model:20} {count:4}/{plan["image_count"]} images  {failed} errors  {state}')


def show_plan(plan,output):
    print('Models: '+', '.join(plan['models']))
    print('Modes: '+', '.join(plan['modes']))
    print(f'Cohort: {plan["image_count"]} images, {plan["image_group_count"]} groups, {plan["tooth_count"]} teeth ({plan["split"]}, seed {plan["seed"]})')
    print(f'Compute: {plan["device"]}, float32, {plan["threads"]} CPU threads, sequential jobs')
    if 'auto' in plan['modes']:
        print(f'Automatic grid: {plan["auto"]["points_per_side"]} x {plan["auto"]["points_per_side"]}; matching IoU >= {plan["matching_iou"]}')
    print('Output: '+str(output))
    print('Image IDs: '+', '.join(str(im['id']) for im in plan['images'][:20])+(' ...' if plan['image_count']>20 else ''))


def prepare_output(output,plan):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (output/'plan.json').exists():
            if read(output/'plan.json')!=plan:
                raise ValueError('Existing output uses different arguments/data/code. Use resume to keep its saved settings, or choose a new --output.')
            return
        if any(output.iterdir()) and any(p.name!='prepare.lock' for p in output.iterdir()):
            raise ValueError('Output directory is not empty and has no v2 plan. Choose a new directory.')
        root=Path(plan['root'])
        snapshot=output/'source_snapshot'
        for name in list(plan['source_sha256'])+['requirements.lock.txt','requirements-sam3.txt','source_revisions.json']:
            src=root/name
            if src.exists():
                dest=snapshot/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(src,dest)
        write(output/'plan.json',plan)


def execute(output,lock_fd=None):
    from .report import report
    plan=read(output/'plan.json')
    verify_plan(plan)
    lock=os.fdopen(lock_fd,'a') if lock_fd is not None else (output/'execution.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('This run is already running. Use status instead of launching a duplicate.')
    active=None
    failures=[]
    def interrupt(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try:
        for mode in plan['modes']:
            for model in plan['models']:
                verify_plan(plan)
                job=output/mode/model
                job.mkdir(parents=True,exist_ok=True)
                write(output/'state.json',{'state':'running','active_mode':mode,'active_model':model})
                command=[sys.executable,'-u',str(Path(plan['root'])/'benchmark.py'),'_worker',str(output),model,mode]
                print(f'\nStarting {model} / {mode}',flush=True)
                with (job/'inference.log').open('a') as log:
                    active=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                             text=True,start_new_session=True,bufsize=1)
                    for line in active.stdout:
                        log.write(line)
                        log.flush()
                        print(line,end='',flush=True)
                    code=active.wait()
                    active=None
                if code:
                    failures.append(f'{mode}/{model}')
                    saved=read(job/'status.json') if (job/'status.json').exists() else {}
                    write(job/'status.json',{**saved,'state':'failed','exit_code':code})
                    write(job/'failure.json',{'exit_code':code,'log':str(job/'inference.log')})
                elif (job/'failure.json').exists():
                    (job/'failure.json').unlink()
                report(output)
        write(output/'state.json',{'state':'failed' if failures else 'complete','failed_jobs':failures})
        print(f'\nReport: {output / "REPORT.md"}',flush=True)
        return 1 if failures else 0
    except KeyboardInterrupt:
        if active and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:
                active.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid,signal.SIGKILL)
                active.wait()
        write(output/'state.json',{'state':'interrupted','message':'Completed images are saved; resume is safe.'})
        print('\nInterrupted. Completed images are saved; use resume --output '+str(output),flush=True)
        return 130
    except Exception as exc:
        write(output/'state.json',{'state':'failed','error':str(exc)})
        raise
    finally:
        lock.close()


def launch(output,background):
    verify_plan(read(output/'plan.json'))
    if not background:
        return execute(output)
    lock=(output/'execution.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('This run is already running; use status.')
    with (output/'suite.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'benchmark.py'),'_execute',str(output),str(lock.fileno())],
                              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              start_new_session=True,pass_fds=(lock.fileno(),))
    lock.close()  # Child retains the same flock until execution ends.
    write(output/'launch.json',{'pid':proc.pid,'started_at_utc':datetime.now(timezone.utc).isoformat()})
    print(f'Background PID {proc.pid}; log: {output / "suite.log"}')
    return 0


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0]=='_worker':
        from .worker import run_job
        run_job(Path(argv[1]),argv[2],argv[3])
        return
    if argv and argv[0]=='_execute':
        raise SystemExit(execute(Path(argv[1]),int(argv[2])))
    p=parser()
    args=p.parse_args(argv)
    try:
        if args.command=='models':
            show_models(args.root)
        elif args.command=='doctor':
            import torch
            from segment_anything import SamAutomaticMaskGenerator
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            print(f'Python {sys.version.split()[0]}; torch {torch.__version__}; CUDA available: {torch.cuda.is_available()}')
            manifest=read(args.root/'prepared/manifest.json')
            print(f'Dataset: {len(manifest["images"])} retained images; {sum(im["evaluable"] for im in manifest["images"])} evaluable.')
            show_models(args.root)
        elif args.command=='run':
            if args.match_iou==0:
                raise ValueError('--match-iou must be greater than zero; disjoint masks must never match.')
            plan=build_plan(args)
            output=(args.output or args.root/'results'/f'cli_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}').resolve()
            show_plan(plan,output)
            if not args.dry_run:
                if 'sam3' in plan['models']:
                    from importlib.metadata import version,PackageNotFoundError
                    try:
                        installed=version('transformers')
                    except PackageNotFoundError:
                        installed=None
                    if installed!=plan['model_specs']['sam3']['transformers_version']:
                        raise ValueError('Install SAM 3 dependencies: .venv/bin/python -m pip install -r requirements-sam3.txt')
                if args.device=='cuda':
                    import torch
                    if not torch.cuda.is_available():
                        raise ValueError('CUDA requested but this host has no CUDA GPU. Use --device cpu.')
                prepare_output(output,plan)
                raise SystemExit(launch(output,args.background))
        elif args.command=='status':
            status(args.output.resolve())
        elif args.command=='report':
            from .report import report
            report(args.output.resolve())
            print(args.output.resolve()/'REPORT.md')
        elif args.command=='resume':
            raise SystemExit(launch(args.output.resolve(),args.background))
    except (ValueError,FileNotFoundError,KeyError) as exc:
        p.exit(2,f'Error: {exc}\n')


if __name__=='__main__':
    main()
