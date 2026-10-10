"""Run train.py jobs from a queue file with fixed GPU parallelism; progress in logs/queue.log.

python run_queue.py jobs_phase1.txt --parallel 3 --gpu 0
Each line of the jobs file: <tag> <train.py args...>. Jobs whose tag already appears in results/runs.jsonl are skipped.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.environ.get("PF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def done_tags():
    p = f"{ROOT}/results/runs.jsonl"
    if not os.path.exists(p):
        return set()
    return {json.loads(l)["tag"] for l in open(p) if l.strip()}


def qlog(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f"{ROOT}/logs/queue.log", "a") as fh:
        fh.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    jobs = [l.split(None, 1) for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    finished = done_tags()
    todo = [(t, args) for t, args in jobs if t not in finished]
    qlog(f"queue {a.jobs}: {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run, parallel={a.parallel}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    retried = set()
    running = {}
    t0 = time.time()
    n_done = 0
    while todo or running:
        while todo and len(running) < a.parallel:
            tag, args = todo.pop(0)
            out = open(f"{ROOT}/logs/train/{tag}.stdout", "w")
            cmd = [sys.executable, "train.py", "--tag", tag] + shlex.split(args)
            p = subprocess.Popen(cmd, cwd=f"{ROOT}/src", env=env, stdout=out, stderr=subprocess.STDOUT)
            running[tag] = (p, time.time())
            qlog(f"START {tag} pid={p.pid}: {' '.join(cmd[2:])}")
        time.sleep(20)
        for tag in list(running):
            p, ts = running[tag]
            if p.poll() is not None:
                n_done += 1
                status = "OK" if p.returncode == 0 else f"FAILED rc={p.returncode} (see logs/train/{tag}.stdout)"
                qlog(f"END   {tag} {status} {(time.time() - ts) / 60:.1f} min | {n_done} done, "
                     f"{len(running) - 1} running, {len(todo)} queued, elapsed {(time.time() - t0) / 3600:.2f} h")
                del running[tag]
                if p.returncode != 0 and tag not in retried:
                    retried.add(tag)
                    args = next(x for t, x in jobs if t == tag)
                    todo.append((tag, args))
                    qlog(f"RETRY {tag} queued again at the end")
    qlog("queue finished")


if __name__ == "__main__":
    main()
