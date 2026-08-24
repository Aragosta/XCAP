"""Run every variant, two at a time (4 cores, 2 torch threads each)."""
import argparse, os, subprocess, sys, time
from pathlib import Path

from lab.variants import VARIANTS

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="results")
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()

    names = a.only or list(VARIANTS)
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT/'vendor'}", OMP_NUM_THREADS=str(a.threads))
    logs = ROOT / a.out / "logs"; logs.mkdir(parents=True, exist_ok=True)
    queue, running = list(names), []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < a.jobs:
            name = queue.pop(0)
            cmd = [sys.executable, "-m", "lab.train", name, "--steps", str(a.steps),
                   "--batch-size", str(a.batch_size), "--seq-len", str(a.seq_len),
                   "--lr", str(a.lr), "--seed", str(a.seed), "--threads", str(a.threads),
                   "--out", str(ROOT / a.out)]
            log = open(logs / f"{name}.log", "w")
            print(f"launch {name}", flush=True)
            running.append((name, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(5)
        for item in list(running):
            name, proc, log = item
            if proc.poll() is not None:
                log.close(); running.remove(item)
                print(f"done   {name} rc={proc.returncode} ({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"all finished in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
