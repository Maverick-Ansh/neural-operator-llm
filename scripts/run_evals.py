"""Run the evaluation sweep for every trained model, two GPUs at a time.

Each GPU gets its own serial queue. A job waits for its checkpoint to appear
(`gate`) before starting, so this can be launched while training is still
running and will simply pick each model up as it lands.

Detached and self-deadlining, like `orchestrate.py`: it must never block the
notebook kernel, and it must not spin forever if a training job dies.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_worker(root, gpu, jobs, deadline):
    for job in jobs:
        gate = os.path.join(root, job["gate"])
        while not os.path.exists(gate):
            if time.time() > deadline:
                log(f"gpu{gpu}: deadline waiting for {job['gate']}, skipping {job['name']}")
                break
            time.sleep(20)
        if not os.path.exists(gate):
            continue

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        cmd = f'{sys.executable} -m nolm.evaluate {job["args"]}'
        log(f"gpu{gpu}: START {job['name']}")
        t0 = time.time()
        with open(os.path.join(root, "results", f"{job['name']}.log"), "w") as lf:
            rc = subprocess.run(cmd, shell=True, cwd=root, env=env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        log(f"gpu{gpu}: END   {job['name']} rc={rc} in {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/neural-operator-llm")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--deadline-min", type=float, default=300.0)
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.root, "results"), exist_ok=True)
    jobs = json.load(open(args.jobs))
    deadline = time.time() + args.deadline_min * 60

    by_gpu = {}
    for j in jobs:
        by_gpu.setdefault(j["gpu"], []).append(j)

    threads = [threading.Thread(target=gpu_worker, args=(args.root, g, js, deadline))
               for g, js in by_gpu.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if args.make_plots:
        log("generating figures")
        r = subprocess.run(f"{sys.executable} -m nolm.plots", shell=True, cwd=args.root,
                           capture_output=True, text=True)
        print(r.stdout[-4000:], flush=True)
        print(r.stderr[-2000:], flush=True)
    log("all evaluations complete")


if __name__ == "__main__":
    main()
