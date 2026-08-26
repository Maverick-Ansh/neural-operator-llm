"""Start the next training run on a GPU as soon as the current one frees it.

Two GPUs, four runs, and a hard wall-clock budget: leaving a card idle while
waiting for a human to notice the first run finished wastes a meaningful
fraction of the day. This polls for each round-1 run's `final.pt` and launches
its round-2 successor on the same card.

Runs detached in the background. It never blocks the notebook kernel, and it
carries its own overall deadline so a stuck job cannot leave it spinning.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def launch(root, job):
    out = os.path.join(root, "runs", job["name"])
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "stdout.log"), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job["gpu"]), PYTHONUNBUFFERED="1")
    cmd = f'{sys.executable} -m nolm.train {job["args"]} --out runs/{job["name"]}'
    p = subprocess.Popen(cmd, shell=True, cwd=root, env=env,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[{time.strftime('%H:%M:%S')}] launched {job['name']} on gpu{job['gpu']} pid={p.pid}",
          flush=True)
    return p.pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/neural-operator-llm")
    ap.add_argument("--jobs", required=True, help="JSON file: list of jobs")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--deadline-min", type=float, default=300.0)
    args = ap.parse_args()

    jobs = json.load(open(args.jobs))
    pending = list(jobs)
    t0 = time.time()
    print(f"orchestrator up, {len(pending)} job(s) queued", flush=True)

    while pending and (time.time() - t0) < args.deadline_min * 60:
        for job in list(pending):
            gate = os.path.join(args.root, job["wait_for"])
            if os.path.exists(gate):
                # The predecessor writes final.pt last; give it a moment to
                # release the GPU allocator before the successor grabs memory.
                time.sleep(10)
                launch(args.root, job)
                pending.remove(job)
        time.sleep(args.poll)

    if pending:
        print("deadline reached with jobs still queued:",
              [j["name"] for j in pending], flush=True)
    print("orchestrator done", flush=True)


if __name__ == "__main__":
    main()
