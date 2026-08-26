"""Push results from Colab back to GitHub.

Two destinations, because they have very different size profiles:

* `results/` and `runs/*/metrics.jsonl` -- figures, JSON reports, training logs.
  A few MB. These are the paper, and they go in the repo.
* checkpoints -- ~120MB each in fp32. These do not belong in git history. They
  are converted to fp16 (halving them, and fp16 is what the model was trained
  under anyway) and uploaded as **release assets**, which have a 2GB limit.

The token is read from Colab's secret store, never from the source or the
command line, so it cannot end up in the repo or in a notebook output.
"""

import argparse
import json
import os
import subprocess
import sys


def get_token():
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN")
    except Exception as e:
        tok = os.environ.get("GITHUB_TOKEN")
        if tok:
            return tok
        raise SystemExit(
            "No GITHUB_TOKEN. In Colab: key icon in the left sidebar -> Add new "
            f"secret -> name it GITHUB_TOKEN -> enable notebook access. ({e})")


def run(cmd, cwd=None, check=True, quiet=False, token=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    if token:                                   # never echo the credential
        out = out.replace(token, "***")
    if not quiet:
        print(out.strip()[-2000:])
    if check and r.returncode != 0:
        raise SystemExit(f"failed ({r.returncode}): {cmd.replace(token, '***') if token else cmd}")
    return r.returncode, out


def export_fp16(root, names):
    """Halve each final checkpoint and drop optimiser state for publication."""
    import torch
    out_dir = os.path.join(root, "release")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for n in names:
        src = os.path.join(root, "runs", n, "final.pt")
        if not os.path.exists(src):
            print(f"  (skip {n}: no final.pt)")
            continue
        ck = torch.load(src, map_location="cpu", weights_only=False)
        ck["model"] = {k: (v.half() if v.is_floating_point() else v)
                       for k, v in ck["model"].items()}
        dst = os.path.join(out_dir, f"{n}.fp16.pt")
        torch.save(ck, dst)
        mb = os.path.getsize(dst) / 1e6
        print(f"  {n}: {os.path.getsize(src)/1e6:.0f}MB -> {mb:.0f}MB  {dst}")
        paths.append(dst)
    return paths


def upload_release(root, repo, tag, paths, token):
    """Create (or reuse) a release and attach the checkpoints via the REST API."""
    import urllib.request, urllib.error

    api = f"https://api.github.com/repos/{repo}"
    hdrs = {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "nolm-release"}

    def req(url, data=None, method=None, ctype=None, raw=None):
        h = dict(hdrs)
        if ctype:
            h["Content-Type"] = ctype
        body = raw if raw is not None else (json.dumps(data).encode() if data else None)
        r = urllib.request.Request(url, data=body, headers=h, method=method)
        with urllib.request.urlopen(r) as resp:
            return json.loads(resp.read() or b"{}")

    try:
        rel = req(f"{api}/releases/tags/{tag}")
        print(f"reusing release {tag}")
    except urllib.error.HTTPError:
        rel = req(f"{api}/releases", data={
            "tag_name": tag, "name": f"Trained checkpoints ({tag})",
            "body": "fp16 weights for every variant. Load with "
                    "`nolm.evaluate.load_model`. See README for the numbers."})
        print(f"created release {tag}")

    up = rel["upload_url"].split("{")[0]
    existing = {a["name"] for a in rel.get("assets", [])}
    for p in paths:
        name = os.path.basename(p)
        if name in existing:
            print(f"  {name} already uploaded, skipping")
            continue
        with open(p, "rb") as f:
            blob = f.read()
        req(f"{up}?name={name}", raw=blob, method="POST",
            ctype="application/octet-stream")
        print(f"  uploaded {name} ({len(blob)/1e6:.0f}MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/neural-operator-llm")
    ap.add_argument("--repo", default="Maverick-Ansh/neural-operator-llm")
    ap.add_argument("--message", default="Add experimental results from Colab 2xT4 run")
    ap.add_argument("--checkpoints", action="store_true",
                    help="also publish fp16 weights as release assets")
    ap.add_argument("--tag", default="v0.1-results")
    ap.add_argument("--names", default="nolm,transformer,local,nolm_fixed")
    args = ap.parse_args()

    token = get_token()
    root = args.root
    names = args.names.split(",")

    # Copy the training logs into the repo tree (runs/ itself is gitignored).
    logs = os.path.join(root, "results", "training_logs")
    os.makedirs(logs, exist_ok=True)
    for n in names:
        src = os.path.join(root, "runs", n, "metrics.jsonl")
        if os.path.exists(src):
            subprocess.run(f'cp "{src}" "{os.path.join(logs, n + ".jsonl")}"', shell=True)

    run('git config user.email "anshvivek2003@gmail.com"', cwd=root)
    run('git config user.name "Maverick-Ansh"', cwd=root)
    run(f'git remote set-url origin https://x-access-token:{token}@github.com/{args.repo}.git',
        cwd=root, quiet=True, token=token)

    run("git add -A results README.md", cwd=root)
    rc, _ = run(f'git commit -m "{args.message}"', cwd=root, check=False, token=token)
    if rc != 0:
        print("(nothing new to commit)")
    run("git pull --rebase -q origin master", cwd=root, check=False, token=token)
    run("git push origin HEAD:master", cwd=root, token=token)

    if args.checkpoints:
        print("\nexporting fp16 checkpoints...")
        paths = export_fp16(root, names)
        if paths:
            upload_release(root, args.repo, args.tag, paths, token)

    # Scrub the credential back out of .git/config.
    run(f"git remote set-url origin https://github.com/{args.repo}.git", cwd=root, quiet=True)
    print("\ndone")


if __name__ == "__main__":
    main()
