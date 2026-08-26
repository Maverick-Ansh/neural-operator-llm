"""Emit a compact numeric digest of every result, small enough to move as text.

The full reports carry the learned kernel sampled on a 16,384-point grid, which
is megabytes of JSON and cannot travel through a text channel. It also does not
need to: the kernel is band-limited to `n_modes` Fourier components, so by
Nyquist any grid finer than 2*n_modes points reproduces it exactly. A few
hundred display points are a lossless rendering of the same function.

Everything else (bits/byte, copy gain, timings, training curves) is a few
hundred numbers to begin with.
"""

import argparse
import glob
import json
import os


def subsample(row, n):
    """Uniformly pick n points from a list, always including both endpoints."""
    L = len(row)
    if L <= n:
        return [round(v, 5) for v in row]
    idx = [round(i * (L - 1) / (n - 1)) for i in range(n)]
    return [round(row[i], 5) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/digest.json")
    ap.add_argument("--fine-points", type=int, default=384)
    ap.add_argument("--coarse-points", type=int, default=48)
    ap.add_argument("--sample-chars", type=int, default=700)
    args = ap.parse_args()

    digest = {"models": {}, "training": {}}

    for p in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name in ("summary", "digest"):
            continue
        r = json.load(open(p, encoding="utf-8"))
        m = {
            "params": r.get("params"),
            "cfg": {k: r.get("cfg", {}).get(k) for k in
                    ("variant", "op_mode", "d_model", "n_layers", "window",
                     "n_modes", "train_len")},
            "trained_op_mode": r.get("trained_op_mode"),
            "eval_op_mode": r.get("eval_op_mode"),
            "op_mode_overridden": r.get("op_mode_overridden"),
            "train_step": r.get("train_step"),
            "bpb_vs_length": r.get("bpb_vs_length", []),
            "copy_probe": r.get("copy_probe", []),
            "cost_vs_length": r.get("cost_vs_length", []),
        }
        if r.get("sample"):
            m["sample"] = r["sample"][:args.sample_chars]

        # Kernel curves: keep both grids, but only as many points as are needed
        # to draw them. The fine grid becomes a line, the coarse grid markers;
        # markers landing on the line is the resolution-invariance claim.
        ks = r.get("kernels") or []
        if ks:
            k = ks[0]
            grids = sorted(k["curves"].keys(), key=int)
            lo, hi = grids[0], grids[-1]
            m["kernel"] = {
                "layer": k.get("layer"),
                "n_operator_layers": k.get("n_operator_layers"),
                "grid_coarse": int(lo), "grid_fine": int(hi),
                "coarse": [subsample(c, args.coarse_points) for c in k["curves"][lo]],
                "fine": [subsample(c, args.fine_points) for c in k["curves"][hi]],
                "decay_first_layer": (list(k.get("decay_by_layer", {}).values()) or [[]])[0][:16],
            }
        digest["models"][name] = m

    for d in sorted(glob.glob(os.path.join(args.runs, "*"))):
        f = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(f):
            continue
        recs = [json.loads(l) for l in open(f) if l.strip()]
        name = os.path.basename(d)
        digest["training"][name] = {
            "valid": [[r["step"], r["val_bpb"]] for r in recs if r.get("event") == "valid"],
            "final": next((r for r in recs if r.get("event") == "done"), None),
            "tok_per_s": next((r["tok_per_s"] for r in reversed(recs)
                               if r.get("event") == "train"), None),
            "params": next((r.get("params") for r in recs if r.get("event") == "start"), None),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(digest, f, separators=(",", ":"))
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")
    print("models:", list(digest["models"]))
    print("training runs:", list(digest["training"]))


if __name__ == "__main__":
    main()
