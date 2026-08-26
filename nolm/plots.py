"""Turn the JSON reports into figures and a markdown results table.

Figures are always written to disk and never shown. Rendering a figure inline
in this Colab setup pushes the encoded PNG into the tool-result channel, which
is both enormous and useless.
"""

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "nolm":         dict(color="#1f77b4", marker="o", label="NOLM (operator, stretch)"),
    "nolm_fixed":   dict(color="#17becf", marker="v", label="NOLM (operator, fixed)"),
    "nolm_as_fixed": dict(color="#9467bd", marker="D",
                          label="NOLM weights re-addressed as fixed"),
    "transformer":  dict(color="#d62728", marker="s", label="Transformer (RoPE, full attn)"),
    "local":        dict(color="#7f7f7f", marker="^", label="Local attention only"),
}


def style(name):
    return STYLE.get(name, dict(color="#333333", marker="x", label=name))


def fmt_len(L):
    """1024 -> '1K'; anything smaller keeps its exact value rather than '0K'."""
    return f"{L // 1024}K" if L >= 1024 and L % 1024 == 0 else str(L)


def load_reports(results_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        n = os.path.splitext(os.path.basename(p))[0]
        if n == "summary":
            continue
        out[n] = json.load(open(p))
    return out


def _finish(ax, path, title, xlabel, ylabel, train_len=None, logx=True):
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if train_len:
        ax.axvline(train_len, ls="--", c="k", alpha=0.45, lw=1)
        ax.annotate("trained here", xy=(train_len, ax.get_ylim()[1]),
                    xytext=(-4, -12), textcoords="offset points",
                    rotation=90, ha="right", va="top", fontsize=8, alpha=0.7)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print("wrote", path)


def fig_bpb(reports, out, train_len=2048):
    for key, ylab, fname, sub in [
        ("bpb_tail", "bits / byte (final 512 bytes of window)", "fig_bpb_tail.png",
         "positions that actually have ~L bytes of context"),
        ("bpb_all", "bits / byte (all positions)", "fig_bpb_all.png",
         "averaged over every position, including context-starved early ones"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for n, r in reports.items():
            rows = [d for d in r.get("bpb_vs_length", [])
                    if not d.get("oom") and key in d]
            if rows:
                xs = [d["length"] for d in rows]
                ys = [d[key] for d in rows]
                # Only bpb_tail carries a stderr; bpb_all averages millions of
                # positions and its error bar would be invisible anyway.
                es = [d.get("bpb_tail_stderr", 0) for d in rows] if key == "bpb_tail" else None
                ax.errorbar(xs, ys, yerr=es, capsize=2, **style(n))
            oom = [d["length"] for d in r.get("bpb_vs_length", []) if d.get("oom")]
            for L in oom:
                ax.scatter([L], [ax.get_ylim()[1]], marker="x", s=60,
                           color=style(n)["color"])
        _finish(ax, os.path.join(out, fname),
                f"Length generalisation\n({sub})",
                "evaluation context length (bytes)", ylab, train_len)


def fig_copy(reports, out, train_len=2048, window=128):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n, r in reports.items():
        pts = [(d["separation"], d["copy_gain_bits"]) for d in r.get("copy_probe", [])
               if not d.get("oom")]
        errs = [d.get("copy_gain_stderr", 0) for d in r.get("copy_probe", [])
                if not d.get("oom")]
        if pts:
            xs, ys = zip(*pts)
            ax.errorbar(xs, ys, yerr=errs, capsize=2, **style(n))
    ax.axhline(0, c="k", lw=1, alpha=0.5)
    ax.axvline(window, ls=":", c="gray", lw=1)
    ax.annotate("local window", xy=(window, 0), xytext=(3, 6),
                textcoords="offset points", fontsize=8, color="gray")
    _finish(ax, os.path.join(out, "fig_copy_probe.png"),
            "Long-range retrieval\n(bits saved on a repeated passage vs. its first occurrence)",
            "distance between the two copies (bytes)", "copy gain (bits / byte)", train_len)


def fig_cost(reports, out, train_len=2048):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for n, r in reports.items():
        rows = [d for d in r.get("cost_vs_length", []) if not d.get("oom")]
        if not rows:
            continue
        axes[0].plot([d["length"] for d in rows], [d["sec_per_forward"] for d in rows], **style(n))
        axes[1].plot([d["length"] for d in rows], [d["peak_mem_GB"] for d in rows], **style(n))
    for ax, ylab, ttl in [(axes[0], "seconds / forward pass", "Time"),
                          (axes[1], "peak GPU memory (GB)", "Memory")]:
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("context length (bytes)"); ax.set_ylabel(ylab)
        ax.set_title(ttl); ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
        ax.axvline(train_len, ls="--", c="k", alpha=0.45, lw=1)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_cost.png"), dpi=150); plt.close()
    print("wrote", os.path.join(out, "fig_cost.png"))


def fig_kernel(reports, out):
    """Overlay kappa(t) sampled on a coarse and a fine grid.

    If the spectral parameterisation means what it claims, the 2048-point and
    16384-point curves are the same function and lie exactly on top of one
    another -- this figure is the visual form of test_kernel_is_resolution_invariant.
    """
    r = reports.get("nolm")
    if not r or "kernels" not in r or not r["kernels"]:
        return
    ks = r["kernels"]
    pick = ks[len(ks) // 2]
    curves = pick["curves"]
    lo, hi = sorted(curves.keys(), key=int)[:2]

    n_show = min(3, len(curves[lo]))
    fig, axes = plt.subplots(1, n_show, figsize=(4.0 * n_show, 3.4), squeeze=False)
    for c in range(n_show):
        ax = axes[0][c]
        ylo, yhi = curves[lo][c], curves[hi][c]
        tlo = [i / len(ylo) for i in range(len(ylo))]
        thi = [i / len(yhi) for i in range(len(yhi))]
        ax.plot(thi, yhi, lw=2.2, alpha=0.45, color="#1f77b4", label=f"N = {hi}")
        ax.plot(tlo, ylo, lw=0.9, ls="--", color="#d62728", label=f"N = {lo}")
        ax.set_xlabel("normalised lag  t = m/N"); ax.set_title(f"channel {c}")
        ax.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel(r"$\kappa(t)$"); ax.legend(fontsize=8)
    fig.suptitle(f"One continuous kernel, two discretisations  ({pick['layer']})", y=1.02)
    plt.tight_layout()
    p = os.path.join(out, "fig_kernel.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", p)


def fig_training(runs_dir, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        f = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(f):
            continue
        n = os.path.basename(d)
        recs = [json.loads(l) for l in open(f) if l.strip()]
        va = [(r["step"], r["val_bpb"]) for r in recs if r.get("event") == "valid"]
        if va:
            ax.plot(*zip(*va), **style(n))
    ax.set_xlabel("optimiser step (equal tokens for all runs)")
    ax.set_ylabel("validation bits / byte @ 2048")
    ax.set_title("Training"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(out, "fig_training.png")
    plt.savefig(p, dpi=150); plt.close(); print("wrote", p)


def summary_table(reports, out, train_len=2048):
    lines = []
    lines.append("### Bits per byte vs. evaluation context length\n")
    lengths = sorted({d["length"] for r in reports.values()
                      for d in r.get("bpb_vs_length", [])})
    head = "| model | params | " + " | ".join(fmt_len(L) for L in lengths) + " |"
    lines += [head, "|" + "---|" * (len(lengths) + 2)]
    for n, r in reports.items():
        by = {d["length"]: d for d in r.get("bpb_vs_length", [])}
        cells = []
        for L in lengths:
            d = by.get(L)
            cells.append("OOM" if (d and d.get("oom")) else
                         (f"{d['bpb_tail']:.3f}" if d else "--"))
        lines.append(f"| {style(n)['label']} | {r['params']/1e6:.1f}M | " + " | ".join(cells) + " |")
    lines.append(f"\n_bits/byte over the final 512 bytes of each window; "
                 f"all models trained at {train_len}._\n")

    lines.append("\n### Long-range copy gain (bits/byte saved on the second copy)\n")
    seps = sorted({d["separation"] for r in reports.values()
                   for d in r.get("copy_probe", [])})
    if seps:
        lines += ["| model | " + " | ".join(str(s) for s in seps) + " |",
                  "|" + "---|" * (len(seps) + 1)]
        for n, r in reports.items():
            by = {d["separation"]: d for d in r.get("copy_probe", [])}
            cells = []
            for s in seps:
                d = by.get(s)
                cells.append("OOM" if (d and d.get("oom")) else
                             (f"{d['copy_gain_bits']:+.3f}" if d else "--"))
            lines.append(f"| {style(n)['label']} | " + " | ".join(cells) + " |")

    lines.append("\n### Cost at long context\n")
    lines += ["| model | " + " | ".join(f"{fmt_len(L)} s/fwd" for L in lengths) + " |",
              "|" + "---|" * (len(lengths) + 1)]
    for n, r in reports.items():
        by = {d["length"]: d for d in r.get("cost_vs_length", [])}
        cells = []
        for L in lengths:
            d = by.get(L)
            cells.append("OOM" if (d and d.get("oom")) else
                         (f"{d['sec_per_forward']:.2f}" if d else "--"))
        lines.append(f"| {style(n)['label']} | " + " | ".join(cells) + " |")

    txt = "\n".join(lines)
    p = os.path.join(out, "summary.md")
    open(p, "w").write(txt)
    print("wrote", p)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--train-len", type=int, default=2048)
    args = ap.parse_args()

    reports = load_reports(args.results)
    print("reports:", list(reports))
    fig_bpb(reports, args.results, args.train_len)
    fig_copy(reports, args.results, args.train_len)
    fig_cost(reports, args.results, args.train_len)
    fig_kernel(reports, args.results)
    fig_training(args.runs, args.results)
    print("\n" + summary_table(reports, args.results, args.train_len))


if __name__ == "__main__":
    main()
