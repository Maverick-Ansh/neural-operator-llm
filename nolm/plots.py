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
        if n in ("summary", "digest"):
            continue
        out[n] = json.load(open(p, encoding="utf-8"))
    return out


def load_slim(path):
    """Load `results/slim.json`: the column-oriented form of the digest.

    Written when the full reports (which carry per-window losses and a 16,384-
    point kernel) are too large to move off the training machine. Column arrays
    instead of key/value objects, which is roughly a 4x saving on JSON text.
    """
    d = json.load(open(path, encoding="utf-8"))
    reports = {}
    for name, m in d["models"].items():
        r = {"params": m["params"], "cfg": {"variant": m["variant"], "op_mode": m["op_mode"]},
             "eval_op_mode": m.get("eval_op_mode"),
             "op_mode_overridden": m.get("overridden"),
             "bpb_vs_length": [
                 {"length": a, "bpb_all": b, "bpb_tail": c, "bpb_tail_stderr": e,
                  "windows": f, "oom": g} for a, b, c, e, f, g in m["L"]],
             "copy_probe": [
                 {"separation": a, "copy_gain_bits": b, "copy_gain_stderr": c,
                  "bpb_first_copy": e, "bpb_second_copy": f, "trials": g}
                 for a, b, c, e, f, g in m["C"]],
             "cost_vs_length": [
                 {"length": a, "sec_per_forward": b, "peak_mem_GB": c, "oom": e}
                 for a, b, c, e in m["K"]]}
        if m.get("sample"):
            r["sample"] = m["sample"]
        if m.get("kernel"):
            k = m["kernel"]
            r["kernels"] = [{"layer": k["layer"], "n_operator_layers": k.get("nop"),
                             "grid_coarse": k["gc"], "grid_fine": k["gf"],
                             "coarse": k["coarse"], "fine": k["fine"]}]
        reports[name] = r
    training = {n: v.get("valid", []) for n, v in d.get("training", {}).items()}
    return reports, training


def load_digest(path):
    """Load the compact digest produced by `scripts/digest.py`.

    The digest is what travels off the training machine when the full reports
    are too large to move, so every figure must be reproducible from it alone.
    Returns (reports, training_curves) in the same shapes the figure functions
    already expect.
    """
    d = json.load(open(path, encoding="utf-8"))
    reports = {}
    for name, m in d["models"].items():
        r = dict(m)
        if "kernel" in m:                       # re-wrap into the reports layout
            r["kernels"] = [m["kernel"]]
        reports[name] = r
    training = {n: v.get("valid", []) for n, v in d.get("training", {}).items()}
    return reports, training


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

    # The transformer's collapse compresses every other model into a single flat
    # band, hiding the comparison that actually decides the experiment. Second
    # panel, same data, y-axis scaled to the models that survive.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    drawn = False
    for n, r in reports.items():
        rows = [d for d in r.get("bpb_vs_length", [])
                if not d.get("oom") and "bpb_tail" in d]
        if not rows or max(d["bpb_tail"] for d in rows) > 2.0:
            continue                       # leave the collapsed model out
        ax.errorbar([d["length"] for d in rows], [d["bpb_tail"] for d in rows],
                    yerr=[d.get("bpb_tail_stderr", 0) for d in rows],
                    capsize=2, **style(n))
        drawn = True
    if drawn:
        _finish(ax, os.path.join(out, "fig_bpb_tail_zoom.png"),
                "Length generalisation, models that survive it\n"
                "(the transformer is off-scale above)",
                "evaluation context length (bytes)",
                "bits / byte (final 512 bytes of window)", train_len)


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
    if not r or not r.get("kernels"):
        return
    pick = r["kernels"][len(r["kernels"]) // 2]

    # Accept either the full report layout ({"curves": {grid: rows}}) or the
    # digest layout ({"coarse": rows, "fine": rows}) -- figures must be
    # reproducible from the digest alone.
    if "curves" in pick:
        grids = sorted(pick["curves"].keys(), key=int)
        lo, hi = grids[0], grids[-1]
        coarse, fine = pick["curves"][lo], pick["curves"][hi]
        n_lo, n_hi = int(lo), int(hi)
    else:
        coarse, fine = pick["coarse"], pick["fine"]
        n_lo, n_hi = pick["grid_coarse"], pick["grid_fine"]

    n_show = min(3, len(fine))
    fig, axes = plt.subplots(1, n_show, figsize=(4.0 * n_show, 3.4), squeeze=False)

    # Only the fine grid is drawn. Overlaying the coarse grid as markers would
    # LOOK like a direct demonstration of resolution invariance, but it is not:
    # kappa is band-limited to n_modes=64 components, so Nyquist requires >=128
    # display points and a 32-point marker set is aliased. The markers would sit
    # off the line for a reason that has nothing to do with the property being
    # claimed. The invariance is asserted numerically instead, to 1e-9, by
    # tests/test_core.py::test_kernel_is_resolution_invariant.
    for c in range(n_show):
        ax = axes[0][c]
        y = fine[c]
        t = [i / (len(y) - 1) for i in range(len(y))]
        ax.plot(t, y, lw=1.5, color="#1f77b4")
        ax.axhline(0, c="k", lw=0.8, alpha=0.4)
        ax.set_xlabel("normalised lag  t = m/N"); ax.set_title(f"channel {c}")
        ax.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel(r"$\kappa(t)$")
    fig.suptitle(f"The learned operator kernel, sampled on an N = {n_hi:,} grid "
                 f"({pick.get('layer','')})\n"
                 r"resolution invariance is verified to $10^{-9}$ in the test suite, not by eye",
                 y=1.06, fontsize=10)
    plt.tight_layout()
    p = os.path.join(out, "fig_kernel.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", p)


def fig_training(runs_dir, out, curves=None):
    """Validation curves, from `runs/*/metrics.jsonl` or from a digest."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if curves is None:
        curves = {}
        for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
            f = os.path.join(d, "metrics.jsonl")
            if not os.path.exists(f):
                continue
            recs = [json.loads(l) for l in open(f) if l.strip()]
            curves[os.path.basename(d)] = [[r["step"], r["val_bpb"]]
                                           for r in recs if r.get("event") == "valid"]
    for n, va in sorted(curves.items()):
        if va:
            ax.plot([p[0] for p in va], [p[1] for p in va], **style(n))
    ax.set_xlabel("optimiser step (equal tokens for all runs)")
    ax.set_ylabel("validation bits / byte @ 2048")
    ax.set_title("Training"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(out, "fig_training.png")
    plt.savefig(p, dpi=150); plt.close(); print("wrote", p)


PAIRS = [
    ("nolm", "transformer", "does the operator beat full attention?"),
    ("nolm", "local", "does the global operator path do anything at all?"),
    ("nolm", "nolm_fixed", "is resolution-invariant addressing the active ingredient?"),
    ("nolm", "nolm_as_fixed", "same weights, re-addressed onto absolute lags"),
]


def paired_table(reports, out):
    """Paired per-window comparison of bpb_tail.

    Every model sees the identical, deterministic sequence of windows, so the
    per-window losses pair up one-to-one. Differencing them cancels the
    between-window variance -- which is much larger than the between-model
    variance -- and turns a visual comparison of two noisy curves into a
    quantitative one. Reported as mean difference, its standard error, and t.
    """
    import math as _m
    lines = ["\n### Paired comparison of bits/byte (same windows, differenced)\n",
             "Negative favours the first model. `t` is a paired t-statistic over "
             "the shared windows; |t| > ~2 is the usual bar for significance.\n"]
    any_rows = False
    for a, b, why in PAIRS:
        if a not in reports or b not in reports:
            continue
        ra = {d["length"]: d for d in reports[a].get("bpb_vs_length", [])}
        rb = {d["length"]: d for d in reports[b].get("bpb_vs_length", [])}
        lens = sorted(set(ra) & set(rb))
        rows = []
        for L in lens:
            da, db = ra[L], rb[L]
            va = da.get("tail_per_window") or []
            vb = db.get("tail_per_window") or []
            n = min(len(va), len(vb))
            if da.get("oom") or db.get("oom") or n < 2:
                rows.append((L, None, None, None))
                continue
            d = [va[i] - vb[i] for i in range(n)]
            mu = sum(d) / n
            var = sum((x - mu) ** 2 for x in d) / (n - 1)
            se = _m.sqrt(var / n)
            rows.append((L, mu, se, (mu / se if se > 0 else float("nan"))))
        if not any(r[1] is not None for r in rows):
            continue          # no per-window data available (e.g. slim digest)
        any_rows = True
        # ASCII only: this string is also printed to a Windows console, whose
        # cp1252 codec cannot encode U+2212.
        lines.append(f"\n**{style(a)['label']}  minus  {style(b)['label']}** - _{why}_\n")
        lines.append("| length | delta bpb | +/- stderr | t | n |")
        lines.append("|---|---|---|---|---|")
        for (L, mu, se, t), Lk in zip(rows, lens):
            if mu is None:
                lines.append(f"| {fmt_len(L)} | — | — | — | — |")
            else:
                n = min(len(ra[Lk].get("tail_per_window") or []),
                        len(rb[Lk].get("tail_per_window") or []))
                lines.append(f"| {fmt_len(L)} | {mu:+.4f} | {se:.4f} | {t:+.1f} | {n} |")
    txt = "\n".join(lines) if any_rows else ""
    if txt:
        open(os.path.join(out, "paired.md"), "w", encoding="utf-8").write(txt)
        print("wrote", os.path.join(out, "paired.md"))
    return txt


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

    txt = "\n".join(lines) + "\n" + paired_table(reports, out)
    p = os.path.join(out, "summary.md")
    open(p, "w", encoding="utf-8").write(txt)
    print("wrote", p)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--train-len", type=int, default=2048)
    ap.add_argument("--digest", default=None,
                    help="render from a digest.json instead of the full reports")
    ap.add_argument("--slim", default=None,
                    help="render from a slim.json (column-oriented digest)")
    args = ap.parse_args()

    if args.slim:
        reports, curves = load_slim(args.slim)
    elif args.digest:
        reports, curves = load_digest(args.digest)
    else:
        reports, curves = load_reports(args.results), None
    print("reports:", list(reports))
    fig_bpb(reports, args.results, args.train_len)
    fig_copy(reports, args.results, args.train_len)
    fig_cost(reports, args.results, args.train_len)
    fig_kernel(reports, args.results)
    fig_training(args.runs, args.results, curves)
    print("\n" + summary_table(reports, args.results, args.train_len))


if __name__ == "__main__":
    main()
