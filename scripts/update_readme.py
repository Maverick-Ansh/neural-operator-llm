"""Inject the generated tables and figures into README.md.

Replaces everything between the RESULTS markers. The interpretation prose lives
in `results/discussion.md` and is written by hand -- deciding what a number
*means* is not something to generate from the number.
"""

import argparse
import glob
import json
import os

START = "<!-- RESULTS -->"
END = "<!-- /RESULTS -->"

FIGURES = [
    ("fig_bpb_tail.png",
     "Bits/byte on the final 512 bytes of the window as the evaluation context "
     "grows, all models trained at 2,048. The transformer loses almost everything "
     "one doubling past its training length; the operator models do not."),
    ("fig_bpb_tail_zoom.png",
     "The same data with the collapsed model removed. Note how far the *unpaired* "
     "error bars overlap — between-window variance in enwik8 dwarfs the difference "
     "between these models, which is why the paired table above is the real test."),
    ("fig_train_length.png",
     "Round 3: both architectures retrained at 8192 bytes on the same token "
     "budget. The transformer's failure point tracks its training length with no "
     "slack -- broken at 4K when trained at 2K, broken at 16K when trained at 8K. "
     "The operator has no cliff at either."),
    ("fig_copy_probe.png",
     "Bits saved on the second copy of a planted passage, against the distance "
     "between the copies. The transformer's +0.64 bits at 512 is the positive "
     "control: it proves the probe detects retrieval when retrieval is there. "
     "Every operator variant sits on zero at every distance."),
    ("fig_kernel.png",
     "The learned continuous kernel. Sharp structure concentrated near t = 0 — "
     "the operator taught itself to be mostly local."),
    ("fig_cost.png",
     "Time and peak memory per forward pass. The O(N log N) advantage is real in "
     "time (5.1x at 64K) and reversed in memory (2.2x worse), for implementation "
     "reasons discussed above."),
    ("fig_training.png",
     "Validation bits/byte during training; identical 104.9M-byte budget for every "
     "run. `nolm_fixed` is hidden underneath `nolm` — they are the same model."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    parts = []

    disc = os.path.join(args.results, "discussion.md")
    if os.path.exists(disc):
        parts.append(open(disc, encoding="utf-8").read().strip())
        parts.append("")

    for fn in ("summary.md", "paired.md"):
        p = os.path.join(args.results, fn)
        if os.path.exists(p):
            parts.append(open(p, encoding="utf-8").read().strip())
            parts.append("")

    figs = [f for f in FIGURES if os.path.exists(os.path.join(args.results, f[0]))]
    if figs:
        parts.append("### Figures\n")
        for name, cap in figs:
            parts.append(f"![{name}](results/{name})\n\n_{cap}_\n")

    # A generated text sample, if one was captured.
    slim = os.path.join(args.results, "slim.json")
    sample = None
    if os.path.exists(slim):
        d = json.load(open(slim, encoding="utf-8"))
        sample = (d.get("models", {}).get("nolm") or {}).get("sample")
    if not sample:
        for pth in sorted(glob.glob(os.path.join(args.results, "*.json"))):
            rep = json.load(open(pth, encoding="utf-8"))
            if rep.get("name") == "nolm" and rep.get("sample"):
                sample = rep["sample"]
                break
    if sample:
        parts.append("### A sample from the operator model\n")
        parts.append("Seeded with `<page>\\n    <title>`, temperature 0.8. "
                     "Byte-level, ~30M parameters, ~105M bytes of training — it has "
                     "learned the MediaWiki XML skeleton and locally plausible "
                     "English, which is about what this budget buys:\n")
        parts.append("```\n" + sample[:900] + "\n```\n")

    body = "\n".join(parts) if parts else "_No results yet._"

    txt = open(args.readme, encoding="utf-8").read()
    if START not in txt:
        raise SystemExit(f"{args.readme} has no {START} marker")
    head = txt.split(START)[0]
    tail = txt.split(END)[1] if END in txt else txt.split(START)[1].split("\n---\n", 1)[-1]
    if END not in txt:
        tail = "\n---\n" + tail
    new = f"{head}{START}\n\n{body}\n\n{END}{tail}"
    open(args.readme, "w", encoding="utf-8").write(new)
    print(f"updated {args.readme} ({len(body)} chars of results)")


if __name__ == "__main__":
    main()
