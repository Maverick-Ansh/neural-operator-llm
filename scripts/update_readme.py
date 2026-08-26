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
    ("fig_bpb_tail.png", "Bits/byte on the final 512 bytes of the window, as the "
                         "evaluation context grows. All models trained at 2,048."),
    ("fig_copy_probe.png", "Bits saved on the second copy of a planted passage, "
                           "against the distance between the two copies."),
    ("fig_kernel.png", "The learned kernel sampled on a 2,048-point and a "
                       "16,384-point grid. One continuous function, two discretisations."),
    ("fig_cost.png", "Time and peak memory per forward pass."),
    ("fig_training.png", "Validation bits/byte during training. Equal token budget for every run."),
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

    summ = os.path.join(args.results, "summary.md")
    if os.path.exists(summ):
        parts.append(open(summ, encoding="utf-8").read().strip())
        parts.append("")

    figs = [f for f in FIGURES if os.path.exists(os.path.join(args.results, f[0]))]
    if figs:
        parts.append("### Figures\n")
        for name, cap in figs:
            parts.append(f"![{name}](results/{name})\n\n_{cap}_\n")

    # A generated text sample, if one was captured.
    for pth in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        rep = json.load(open(pth, encoding="utf-8"))
        if rep.get("name") == "nolm" and rep.get("sample"):
            s = rep["sample"][:900]
            parts.append("### A sample from the operator model\n")
            parts.append("Seeded with `<page>\\n    <title>`, temperature 0.8. "
                         "Byte-level, ~30M parameters, ~105M bytes of training:\n")
            parts.append("```\n" + s + "\n```\n")
            break

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
