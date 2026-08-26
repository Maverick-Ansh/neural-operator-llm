"""Evaluation: does the operator actually *use* a context it never trained on?

Three measurements, in increasing order of how hard they are to fake.

1. bits/byte vs evaluation length
   Reported two ways, because the usual single number hides the effect.
   `bpb_all` averages over every position in the window -- but early positions
   in a long window have almost no context no matter how good the model is, so
   `bpb_all` is dominated by them and moves very little. `bpb_tail` averages
   only over the final 512 bytes, i.e. the positions that actually have ~L bytes
   of context available. `bpb_tail` falling as L grows is the signal we want;
   `bpb_tail` rising means the long context is actively hurting.

2. Long-range copy probe
   A model that merely *tolerates* long context looks identical to one that
   *uses* it under (1) -- a bounded-window model quietly ignoring everything
   past W has a perfectly flat curve. So we plant a passage, separate the two
   copies by a controlled distance, and measure how many bits the second copy
   costs relative to the first. Retrieval across the gap is the only way to
   score well, and the gap is swept far past the training length.

3. Cost vs length
   Wall-clock and peak memory. This is where O(N log N) stops being a claim on
   paper.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import ByteData, decode
from .model import NOLM, NOLMConfig

LN2 = math.log(2)


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = NOLMConfig(**ck["cfg"])
    model = NOLM(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg, ck


@torch.no_grad()
def per_position_loss(model, x, y, amp=torch.float16):
    """(1, N) tensor of per-byte losses in nats."""
    with torch.autocast("cuda", dtype=amp):
        logits, _ = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                           y.reshape(-1), reduction="none").view(y.shape)


# --------------------------------------------------------------------------- #
# 1. bits/byte as a function of evaluation length
# --------------------------------------------------------------------------- #
@torch.no_grad()
def bpb_vs_length(model, data, lengths, byte_budget=1_048_576, tail=512, split="valid"):
    out = []
    for L in lengths:
        n_windows = max(1, byte_budget // L)
        tot_all, cnt_all, tot_tail, cnt_tail = 0.0, 0, 0.0, 0
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        oom = False
        try:
            for x, y in data.sequential_windows(split, L, max_windows=n_windows):
                pl = per_position_loss(model, x, y)
                tot_all += pl.sum().item(); cnt_all += pl.numel()
                t = min(tail, L)
                tot_tail += pl[:, -t:].sum().item(); cnt_tail += t
        except torch.cuda.OutOfMemoryError:
            oom = True
            torch.cuda.empty_cache()

        rec = {"length": L, "windows": n_windows, "oom": oom}
        if not oom:
            rec.update({
                "bpb_all": round(tot_all / cnt_all / LN2, 4),
                "bpb_tail": round(tot_tail / cnt_tail / LN2, 4),
                "sec_per_window": round((time.time() - t0) / n_windows, 4),
                "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            })
        out.append(rec)
        print(json.dumps(rec), flush=True)
    return out


# --------------------------------------------------------------------------- #
# 2. the long-range copy probe
# --------------------------------------------------------------------------- #
@torch.no_grad()
def copy_probe(model, data, separations, passage=256, lead=128, trials=24,
               split="valid", seed=0):
    """Plant a passage twice, `sep` bytes apart, and price the second copy.

    Layout:   [ lead filler ][ P ][ gap filler ][ P ]
    We score the same bytes of P in both positions. The first copy is the
    control: identical text, identical local context statistics, but nothing to
    retrieve. The difference is therefore attributable to retrieval across the
    gap and not to P being intrinsically easy.

    Filler is drawn from a distant region of the split so it cannot itself
    contain the passage.
    """
    rng = np.random.default_rng(seed)
    n = data.size(split)
    results = []

    for sep in separations:
        gap = sep - passage
        if gap < 0:
            continue
        total = lead + passage + gap + passage
        if total + 8 > n:
            print(json.dumps({"separation": sep, "skipped": "split too short"}), flush=True)
            continue

        first_bits, second_bits, ok = [], [], 0
        try:
            for _ in range(trials):
                p_start = int(rng.integers(0, n - passage - 2))
                f_start = int(rng.integers(0, n - (lead + gap) - 2))
                P = data.raw(split, p_start, passage)
                lead_f = data.raw(split, f_start, lead)
                gap_f = data.raw(split, f_start + lead, gap)

                seq = np.concatenate([lead_f, P, gap_f, P]).astype(np.int64)
                x = torch.from_numpy(seq[:-1])[None].to("cuda")
                y = torch.from_numpy(seq[1:])[None].to("cuda")
                pl = per_position_loss(model, x, y)[0]          # (total-1,)

                # loss index j predicts seq[j+1]; the copies start at these offsets
                a, b = lead, lead + passage + gap
                first = pl[a:a + passage - 1].mean().item()
                second = pl[b:b + passage - 1].mean().item()
                first_bits.append(first / LN2); second_bits.append(second / LN2)
                ok += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(json.dumps({"separation": sep, "oom": True}), flush=True)
            results.append({"separation": sep, "oom": True})
            continue

        f, s = float(np.mean(first_bits)), float(np.mean(second_bits))
        rec = {"separation": sep, "total_len": total, "trials": ok,
               "bpb_first_copy": round(f, 4), "bpb_second_copy": round(s, 4),
               "copy_gain_bits": round(f - s, 4),
               "copy_gain_stderr": round(float(np.std(np.array(first_bits) - np.array(second_bits),
                                                      ddof=1) / max(np.sqrt(ok), 1)), 4)}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    return results


# --------------------------------------------------------------------------- #
# 3. cost vs length
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cost_vs_length(model, lengths, reps=3):
    out = []
    for L in lengths:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, 256, (1, L), device="cuda")
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)                                  # warm up kernels
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(reps):
                with torch.autocast("cuda", dtype=torch.float16):
                    model(x)
            torch.cuda.synchronize()
            rec = {"length": L, "sec_per_forward": round((time.time() - t0) / reps, 4),
                   "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                   "oom": False}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rec = {"length": L, "oom": True}
        out.append(rec); print(json.dumps(rec), flush=True)
    return out


# --------------------------------------------------------------------------- #
# 4. what the learned kernel looks like at two resolutions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def kernel_snapshot(model, lengths=(2048, 16384), n_channels=6):
    """Dump kappa(t) at two grids to show it is one continuous function.

    Undoes the 1/grid quadrature weight so the two curves are directly
    comparable; if the parameterisation is doing what it claims, the coarse
    curve lies exactly on top of the fine one.
    """
    from .operators import SpectralOperatorMixer
    snaps = []
    for name, mod in model.named_modules():
        if isinstance(mod, SpectralOperatorMixer):
            entry = {"layer": name, "curves": {}}
            for L in lengths:
                k = mod.op.kernel(L, next(mod.parameters()).device).float() * L
                entry["curves"][str(L)] = k[:n_channels].cpu().numpy().tolist()
            entry["decay"] = F.softplus(mod.op.log_decay).detach().cpu().numpy().tolist()
            snaps.append(entry)
    return snaps


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--lengths", default="1024,2048,4096,8192,16384,32768,65536")
    ap.add_argument("--separations", default="512,1024,2048,4096,8192,16384,32768")
    ap.add_argument("--byte-budget", type=int, default=1_048_576)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--skip-copy", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lengths = [int(v) for v in args.lengths.split(",")]
    seps = [int(v) for v in args.separations.split(",")]

    model, cfg, ck = load_model(args.ckpt)
    data = ByteData(args.data_dir, device="cuda")
    print(json.dumps({"model": args.name, "params": model.num_params(),
                      "cfg": cfg.to_dict(), "train_step": ck.get("step")}), flush=True)

    report = {"name": args.name, "params": model.num_params(), "cfg": cfg.to_dict(),
              "train_step": ck.get("step"), "train_len": cfg.train_len}

    print("\n--- bits/byte vs evaluation length ---", flush=True)
    report["bpb_vs_length"] = bpb_vs_length(model, data, lengths,
                                           byte_budget=args.byte_budget, split=args.split)

    if not args.skip_copy:
        print("\n--- long-range copy probe ---", flush=True)
        report["copy_probe"] = copy_probe(model, data, seps, trials=args.trials,
                                          split=args.split)

    print("\n--- cost vs length ---", flush=True)
    report["cost_vs_length"] = cost_vs_length(model, lengths)

    if cfg.variant == "nolm":
        report["kernels"] = kernel_snapshot(model)

    # A short unconditional sample, purely as a qualitative sanity check.
    seed_text = "<page>\n    <title>"
    prompt = torch.tensor([[ord(c) for c in seed_text]], device="cuda")
    try:
        gen = model.generate(prompt, 400, temperature=0.8, top_k=50)
        report["sample"] = decode(gen[0])
    except Exception as e:
        report["sample"] = f"(generation failed: {e})"

    path = os.path.join(args.out, f"{args.name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
