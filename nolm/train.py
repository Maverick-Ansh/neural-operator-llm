"""Training loop.

Designed around two constraints of the environment it runs in:

* **A free Colab T4 can vanish at any moment.** Everything is checkpointed on a
  wall-clock timer, and `--resume` picks the run back up from the last one. The
  metrics log is append-only JSONL flushed every step, so a killed run still
  leaves a complete record of everything it did.

* **The budget is wall-clock, not steps.** `--max-minutes` is the real stopping
  criterion; the LR schedule is therefore defined over a step budget estimated
  from a short timing probe, so the cosine decay actually completes.

Precision note: the GPU is a T4 (compute capability 7.5). It reports
`is_bf16_supported() == True`, but bf16 there is emulated rather than run on the
tensor cores and is several times slower. fp16 with a gradient scaler is the
correct choice on this hardware.
"""

import argparse
import json
import math
import os
import time

import torch

from .data import ByteData
from .model import NOLM, NOLMConfig


def lr_at(step, total, base_lr, warmup, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    p = min(max(p, 0.0), 1.0)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def make_optimizer(model, lr, weight_decay, betas):
    """Decay matrices, do not decay gains/biases/kernels.

    The spectral coefficients and the decay envelope are excluded on purpose:
    weight decay on Fourier modes pulls the kernel towards the zero operator,
    which is a strictly worse inductive bias than leaving it free.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(("log_decay", "D")) or "op.weight" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)


@torch.no_grad()
def quick_eval(model, data, seq_len, iters, device, amp_dtype):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = data.get_batch("valid", 1, seq_len)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    nats = tot / iters
    return nats, nats / math.log(2)          # (nats/byte, bits/byte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="nolm", choices=["nolm", "transformer", "local"])
    ap.add_argument("--op-mode", default="stretch", choices=["stretch", "fixed"])
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=10)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--n-modes", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-minutes", type=float, default=75.0)
    ap.add_argument("--max-steps", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--ckpt-every-min", type=float, default=8.0)
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    amp_dtype = torch.float16                       # see module docstring

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "metrics.jsonl")
    ckpt_path = os.path.join(args.out, "ckpt.pt")

    cfg = NOLMConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                     train_len=args.seq_len, variant=args.variant, window=args.window,
                     n_modes=args.n_modes, op_mode=args.op_mode, dropout=args.dropout)
    model = NOLM(cfg).to(device)
    opt = make_optimizer(model, args.lr, args.weight_decay, (0.9, args.beta2))
    scaler = torch.amp.GradScaler("cuda")
    data = ByteData(args.data_dir, device=device)

    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"]
        print(f"resumed from step {start_step}", flush=True)

    n_params = model.num_params()
    print(json.dumps({"event": "config", "variant": args.variant,
                      "params": n_params, "params_non_embed": model.num_params(True),
                      "breakdown": model.param_breakdown(), "cfg": cfg.to_dict()}), flush=True)

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum

    # ---- timing probe: convert the wall-clock budget into a step budget so the
    # ---- cosine schedule is defined over the run we are actually going to do.
    model.train()
    probe_start = time.time()
    for _ in range(6):
        x, y = data.get_batch("train", args.batch_size, args.seq_len)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(x, y)
        scaler.scale(loss / args.grad_accum).backward()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    sec_per_micro = (time.time() - probe_start) / 6
    est_step_s = sec_per_micro * args.grad_accum
    total_steps = min(args.max_steps, max(50, int(args.max_minutes * 60 / est_step_s * 0.92)))
    print(json.dumps({"event": "budget", "sec_per_step_est": round(est_step_s, 4),
                      "total_steps": total_steps,
                      "tokens_planned": total_steps * tokens_per_step}), flush=True)

    logf = open(log_path, "a", buffering=1)
    logf.write(json.dumps({"event": "start", "args": vars(args), "params": n_params,
                           "total_steps": total_steps}) + "\n")

    t0 = time.time()
    last_ckpt = t0
    step = start_step
    running = None

    while step < total_steps:
        lr = lr_at(step, total_steps, args.lr, args.warmup, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = data.get_batch("train", args.batch_size, args.seq_len)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(x, y)
            scaler.scale(loss / args.grad_accum).backward()
            loss_acc += loss.item() / args.grad_accum

        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        running = loss_acc if running is None else 0.95 * running + 0.05 * loss_acc
        step += 1

        if step % 20 == 0 or step == 1:
            el = time.time() - t0
            rec = {"event": "train", "step": step, "loss": round(loss_acc, 4),
                   "ema": round(running, 4), "bpb": round(running / math.log(2), 4),
                   "lr": round(lr, 6), "gnorm": round(float(gnorm), 3),
                   "tokens": step * tokens_per_step, "elapsed_s": round(el, 1),
                   "tok_per_s": int(step * tokens_per_step / max(el, 1e-9))}
            logf.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)

        if step % args.eval_every == 0:
            nats, bpb = quick_eval(model, data, args.seq_len, args.eval_iters, device, amp_dtype)
            rec = {"event": "valid", "step": step, "val_nats": round(nats, 4),
                   "val_bpb": round(bpb, 4), "elapsed_s": round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)

        now = time.time()
        if now - last_ckpt > args.ckpt_every_min * 60 or step >= total_steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "step": step,
                        "cfg": cfg.to_dict(), "args": vars(args)}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            last_ckpt = now
            print(json.dumps({"event": "ckpt", "step": step}), flush=True)

        if now - t0 > args.max_minutes * 60:
            print(json.dumps({"event": "time_budget_reached", "step": step}), flush=True)
            break

    nats, bpb = quick_eval(model, data, args.seq_len, 50, device, amp_dtype)
    # Weights-only final artefact: small, and all that evaluation needs.
    torch.save({"model": model.state_dict(), "step": step, "cfg": cfg.to_dict(),
                "args": vars(args), "final_val_bpb": bpb},
               os.path.join(args.out, "final.pt"))
    rec = {"event": "done", "step": step, "final_val_bpb": round(bpb, 4),
           "minutes": round((time.time() - t0) / 60, 2),
           "tokens": step * tokens_per_step}
    logf.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    logf.close()


if __name__ == "__main__":
    main()
