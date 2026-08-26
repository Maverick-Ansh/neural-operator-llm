"""enwik8, at the byte level.

Why bytes
---------
Two reasons, both load-bearing for this project.

1. *Long context becomes necessary rather than decorative.* A byte is worth
   roughly a quarter of a BPE token, so 64K bytes is only ~16K tokens of
   ordinary text. Structure that a subword model would see inside its window is
   pushed out past it, which is precisely the regime we want to probe.

2. *There is no tokenizer to lose.* The vocabulary is the 256 byte values,
   defined by the ASCII/UTF-8 standard rather than by an artefact that has to be
   saved alongside the checkpoint. A checkpoint from this repo can always be
   decoded, by anyone, forever.

Splits follow the standard enwik8 protocol (Mahoney): the first 90M bytes are
train, the next 5M validation, the last 5M test. Results are reported in
bits per byte, which is directly comparable to the published literature.
"""

import hashlib
import os
import urllib.request
import zipfile

import numpy as np
import torch

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIK8_MD5 = "a1dcaf68b6c6a1e08c9a5e0d5fdb1cc0"     # advisory only, not enforced

TRAIN_BYTES = 90_000_000
VALID_BYTES = 5_000_000


def download_enwik8(data_dir="data"):
    """Fetch and unzip enwik8 (~36MB zipped, 100MB raw). Idempotent."""
    os.makedirs(data_dir, exist_ok=True)
    raw = os.path.join(data_dir, "enwik8")
    if os.path.exists(raw) and os.path.getsize(raw) > 99_000_000:
        return raw

    zpath = os.path.join(data_dir, "enwik8.zip")
    if not os.path.exists(zpath):
        print(f"downloading {ENWIK8_URL} ...", flush=True)
        urllib.request.urlretrieve(ENWIK8_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extract("enwik8", data_dir)
    print(f"enwik8 ready: {os.path.getsize(raw):,} bytes", flush=True)
    return raw


def prepare(data_dir="data"):
    """Materialise train/valid/test as uint8 .bin files. Returns their paths."""
    paths = {s: os.path.join(data_dir, f"{s}.bin") for s in ("train", "valid", "test")}
    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths.values()):
        return paths

    raw = download_enwik8(data_dir)
    with open(raw, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    splits = {
        "train": data[:TRAIN_BYTES],
        "valid": data[TRAIN_BYTES:TRAIN_BYTES + VALID_BYTES],
        "test": data[TRAIN_BYTES + VALID_BYTES:],
    }
    for name, arr in splits.items():
        arr.tofile(paths[name])
        print(f"{name}: {len(arr):,} bytes -> {paths[name]}", flush=True)
    return paths


class ByteData:
    """Memory-mapped byte stream with random contiguous sampling.

    The whole corpus never enters RAM; np.memmap lets us pull arbitrary 64K-byte
    windows out of a 90MB file at negligible cost, which is what makes the
    long-context evaluation sweep cheap.
    """

    def __init__(self, data_dir="data", device="cuda"):
        self.paths = prepare(data_dir)
        self.device = device
        self._mm = {s: np.memmap(p, dtype=np.uint8, mode="r")
                    for s, p in self.paths.items()}

    def __len__(self):
        return len(self._mm["train"])

    def size(self, split):
        return len(self._mm[split])

    def get_batch(self, split, batch_size, seq_len, generator=None):
        """Random contiguous windows. Returns (x, y) each (B, seq_len) int64."""
        arr = self._mm[split]
        hi = len(arr) - seq_len - 1
        ix = torch.randint(hi, (batch_size,), generator=generator).tolist()
        x = np.stack([arr[i:i + seq_len] for i in ix])
        y = np.stack([arr[i + 1:i + 1 + seq_len] for i in ix])
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def sequential_windows(self, split, seq_len, max_windows=None, stride=None):
        """Yield non-overlapping (x, y) windows for deterministic evaluation.

        Non-overlapping windows mean every byte is predicted exactly once, so
        the reported bits/byte is an honest average over the split rather than
        an average that silently over-weights easy positions.
        """
        arr = self._mm[split]
        stride = stride or seq_len
        n = 0
        i = 0
        while i + seq_len + 1 <= len(arr):
            x = torch.from_numpy(arr[i:i + seq_len].astype(np.int64))[None]
            y = torch.from_numpy(arr[i + 1:i + 1 + seq_len].astype(np.int64))[None]
            yield x.to(self.device), y.to(self.device)
            n += 1
            i += stride
            if max_windows and n >= max_windows:
                return

    def raw(self, split, start, length):
        """Raw bytes as a numpy array -- used to build the retrieval probes."""
        return np.array(self._mm[split][start:start + length])


def decode(t):
    """Tensor/array of byte values -> str, replacing anything undecodable."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return bytes(np.asarray(t, dtype=np.uint8).tolist()).decode("utf-8", errors="replace")


def encode(s):
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long)
