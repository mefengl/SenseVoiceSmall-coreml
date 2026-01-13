#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy<=1.26.4",
#   "coremltools",
# ]
# ///

"""CoreML-only sanity check (no upstream SenseVoice repo needed).

This is intentionally lightweight:
- loads the .mlpackage
- runs a random forward pass
- checks output names/shapes and basic invariants

This does *not* prove semantic accuracy; it proves the artifact is runnable and self-consistent.
"""

from __future__ import annotations

import argparse

import coremltools as ct
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mlpackage", required=True)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--vocab", type=int, default=None, help="Optional expected vocab size")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    B = 1
    T = int(args.frames)
    F = 560

    speech = rng.standard_normal((B, T, F), dtype=np.float32)
    speech_lengths = np.array([T], dtype=np.int32)
    language = np.array([0], dtype=np.int32)
    textnorm = np.array([15], dtype=np.int32)

    mlmodel = ct.models.MLModel(args.mlpackage)
    out = mlmodel.predict(
        {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "language": language,
            "textnorm": textnorm,
        }
    )

    if "ctc_logits" not in out or "encoder_out_lens" not in out:
        raise SystemExit(f"Missing outputs. Got keys={list(out.keys())}")

    logits = np.array(out["ctc_logits"], dtype=np.float32)
    lens = np.array(out["encoder_out_lens"], dtype=np.int64)

    if logits.ndim != 3:
        raise SystemExit(f"ctc_logits must be rank-3, got shape={logits.shape}")
    if lens.shape != (1,):
        raise SystemExit(f"encoder_out_lens must be shape (1,), got shape={lens.shape}")

    if not np.isfinite(logits).all():
        raise SystemExit("ctc_logits contains NaN/Inf")

    if logits.shape[0] != 1:
        raise SystemExit(f"batch must be 1, got {logits.shape[0]}")

    coreml_T = int(logits.shape[1])
    coreml_V = int(logits.shape[2])
    L = int(lens[0])

    if args.vocab is not None and coreml_V != int(args.vocab):
        raise SystemExit(f"vocab mismatch: expected {args.vocab}, got {coreml_V}")

    if not (0 < L <= coreml_T):
        raise SystemExit(f"lens out of range: lens={L} logits_T={coreml_T}")

    print(f"logits shape: {logits.shape}")
    print(f"lens: {lens.tolist()}")
    print("OK")


if __name__ == "__main__":
    main()
